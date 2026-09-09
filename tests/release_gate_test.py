"""Tests for the release gate's impact selection.

Impact selection decides which validation a candidate gets, so a wrong mapping
is a silently under-validated release. These tests pin the mapping: anything
unattributable must select every area.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType

GATE_PATH = Path(__file__).resolve().parent.parent / "release-gate"
RUN_PATH = Path(__file__).resolve().parent.parent / "run"
ALL_AREAS = {"scan", "python", "go", "image"}


# ##################################################################
# load gate module
# the gate is an extensionless executable; import it by explicit file location
def _load_gate() -> ModuleType:
    loader = SourceFileLoader("release_gate", str(GATE_PATH))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve annotations through sys.modules; register before exec.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GATE = _load_gate()


def test_unknown_diff_selects_everything() -> None:
    assert GATE.select_areas(None)[0] == ALL_AREAS
    assert GATE.select_areas(set())[0] == ALL_AREAS


def test_gate_or_config_change_selects_everything() -> None:
    for path in ("release-gate", "greenline.toml", "tests/release_gate_test.py"):
        areas, reason = GATE.select_areas({path})
        assert areas == ALL_AREAS, path
        assert "full" in reason


def test_python_source_change_selects_python_not_go() -> None:
    areas, _ = GATE.select_areas({"src/daz_agent_sdk/registry.py"})
    assert areas == {"scan", "python"}


def test_python_image_change_selects_image_canaries() -> None:
    areas, _ = GATE.select_areas({"src/daz_agent_sdk/capabilities/image.py"})
    assert areas == {"scan", "python", "image"}


def test_dependency_change_selects_python_and_image() -> None:
    for path in ("requirements.txt", "pyproject.toml"):
        areas, _ = GATE.select_areas({path})
        assert areas == {"scan", "python", "image"}, path


def test_go_source_change_selects_go_not_python() -> None:
    areas, _ = GATE.select_areas({"go/registry.go"})
    assert areas == {"scan", "go"}


def test_go_capability_change_selects_image_canaries() -> None:
    areas, _ = GATE.select_areas({"go/capability/image.go"})
    assert areas == {"scan", "go", "image"}


def test_documentation_change_selects_only_scans() -> None:
    areas, reason = GATE.select_areas({"README.md", "docs/DOCTRINE.md"})
    assert areas == {"scan"}
    assert reason == "documentation only"


def test_mixed_change_unions_areas() -> None:
    areas, _ = GATE.select_areas(
        {"src/daz_agent_sdk/registry.py", "go/registry.go", "README.md"}
    )
    assert areas == {"scan", "python", "go"}


def test_every_declared_area_has_phases(tmp_path: Path) -> None:
    for area in ("python", "go", "image"):
        phases = GATE.build_phases({"scan", area}, tmp_path, {})
        names = {phase.name for phase in phases}
        assert names, area
        if area == "python":
            assert {"python-tests", "python-types", "python-lint"} <= names
        if area == "go":
            assert {"go-tests", "go-vet", "go-build"} <= names
        if area == "image":
            assert {"real-remote-go-igs-healthz", "real-local-python-igs-healthz"} <= names


def test_phase_dependencies_all_resolve(tmp_path: Path) -> None:
    phases = GATE.build_phases(ALL_AREAS, tmp_path, {})
    names = {phase.name for phase in phases}
    for phase in phases:
        for need in phase.needs:
            assert need in names, f"{phase.name} needs unknown phase {need}"


def test_release_keeps_pypi_token_in_process_memory() -> None:
    source = RUN_PATH.read_text()
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    process_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "subprocess"
        and call.func.attr == "Popen"
    ]

    assert '"/Users/darrenoakey/.local/bin/daz-secrets"' in source
    assert "TWINE_PASSWORD" not in source
    assert "TWINE_USERNAME" not in source
    assert len(process_calls) == 1
    process_keywords = {keyword.arg: keyword.value for keyword in process_calls[0].keywords}
    assert ast.literal_eval(process_keywords["env"]) == {}
    assert ast.literal_eval(process_keywords["start_new_session"]) is True
    assert "stderr=subprocess.DEVNULL" in source
    assert "os.killpg(process.pid, signal.SIGKILL)" in source
    assert "4097 - len(output)" in source
    assert "Settings(username=\"__token__\", password=token, non_interactive=True)" in source
    assert "twine_upload.upload(" in source


def test_release_installs_and_verifies_with_same_global_python() -> None:
    source = RUN_PATH.read_text()
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

    install_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "subprocess"
        and call.func.attr == "run"
        and call.args
        and isinstance(call.args[0], ast.List)
        and any(
            isinstance(element, ast.JoinedStr)
            and "daz-agent-sdk==" in ast.unparse(element)
            for element in call.args[0].elts
        )
    ]
    verification_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "subprocess"
        and call.func.attr == "check_output"
    ]

    assert 'shutil.which("python3")' in source
    assert "str(Path(global_python_command).resolve())" in source
    assert len(install_calls) == 1
    install_command = install_calls[0].args[0]
    assert isinstance(install_command, ast.List)
    assert isinstance(install_command.elts[0], ast.Name)
    assert install_command.elts[0].id == "global_python"
    assert [ast.literal_eval(element) for element in install_command.elts[1:4]] == [
        "-m",
        "pip",
        "install",
    ]
    assert len(verification_calls) == 1
    verification_command = verification_calls[0].args[0]
    assert isinstance(verification_command, ast.List)
    assert isinstance(verification_command.elts[0], ast.Name)
    assert verification_command.elts[0].id == "global_python"
