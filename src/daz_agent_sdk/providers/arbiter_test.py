from __future__ import annotations

import json
import time

import pytest
from pydantic import BaseModel

from daz_agent_sdk.providers.arbiter import _KNOWN_TIERS, ArbiterProvider
from daz_agent_sdk.types import (
    AgentError,
    Capability,
    ErrorKind,
    Message,
    ModelInfo,
    Response,
    StructuredResponse,
    Tier,
)

# ##################################################################
# test model selection
# use the production semantic route so concrete model replacements do not
# invalidate behavior coverage.
TEST_MODEL_ID = "local-coder"


# ##################################################################
# helper — make model info
# build a ModelInfo for the named arbiter LLM. used when passing a
# model into provider methods without first calling list_models().
def _make_model(model_id: str = TEST_MODEL_ID) -> ModelInfo:
    return ModelInfo(
        provider="arbiter",
        model_id=model_id,
        display_name="Test Model",
        capabilities=frozenset({Capability.TEXT, Capability.STRUCTURED}),
        tier=Tier.FREE_THINKING,
    )


# ##################################################################
# known tiers — unit tests (no arbiter required)
# verify the hardcoded tier mapping covers the registered llm_names.
def test_known_tiers_contains_nemotron() -> None:
    assert _KNOWN_TIERS["nemotron-30b-a3b"] == Tier.FREE_FAST


def test_known_tiers_contains_ornith() -> None:
    assert _KNOWN_TIERS["ornith-1.5-35b"] == Tier.FREE_THINKING


def test_known_tiers_maps_category_aliases() -> None:
    assert _KNOWN_TIERS["local-chat"] == Tier.FREE_FAST
    assert _KNOWN_TIERS["local-summariser"] == Tier.SUMMARIES
    assert _KNOWN_TIERS["local-coder"] == Tier.FREE_THINKING
    assert _KNOWN_TIERS["local-extract"] == Tier.FREE_THINKING
    assert _KNOWN_TIERS["local-vision"] == Tier.FREE_THINKING


def test_known_tiers_keeps_concrete_model_backcompat() -> None:
    # concrete arbiter model IDs must still resolve to their historical tiers
    # even after the category-alias migration.
    assert _KNOWN_TIERS["qwen3.8-27b"] == Tier.FREE_THINKING
    assert _KNOWN_TIERS["qwen3.6-35b"] == Tier.FREE_THINKING
    assert _KNOWN_TIERS["gpt-oss-20b"] == Tier.FREE_FAST


# ##################################################################
# constructor — unit tests
def test_constructor_defaults_to_spark_arbiter() -> None:
    provider = ArbiterProvider()
    assert provider._base_url == "http://10.0.0.254:8400"


def test_constructor_strips_trailing_slash() -> None:
    provider = ArbiterProvider(base_url="http://example:9000/")
    assert provider._base_url == "http://example:9000"


# ##################################################################
# available — online tests
@pytest.mark.asyncio
async def test_available_returns_true(arbiter_tunnel_url: str) -> None:
    provider = ArbiterProvider(base_url=arbiter_tunnel_url)
    result = await provider.available()
    assert result is True


@pytest.mark.asyncio
async def test_available_wrong_port_returns_false() -> None:
    provider = ArbiterProvider(base_url="http://localhost:19999")
    result = await provider.available()
    assert result is False


# ##################################################################
# list models — online tests
@pytest.mark.asyncio
async def test_list_models_returns_model_info_instances(
    arbiter_tunnel_url: str,
) -> None:
    provider = ArbiterProvider(base_url=arbiter_tunnel_url)
    models = await provider.list_models()
    assert len(models) > 0
    for m in models:
        assert isinstance(m, ModelInfo)
        assert m.provider == "arbiter"
        assert m.model_id != ""
        assert Capability.TEXT in m.capabilities
        assert Capability.STRUCTURED in m.capabilities
        assert isinstance(m.tier, Tier)


@pytest.mark.asyncio
async def test_list_models_includes_current_models(arbiter_tunnel_url: str) -> None:
    provider = ArbiterProvider(base_url=arbiter_tunnel_url)
    models = await provider.list_models()
    names = {m.model_id for m in models}
    assert "nemotron-30b-a3b" in names
    assert "ornith-1.5-35b" in names


@pytest.mark.asyncio
async def test_list_models_wrong_port_returns_empty() -> None:
    provider = ArbiterProvider(base_url="http://localhost:19999")
    models = await provider.list_models()
    assert models == []


# ##################################################################
# complete — online tests
# generous timeouts so the first call can warm a cold model.
@pytest.mark.asyncio
async def test_complete_simple_prompt(arbiter_tunnel_url: str) -> None:
    provider = ArbiterProvider(base_url=arbiter_tunnel_url)
    messages = [
        Message(role="user", content="What is 2+2? Reply with just the number.")
    ]
    model = _make_model()
    resp = await provider.complete(messages, model, timeout=180.0)
    assert isinstance(resp, Response)
    assert "4" in resp.text
    assert resp.model_used is model
    assert resp.conversation_id is not None
    assert resp.turn_id is not None


@pytest.mark.asyncio
async def test_complete_with_system_message(arbiter_tunnel_url: str) -> None:
    provider = ArbiterProvider(base_url=arbiter_tunnel_url)
    messages = [
        Message(role="system", content="You are a terse assistant. Be brief."),
        Message(role="user", content="What is the capital of France?"),
    ]
    model = _make_model()
    resp = await provider.complete(messages, model, timeout=180.0)
    assert isinstance(resp, Response)
    assert "Paris" in resp.text


# ##################################################################
# structured output — online tests
@pytest.mark.asyncio
async def test_complete_structured_output(arbiter_tunnel_url: str) -> None:
    class MathAnswer(BaseModel):
        result: int
        explanation: str

    provider = ArbiterProvider(base_url=arbiter_tunnel_url)
    messages = [Message(role="user", content="What is 3 multiplied by 7?")]
    model = _make_model()
    resp = await provider.complete(messages, model, schema=MathAnswer, timeout=300.0)
    assert isinstance(resp, StructuredResponse)
    assert isinstance(resp.parsed, MathAnswer)
    assert resp.parsed.result == 21


# ##################################################################
# stream — online tests
@pytest.mark.asyncio
async def test_stream_yields_string_chunks(arbiter_tunnel_url: str) -> None:
    provider = ArbiterProvider(base_url=arbiter_tunnel_url)
    messages = [Message(role="user", content="Count from 1 to 5, one number per line.")]
    model = _make_model()
    chunks: list[str] = []
    async for chunk in provider.stream(messages, model, timeout=180.0):
        assert isinstance(chunk, str)
        chunks.append(chunk)
    assert len(chunks) > 0
    full = "".join(chunks)
    assert full.strip() != ""


@pytest.mark.asyncio
async def test_stream_produces_complete_response(arbiter_tunnel_url: str) -> None:
    provider = ArbiterProvider(base_url=arbiter_tunnel_url)
    messages = [
        Message(
            role="user", content="What is 10 divided by 2? Reply with just the number."
        )
    ]
    model = _make_model()
    full = ""
    async for chunk in provider.stream(messages, model, timeout=180.0):
        full += chunk
    assert "5" in full


# ##################################################################
# connection error handling
@pytest.mark.asyncio
async def test_complete_wrong_port_raises_agent_error() -> None:
    provider = ArbiterProvider(base_url="http://localhost:19999")
    messages = [Message(role="user", content="hello")]
    model = _make_model()
    with pytest.raises(AgentError) as exc_info:
        await provider.complete(messages, model, timeout=5.0)
    assert exc_info.value.kind == ErrorKind.NOT_AVAILABLE


@pytest.mark.asyncio
async def test_stream_wrong_port_raises_agent_error() -> None:
    provider = ArbiterProvider(base_url="http://localhost:19999")
    messages = [Message(role="user", content="hello")]
    model = _make_model()
    with pytest.raises(AgentError) as exc_info:
        async for _ in provider.stream(messages, model, timeout=5.0):
            pass
    assert exc_info.value.kind == ErrorKind.NOT_AVAILABLE


# ##################################################################
# local-coder — dedicated reasoning-model test
# verifies the reasoning→content fallback path with the default
# low/free_fast/free_thinking tier model. marked slow because a cold
# qwen load is ~10 minutes.
@pytest.mark.asyncio
async def test_complete_qwen_reasoning_model(arbiter_tunnel_url: str) -> None:
    provider = ArbiterProvider(base_url=arbiter_tunnel_url)
    messages = [
        Message(role="user", content="What is 2+2? Answer with just the number.")
    ]
    model = ModelInfo(
        provider="arbiter",
        model_id="local-coder",
        display_name="Local Coder",
        capabilities=frozenset({Capability.TEXT, Capability.STRUCTURED}),
        tier=Tier.FREE_THINKING,
    )
    resp = await provider.complete(messages, model, timeout=900.0)
    assert isinstance(resp, Response)
    # content may be null when max_tokens runs out mid-reasoning; our
    # provider falls back to the reasoning field so text is never empty
    assert resp.text.strip() != ""


# ##################################################################
# answer from message — reasoning-vs-content extraction (pure unit tests)
def test_answer_from_message_prefers_content() -> None:
    from daz_agent_sdk.providers.arbiter import _answer_from_message

    msg = {"content": "The story text.", "reasoning": "planning notes"}
    assert _answer_from_message(msg) == "The story text."


def test_answer_from_message_empty_content_with_reasoning_raises_retryable() -> None:
    # an interrupted generation has reasoning but no answer — that must be a
    # retryable failure, never silently handed back as the answer (observed
    # live: chain-of-thought saved as a novel section's prose).
    import pytest as _pytest

    from daz_agent_sdk.providers.arbiter import _answer_from_message
    from daz_agent_sdk.types import AgentError, ErrorKind

    msg = {"content": "", "reasoning": "1. Analyze the user input..."}
    with _pytest.raises(AgentError) as exc_info:
        _answer_from_message(msg)
    assert exc_info.value.kind == ErrorKind.INTERNAL


def test_answer_from_message_both_empty_returns_empty() -> None:
    from daz_agent_sdk.providers.arbiter import _answer_from_message

    assert _answer_from_message({}) == ""
    assert _answer_from_message({"content": "", "reasoning": ""}) == ""


# ##################################################################
# max_tokens plumbing — the parameter must actually reach the server
@pytest.mark.asyncio
async def test_complete_max_tokens_caps_output(arbiter_tunnel_url: str) -> None:
    provider = ArbiterProvider(base_url=arbiter_tunnel_url)
    assert await provider.available(), (
        "arbiter is unreachable through its loopback tunnel"
    )
    models = await provider.list_models()
    target = next((m for m in models if m.model_id == "qwen3.8-27b"), None)
    assert target is not None, "qwen3.8-27b is not registered"
    messages = [
        Message(
            role="user",
            content="Count from one to one thousand in words, without stopping.",
        )
    ]
    # a 32-token cap must be honored by the server: either a tiny visible
    # response, or (for a reasoning model) the thinking consumes the whole
    # budget and the empty-content guard raises — both prove the cap arrived,
    # since the un-capped default (4096) would produce a long response.
    try:
        result = await provider.complete(messages, target, timeout=300.0, max_tokens=32)
        assert len((result.text or "").split()) < 200
    except AgentError as exc:
        assert "no answer content" in str(exc)


# ##################################################################
# reasoning_effort="none" + schema — a reasoning model's structured answer
# must be the JSON object itself, produced without a thinking transcript.
# Live: nemotron through the arbiter (Ollama on boringstack),
# which with thinking on spends minutes and emits malformed JSON (2026-09-04).
class _Bounds(BaseModel):
    index: int
    framing: str
    face_slot: str


class _BoundsAnswer(BaseModel):
    boundary_images: list[_Bounds]


@pytest.mark.asyncio
async def test_complete_reasoning_none_with_schema_returns_pure_json(
    arbiter_tunnel_url: str,
) -> None:
    provider = ArbiterProvider(base_url=arbiter_tunnel_url)
    assert await provider.available(), (
        "arbiter is unreachable through its loopback tunnel"
    )
    models = await provider.list_models()
    target = next((m for m in models if m.model_id == "nemotron-30b-a3b"), None)
    assert target is not None, "nemotron-30b-a3b is not registered"
    messages = [
        Message(
            role="user",
            content=(
                'Return JSON only: {"boundary_images": [...]} holding exactly 3 objects '
                "for indices 0..2, each with index, framing (CU, MCU or MS) and face_slot "
                "(left, center or right)."
            ),
        )
    ]
    started = time.monotonic()
    result = await provider.complete(
        messages,
        target,
        timeout=300.0,
        max_tokens=2048,
        reasoning_effort="none",
        schema=_BoundsAnswer,
    )
    elapsed = time.monotonic() - started
    assert isinstance(result, StructuredResponse)
    assert [row.index for row in result.parsed.boundary_images] == [0, 1, 2]
    # the text IS the object: no thinking prose before it
    assert result.text.lstrip().startswith("{")
    json.loads(result.text)
    assert elapsed < 120, f"no-think structured answer took {elapsed:.0f}s"


class TestProvenance:
    def test_who_defaults_to_sdk_name(self, monkeypatch):
        monkeypatch.delenv("ARBITER_WHO", raising=False)
        monkeypatch.delenv("ARBITER_WHY", raising=False)
        provider = ArbiterProvider.__new__(ArbiterProvider)
        who, why = provider._provenance()
        assert who == "daz-agent-sdk"
        assert why is None

    def test_ambient_env_overrides_who_and_sets_why(self, monkeypatch):
        monkeypatch.setenv("ARBITER_WHO", "beezle3")
        monkeypatch.setenv("ARBITER_WHY", "hang williams video")
        provider = ArbiterProvider.__new__(ArbiterProvider)
        who, why = provider._provenance()
        assert who == "beezle3"
        assert why == "hang williams video"

    def test_values_are_capped(self, monkeypatch):
        monkeypatch.setenv("ARBITER_WHO", "x" * 300)
        monkeypatch.setenv("ARBITER_WHY", "y" * 300)
        provider = ArbiterProvider.__new__(ArbiterProvider)
        who, why = provider._provenance()
        assert who is not None and len(who) == 128
        assert why is not None and len(why) == 256
