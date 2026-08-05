<claude-mem-context>
# Memory Context

# [daz-agent-sdk] recent context, 2026-05-22 6:22am GMT+10

No previous sessions found.
</claude-mem-context>

## Image generation architecture

- All SDK image generation goes through the single Mac mini `image_generation_service` on port 8830. Do not add provider-specific CLI/model implementations back into the SDK; callers receive one stable capability while the service owns the backing image model and serialization.
- The SDK always dials the single canonical Mac mini IGS origin `http://10.0.0.46:8830`. No caller, host detection, environment variable, tunnel, or configuration may select another still-image service origin.
- A release is not verified by a successful PyPI upload alone. Exercise `generate_image` against the real service and validate the resulting image dimensions/format before publishing or pinning the version downstream.

<!-- >>> greenline >>> -->
## Greenline gate — how merges work here

This repo is gated by **greenline**. Read `docs/greenline.md` and `docs/DOCTRINE.md`
before writing code or tests.

**Invariants (never violate):**
- `main` == what prod runs == green, always.
- The canonical checkout is pristine — never edit it by hand.
- All work happens in worktrees branched from last-green.
- Every merge goes through the serialized gate: full `check` + real `deploy`.

**Your workflow:**
1. `greenline worktree <name>` — get a worktree at `/Volumes/Gumby/worktrees/greenline/daz-agent-sdk/<name>` on branch `gl/<name>`.
2. Do your work there. Co-design tests + code per docs/DOCTRINE.md (parallel-safe, namespaced, no global-state assertions, OS-assigned ports; never mock other services — make real calls fast with a content-addressed record/replay cache).
3. Commit in your worktree. Then `greenline submit` (from that worktree).
4. The gate squash-merges, runs `./run check`, fast-forwards `main`, runs `./run deploy`, and publishes. It rolls back prod automatically if deploy fails.
5. On success: `greenline done` to remove your worktree + branch.

**Never** commit or push on `main` — hooks hard-lock it (reference-transaction
cannot be bypassed with `--no-verify`; pre-commit/pre-push refuse too). Never edit the
canonical checkout. If the gate reports a conflict, rebase your worktree on
`main` and resubmit. If commits somehow reached `main` outside the
gate (legacy workflow, hotfix), run `greenline adopt` to gate them in place —
greenline never discards commits on `main`.

Diagnose with `greenline status` and `greenline doctor` (`--fix` to reconcile).
<!-- <<< greenline <<< -->
