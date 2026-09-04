# Changelog

## Unreleased

- `Conversation(max_tokens=...)` / `Agent.conversation(max_tokens=...)` / `say(max_tokens=...)` forward a completion budget to the provider on every turn, so reasoning models (arbiter default cap 4096) cannot crowd a long structured answer out of a conversation.

## 0.2.19

- Report expected CLI image validation, execution, and recovery failures as one actionable `Error: ...` line without a Python traceback.
- Retain legacy image provider, model, step, finite-timeout, and configuration controls only as fail-closed compatibility options that direct callers to the durable Mac mini Codex image service.
