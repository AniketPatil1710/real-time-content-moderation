# ModGuard — Real-Time Toxic Content Moderation API

A production-style ML service that classifies social media text as toxic or
policy-violating in under 50ms, using a fine-tuned DistilBERT model optimized
with ONNX Runtime.

> **Status:** Phase 0 (project skeleton) complete. See [Phases.md](Phases.md) for the
> build plan and [Memory.md](Memory.md) for current progress. Results table below
> will be filled in once Phase 5 benchmarks exist — see [Rules.md](Rules.md).

## Docs

- [PRD.md](PRD.md) — goals, scope, success metrics
- [Architecture.md](Architecture.md) — system design, folder structure, tech stack
- [Design.md](Design.md) — demo UI design
- [Phases.md](Phases.md) — build order and acceptance criteria
- [Rules.md](Rules.md) — engineering conventions and guardrails
- [Memory.md](Memory.md) — session-to-session project state

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Results table (baseline vs PyTorch vs ONNX vs ONNX-INT8) will be added once
Phase 5 produces real benchmark numbers.
