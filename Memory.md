# Memory.md — Project State (update at the END of every AI session)

> Purpose: lets a fresh AI session resume instantly without re-reading the codebase.
> The AI must update this file before ending any working session (see Rules.md).
> Keep it under ~150 lines — this is a summary, not a log. Prune stale detail.

## Current Status
- **Current phase:** Phase 2 done; Phase 3 (DistilBERT) code written, NOT yet run (needs Colab GPU)
- **Last session date:** 2026-07-20/21
- **Last thing completed:** `src/models/train.py` written — HF `Trainer` subclass (`WeightedTrainer`) with `BCEWithLogitsLoss(pos_weight=...)`, pos_weight computed from real training-set label frequencies via `compute_pos_weight()`, early stopping on val macro-F1 (`configs/training.yaml: training.early_stopping_metric/patience`), saves to `models/distilbert-finetuned/`, evaluates on test via the SAME `src.evaluation.evaluate.evaluate()` used for the baseline, and logs a loud pass/fail comparison against `metrics/baseline.json` (macro F1 must beat 0.4449 per Phases.md acceptance). All torch/transformers imports are deferred inside functions (not at module level) so the file still imports and its pure-logic helper (`compute_pos_weight`) is unit-testable on the lightweight local venv without installing torch — `tests/test_train.py` covers it. **NOT YET RUN** — this requires a GPU (Colab), which is outside this local session entirely. User needs to: upload `data/processed/*.parquet` to Drive, clone/upload the repo into Colab, `pip install -r requirements.txt` there (the heavy torch/transformers install, fine on Colab), then `python -m src.models.train`.
- **Next immediate task:** Run Phase 3 training in Colab (see above), then come back with `metrics/distilbert.json` + `models/distilbert-finetuned/` results to confirm and move to Phase 4.

## Phase Checklist
- [x] Phase 0 — Skeleton
- [x] Phase 1 — Data pipeline
- [x] Phase 2 — Baseline (`metrics/baseline.json`: macro F1 0.4449, macro PR-AUC 0.4775)
- [ ] Phase 3 — DistilBERT fine-tune (code ready in `src/models/train.py`; awaiting a Colab GPU run from the user)
- [ ] Phase 4 — Evaluation & thresholds
- [ ] Phase 5 — ONNX + benchmark
- [ ] Phase 6 — FastAPI serving
- [ ] Phase 7 — Demo + README
- [ ] Phase 8 — Extras (optional)

## Key Numbers So Far (source of truth: metrics/*.json)
| Metric | Baseline | DistilBERT | ONNX-INT8 |
|--------|----------|------------|-----------|
| Macro F1 | 0.4449 | TBD | TBD |
| Macro PR-AUC | 0.4775 | TBD | TBD |
| Micro F1 | 0.5564 | TBD | TBD |
| Precision @ recall | (see per-label in metrics/baseline.json) | TBD | TBD |
| p95 latency (CPU, bs=1, len=128) | — | TBD | TBD |

## Decisions Made (append-only; never silently reverse — see Rules.md)
1. 2026-07-20: Civil Comments toxicity binarized at ≥0.5 (per `configs/training.yaml: data.civil_comments_binarize_threshold`) — matches Jigsaw 2018 convention, documented in `preprocess.py` module docstring.
2. 2026-07-20: Civil Comments' `identity_attack` continuous score is mapped onto the Jigsaw 2018 `identity_hate` label (same underlying concept, different naming) — see `CIVIL_COMMENTS_LABEL_MAP` in `preprocess.py`.
3. 2026-07-20: Civil Comments rows with a missing subtype score are treated as 0 (non-toxic on that subtype), since only a subset of comments received full subtype annotation. Count of fills is logged, not silently dropped.
4. 2026-07-20: `download.py` fetches only `train.csv` per competition (not the full competition bundle) — cuts disk usage from ~2.3GB to ~844MB by skipping test sets/annotation files `preprocess.py` never reads.
5. 2026-07-20: `id` column is force-cast to `str` immediately after each CSV load. Real data exposed a bug: Jigsaw 2018 ids are hex strings (e.g. `dd47c73e9e4e11ad`) while Civil Comments ids are pure digits; pandas left the merged column as mixed-type `object`, and pyarrow's `to_parquet` guessed `int64` from the numeric majority then crashed on the first hex id. Fixed in both loaders.
6. 2026-07-20: Fixed two bugs in `tests/test_preprocess.py` found by a real (slow, 26-min) pytest run: (a) `test_merge_and_dedupe_...` passed raw whitespace-only text directly to `merge_and_dedupe`, but blank-filtering to `""` is `clean_text`'s job (always called upstream in the real pipeline) — test now runs `clean_text` first. (b) `test_rarest_label_...` gave `threat` 1 positive while every other label had 0, making `threat` the MOST common label, not the rarest — `rarest_label()` correctly returned `toxic` (tied at 0, first in label order); test now gives every other label more positives than `threat`. Neither was a bug in the actual pipeline code — the real end-to-end run already succeeded before these were found.
7. 2026-07-20: Baseline evaluator threshold fixed at 0.5 for reporting (`evaluate()` default) — this is NOT the same as the operational allow/flag/block thresholds, which come from Phase 4's PR-curve analysis and live in `configs/thresholds.json`. Don't conflate the two.
8. 2026-07-21: `pos_weight` in `train.py` is the literal, uncapped `n_negative/n_positive` per label, straight from Phases.md's spec ("pos_weight from training-set label frequencies") — NOT capped or sqrt-scaled. Given `severe_toxic`'s ~0.08% positive rate in train, expect a pos_weight around 1200+ for it. This could cause training instability (huge gradients from rare-label examples); if that happens in Colab, it's a known risk to investigate (try capping pos_weight, e.g. `np.minimum(pos_weight, some_max)`) rather than a surprise bug.
9. 2026-07-21: `train.py` defers ALL torch/transformers imports inside functions (never at module top-level) specifically so `compute_pos_weight` stays unit-testable on the lightweight local venv (`.venv_phase1`, no torch installed). Keep this pattern if train.py is edited — don't move torch imports to the top of the file.

## Known Issues / Tech Debt
- **`pytest` (and likely other pandas/sklearn-heavy scripts) genuinely takes ~26 minutes to run on this host, period — this is NOT a tool/sandbox-specific artifact.** Earlier this session it looked like automated/backgrounded runs were uniquely "stalled" while the user's own Terminal was fast; that theory was WRONG. A full `pytest -q` run timed end-to-end in the user's own Terminal took exactly 1580s (26m 19s) for just 18 tests. The host is simply very slow for this kind of workload (likely disk + memory pressure — see disk space note below). **Practical implication: don't try to "fix" the slowness or diagnose it further — it's real, budget for it.** When running anything nontrivial (pytest, preprocess, baseline training), either wait it out patiently in the background, or ask the user to kick it off in their own terminal and check back later — don't assume 10-15 min of low CPU means something is broken.
- `pytest` full-suite pass IS now confirmed (18 tests, all pass after fixing 2 test-only bugs — see Decisions #6). Don't re-run it reflexively given the 26-min cost; only re-run when something specific needs verifying.
- Host disk space is tight (has fluctuated between ~400MB and ~6.6GB free over this project's sessions — user has been freeing space periodically). `data/raw/` now holds ~844MB (both `train.csv` files only). Check `df -h /` before any large local install or download.

## Environment Notes
- Training: Colab (GPU type: TBD). Inference benchmarks: (CPU model: TBD — record exactly, it goes in the README).
- Kaggle credentials: `~/.kaggle/kaggle.json`, built from username `aniketpatil0904` + a newer-style `KGAT_`-prefixed API token (Kaggle's classic `kaggle.json` schema still works with these tokens in the `key` field). Never committed. Both competition rules accepted by the user.
- Local verification uses a lightweight venv (`.venv_phase1`, gitignored) with just `pandas numpy pyarrow scikit-learn pytest pyyaml kaggle tqdm` — NOT the full `requirements.txt` (torch/transformers/onnxruntime are only needed from Phase 3 onward and are heavy; install those only when actually training).

## File Map Quick Reference (update if structure drifts from Architecture.md)
- Entry points: `src/data/preprocess.py`, `src/models/baseline.py`, `src/models/train.py`, `src/models/export_onnx.py`, `src/evaluation/benchmark.py`, `src/serving/app.py`, `demo/streamlit_app.py`
- Shared library (no CLI, imported by entry points): `src/evaluation/evaluate.py` (model-agnostic `evaluate()`, used by baseline now, will be reused unchanged for DistilBERT/ONNX)
- Configs: `configs/training.yaml` (now also has a `baseline:` section for TF-IDF/LogReg hyperparams), `configs/thresholds.json`, `configs/labels.json`
