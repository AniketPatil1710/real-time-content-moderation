# Rules.md — Boundaries for the AI Coding Assistant

## Libraries

**Allowed (only these, unless I explicitly approve additions):**
- `transformers`, `optimum[onnxruntime]`, `onnxruntime`, `torch`, `datasets`
- `scikit-learn`, `pandas`, `numpy`, `pyarrow`
- `fastapi`, `uvicorn`, `pydantic`
- `streamlit`, `matplotlib`
- `pytest`, `httpx` (for API tests)
- `kaggle` (dataset download), `pyyaml`, `tqdm`

**Forbidden:**
- TensorFlow / Keras (single-framework project — PyTorch only)
- `pypdf`, LangChain, LlamaIndex, or any LLM-orchestration framework (this is a classifier, not an LLM app)
- Heavy experiment trackers (MLflow, W&B) unless I ask — metrics go to JSON files
- Any paid API or service
- `pickle` for anything that crosses a trust boundary; use `safetensors`/ONNX for models, parquet/JSON for data

## Code Conventions

- Python 3.10+, type hints on all function signatures, docstrings on public functions.
- Every script runnable as `python -m src.module.name --args` with `argparse`; no logic that only works inside a notebook.
- All hyperparameters, paths, and thresholds come from `configs/` — **no magic numbers in code**.
- Set and log random seeds (`torch`, `numpy`, `random`) in every training/eval script. Reproducibility is a stated project goal.
- `logging` module, not `print`, in `src/` (print is fine in notebooks).
- Keep functions under ~50 lines; split when longer.

## Error Handling

- FastAPI: validate input with Pydantic (`text`: non-empty, max 5000 chars). Return 422 for bad input, 503 if model not loaded, 500 with a generic message otherwise — never leak stack traces in responses.
- Data pipeline: fail loudly and early (assert expected columns, row counts, label ranges after each step). A silent data bug invalidates every downstream metric.
- Inference wrapper: catch tokenizer/runtime errors and return a structured error, never crash the server.
- Never use bare `except:`. Catch specific exceptions.

## Metrics Integrity (most important section)

- **Never fabricate, estimate, or placeholder a metric.** If a number isn't computed yet, write `TBD` — never an invented value.
- Every number in README must map to a JSON file in `metrics/` produced by a script in `src/`.
- Latency benchmarks must include: hardware description, batch size, sequence length, warmup count, number of runs, and p50/p95/p99 — all logged into the output JSON.
- Precision must always be reported alongside recall and the threshold used.
- When comparing models, use the identical test set and identical evaluation code.

## What the AI Should Do

- Follow Phases.md strictly — complete and verify the current phase before touching the next.
- Write the test alongside the feature, not after.
- Update Memory.md at the end of every working session (see Memory.md format).
- Ask before: adding a dependency, changing the folder structure, changing a config schema, or deleting files.
- When uncertain about a design choice, present 2 options with tradeoffs instead of silently picking one.

## What the AI Should NOT Do

- Do not skip ahead to serving/demo before evaluation is done.
- Do not "improve" metrics by changing the test set, threshold, or metric definition without flagging it explicitly.
- Do not commit `data/`, `models/`, `.env`, or Kaggle credentials. Kaggle API key comes from the standard `~/.kaggle/kaggle.json` location or env vars only.
- Do not write toxic example strings directly into code or tests beyond the minimal necessary mild examples; load edge-case test inputs from the dataset instead.
- Do not refactor working code from previous phases unless the current phase requires it.
- Do not generate the README results table until Phase 5 benchmarks exist.

## Definition of Done (per phase)

A phase is done only when: (1) its scripts run end-to-end from a clean state, (2) its tests pass, (3) its outputs/artifacts exist where Architecture.md says they should, (4) Memory.md is updated.
