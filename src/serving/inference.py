"""ONNX Runtime session wrapper: load once, predict. Phase 6.

Instantiated as a module-level singleton in app.py's lifespan handler
(Architecture.md Decision #1) — never per-request, since per-request loading
would blow the latency budget by 100x.

optimum/transformers/numpy imports are deferred inside ModerationModel so
this file stays importable, and `decide()` (pure threshold logic) stays
unit-testable, without the heavy deps installed — same pattern as
train.py/export_onnx.py/benchmark.py.
"""

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DECISION_SEVERITY = {"allow": 0, "flag": 1, "block": 2}


class InferenceError(Exception):
    """Raised when tokenization or model inference fails (Rules.md: never crash the server)."""


class ModerationModel:
    """Wraps a loaded ONNX session + tokenizer for single-text inference."""

    def __init__(self, model_dir: Path, file_name: str, max_length: int, label_names: list[str]) -> None:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = ORTModelForSequenceClassification.from_pretrained(
            str(model_dir), provider="CPUExecutionProvider", file_name=file_name
        )
        self.max_length = max_length
        self.label_names = label_names

    def predict(self, text: str) -> tuple[dict[str, float], float]:
        """Return (per-label scores, latency_ms) for one text."""
        import numpy as np

        start = time.perf_counter()
        try:
            encoding = self.tokenizer(
                text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt"
            )
            logits = self.model(**encoding).logits
            probs = 1 / (1 + np.exp(-logits.detach().cpu().numpy()[0]))
        except Exception as exc:
            logger.exception("Inference failed")
            raise InferenceError("Model inference failed") from exc
        latency_ms = (time.perf_counter() - start) * 1000
        scores = {label: float(probs[i]) for i, label in enumerate(self.label_names)}
        return scores, latency_ms


def decide(scores: dict[str, float], thresholds: dict[str, Any]) -> str:
    """Per-label decision from configs/thresholds.json, overall = most severe label.

    Per-label rather than a single global cutoff, since each label's PR-curve-derived
    thresholds differ (Memory.md Decision #10) — Architecture.md's "max label score"
    simplification is realized here as "most severe per-label decision wins".
    """
    overall = "allow"
    for label, score in scores.items():
        label_thresholds = thresholds["labels"][label]
        if score >= label_thresholds["block_threshold"]:
            label_decision = "block"
        elif score >= label_thresholds["flag_threshold"]:
            label_decision = "flag"
        else:
            label_decision = "allow"
        if DECISION_SEVERITY[label_decision] > DECISION_SEVERITY[overall]:
            overall = label_decision
    return overall
