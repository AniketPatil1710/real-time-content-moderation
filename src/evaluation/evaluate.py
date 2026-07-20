"""Model-agnostic evaluator: takes a predict function, scores any model. Phase 2.

`evaluate()` takes a predict_fn (texts -> per-label probability matrix), not a
specific model type, so the exact same code scores the TF-IDF baseline,
PyTorch DistilBERT, and ONNX models — guaranteeing the comparison table is
apples-to-apples (see Rules.md: identical test set, identical eval code).
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PredictFn = Callable[[list[str]], np.ndarray]


def evaluate(
    predict_fn: PredictFn,
    texts: list[str],
    y_true: np.ndarray,
    label_names: list[str],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Score predict_fn's output against y_true at a fixed decision threshold.

    Returns per-label precision/recall/F1/PR-AUC plus micro/macro F1 and
    macro PR-AUC. `threshold` here is a fixed reporting threshold (0.5);
    the operational allow/flag/block thresholds are selected separately in
    Phase 4 from PR curves and never hardcoded.
    """
    y_prob = predict_fn(texts)
    assert y_prob.shape == y_true.shape, f"predict_fn output shape {y_prob.shape} != y_true shape {y_true.shape}"
    y_pred = (y_prob >= threshold).astype(int)

    per_label = {}
    for i, label in enumerate(label_names):
        per_label[label] = {
            "precision": float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "recall": float(recall_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "f1": float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "pr_auc": float(average_precision_score(y_true[:, i], y_prob[:, i])),
        }

    return {
        "threshold": threshold,
        "n_examples": len(texts),
        "per_label": per_label,
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_pr_auc": float(np.mean([per_label[label]["pr_auc"] for label in label_names])),
    }


def load_split(processed_dir: Path, split: str, label_names: list[str]) -> tuple[list[str], np.ndarray]:
    """Load one split's texts and label matrix from processed parquet."""
    df = pd.read_parquet(processed_dir / f"{split}.parquet")
    texts = df["comment_text"].tolist()
    y = df[label_names].to_numpy(dtype=int)
    return texts, y


def save_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    """Write a metrics dict to JSON, creating parent dirs if needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", output_path)
