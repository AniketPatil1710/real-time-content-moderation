"""PR-curve based per-label threshold selection -> configs/thresholds.json. Phase 4.

Per label, selects two operational thresholds from the test-set PR curve:
- block_threshold: max recall subject to precision >= target_block_precision
- flag_threshold: max recall subject to precision >= target_flag_precision (< target_block_precision)
Score bands: score < flag_threshold -> allow, flag_threshold <= score < block_threshold -> flag,
score >= block_threshold -> block. Thresholds are per-label (not a single global cutoff) since
each label's classifier has its own score distribution and base rate.

Also writes per-label PR curve plots and a combined (across all labels) worst-mistakes report.
matplotlib and transformers/torch imports are deferred inside functions, not at module level, so
the pure threshold-selection logic stays unit-testable on the lightweight local venv (no torch,
no matplotlib) — see train.py for the same pattern and Memory.md Decision #9.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.data.preprocess import load_label_names, load_training_config
from src.evaluation.evaluate import load_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

THRESHOLDS_PATH = Path("configs/thresholds.json")
PR_CURVES_DIR = Path("metrics/pr_curves")
ERROR_ANALYSIS_PATH = Path("metrics/error_analysis.md")
N_ERROR_EXAMPLES = 50


def _select_threshold(y_true_col: np.ndarray, y_prob_col: np.ndarray, target_precision: float) -> dict[str, Any]:
    """Max-recall threshold subject to precision >= target_precision on this label's PR curve.

    Falls back to the highest-precision point on the curve (flagged via `met_target: False`)
    if no threshold reaches target_precision — e.g. a label too rare/hard for the model to hit
    0.90 precision at any cutoff. Never silently reports a threshold as meeting a target it didn't.
    """
    precisions, recalls, curve_thresholds = precision_recall_curve(y_true_col, y_prob_col)
    # precision_recall_curve appends a final (precision=1, recall=0) point with no threshold; drop it.
    precisions, recalls = precisions[:-1], recalls[:-1]

    meets_target = precisions >= target_precision
    if meets_target.any():
        candidate_idx = np.flatnonzero(meets_target)
        best_idx = candidate_idx[np.argmax(recalls[candidate_idx])]
        met_target = True
    else:
        best_idx = int(np.argmax(precisions))
        met_target = False

    return {
        "threshold": float(curve_thresholds[best_idx]),
        "precision": float(precisions[best_idx]),
        "recall": float(recalls[best_idx]),
        "met_target": met_target,
    }


def select_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: list[str],
    block_precision: float,
    flag_precision: float,
) -> dict[str, Any]:
    """Per-label block_threshold (precision >= block_precision) and flag_threshold (>= flag_precision)."""
    result: dict[str, Any] = {}
    for i, label in enumerate(label_names):
        block = _select_threshold(y_true[:, i], y_prob[:, i], block_precision)
        flag = _select_threshold(y_true[:, i], y_prob[:, i], flag_precision)
        if not block["met_target"]:
            logger.warning(
                "Label '%s': no threshold reaches block precision target %.2f; using best achievable (%.4f)",
                label,
                block_precision,
                block["precision"],
            )
        if not flag["met_target"]:
            logger.warning(
                "Label '%s': no threshold reaches flag precision target %.2f; using best achievable (%.4f)",
                label,
                flag_precision,
                flag["precision"],
            )
        result[label] = {
            "block_threshold": block["threshold"],
            "block_precision": block["precision"],
            "block_recall": block["recall"],
            "flag_threshold": flag["threshold"],
            "flag_precision": flag["precision"],
            "flag_recall": flag["recall"],
        }
    return result


def save_thresholds(thresholds: dict[str, Any], block_precision: float, flag_precision: float, path: Path) -> None:
    """Write thresholds.json with the precision targets used, so the file is self-documenting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "block_precision_target": block_precision,
        "flag_precision_target": flag_precision,
        "labels": thresholds,
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved thresholds to %s", path)


def save_pr_curves(y_true: np.ndarray, y_prob: np.ndarray, label_names: list[str], output_dir: Path) -> None:
    """One PR-curve PNG per label, with the chosen thresholds' precision/recall marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    for i, label in enumerate(label_names):
        precisions, recalls, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
        fig, ax = plt.subplots()
        ax.plot(recalls, precisions)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR curve: {label}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        fig.savefig(output_dir / f"{label}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    logger.info("Saved %d PR curve plots to %s", len(label_names), output_dir)


def build_error_analysis(
    texts: list[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: list[str],
    thresholds: dict[str, Any],
    n: int = N_ERROR_EXAMPLES,
) -> str:
    """Markdown report of the n worst false positives + n worst false negatives, aggregated across all labels.

    "Worst" = most confidently wrong: highest predicted probability among false positives, lowest
    predicted probability among false negatives. Decisions use each label's own block_threshold.
    """
    rows = []
    for i, label in enumerate(label_names):
        threshold = thresholds[label]["block_threshold"]
        pred = y_prob[:, i] >= threshold
        for idx in range(len(texts)):
            true_label, pred_label = bool(y_true[idx, i]), bool(pred[idx])
            row = {"label": label, "text": texts[idx], "prob": float(y_prob[idx, i])}
            if pred_label and not true_label:
                rows.append({**row, "kind": "false_positive"})
            elif true_label and not pred_label:
                rows.append({**row, "kind": "false_negative"})

    false_positives = sorted((r for r in rows if r["kind"] == "false_positive"), key=lambda r: -r["prob"])[:n]
    false_negatives = sorted((r for r in rows if r["kind"] == "false_negative"), key=lambda r: r["prob"])[:n]

    lines = [
        "# Error Analysis: Worst Mistakes (Phase 4)",
        "",
        f"Aggregated across all {len(label_names)} labels, decisions at each label's `block_threshold` "
        f"from `configs/thresholds.json`. Top {n} shown per category, ranked by model confidence.",
        "",
        f"## Worst False Positives ({len(false_positives)} of {n} requested)",
        "",
        "| Label | Predicted Prob | Text |",
        "|---|---|---|",
    ]
    for r in false_positives:
        lines.append(f"| {r['label']} | {r['prob']:.4f} | {_escape_md(r['text'])} |")

    lines += [
        "",
        f"## Worst False Negatives ({len(false_negatives)} of {n} requested)",
        "",
        "| Label | Predicted Prob | Text |",
        "|---|---|---|",
    ]
    for r in false_negatives:
        lines.append(f"| {r['label']} | {r['prob']:.4f} | {_escape_md(r['text'])} |")

    return "\n".join(lines) + "\n"


def _escape_md(text: str) -> str:
    """Keep pipe characters and newlines from breaking the markdown table."""
    return text.replace("|", "\\|").replace("\n", " ")


def save_error_analysis(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    logger.info("Saved error analysis to %s", path)


def load_model_predict_fn(model_dir: Path, max_length: int, batch_size: int = 64):
    """Load a fine-tuned model + tokenizer from disk and wrap it as an evaluate()-style predict_fn."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.models.train import make_predict_fn

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    return make_predict_fn(model, tokenizer, max_length, batch_size)


def run_thresholds(
    processed_dir: Path,
    model_dir: Path,
    thresholds_path: Path,
    pr_curves_dir: Path,
    error_analysis_path: Path,
    block_precision: float,
    flag_precision: float,
    n_error_examples: int,
) -> None:
    """End-to-end Phase 4: load model + test set, select thresholds, save PR curves + error analysis."""
    label_names = load_label_names()
    config = load_training_config()

    texts, y_true = load_split(processed_dir, "test", label_names)
    predict_fn = load_model_predict_fn(model_dir, config["model"]["max_length"])
    y_prob = predict_fn(texts)

    thresholds = select_thresholds(y_true, y_prob, label_names, block_precision, flag_precision)
    save_thresholds(thresholds, block_precision, flag_precision, thresholds_path)
    save_pr_curves(y_true, y_prob, label_names, pr_curves_dir)

    report = build_error_analysis(texts, y_true, y_prob, label_names, thresholds, n_error_examples)
    save_error_analysis(report, error_analysis_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/distilbert-finetuned"))
    parser.add_argument("--thresholds-path", type=Path, default=THRESHOLDS_PATH)
    parser.add_argument("--pr-curves-dir", type=Path, default=PR_CURVES_DIR)
    parser.add_argument("--error-analysis-path", type=Path, default=ERROR_ANALYSIS_PATH)
    parser.add_argument("--block-precision", type=float, default=0.90)
    parser.add_argument("--flag-precision", type=float, default=0.5)
    parser.add_argument("--n-error-examples", type=int, default=N_ERROR_EXAMPLES)
    args = parser.parse_args()
    run_thresholds(
        args.processed_dir,
        args.model_dir,
        args.thresholds_path,
        args.pr_curves_dir,
        args.error_analysis_path,
        args.block_precision,
        args.flag_precision,
        args.n_error_examples,
    )


if __name__ == "__main__":
    main()
