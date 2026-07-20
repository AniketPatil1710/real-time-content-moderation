"""Tests for src.evaluation.evaluate."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.evaluate import evaluate, load_split, save_metrics

LABELS = ["toxic", "insult"]


def test_evaluate_perfect_predictions_score_one() -> None:
    y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])

    def predict_fn(texts: list[str]) -> np.ndarray:
        return y_true.astype(float)

    result = evaluate(predict_fn, ["a", "b", "c", "d"], y_true, LABELS)

    for label in LABELS:
        assert result["per_label"][label]["precision"] == 1.0
        assert result["per_label"][label]["recall"] == 1.0
        assert result["per_label"][label]["f1"] == 1.0
        assert result["per_label"][label]["pr_auc"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["micro_f1"] == 1.0


def test_evaluate_known_precision_recall() -> None:
    # toxic: true [1,1,0,0], pred [1,0,0,1] -> TP=1, FP=1, FN=1 -> P=0.5, R=0.5, F1=0.5
    y_true = np.array([[1, 0], [1, 0], [0, 0], [0, 0]])
    y_prob = np.array([[0.9, 0.1], [0.2, 0.1], [0.1, 0.1], [0.8, 0.1]])

    def predict_fn(texts: list[str]) -> np.ndarray:
        return y_prob

    result = evaluate(predict_fn, ["a", "b", "c", "d"], y_true, LABELS, threshold=0.5)

    assert result["per_label"]["toxic"]["precision"] == pytest.approx(0.5)
    assert result["per_label"]["toxic"]["recall"] == pytest.approx(0.5)
    assert result["per_label"]["toxic"]["f1"] == pytest.approx(0.5)
    assert result["per_label"]["insult"]["precision"] == 0.0
    assert result["per_label"]["insult"]["recall"] == 0.0


def test_evaluate_shape_mismatch_raises() -> None:
    y_true = np.array([[1, 0], [0, 1]])

    def predict_fn(texts: list[str]) -> np.ndarray:
        return np.zeros((3, 2))  # wrong number of rows

    with pytest.raises(AssertionError):
        evaluate(predict_fn, ["a", "b"], y_true, LABELS)


def test_load_split_reads_texts_and_label_matrix(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "id": ["1", "2"],
            "comment_text": ["fine comment", "another comment"],
            "toxic": [1, 0],
            "insult": [0, 0],
            "source": ["synthetic", "synthetic"],
        }
    )
    df.to_parquet(tmp_path / "test.parquet", index=False)

    texts, y = load_split(tmp_path, "test", LABELS)

    assert texts == ["fine comment", "another comment"]
    assert y.tolist() == [[1, 0], [0, 0]]


def test_save_metrics_writes_valid_json(tmp_path: Path) -> None:
    metrics = {"macro_f1": 0.5, "per_label": {"toxic": {"precision": 1.0}}}
    output_path = tmp_path / "nested" / "metrics.json"

    save_metrics(metrics, output_path)

    assert output_path.exists()
    with output_path.open() as f:
        loaded = json.load(f)
    assert loaded == metrics
