"""Tests for src.evaluation.thresholds. Pure numpy/sklearn logic only — no torch or matplotlib
needed, so this runs on the lightweight local venv (see Memory.md Decision #9 for the pattern)."""

import numpy as np
import pytest

from src.evaluation.thresholds import build_error_analysis, save_thresholds, select_thresholds

LABELS = ["toxic", "insult"]


def test_select_thresholds_meets_target_precision() -> None:
    # toxic: 10 positives at high scores, 10 negatives at low scores -> perfectly separable.
    y_true = np.array([[1, 0]] * 10 + [[0, 0]] * 10)
    y_prob = np.array([[0.9, 0.1]] * 10 + [[0.1, 0.1]] * 10)

    result = select_thresholds(y_true, y_prob, LABELS, block_precision=0.90, flag_precision=0.5)

    assert result["toxic"]["block_precision"] >= 0.90
    assert result["toxic"]["block_recall"] == pytest.approx(1.0)
    assert result["toxic"]["flag_threshold"] <= result["toxic"]["block_threshold"]


def test_select_thresholds_falls_back_when_target_unreachable() -> None:
    # insult: positives and negatives fully overlap in score -> no threshold reaches 0.90 precision.
    y_true = np.array([[0, 1], [0, 0], [0, 1], [0, 0]])
    y_prob = np.array([[0.1, 0.5], [0.1, 0.5], [0.1, 0.5], [0.1, 0.5]])

    result = select_thresholds(y_true, y_prob, LABELS, block_precision=0.90, flag_precision=0.5)

    # Best achievable precision here is 0.5 (2 true positives out of 4 predicted positive at any cutoff).
    assert result["insult"]["block_precision"] == pytest.approx(0.5)


def test_select_thresholds_block_precision_at_least_flag_precision_target() -> None:
    y_true = np.array([[1, 0], [1, 0], [0, 0], [0, 0], [1, 0]])
    y_prob = np.array([[0.95, 0.1], [0.8, 0.1], [0.4, 0.1], [0.2, 0.1], [0.6, 0.1]])

    result = select_thresholds(y_true, y_prob, LABELS, block_precision=0.90, flag_precision=0.5)

    assert result["toxic"]["block_threshold"] >= result["toxic"]["flag_threshold"]


def test_save_thresholds_writes_precision_targets(tmp_path) -> None:
    thresholds = {"toxic": {"block_threshold": 0.8, "block_precision": 0.9, "block_recall": 0.6,
                             "flag_threshold": 0.3, "flag_precision": 0.5, "flag_recall": 0.9}}
    path = tmp_path / "thresholds.json"

    save_thresholds(thresholds, block_precision=0.90, flag_precision=0.5, path=path)

    import json
    with path.open() as f:
        payload = json.load(f)
    assert payload["block_precision_target"] == 0.90
    assert payload["flag_precision_target"] == 0.5
    assert payload["labels"] == thresholds


def test_build_error_analysis_ranks_by_confidence() -> None:
    texts = ["clearly toxic missed", "borderline fp", "obviously fine", "confidently wrong fp"]
    y_true = np.array([[1, 0], [0, 0], [0, 0], [0, 0]])
    y_prob = np.array([[0.05, 0.0], [0.6, 0.0], [0.1, 0.0], [0.95, 0.0]])
    thresholds = {"toxic": {"block_threshold": 0.5}, "insult": {"block_threshold": 0.5}}

    report = build_error_analysis(texts, y_true, y_prob, ["toxic", "insult"], thresholds, n=5)

    fp_section = report.split("Worst False Positives")[1].split("Worst False Negatives")[0]
    fn_section = report.split("Worst False Negatives")[1]
    assert "confidently wrong fp" in fp_section
    assert "clearly toxic missed" in fn_section
    # Most-confident FP ("confidently wrong fp", prob 0.95) should be listed before the weaker one.
    assert fp_section.index("confidently wrong fp") < fp_section.index("borderline fp")


def test_build_error_analysis_escapes_pipe_characters() -> None:
    texts = ["a | b"]
    y_true = np.array([[0]])
    y_prob = np.array([[0.9]])
    thresholds = {"toxic": {"block_threshold": 0.5}}

    report = build_error_analysis(texts, y_true, y_prob, ["toxic"], thresholds, n=5)

    assert "a \\| b" in report
