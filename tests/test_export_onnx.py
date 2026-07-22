"""Tests for src.models.export_onnx. Only the pure accuracy-gate logic — no torch/optimum
needed, so this runs on the lightweight local venv (see Memory.md Decision #9 for the pattern)."""

import json
from pathlib import Path

import pytest

from src.models.export_onnx import compare_to_pytorch, f1_drop_within_budget


def test_f1_drop_within_budget_true_when_drop_small() -> None:
    assert f1_drop_within_budget(quantized_f1=0.549, pytorch_f1=0.5513, max_drop=0.01) is True


def test_f1_drop_within_budget_false_when_drop_large() -> None:
    assert f1_drop_within_budget(quantized_f1=0.53, pytorch_f1=0.5513, max_drop=0.01) is False


def test_f1_drop_within_budget_exact_boundary_is_not_within_budget() -> None:
    # drop == max_drop exactly should NOT count as "within budget" (strict <, not <=).
    assert f1_drop_within_budget(quantized_f1=0.5413, pytorch_f1=0.5513, max_drop=0.01) is False


def test_compare_to_pytorch_missing_pytorch_metrics_does_not_raise(tmp_path: Path) -> None:
    compare_to_pytorch({"macro_f1": 0.5}, tmp_path / "nonexistent.json")


def test_compare_to_pytorch_reads_and_compares(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    pytorch_metrics_path = tmp_path / "distilbert.json"
    pytorch_metrics_path.write_text(json.dumps({"macro_f1": 0.5513}))

    with caplog.at_level("INFO"):
        compare_to_pytorch({"macro_f1": 0.549}, pytorch_metrics_path)
    assert "within budget" in caplog.text

    caplog.clear()
    with caplog.at_level("ERROR"):
        compare_to_pytorch({"macro_f1": 0.50}, pytorch_metrics_path)
    assert "exceeds the" in caplog.text
