"""Tests for src.models.train's pure-Python helpers (no torch/GPU required).

train.py defers all torch/transformers imports inside functions so this file
stays importable and testable on the lightweight local venv.
"""

import numpy as np
import pytest

from src.models.train import compute_pos_weight


def test_compute_pos_weight_matches_manual_calculation() -> None:
    # label 0: 2 positive / 2 negative -> pos_weight 1.0
    # label 1: 1 positive / 3 negative -> pos_weight 3.0
    y = np.array([[1, 0], [1, 0], [0, 0], [0, 1]])
    pos_weight = compute_pos_weight(y)
    assert pos_weight.tolist() == pytest.approx([1.0, 3.0])


def test_compute_pos_weight_rejects_all_zero_label() -> None:
    y = np.array([[1, 0], [0, 0], [1, 0]])  # label 1 has zero positives
    with pytest.raises(AssertionError):
        compute_pos_weight(y)
