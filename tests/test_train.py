"""Tests for src.models.train's pure-Python helpers (compute_pos_weight/compute_focal_alpha
need no torch/GPU) plus correctness checks for _make_loss_fn's focal loss math (needs torch,
which is installed in .venv_phase1 as of Phase 6 — see Memory.md Environment Notes).

train.py defers all torch/transformers imports inside functions so the module itself stays
importable without torch; only the _make_loss_fn tests below actually require it at runtime.
"""

import numpy as np
import pytest

from src.models.train import compute_focal_alpha, compute_pos_weight


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


def test_compute_focal_alpha_matches_manual_calculation() -> None:
    # label 0: 2 positive / 4 total -> alpha 0.5
    # label 1: 1 positive / 4 total -> alpha 0.75
    y = np.array([[1, 0], [1, 0], [0, 0], [0, 1]])
    alpha = compute_focal_alpha(y)
    assert alpha.tolist() == pytest.approx([0.5, 0.75])


def test_compute_focal_alpha_is_bounded_unlike_pos_weight() -> None:
    # A label with 1 positive out of 1000 gives pos_weight=999 (unbounded) but
    # alpha stays in [0,1] by construction (Memory.md Decision #8's instability risk).
    y = np.zeros((1000, 1), dtype=int)
    y[0, 0] = 1
    assert compute_pos_weight(y)[0] == pytest.approx(999.0)
    assert 0.0 <= compute_focal_alpha(y)[0] <= 1.0


def test_compute_focal_alpha_rejects_all_zero_label() -> None:
    y = np.array([[1, 0], [0, 0], [1, 0]])  # label 1 has zero positives
    with pytest.raises(AssertionError):
        compute_focal_alpha(y)


def test_make_loss_fn_focal_matches_manual_calculation() -> None:
    import torch

    from src.models.train import _make_loss_fn

    config = {"training": {"loss": "focal", "focal_gamma": 2.0}}
    pos_weight = np.array([1.0, 3.0])  # unused by the focal branch
    focal_alpha = np.array([0.5, 0.75])
    loss_fn = _make_loss_fn(config, pos_weight, focal_alpha)

    logits = torch.tensor([[0.0, 0.0]])
    labels = torch.tensor([[1.0, 0.0]])
    loss = loss_fn(logits, labels)

    # Manually derived: p=0.5 for both labels at logit=0.
    # label 0 (y=1): BCE=-log(0.5)=0.693147, pt=0.5, alpha_t=alpha[0]=0.5
    #   -> FL = 0.5 * 0.5^2 * 0.693147 = 0.0866434
    # label 1 (y=0): BCE=-log(0.5)=0.693147, pt=0.5, alpha_t=1-alpha[1]=0.25
    #   -> FL = 0.25 * 0.5^2 * 0.693147 = 0.0433217
    # mean over both elements:
    expected = (0.0866434 + 0.0433217) / 2
    assert loss.item() == pytest.approx(expected, abs=1e-5)


def test_make_loss_fn_bce_pos_weight_uses_configured_pos_weight() -> None:
    import torch
    from torch import nn

    from src.models.train import _make_loss_fn

    config = {"training": {"loss": "bce_pos_weight"}}
    pos_weight = np.array([1.0, 3.0])
    focal_alpha = np.array([0.5, 0.75])  # unused by the bce_pos_weight branch
    loss_fn = _make_loss_fn(config, pos_weight, focal_alpha)

    assert isinstance(loss_fn, nn.BCEWithLogitsLoss)
    assert loss_fn.pos_weight.tolist() == pytest.approx([1.0, 3.0])


def test_make_loss_fn_unknown_loss_raises() -> None:
    from src.models.train import _make_loss_fn

    config = {"training": {"loss": "not_a_real_loss"}}
    with pytest.raises(ValueError, match="not_a_real_loss"):
        _make_loss_fn(config, np.array([1.0]), np.array([0.5]))
