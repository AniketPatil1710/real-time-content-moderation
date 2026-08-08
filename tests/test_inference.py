"""Tests for src.serving.inference's pure decision logic — no ONNX/torch needed
(the ModerationModel class itself is not exercised here, only decide())."""

from src.serving.inference import decide

LABELS = ["toxic", "insult"]


def _thresholds(block: float = 0.9, flag: float = 0.5) -> dict:
    return {"labels": {label: {"block_threshold": block, "flag_threshold": flag} for label in LABELS}}


def test_decide_all_low_scores_allows() -> None:
    assert decide({"toxic": 0.1, "insult": 0.05}, _thresholds()) == "allow"


def test_decide_one_label_over_block_threshold_blocks() -> None:
    assert decide({"toxic": 0.95, "insult": 0.05}, _thresholds()) == "block"


def test_decide_one_label_over_flag_threshold_flags() -> None:
    assert decide({"toxic": 0.6, "insult": 0.05}, _thresholds()) == "flag"


def test_decide_most_severe_label_wins() -> None:
    # insult blocks, toxic only flags -> overall must be the more severe of the two.
    assert decide({"toxic": 0.6, "insult": 0.95}, _thresholds()) == "block"


def test_decide_boundary_scores_are_inclusive() -> None:
    assert decide({"toxic": 0.9, "insult": 0.0}, _thresholds(block=0.9)) == "block"
    assert decide({"toxic": 0.5, "insult": 0.0}, _thresholds(flag=0.5)) == "flag"
