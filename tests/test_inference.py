"""Tests for src.serving.inference's pure decision logic — no ONNX/torch needed
(the ModerationModel class itself is not exercised here, only decide()/label_band())."""

from src.serving.inference import decide, label_band

LABELS = ["toxic", "insult"]


def _thresholds(block: float = 0.9, flag: float = 0.5) -> dict:
    return {"labels": {label: {"block_threshold": block, "flag_threshold": flag} for label in LABELS}}


def _label_thresholds(block: float = 0.9, flag: float = 0.5) -> dict:
    return {"block_threshold": block, "flag_threshold": flag}


def test_label_band_below_flag_is_allow() -> None:
    assert label_band(0.1, _label_thresholds()) == "allow"


def test_label_band_at_flag_threshold_is_flag() -> None:
    assert label_band(0.5, _label_thresholds(flag=0.5)) == "flag"


def test_label_band_at_block_threshold_is_block() -> None:
    assert label_band(0.9, _label_thresholds(block=0.9)) == "block"


def test_label_band_between_flag_and_block_is_flag() -> None:
    assert label_band(0.7, _label_thresholds(block=0.9, flag=0.5)) == "flag"


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
