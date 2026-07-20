"""Tests for src.data.preprocess. Uses small synthetic, non-toxic text only."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.preprocess import (
    assert_no_text_overlap,
    cap_dataset_size,
    clean_text,
    load_civil_comments,
    load_jigsaw_toxic,
    load_label_names,
    merge_and_dedupe,
    rarest_label,
    stratified_split,
)

LABELS = load_label_names()


def _zero_label_frame(texts: list[str], positive_label: str | None = None) -> pd.DataFrame:
    """Build a minimal unified-schema frame; all labels 0 except one optional positive."""
    data = {"id": [str(i) for i in range(len(texts))], "comment_text": texts}
    for label in LABELS:
        data[label] = [1 if label == positive_label else 0] * len(texts)
    data["source"] = ["synthetic"] * len(texts)
    return pd.DataFrame(data)


def test_clean_text_collapses_whitespace_and_strips() -> None:
    series = pd.Series(["  hello   world  ", "line1\nline2\t\tline3"])
    cleaned = clean_text(series)
    assert cleaned.tolist() == ["hello world", "line1 line2 line3"]


def test_clean_text_handles_missing_values() -> None:
    series = pd.Series(["fine", None, float("nan")])
    cleaned = clean_text(series)
    assert cleaned.tolist() == ["fine", "", ""]


def test_merge_and_dedupe_drops_exact_duplicates_and_empty_text() -> None:
    # merge_and_dedupe only drops rows already blank (len 0); whitespace-only
    # text is normalized to "" by clean_text upstream in the real pipeline,
    # so mirror that ordering here rather than passing raw "  " directly.
    frame_a = _zero_label_frame(clean_text(pd.Series(["hello there", "  ", "unique a"])).tolist())
    frame_b = _zero_label_frame(clean_text(pd.Series(["hello there", "unique b"])).tolist())
    merged = merge_and_dedupe([frame_a, frame_b])
    assert sorted(merged["comment_text"]) == ["hello there", "unique a", "unique b"]


def test_rarest_label_finds_min_positive_label() -> None:
    df = _zero_label_frame(["a", "b", "c", "d"])
    for label in LABELS:
        df[label] = 0 if label == "threat" else 1
    df.loc[0, "threat"] = 1
    assert rarest_label(df, LABELS) == "threat"


def test_cap_dataset_size_no_op_when_under_cap() -> None:
    df = _zero_label_frame(["a", "b", "c"])
    capped = cap_dataset_size(df, max_rows=10, stratify_col="toxic", seed=42)
    assert len(capped) == 3


def test_cap_dataset_size_stratified_subsample() -> None:
    texts = [f"comment {i}" for i in range(100)]
    df = _zero_label_frame(texts)
    df.loc[:19, "toxic"] = 1  # 20% positive rate
    capped = cap_dataset_size(df, max_rows=50, stratify_col="toxic", seed=42)
    assert len(capped) == 50
    assert capped["toxic"].mean() == pytest.approx(0.2, abs=0.05)


def test_stratified_split_ratios_and_no_overlap() -> None:
    texts = [f"comment {i}" for i in range(200)]
    df = _zero_label_frame(texts)
    df.loc[:39, "toxic"] = 1  # 20% positive rate
    split_ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    train_df, val_df, test_df = stratified_split(df, "toxic", split_ratios, seed=42)

    assert len(train_df) == 160
    assert len(val_df) == 20
    assert len(test_df) == 20
    assert_no_text_overlap(train_df, val_df, test_df)


def test_assert_no_text_overlap_raises_on_overlap() -> None:
    df = _zero_label_frame(["shared text", "other"])
    with pytest.raises(AssertionError):
        assert_no_text_overlap(df, df, df)


def test_load_jigsaw_toxic_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_jigsaw_toxic(tmp_path, LABELS)


def test_load_jigsaw_toxic_missing_columns_raises(tmp_path: Path) -> None:
    raw_dir = tmp_path / "jigsaw_toxic_comment"
    raw_dir.mkdir()
    pd.DataFrame({"id": ["1"], "comment_text": ["fine"]}).to_csv(raw_dir / "train.csv", index=False)
    with pytest.raises(ValueError):
        load_jigsaw_toxic(tmp_path, LABELS)


def test_load_jigsaw_toxic_reads_unified_schema(tmp_path: Path) -> None:
    raw_dir = tmp_path / "jigsaw_toxic_comment"
    raw_dir.mkdir()
    row = {"id": "1", "comment_text": "this is fine"}
    row.update({label: 0 for label in LABELS})
    row["toxic"] = 1
    pd.DataFrame([row]).to_csv(raw_dir / "train.csv", index=False)

    df = load_jigsaw_toxic(tmp_path, LABELS)
    assert list(df.columns) == ["id", "comment_text", *LABELS, "source"]
    assert df.loc[0, "toxic"] == 1
    assert df.loc[0, "source"] == "jigsaw_toxic"


def test_load_civil_comments_binarizes_at_threshold(tmp_path: Path) -> None:
    raw_dir = tmp_path / "civil_comments"
    raw_dir.mkdir()
    rows = pd.DataFrame(
        {
            "id": ["1", "2"],
            "comment_text": ["mildly annoyed comment", "perfectly pleasant comment"],
            "target": [0.6, 0.1],
            "severe_toxicity": [0.0, 0.0],
            "obscene": [0.0, 0.0],
            "threat": [0.0, 0.0],
            "insult": [0.4, 0.0],
            "identity_attack": [0.0, 0.0],
        }
    )
    rows.to_csv(raw_dir / "train.csv", index=False)

    df = load_civil_comments(tmp_path, LABELS, threshold=0.5)
    assert df.loc[0, "toxic"] == 1
    assert df.loc[0, "insult"] == 0  # 0.4 < 0.5 threshold
    assert df.loc[1, "toxic"] == 0
    assert df.loc[0, "source"] == "civil_comments"
