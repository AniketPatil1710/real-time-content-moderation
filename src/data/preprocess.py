"""Clean, merge, dedupe, stratified split into train/val/test parquet. Phase 1.

Key decisions (see Memory.md for the append-only decision log):
- Civil Comments (jigsaw-unintended-bias-in-toxicity-classification) ships
  continuous crowd-rated scores, not the Jigsaw 2018 binary labels. Each
  subtype score is binarized at `data.civil_comments_binarize_threshold`
  (configs/training.yaml) to match the Jigsaw 2018 convention.
- Civil Comments' `identity_attack` column is treated as the same concept as
  Jigsaw 2018's `identity_hate` label and mapped onto it directly.
- Rows with a missing subtype score are treated as 0 (non-toxic on that
  subtype) since Civil Comments only ran full subtype annotation on a subset
  of comments; the count of such fills is logged, not silently dropped.
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LABELS_CONFIG_PATH = Path("configs/labels.json")
TRAINING_CONFIG_PATH = Path("configs/training.yaml")

# Unified schema columns, in output-file order.
ID_COL = "id"
TEXT_COL = "comment_text"
SOURCE_COL = "source"

# Civil Comments subtype score -> unified label name.
CIVIL_COMMENTS_LABEL_MAP = {
    "target": "toxic",
    "severe_toxicity": "severe_toxic",
    "obscene": "obscene",
    "threat": "threat",
    "insult": "insult",
    "identity_attack": "identity_hate",
}


def load_label_names(config_path: Path = LABELS_CONFIG_PATH) -> list[str]:
    """Load the ordered label list — the single source of truth for label order."""
    with config_path.open() as f:
        config = json.load(f)
    return config["labels"]


def load_training_config(config_path: Path = TRAINING_CONFIG_PATH) -> dict[str, Any]:
    """Load seed, binarization threshold, dataset cap, and split ratios."""
    with config_path.open() as f:
        return yaml.safe_load(f)


def clean_text(series: pd.Series) -> pd.Series:
    """Minimal cleaning: keep noisy text realistic, just normalize whitespace.

    Casts to string, strips leading/trailing whitespace, and collapses
    internal whitespace runs (including newlines/tabs) to a single space.
    Does NOT lowercase, strip punctuation, or remove special characters —
    the model should see text as users actually write it. Missing text
    becomes an empty string so it's dropped by the merge step, not stringified.
    """
    return series.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def load_jigsaw_toxic(raw_dir: Path, label_names: list[str]) -> pd.DataFrame:
    """Load the Jigsaw 2018 Toxic Comment Classification Challenge train split."""
    path = raw_dir / "jigsaw_toxic_comment" / "train.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing raw file: {path}. Run src.data.download first.")

    df = pd.read_csv(path)
    required = {ID_COL, TEXT_COL, *label_names}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {sorted(missing)}")

    df[ID_COL] = df[ID_COL].astype(str)
    df[TEXT_COL] = clean_text(df[TEXT_COL])
    for label in label_names:
        df[label] = df[label].astype(int)
        assert df[label].isin([0, 1]).all(), f"{path} column '{label}' has values outside {{0,1}}"

    df[SOURCE_COL] = "jigsaw_toxic"
    return df[[ID_COL, TEXT_COL, *label_names, SOURCE_COL]]


def load_civil_comments(raw_dir: Path, label_names: list[str], threshold: float) -> pd.DataFrame:
    """Load Civil Comments and binarize its continuous subtype scores.

    See module docstring for the score -> label mapping and the missing-value
    convention.
    """
    path = raw_dir / "civil_comments" / "train.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing raw file: {path}. Run src.data.download first.")

    df = pd.read_csv(path)
    required = {ID_COL, TEXT_COL, *CIVIL_COMMENTS_LABEL_MAP.keys()}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {sorted(missing)}")

    df[ID_COL] = df[ID_COL].astype(str)
    df[TEXT_COL] = clean_text(df[TEXT_COL])

    for source_col in CIVIL_COMMENTS_LABEL_MAP:
        n_missing = df[source_col].isna().sum()
        if n_missing:
            logger.info("Civil Comments: filling %d missing '%s' scores with 0.0", n_missing, source_col)
        df[source_col] = df[source_col].fillna(0.0)
        assert df[source_col].between(0.0, 1.0).all(), f"{path} column '{source_col}' has values outside [0,1]"

    for source_col, label in CIVIL_COMMENTS_LABEL_MAP.items():
        df[label] = (df[source_col] >= threshold).astype(int)

    df[SOURCE_COL] = "civil_comments"
    return df[[ID_COL, TEXT_COL, *label_names, SOURCE_COL]]


def merge_and_dedupe(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate sources, drop empty text, and dedupe on comment text."""
    merged = pd.concat(frames, ignore_index=True)
    merged = merged[merged[TEXT_COL].str.len() > 0]
    before = len(merged)
    merged = merged.drop_duplicates(subset=TEXT_COL, keep="first").reset_index(drop=True)
    logger.info("Deduped %d -> %d rows (%d exact-text duplicates dropped)", before, len(merged), before - len(merged))
    return merged


def rarest_label(df: pd.DataFrame, label_names: list[str]) -> str:
    """Return the label with the fewest positive examples.

    Used as the stratification key throughout so the minority-class rate is
    preserved in the capped sample and in every split.
    """
    positive_counts = df[label_names].sum()
    return str(positive_counts.idxmin())


def cap_dataset_size(df: pd.DataFrame, max_rows: int, stratify_col: str, seed: int) -> pd.DataFrame:
    """Stratified-subsample down to max_rows if the merged dataset exceeds it."""
    if len(df) <= max_rows:
        return df
    sampled, _ = train_test_split(df, train_size=max_rows, stratify=df[stratify_col], random_state=seed)
    logger.info("Capped dataset %d -> %d rows via stratified sampling on '%s'", len(df), len(sampled), stratify_col)
    return sampled.reset_index(drop=True)


def stratified_split(
    df: pd.DataFrame, stratify_col: str, split_ratios: dict[str, float], seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """80/10/10 (ratios from config) split, stratified by stratify_col."""
    train_ratio = split_ratios["train"]
    val_ratio = split_ratios["val"]
    test_ratio = split_ratios["test"]
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, "split_ratios must sum to 1.0"

    train_df, holdout_df = train_test_split(
        df, test_size=(val_ratio + test_ratio), stratify=df[stratify_col], random_state=seed
    )
    val_share_of_holdout = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        holdout_df, train_size=val_share_of_holdout, stratify=holdout_df[stratify_col], random_state=seed
    )
    return (train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True))


def assert_no_text_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Fail loudly if any comment text leaked across splits."""
    train_texts = set(train_df[TEXT_COL])
    val_texts = set(val_df[TEXT_COL])
    test_texts = set(test_df[TEXT_COL])
    assert not (train_texts & val_texts), "Text overlap detected between train and val splits"
    assert not (train_texts & test_texts), "Text overlap detected between train and test splits"
    assert not (val_texts & test_texts), "Text overlap detected between val and test splits"


def summarize(name: str, df: pd.DataFrame, label_names: list[str]) -> None:
    """Log row count and per-label positive rate for one split."""
    lines = [f"{name}: {len(df)} rows"]
    for label in label_names:
        rate = df[label].mean() * 100
        lines.append(f"  {label}: {rate:.2f}% positive")
    logger.info("\n".join(lines))


def run_preprocess(raw_dir: Path, processed_dir: Path) -> None:
    """End-to-end pipeline: load, clean, merge, dedupe, cap, split, save, summarize."""
    label_names = load_label_names()
    config = load_training_config()
    seed = config["seed"]
    logger.info("Using seed=%d", seed)
    random.seed(seed)
    np.random.seed(seed)

    threshold = config["data"]["civil_comments_binarize_threshold"]
    max_rows = config["data"]["max_rows"]
    split_ratios = config["data"]["split_ratios"]

    jigsaw_df = load_jigsaw_toxic(raw_dir, label_names)
    civil_df = load_civil_comments(raw_dir, label_names, threshold)
    merged = merge_and_dedupe([jigsaw_df, civil_df])
    assert len(merged) > 0, "Merged dataset is empty after cleaning/dedup — check raw inputs"

    strat_label = rarest_label(merged, label_names)
    logger.info("Stratifying on rarest label: '%s'", strat_label)

    capped = cap_dataset_size(merged, max_rows, strat_label, seed)
    train_df, val_df, test_df = stratified_split(capped, strat_label, split_ratios, seed)
    assert_no_text_overlap(train_df, val_df, test_df)

    processed_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(processed_dir / "train.parquet", index=False)
    val_df.to_parquet(processed_dir / "val.parquet", index=False)
    test_df.to_parquet(processed_dir / "test.parquet", index=False)

    summarize("train", train_df, label_names)
    summarize("val", val_df, label_names)
    summarize("test", test_df, label_names)
    logger.info("Saved train/val/test parquet files to %s", processed_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Raw Kaggle CSVs (default: data/raw)")
    parser.add_argument(
        "--processed-dir", type=Path, default=Path("data/processed"), help="Output dir for parquet files"
    )
    args = parser.parse_args()
    run_preprocess(args.raw_dir, args.processed_dir)


if __name__ == "__main__":
    main()
