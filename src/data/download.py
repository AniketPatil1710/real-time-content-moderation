"""Kaggle API download of Jigsaw Toxic Comment + Civil Comments datasets. Phase 1.

Requires a valid Kaggle API token at ~/.kaggle/kaggle.json (or KAGGLE_USERNAME /
KAGGLE_KEY env vars) AND prior acceptance of each competition's rules on
kaggle.com — the API returns 403 Forbidden otherwise. This script cannot do
that acceptance step for you; visit the competition page and click
"I Understand and Accept" first.

Downloads only train.csv per competition (not the full competition bundle,
which also includes test sets and annotation files preprocess.py never
reads) to keep disk usage minimal.
"""

import argparse
import logging
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

JIGSAW_TOXIC_COMPETITION = "jigsaw-toxic-comment-classification-challenge"
CIVIL_COMMENTS_COMPETITION = "jigsaw-unintended-bias-in-toxicity-classification"


def download_train_file(competition: str, dest_dir: Path) -> None:
    """Download only train.csv for a Kaggle competition into dest_dir.

    Imports the kaggle package lazily so this module can be imported (e.g. by
    tests) without the kaggle package or credentials being present. Kaggle
    serves some competitions' files pre-zipped and others as plain CSV, so
    this unzips only if a .zip actually came back.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    dest_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()

    logger.info("Downloading %s/train.csv to %s", competition, dest_dir)
    api.competition_download_file(competition, "train.csv", path=str(dest_dir))

    zip_path = dest_dir / "train.csv.zip"
    if zip_path.exists():
        _extract_and_remove(zip_path, dest_dir)

    csv_path = dest_dir / "train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Expected {csv_path} after download. Confirm you accepted the competition rules on kaggle.com."
        )
    logger.info("Downloaded %s/train.csv (%d bytes)", competition, csv_path.stat().st_size)


def _extract_and_remove(zip_path: Path, dest_dir: Path) -> None:
    """Extract a zip file into dest_dir, then delete the archive."""
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    zip_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to download raw Kaggle files into (default: data/raw)",
    )
    parser.add_argument(
        "--skip-jigsaw", action="store_true", help="Skip the Jigsaw Toxic Comment Classification Challenge download"
    )
    parser.add_argument(
        "--skip-civil-comments",
        action="store_true",
        help="Skip the Civil Comments (Jigsaw Unintended Bias) download",
    )
    args = parser.parse_args()

    if not args.skip_jigsaw:
        download_train_file(JIGSAW_TOXIC_COMPETITION, args.raw_dir / "jigsaw_toxic_comment")
    if not args.skip_civil_comments:
        download_train_file(CIVIL_COMMENTS_COMPETITION, args.raw_dir / "civil_comments")


if __name__ == "__main__":
    main()
