"""TF-IDF + One-vs-Rest Logistic Regression baseline, train + eval. Phase 2."""

import argparse
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from src.data.preprocess import load_label_names, load_training_config
from src.evaluation.evaluate import evaluate, load_split, save_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

METRICS_PATH = Path("metrics/baseline.json")


def build_vectorizers(config: dict[str, Any]) -> tuple[TfidfVectorizer, TfidfVectorizer]:
    """Word (1-2 gram) + char (3-5 gram) TF-IDF vectorizers, per configs/training.yaml."""
    tfidf_cfg = config["baseline"]["tfidf"]
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=tuple(tfidf_cfg["word_ngram_range"]),
        max_features=tfidf_cfg["max_features_word"],
    )
    char_vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=tuple(tfidf_cfg["char_ngram_range"]),
        max_features=tfidf_cfg["max_features_char"],
    )
    return word_vec, char_vec


def vectorize(
    texts: list[str], word_vec: TfidfVectorizer, char_vec: TfidfVectorizer, fit: bool
) -> sparse.csr_matrix:
    """Fit (if fit=True) or transform texts through both vectorizers and hstack them."""
    if fit:
        word_features = word_vec.fit_transform(texts)
        char_features = char_vec.fit_transform(texts)
    else:
        word_features = word_vec.transform(texts)
        char_features = char_vec.transform(texts)
    return sparse.hstack([word_features, char_features]).tocsr()


def train_baseline(
    config: dict[str, Any], train_texts: list[str], y_train: np.ndarray
) -> tuple[OneVsRestClassifier, TfidfVectorizer, TfidfVectorizer]:
    """Fit the TF-IDF vectorizers and a One-vs-Rest Logistic Regression classifier."""
    word_vec, char_vec = build_vectorizers(config)
    x_train = vectorize(train_texts, word_vec, char_vec, fit=True)

    lr_cfg = config["baseline"]["logistic_regression"]
    base_clf = LogisticRegression(
        C=lr_cfg["C"],
        max_iter=lr_cfg["max_iter"],
        class_weight=lr_cfg["class_weight"],
        random_state=config["seed"],
    )
    clf = OneVsRestClassifier(base_clf)
    clf.fit(x_train, y_train)
    return clf, word_vec, char_vec


def make_predict_fn(clf: OneVsRestClassifier, word_vec: TfidfVectorizer, char_vec: TfidfVectorizer):
    """Wrap the fitted vectorizers + classifier into an evaluate()-compatible predict_fn."""

    def predict_fn(texts: list[str]) -> np.ndarray:
        x = vectorize(texts, word_vec, char_vec, fit=False)
        return clf.predict_proba(x)

    return predict_fn


def run_baseline(processed_dir: Path, metrics_path: Path) -> None:
    """Train the baseline on train, evaluate on test, and save metrics/baseline.json."""
    label_names = load_label_names()
    config = load_training_config()
    seed = config["seed"]
    logger.info("Using seed=%d", seed)
    random.seed(seed)
    np.random.seed(seed)

    train_texts, y_train = load_split(processed_dir, "train", label_names)
    test_texts, y_test = load_split(processed_dir, "test", label_names)

    clf, word_vec, char_vec = train_baseline(config, train_texts, y_train)
    predict_fn = make_predict_fn(clf, word_vec, char_vec)

    metrics = evaluate(predict_fn, test_texts, y_test, label_names)
    save_metrics(metrics, metrics_path)
    logger.info("Baseline macro F1: %.4f, macro PR-AUC: %.4f", metrics["macro_f1"], metrics["macro_pr_auc"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"), help="Parquet splits dir")
    parser.add_argument("--metrics-path", type=Path, default=METRICS_PATH, help="Output metrics JSON path")
    args = parser.parse_args()
    run_baseline(args.processed_dir, args.metrics_path)


if __name__ == "__main__":
    main()
