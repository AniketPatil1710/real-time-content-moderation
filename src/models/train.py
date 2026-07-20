"""DistilBERT multi-label fine-tuning entrypoint (HF Trainer). Phase 3.

Runs on GPU (Colab) — actual training cannot run on this project's local
host. `compute_pos_weight` has no torch/transformers dependency and is
imported eagerly so it stays unit-testable without a GPU environment;
everything else that needs torch/transformers is imported lazily inside the
functions that use it.
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.preprocess import load_label_names, load_training_config
from src.evaluation.evaluate import evaluate, load_split, save_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("models/distilbert-finetuned")
METRICS_PATH = Path("metrics/distilbert.json")
BASELINE_METRICS_PATH = Path("metrics/baseline.json")


def compute_pos_weight(y: np.ndarray) -> np.ndarray:
    """pos_weight[i] = n_negative_i / n_positive_i (BCEWithLogitsLoss convention).

    Rare positive labels get a larger weight so the loss penalizes false
    negatives on them more heavily, per Phases.md's class-imbalance handling.
    """
    n_pos = y.sum(axis=0)
    assert (n_pos > 0).all(), "compute_pos_weight: a label has zero positive examples in training data"
    n_neg = len(y) - n_pos
    return n_neg / n_pos


def _compute_metrics(eval_pred: Any) -> dict[str, float]:
    """HF Trainer eval callback: macro F1 at threshold 0.5, for early-stopping monitoring only.

    Final reported metrics always come from the shared evaluate() on the test
    set (see run_train), never from this callback.
    """
    from sklearn.metrics import f1_score

    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    return {"macro_f1": f1_score(labels, preds, average="macro", zero_division=0)}


def _build_training_args(config: dict[str, Any], checkpoint_dir: Path):
    """TrainingArguments from configs/training.yaml — no hardcoded hyperparameters."""
    from transformers import TrainingArguments

    training_cfg = config["training"]
    metric_name = training_cfg["early_stopping_metric"].removeprefix("eval_")
    return TrainingArguments(
        output_dir=str(checkpoint_dir / "checkpoints"),
        per_device_train_batch_size=training_cfg["batch_size"],
        per_device_eval_batch_size=training_cfg["batch_size"],
        num_train_epochs=training_cfg["epochs"],
        learning_rate=training_cfg["learning_rate"],
        fp16=training_cfg["fp16"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        greater_is_better=True,
        seed=config["seed"],
        logging_steps=100,
        report_to=[],  # metrics go to JSON files, not W&B/MLflow — see Rules.md
    )


def build_trainer(
    config: dict[str, Any],
    label_names: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    pos_weight: np.ndarray,
    checkpoint_dir: Path,
):
    """Build the HF Trainer: model, tokenizer, datasets, pos_weight-ed loss, early stopping."""
    import torch
    from torch import nn
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback, Trainer

    from src.data.dataset import ToxicCommentsDataset

    model_name = config["model"]["name"]
    max_length = config["model"]["max_length"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label_names), problem_type="multi_label_classification"
    )
    train_dataset = ToxicCommentsDataset(train_df, tokenizer, max_length, label_names)
    val_dataset = ToxicCommentsDataset(val_df, tokenizer, max_length, label_names)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32)

    class WeightedTrainer(Trainer):
        """Trainer with BCEWithLogitsLoss(pos_weight=...) instead of HF's unweighted default."""

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor.to(outputs.logits.device))
            loss = loss_fct(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    patience = config["training"]["early_stopping_patience"]
    trainer = WeightedTrainer(
        model=model,
        args=_build_training_args(config, checkpoint_dir),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )
    return trainer, tokenizer


def make_predict_fn(model: Any, tokenizer: Any, max_length: int, batch_size: int = 64):
    """Wrap a fine-tuned model into an evaluate()-compatible predict_fn (batched, sigmoid)."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    def predict_fn(texts: list[str]) -> np.ndarray:
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                encoding = tokenizer(
                    batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt"
                ).to(device)
                logits = model(**encoding).logits
                all_probs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(all_probs, axis=0)

    return predict_fn


def compare_to_baseline(metrics: dict[str, Any], baseline_metrics_path: Path) -> None:
    """Log loudly whether DistilBERT beat the baseline macro F1 (Phases.md acceptance)."""
    if not baseline_metrics_path.exists():
        logger.warning("No baseline metrics found at %s; skipping beat-baseline check", baseline_metrics_path)
        return
    with baseline_metrics_path.open() as f:
        baseline = json.load(f)
    delta = metrics["macro_f1"] - baseline["macro_f1"]
    if delta <= 0:
        logger.error(
            "DistilBERT macro F1 (%.4f) did NOT beat baseline (%.4f). "
            "Per Phases.md: stop and investigate before proceeding to Phase 4.",
            metrics["macro_f1"],
            baseline["macro_f1"],
        )
    else:
        logger.info(
            "DistilBERT macro F1 (%.4f) beats baseline (%.4f) by %.4f",
            metrics["macro_f1"],
            baseline["macro_f1"],
            delta,
        )


def run_train(processed_dir: Path, checkpoint_dir: Path, metrics_path: Path, baseline_metrics_path: Path) -> None:
    """Fine-tune DistilBERT, save it, evaluate on test with the shared evaluator, compare to baseline."""
    import torch

    label_names = load_label_names()
    config = load_training_config()
    seed = config["seed"]
    logger.info("Using seed=%d", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")

    pos_weight = compute_pos_weight(train_df[label_names].to_numpy(dtype=int))
    logger.info("pos_weight per label: %s", dict(zip(label_names, pos_weight.round(2).tolist())))

    trainer, tokenizer = build_trainer(config, label_names, train_df, val_df, pos_weight, checkpoint_dir)
    trainer.train()

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    logger.info("Saved fine-tuned model + tokenizer to %s", checkpoint_dir)

    test_texts, y_test = load_split(processed_dir, "test", label_names)
    predict_fn = make_predict_fn(trainer.model, tokenizer, config["model"]["max_length"])
    metrics = evaluate(predict_fn, test_texts, y_test, label_names)
    save_metrics(metrics, metrics_path)
    logger.info("DistilBERT macro F1: %.4f, macro PR-AUC: %.4f", metrics["macro_f1"], metrics["macro_pr_auc"])

    compare_to_baseline(metrics, baseline_metrics_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--metrics-path", type=Path, default=METRICS_PATH)
    parser.add_argument("--baseline-metrics-path", type=Path, default=BASELINE_METRICS_PATH)
    args = parser.parse_args()
    run_train(args.processed_dir, args.checkpoint_dir, args.metrics_path, args.baseline_metrics_path)


if __name__ == "__main__":
    main()
