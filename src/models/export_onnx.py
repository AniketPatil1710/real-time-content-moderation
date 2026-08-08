"""Optimum ONNX export + dynamic INT8 quantization -> models/onnx/. Phase 5.

Exports the Phase 3 fine-tuned model to ONNX (FP32), then dynamically
quantizes it to INT8 (AVX2 config — broadly compatible across x86_64 CPUs,
not tied to a specific Colab runtime's exact instruction set support).
Re-evaluates the quantized model with the SAME evaluator used for
baseline/DistilBERT (Rules.md: identical test set, identical eval code) and
compares macro F1 against metrics/distilbert.json.

torch/transformers/optimum imports are deferred inside functions (same
pattern as train.py) so this file stays importable without the heavy deps
installed.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.data.preprocess import load_label_names, load_training_config
from src.evaluation.evaluate import evaluate, load_split, save_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ONNX_DIR = Path("models/onnx")
METRICS_PATH = Path("metrics/onnx_quantized.json")
PYTORCH_METRICS_PATH = Path("metrics/distilbert.json")
MAX_F1_DROP = 0.01
# ORTQuantizer.quantize() names its output after the source file's stem + "_quantized"
# (source is always "model.onnx" from export_fp32) — not the "model.onnx" default
# ORTModelForSequenceClassification.from_pretrained() looks for.
QUANTIZED_FILE_NAME = "model_quantized.onnx"


def export_fp32(model_dir: Path, output_dir: Path) -> Path:
    """Export the fine-tuned PyTorch model to ONNX (FP32) via Optimum."""
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    model = ORTModelForSequenceClassification.from_pretrained(str(model_dir), export=True)
    model.save_pretrained(str(output_dir))
    AutoTokenizer.from_pretrained(str(model_dir)).save_pretrained(str(output_dir))
    logger.info("Exported FP32 ONNX model to %s", output_dir)
    return output_dir


def quantize_int8(fp32_dir: Path, output_dir: Path) -> Path:
    """Dynamic INT8 quantization of the FP32 ONNX model via Optimum."""
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    quantizer = ORTQuantizer.from_pretrained(str(fp32_dir))
    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=str(output_dir), quantization_config=qconfig)
    AutoTokenizer.from_pretrained(str(fp32_dir)).save_pretrained(str(output_dir))
    logger.info("Saved INT8-quantized ONNX model to %s", output_dir)
    return output_dir


def make_onnx_predict_fn(model_dir: Path, max_length: int, batch_size: int = 64, file_name: str | None = None):
    """Wrap a saved ONNX model (FP32 or INT8) into an evaluate()-compatible predict_fn.

    `file_name` must be set to QUANTIZED_FILE_NAME for a directory produced by
    quantize_int8() — Optimum's default from_pretrained() lookup only finds "model.onnx".
    """
    import torch
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = ORTModelForSequenceClassification.from_pretrained(
        str(model_dir), provider="CPUExecutionProvider", file_name=file_name
    )

    def predict_fn(texts: list[str]) -> np.ndarray:
        all_probs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoding = tokenizer(batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
            logits = model(**encoding).logits
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        return np.concatenate(all_probs, axis=0)

    return predict_fn


def f1_drop_within_budget(quantized_f1: float, pytorch_f1: float, max_drop: float = MAX_F1_DROP) -> bool:
    """True if quantization's macro-F1 drop vs PyTorch stays under max_drop (Phases.md Phase 5 gate)."""
    return (pytorch_f1 - quantized_f1) < max_drop


def compare_to_pytorch(metrics: dict[str, Any], pytorch_metrics_path: Path) -> None:
    """Log loudly whether the quantized model's F1 drop vs PyTorch stays under MAX_F1_DROP."""
    if not pytorch_metrics_path.exists():
        logger.warning("No PyTorch metrics found at %s; skipping accuracy-drop check", pytorch_metrics_path)
        return
    with pytorch_metrics_path.open() as f:
        pytorch_metrics = json.load(f)
    drop = pytorch_metrics["macro_f1"] - metrics["macro_f1"]
    if f1_drop_within_budget(metrics["macro_f1"], pytorch_metrics["macro_f1"]):
        logger.info(
            "ONNX-INT8 macro F1 (%.4f) within budget vs PyTorch (%.4f), drop=%.4f",
            metrics["macro_f1"],
            pytorch_metrics["macro_f1"],
            drop,
        )
    else:
        logger.error(
            "ONNX-INT8 macro F1 (%.4f) dropped %.4f vs PyTorch (%.4f) — exceeds the %.0f%% budget. "
            "Per Phases.md: fall back to the FP32 ONNX model for serving, do not use INT8.",
            metrics["macro_f1"],
            drop,
            pytorch_metrics["macro_f1"],
            MAX_F1_DROP * 100,
        )


def run_export(
    processed_dir: Path,
    model_dir: Path,
    onnx_dir: Path,
    metrics_path: Path,
    pytorch_metrics_path: Path,
) -> None:
    """End-to-end Phase 5 export: FP32 export, INT8 quantize, evaluate, compare to PyTorch."""
    label_names = load_label_names()
    config = load_training_config()
    max_length = config["model"]["max_length"]

    fp32_dir = export_fp32(model_dir, onnx_dir / "fp32")
    int8_dir = quantize_int8(fp32_dir, onnx_dir / "int8")

    test_texts, y_test = load_split(processed_dir, "test", label_names)
    predict_fn = make_onnx_predict_fn(int8_dir, max_length, file_name=QUANTIZED_FILE_NAME)
    metrics = evaluate(predict_fn, test_texts, y_test, label_names)
    save_metrics(metrics, metrics_path)
    logger.info("ONNX-INT8 macro F1: %.4f, macro PR-AUC: %.4f", metrics["macro_f1"], metrics["macro_pr_auc"])

    compare_to_pytorch(metrics, pytorch_metrics_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/distilbert-finetuned"))
    parser.add_argument("--onnx-dir", type=Path, default=ONNX_DIR)
    parser.add_argument("--metrics-path", type=Path, default=METRICS_PATH)
    parser.add_argument("--pytorch-metrics-path", type=Path, default=PYTORCH_METRICS_PATH)
    args = parser.parse_args()
    run_export(args.processed_dir, args.model_dir, args.onnx_dir, args.metrics_path, args.pytorch_metrics_path)


if __name__ == "__main__":
    main()
