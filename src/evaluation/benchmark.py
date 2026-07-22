"""Latency benchmark: warmup + timed runs, p50/p95/p99, hardware info. Phase 5.

Benchmarks batch=1, seq_len=128 CPU inference for PyTorch, ONNX (FP32), and
ONNX-INT8 — forces CPU execution explicitly for all three (`model.to("cpu")`
for PyTorch, `provider="CPUExecutionProvider"` for ONNX Runtime) regardless
of whether a GPU is present in the run environment, since this measures the
CPU-serving scenario Phase 6's FastAPI endpoint actually runs under.

torch/transformers/optimum imports are deferred inside functions (same
pattern as train.py/export_onnx.py) so this file stays importable, and its
pure-logic pieces (percentiles, timing loop, hardware info) stay
unit-testable, without the heavy deps installed.
"""

import argparse
import json
import logging
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.data.preprocess import load_training_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

METRICS_PATH = Path("metrics/latency_benchmark.json")
N_WARMUP = 50
N_RUNS = 1000
BATCH_SIZE = 1


def get_hardware_info() -> dict[str, Any]:
    """Best-effort CPU description, working on both Linux (Colab) and macOS."""
    cpu_model = None
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        for line in cpuinfo_path.read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    if cpu_model is None and platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
            )
            cpu_model = result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    return {
        "platform": platform.platform(),
        "cpu_model": cpu_model or "unknown",
        "cpu_count": os.cpu_count(),
    }


def compute_percentiles(latencies_ms: list[float]) -> dict[str, float]:
    """p50/p95/p99/mean over a list of per-call latencies in milliseconds."""
    arr = np.array(latencies_ms)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
    }


def time_predict_fn(predict_one: Callable[[], None], n_warmup: int = N_WARMUP, n_runs: int = N_RUNS) -> dict[str, float]:
    """Call predict_one n_warmup times (discarded, for JIT/cache warmup) then n_runs timed times."""
    for _ in range(n_warmup):
        predict_one()

    latencies_ms = []
    for _ in range(n_runs):
        start = time.perf_counter()
        predict_one()
        latencies_ms.append((time.perf_counter() - start) * 1000)

    return compute_percentiles(latencies_ms)


def build_pytorch_predict_one(model_dir: Path, text: str, max_length: int) -> Callable[[], None]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to("cpu")
    model.eval()
    encoding = tokenizer(text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")

    def predict_one() -> None:
        with torch.no_grad():
            model(**encoding)

    return predict_one


def build_onnx_predict_one(model_dir: Path, text: str, max_length: int) -> Callable[[], None]:
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = ORTModelForSequenceClassification.from_pretrained(str(model_dir), provider="CPUExecutionProvider")
    encoding = tokenizer(text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")

    def predict_one() -> None:
        model(**encoding)

    return predict_one


def run_benchmark(
    processed_dir: Path,
    pytorch_model_dir: Path,
    onnx_fp32_dir: Path,
    onnx_int8_dir: Path,
    metrics_path: Path,
) -> None:
    """Benchmark PyTorch, ONNX-FP32, and ONNX-INT8 on one real test-set example, save results."""
    config = load_training_config()
    max_length = config["model"]["max_length"]
    text = str(pd.read_parquet(processed_dir / "test.parquet").iloc[0]["comment_text"])

    builders = {
        "pytorch": lambda: build_pytorch_predict_one(pytorch_model_dir, text, max_length),
        "onnx_fp32": lambda: build_onnx_predict_one(onnx_fp32_dir, text, max_length),
        "onnx_int8": lambda: build_onnx_predict_one(onnx_int8_dir, text, max_length),
    }

    results = {}
    for name, build in builders.items():
        logger.info("Benchmarking %s: %d warmup + %d timed runs", name, N_WARMUP, N_RUNS)
        predict_one = build()
        results[name] = time_predict_fn(predict_one)
        logger.info(
            "%s p50=%.2fms p95=%.2fms p99=%.2fms",
            name,
            results[name]["p50_ms"],
            results[name]["p95_ms"],
            results[name]["p99_ms"],
        )

    payload = {
        "hardware": get_hardware_info(),
        "batch_size": BATCH_SIZE,
        "seq_len": max_length,
        "n_warmup": N_WARMUP,
        "n_runs": N_RUNS,
        "results": results,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved latency benchmark to %s", metrics_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--pytorch-model-dir", type=Path, default=Path("models/distilbert-finetuned"))
    parser.add_argument("--onnx-fp32-dir", type=Path, default=Path("models/onnx/fp32"))
    parser.add_argument("--onnx-int8-dir", type=Path, default=Path("models/onnx/int8"))
    parser.add_argument("--metrics-path", type=Path, default=METRICS_PATH)
    args = parser.parse_args()
    run_benchmark(args.processed_dir, args.pytorch_model_dir, args.onnx_fp32_dir, args.onnx_int8_dir, args.metrics_path)


if __name__ == "__main__":
    main()
