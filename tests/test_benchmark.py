"""Tests for src.evaluation.benchmark. Only the pure timing/percentile/hardware-info logic —
no torch/optimum needed, so this runs on the lightweight local venv."""

from src.evaluation.benchmark import compute_percentiles, get_hardware_info, time_predict_fn


def test_compute_percentiles_known_values() -> None:
    latencies = list(range(1, 101))  # 1..100 ms

    result = compute_percentiles(latencies)

    assert result["p50_ms"] == 50.5
    assert result["p95_ms"] == 95.05
    assert result["mean_ms"] == 50.5


def test_time_predict_fn_calls_warmup_and_timed_runs() -> None:
    call_count = {"n": 0}

    def predict_one() -> None:
        call_count["n"] += 1

    result = time_predict_fn(predict_one, n_warmup=5, n_runs=10)

    assert call_count["n"] == 15
    assert set(result.keys()) == {"p50_ms", "p95_ms", "p99_ms", "mean_ms"}
    assert all(v >= 0 for v in result.values())


def test_get_hardware_info_has_expected_keys() -> None:
    info = get_hardware_info()

    assert set(info.keys()) == {"platform", "cpu_model", "cpu_count"}
    assert info["cpu_count"] is None or info["cpu_count"] > 0
    assert isinstance(info["cpu_model"], str) and info["cpu_model"]
