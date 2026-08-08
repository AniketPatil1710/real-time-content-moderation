"""Tests for src.serving.app (FastAPI TestClient). Uses a fake in-memory model injected
directly into app._state — no real ONNX weights needed (those only exist on Drive, see
Memory.md), and no toxic text is written here since the fake model's scores are set
directly rather than derived from real classification (Rules.md: don't hardcode toxic
example strings beyond minimal mild ones)."""

import pytest
from fastapi.testclient import TestClient

from src.data.preprocess import load_label_names
from src.serving import app as app_module

LABELS = load_label_names()


class FakeModel:
    """Stands in for ModerationModel — returns fixed per-label scores, no real inference."""

    def __init__(self, scores: dict[str, float]):
        self._scores = scores

    def predict(self, text: str) -> tuple[dict[str, float], float]:
        return self._scores, 1.23


def _thresholds(block: float = 0.9, flag: float = 0.5) -> dict:
    return {
        "block_precision_target": 0.9,
        "flag_precision_target": 0.5,
        "labels": {label: {"block_threshold": block, "flag_threshold": flag} for label in LABELS},
    }


@pytest.fixture(autouse=True)
def _reset_state():
    yield
    app_module._state["model"] = None
    app_module._state["thresholds"] = None


def test_health_ok_when_model_loaded() -> None:
    app_module._state["model"] = FakeModel({label: 0.0 for label in LABELS})
    app_module._state["thresholds"] = _thresholds()
    client = TestClient(app_module.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_reports_not_loaded_when_model_missing() -> None:
    app_module._state["model"] = None
    client = TestClient(app_module.app)

    response = client.get("/health")

    assert response.json()["status"] == "model_not_loaded"


def test_moderate_clean_text_allows() -> None:
    app_module._state["model"] = FakeModel({label: 0.01 for label in LABELS})
    app_module._state["thresholds"] = _thresholds()
    client = TestClient(app_module.app)

    response = client.post("/moderate", json={"text": "Have a wonderful day!"})

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "allow"
    assert set(body["scores"].keys()) == set(LABELS)
    assert body["latency_ms"] >= 0


def test_moderate_high_score_blocks() -> None:
    app_module._state["model"] = FakeModel({label: 0.99 for label in LABELS})
    app_module._state["thresholds"] = _thresholds()
    client = TestClient(app_module.app)

    response = client.post("/moderate", json={"text": "placeholder text"})

    assert response.json()["decision"] == "block"


def test_moderate_mid_score_flags() -> None:
    app_module._state["model"] = FakeModel({label: 0.6 for label in LABELS})
    app_module._state["thresholds"] = _thresholds()
    client = TestClient(app_module.app)

    response = client.post("/moderate", json={"text": "placeholder text"})

    assert response.json()["decision"] == "flag"


def test_moderate_empty_text_422() -> None:
    app_module._state["model"] = FakeModel({label: 0.0 for label in LABELS})
    app_module._state["thresholds"] = _thresholds()
    client = TestClient(app_module.app)

    response = client.post("/moderate", json={"text": ""})

    assert response.status_code == 422


def test_moderate_too_long_text_422() -> None:
    app_module._state["model"] = FakeModel({label: 0.0 for label in LABELS})
    app_module._state["thresholds"] = _thresholds()
    client = TestClient(app_module.app)

    response = client.post("/moderate", json={"text": "a" * 5001})

    assert response.status_code == 422


def test_moderate_model_not_loaded_503() -> None:
    app_module._state["model"] = None
    app_module._state["thresholds"] = None
    client = TestClient(app_module.app)

    response = client.post("/moderate", json={"text": "hello"})

    assert response.status_code == 503
