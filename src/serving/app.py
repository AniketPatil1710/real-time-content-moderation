"""FastAPI application: POST /moderate, GET /health. Phase 6.

ONNX session loaded once at startup via the lifespan handler
(Architecture.md Decision #1), never per-request.
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from src.data.preprocess import load_label_names, load_training_config
from src.models.export_onnx import QUANTIZED_FILE_NAME
from src.serving.inference import InferenceError, ModerationModel, decide
from src.serving.schemas import HealthResponse, ModerateRequest, ModerateResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# INT8 model: Phase 5 confirmed its macro F1 beats PyTorch, so it's the one to serve
# (see Memory.md Decisions #11/#12 — file_name must be QUANTIZED_FILE_NAME, not "model.onnx").
MODEL_DIR = Path("models/onnx/int8")
THRESHOLDS_PATH = Path("configs/thresholds.json")

_state: dict = {"model": None, "thresholds": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    label_names = load_label_names()
    config = load_training_config()
    try:
        _state["model"] = ModerationModel(MODEL_DIR, QUANTIZED_FILE_NAME, config["model"]["max_length"], label_names)
        with THRESHOLDS_PATH.open() as f:
            _state["thresholds"] = json.load(f)
        logger.info("Model and thresholds loaded from %s / %s", MODEL_DIR, THRESHOLDS_PATH)
    except Exception:
        logger.exception("Failed to load model/thresholds at startup — /moderate will return 503")
        _state["model"] = None
        _state["thresholds"] = None
    yield
    _state["model"] = None
    _state["thresholds"] = None


app = FastAPI(title="ModGuard Content Moderation API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    status = "ok" if _state.get("model") is not None else "model_not_loaded"
    return HealthResponse(status=status)


@app.post("/moderate", response_model=ModerateResponse)
async def moderate(request: ModerateRequest) -> ModerateResponse:
    model = _state.get("model")
    thresholds = _state.get("thresholds")
    if model is None or thresholds is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        scores, latency_ms = model.predict(request.text)
    except InferenceError:
        raise HTTPException(status_code=500, detail="Inference failed")

    decision = decide(scores, thresholds)
    return ModerateResponse(scores=scores, decision=decision, latency_ms=latency_ms)
