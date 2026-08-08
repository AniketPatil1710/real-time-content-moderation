"""Pydantic request/response models for the moderation API. Phase 6."""

from pydantic import BaseModel, Field

MAX_TEXT_LENGTH = 5000


class ModerateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)


class ModerateResponse(BaseModel):
    scores: dict[str, float]
    decision: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
