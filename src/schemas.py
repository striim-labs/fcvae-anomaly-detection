"""Pydantic request/response models for the FCVAE scoring API."""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

WINDOW_SIZE = 24

VALID_COMBOS = ["Penny_All", "Accel_CMP", "Accel_nopin", "Star_CMP", "Star_nopin"]


class ScoreRequest(BaseModel):
    """Single scoring request: a 24-hour window of transaction counts."""

    values: List[float] = Field(
        ...,
        description="Exactly 24 hourly transaction counts, chronological order. "
        "Position [-1] is the hour being scored.",
    )
    combo: str = Field(
        default="Penny_All",
        description="Model combo to score against. One of: Penny_All, Accel_CMP, "
        "Accel_nopin, Star_CMP, Star_nopin.",
    )
    timestamps: Optional[List[str]] = Field(
        default=None,
        description="ISO-format timestamps for each value. Used for response metadata only.",
    )

    @field_validator("values")
    @classmethod
    def validate_values_length(cls, v: List[float]) -> List[float]:
        if len(v) != WINDOW_SIZE:
            raise ValueError(
                f"values must contain exactly {WINDOW_SIZE} elements, got {len(v)}"
            )
        return v

    @field_validator("combo")
    @classmethod
    def validate_combo(cls, v: str) -> str:
        if v not in VALID_COMBOS:
            raise ValueError(
                f"combo must be one of {VALID_COMBOS}, got '{v}'"
            )
        return v

    @field_validator("timestamps")
    @classmethod
    def validate_timestamps_length(cls, v: Optional[List[str]], info) -> Optional[List[str]]:
        if v is not None and len(v) != WINDOW_SIZE:
            raise ValueError(
                f"timestamps must contain exactly {WINDOW_SIZE} elements, got {len(v)}"
            )
        return v


class ScoreResponse(BaseModel):
    """Response from a single scoring request."""

    is_anomaly: bool = Field(
        ..., description="Whether the last point crossed below the calibrated threshold"
    )
    last_point_score: float = Field(
        ...,
        description="NLL score at position [-1]. More negative = more anomalous.",
    )
    combo: str = Field(
        ..., description="Which model combo was used for scoring"
    )
    threshold: float = Field(
        ..., description="Calibrated threshold for the scored model"
    )
    scored_timestamp: Optional[str] = Field(
        default=None,
        description="Timestamp of the scored hour (last element of timestamps)",
    )
    all_point_scores: List[float] = Field(
        ...,
        description="NLL scores for all 24 positions. "
        "Positions [0..22] are reconstructions; [-1] is the genuine prediction.",
    )


class HealthResponse(BaseModel):
    """Service health and model load status."""

    status: Literal["healthy", "unhealthy"]
    models_loaded: List[str] = Field(default_factory=list)
    default_model: str = "Penny_All"


class ModelInfoResponse(BaseModel):
    """Metadata about loaded model and configuration."""

    model_config_info: Dict
    scorer_config: Dict
    threshold: Optional[float] = None
    last_point_threshold: Optional[float] = None
