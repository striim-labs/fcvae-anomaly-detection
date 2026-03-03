"""Pydantic request/response models for the FCVAE scoring API."""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

VALID_COMBOS = {"Accel_CMP", "Accel_nopin", "Star_CMP", "Star_nopin"}

WINDOW_SIZE = 24


class ScoreRequest(BaseModel):
    """Single scoring request: a 24-hour window for one combo."""

    combo: str = Field(
        ...,
        description="Model combo key: Accel_CMP, Accel_nopin, Star_CMP, or Star_nopin",
    )
    values: List[float] = Field(
        ...,
        description="Exactly 24 hourly transaction counts, chronological order. "
        "Position [-1] is the hour being scored.",
    )
    timestamps: Optional[List[str]] = Field(
        default=None,
        description="ISO-format timestamps for each value. Used for response metadata only.",
    )

    @field_validator("combo")
    @classmethod
    def validate_combo(cls, v: str) -> str:
        if v not in VALID_COMBOS:
            raise ValueError(
                f"Invalid combo '{v}'. Must be one of: {sorted(VALID_COMBOS)}"
            )
        return v

    @field_validator("values")
    @classmethod
    def validate_values_length(cls, v: List[float]) -> List[float]:
        if len(v) != WINDOW_SIZE:
            raise ValueError(
                f"values must contain exactly {WINDOW_SIZE} elements, got {len(v)}"
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
    threshold: float = Field(
        ..., description="Calibrated threshold for this combo"
    )
    combo: str = Field(..., description="Echo of the requested combo")
    scored_timestamp: Optional[str] = Field(
        default=None,
        description="Timestamp of the scored hour (last element of timestamps)",
    )
    all_point_scores: List[float] = Field(
        ...,
        description="NLL scores for all 24 positions. "
        "Positions [0..22] are reconstructions; [-1] is the genuine prediction.",
    )
    model_version: str = Field(..., description="Version of the loaded model artifacts")


class BatchScoreRequest(BaseModel):
    """Batch scoring request: multiple windows in a single call."""

    requests: List[ScoreRequest] = Field(
        ..., description="List of scoring requests", min_length=1
    )


class BatchScoreResponse(BaseModel):
    """Response from a batch scoring request."""

    results: List[ScoreResponse]


class HealthResponse(BaseModel):
    """Service health and model load status."""

    status: Literal["healthy", "unhealthy"]
    models_loaded: Dict[str, bool]
    model_version: str
    uptime_seconds: float


class ComboInfo(BaseModel):
    """Per-combo model metadata."""

    threshold: Optional[float] = None
    scorer_config: Dict


class ModelInfoResponse(BaseModel):
    """Metadata about loaded models and configuration."""

    model_config_info: Dict
    combos: Dict[str, ComboInfo]
    model_version: str


# --- Retrain schemas ---


class RetrainRequest(BaseModel):
    """Request to trigger a retrain job."""

    mode: str = Field(
        default="weekly",
        description="Retrain mode: 'weekly' (lookback_days) or 'manual' (date range)",
    )
    combos: Optional[List[str]] = Field(
        default=None,
        description="Combo keys to retrain. Default: all 4.",
    )
    lookback_days: int = Field(default=7, description="Lookback days for weekly mode")
    start_date: Optional[str] = Field(default=None, description="Start date for manual mode (ISO)")
    end_date: Optional[str] = Field(default=None, description="End date for manual mode (ISO)")
    auto_reload: bool = Field(default=True, description="Auto-promote and hot-swap on success")
    min_windows: int = Field(default=50, description="Minimum normal windows required per combo")

    @field_validator("combos")
    @classmethod
    def validate_combos(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            for combo in v:
                if combo not in VALID_COMBOS:
                    raise ValueError(
                        f"Invalid combo '{combo}'. Must be one of: {sorted(VALID_COMBOS)}"
                    )
        return v


class RetrainResponse(BaseModel):
    """Response from POST /v1/retrain."""

    job_id: str
    status: str
    message: str


class RetrainStatusResponse(BaseModel):
    """Response from GET /v1/retrain/status/{job_id}."""

    job_id: str
    status: str  # "running", "completed", "failed"
    combo_results: Optional[List[dict]] = None
    started_at: str
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None


class ReloadRequest(BaseModel):
    """Request to manually reload models from staged artifacts."""

    combos: Optional[List[str]] = Field(
        default=None,
        description="Combo keys to reload. Default: all 4.",
    )
    staging_dir: str = Field(
        default="models/fcvae_staging",
        description="Path to staged artifacts directory",
    )

    @field_validator("combos")
    @classmethod
    def validate_combos(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            for combo in v:
                if combo not in VALID_COMBOS:
                    raise ValueError(
                        f"Invalid combo '{combo}'. Must be one of: {sorted(VALID_COMBOS)}"
                    )
        return v


class ReloadResponse(BaseModel):
    """Response from POST /v1/model/reload."""

    reloaded_combos: List[str]
    message: str
