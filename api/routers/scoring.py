"""Scoring endpoints for the FCVAE anomaly detection API."""

import logging
import time

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from api.dependencies import model_store, settings
from api.schemas import (
    BatchScoreRequest,
    BatchScoreResponse,
    ScoreRequest,
    ScoreResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1")


def _score_single(req: ScoreRequest) -> ScoreResponse:
    """Score a single 24-hour window. Shared by single and batch endpoints."""
    detector = model_store.get_detector(req.combo)
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model for combo '{req.combo}' is not loaded",
        )

    # Build a DataFrame that score_window_detailed expects
    timestamps = req.timestamps or [str(i) for i in range(len(req.values))]
    df = pd.DataFrame({"timestamp": timestamps, "value": req.values})

    start = time.perf_counter()
    result = detector.score_window_detailed(df)
    latency_ms = (time.perf_counter() - start) * 1000

    if result is None:
        raise HTTPException(
            status_code=500,
            detail=f"Scoring failed for combo '{req.combo}'",
        )

    # Convert numpy arrays to plain Python types
    all_point_scores = result["point_scores"]
    if isinstance(all_point_scores, np.ndarray):
        all_point_scores = all_point_scores.tolist()

    scored_timestamp = timestamps[-1] if req.timestamps else None

    logger.info(
        "score_request combo=%s last_point_score=%.4f is_anomaly=%s latency_ms=%.1f",
        req.combo,
        result["last_point_score"],
        result["is_anomaly"],
        latency_ms,
    )

    return ScoreResponse(
        is_anomaly=result["is_anomaly"],
        last_point_score=result["last_point_score"],
        threshold=result["point_threshold"],
        combo=req.combo,
        scored_timestamp=scored_timestamp,
        all_point_scores=all_point_scores,
        model_version=settings.model_version,
    )


@router.post("/score", response_model=ScoreResponse)
async def score(req: ScoreRequest):
    """Score a single 24-hour window for one combo.

    Accepts exactly 24 hourly transaction counts and returns the anomaly
    decision based on the last-point NLL score.
    """
    return _score_single(req)


@router.post("/score/batch", response_model=BatchScoreResponse)
async def score_batch(req: BatchScoreRequest):
    """Score multiple 24-hour windows in a single call."""
    results = [_score_single(r) for r in req.requests]
    return BatchScoreResponse(results=results)
