"""Retrain and model reload endpoints."""

import logging
import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from api.dependencies import (
    COMBO_KEY_MAP,
    data_store,
    model_store,
    retrain_job_manager,
    settings,
)
from api.retrain import RetrainConfig, RetrainPipeline
from api.schemas import (
    ReloadRequest,
    ReloadResponse,
    RetrainRequest,
    RetrainResponse,
    RetrainStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1")

ALL_COMBOS: List[str] = list(COMBO_KEY_MAP.keys())


@router.post("/retrain", response_model=RetrainResponse, status_code=202)
async def trigger_retrain(req: RetrainRequest):
    """Trigger a background retrain job.

    Returns 202 with a job_id that can be polled via GET /v1/retrain/status/{job_id}.
    Returns 409 if a retrain job is already running.
    """
    if data_store is None:
        raise HTTPException(status_code=503, detail="Data store is disabled")

    combos = req.combos or ALL_COMBOS

    config = RetrainConfig(
        staging_dir=settings.model_path.replace("fcvae", "fcvae_staging"),
        production_dir=settings.model_path,
    )
    pipeline = RetrainPipeline(
        data_store=data_store,
        config=config,
        device=settings.device,
    )

    try:
        job_id = retrain_job_manager.submit_job(
            pipeline=pipeline,
            combos=combos,
            mode=req.mode,
            lookback_days=req.lookback_days,
            start_date=req.start_date,
            end_date=req.end_date,
            auto_reload=req.auto_reload,
            model_store=model_store,
            min_windows=req.min_windows,
        )
    except RuntimeError:
        raise HTTPException(status_code=409, detail="A retrain job is already running")

    return RetrainResponse(
        job_id=job_id,
        status="running",
        message=f"Retrain job started for combos: {combos}",
    )


@router.get("/retrain/status/{job_id}", response_model=RetrainStatusResponse)
async def retrain_status(job_id: str):
    """Poll the status of a retrain job."""
    job = retrain_job_manager.get_job_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    return RetrainStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        combo_results=job.get("combo_results"),
        started_at=job["started_at"],
        completed_at=job.get("completed_at"),
        duration_seconds=job.get("duration_seconds"),
    )


@router.post("/model/reload", response_model=ReloadResponse)
async def reload_models(req: ReloadRequest):
    """Manually reload models from staged artifacts.

    Copies staged artifacts to production dir and hot-swaps detectors.
    """
    combos = req.combos or ALL_COMBOS
    staging_base = Path(req.staging_dir)
    production_base = Path(settings.model_path)

    reloaded: List[str] = []
    for combo in combos:
        staging_combo = staging_base / combo
        if not staging_combo.exists():
            logger.warning(f"reload: staging dir not found for {combo}: {staging_combo}")
            continue

        production_combo = production_base / combo
        try:
            shutil.copytree(
                str(staging_combo), str(production_combo), dirs_exist_ok=True,
            )
        except Exception:
            logger.exception(f"reload: failed to copy artifacts for {combo}")
            continue

        ok = model_store.reload_combo(
            combo_key=combo,
            model_path=str(production_base),
            device=settings.device,
            n_samples=settings.n_samples,
        )
        if ok:
            reloaded.append(combo)

    return ReloadResponse(
        reloaded_combos=reloaded,
        message=f"Reloaded {len(reloaded)}/{len(combos)} combos",
    )
