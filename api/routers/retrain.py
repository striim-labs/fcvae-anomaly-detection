"""Retrain and model reload endpoints."""

import asyncio
import logging
import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from api.dependencies import (
    COMBO_KEY_MAP,
    data_store,
    model_event_log,
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
    If a retrain job is already running, queues the request and waits up to 120s
    for the current job to finish before submitting. Returns 409 only if the
    wait times out.
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

    # If a job is already running, wait for it to finish before submitting.
    # This serializes concurrent retrain requests (e.g., from per-combo triggers)
    # instead of rejecting them with 409.
    max_wait_seconds = 120
    poll_interval = 2.0
    waited = 0.0

    while True:
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
                event_log=model_event_log,
            )
            break
        except RuntimeError:
            if waited >= max_wait_seconds:
                raise HTTPException(
                    status_code=409,
                    detail=f"A retrain job is still running after waiting {max_wait_seconds}s",
                )
            logger.info(
                f"Retrain queue: waiting for current job to finish "
                f"(combo={combos}, waited={waited:.0f}s)"
            )
            await asyncio.sleep(poll_interval)
            waited += poll_interval

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
            reread_oracle_thresholds=True,
        )
        if ok:
            reloaded.append(combo)
            model_event_log.log_event(
                "model_reloaded",
                combo=combo,
                details={"source": "manual", "version": model_store.combo_versions.get(combo, 0)},
            )

    return ReloadResponse(
        reloaded_combos=reloaded,
        message=f"Reloaded {len(reloaded)}/{len(combos)} combos",
    )