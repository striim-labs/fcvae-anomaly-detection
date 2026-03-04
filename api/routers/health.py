"""Health check and model info endpoints."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.dependencies import COMBO_KEY_MAP, model_store
from api.schemas import ComboInfo, HealthResponse, ModelInfoResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    """Return service health and per-combo model load status."""
    models_loaded = {
        key: model_store.is_combo_loaded(key) for key in COMBO_KEY_MAP
    }
    healthy = model_store.any_loaded
    response = HealthResponse(
        status="healthy" if healthy else "unhealthy",
        models_loaded=models_loaded,
        model_version=model_store.effective_version,
        uptime_seconds=round(model_store.uptime_seconds, 1),
    )
    status_code = 200 if healthy else 503
    return JSONResponse(content=response.model_dump(), status_code=status_code)


@router.get("/v1/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Return metadata about loaded models, thresholds, and configuration."""
    combos = {}
    for combo_key in COMBO_KEY_MAP:
        threshold = model_store.get_threshold(combo_key)
        scorer_config = model_store.get_scorer_config_dict(combo_key)
        combos[combo_key] = ComboInfo(threshold=threshold, scorer_config=scorer_config)

    return ModelInfoResponse(
        model_config_info=model_store.get_model_config_dict(),
        combos=combos,
        model_version=model_store.effective_version,
        retrain_count=model_store.retrain_counter,
        last_retrain_time=model_store.last_retrain_time,
        combo_versions=dict(model_store.combo_versions),
    )
