"""FCVAE Scoring API — FastAPI application entry point.

Wraps the existing FCVAEStreamingDetector in a stateless REST API.
Models are loaded once at startup via the lifespan context manager.

Run locally:
    PYTHONPATH=app:$PYTHONPATH uvicorn api.main:app --reload --port 8000

Or via UV from the project root:
    cd api && uv run uvicorn api.main:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.dependencies import data_store, model_store, settings
from api.routers import health, retrain, scoring

# Configure structured logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all combo models at startup, clean up on shutdown."""
    logger.info(
        "Starting FCVAE Scoring API — model_path=%s device=%s n_samples=%d",
        settings.model_path,
        settings.device,
        settings.n_samples,
    )
    model_store.load_all(
        model_path=settings.model_path,
        device=settings.device,
        n_samples=settings.n_samples,
    )
    loaded = [k for k, v in model_store.detectors.items()]
    logger.info("Models loaded: %s", loaded)
    if data_store is not None:
        logger.info("Data store initialized: %s", data_store.get_stats())
    else:
        logger.info("Data store disabled")
    yield
    logger.info("Shutting down FCVAE Scoring API")


app = FastAPI(
    title="FCVAE Anomaly Detection API",
    version=settings.model_version,
    description=(
        "Real-time anomaly scoring on transaction frequency time series "
        "using a Frequency-enhanced Conditional VAE (FCVAE)."
    ),
    lifespan=lifespan,
)

# --- Optional API key auth ---------------------------------------------------

if settings.api_key:

    @app.middleware("http")
    async def check_api_key(request: Request, call_next):
        # Skip auth for health, docs, and openapi schema
        if request.url.path in ("/health", "/docs", "/redoc", "/openapi.json"):
            return await call_next(request)
        api_key = request.headers.get("x-api-key", "")
        if api_key != settings.api_key:
            return JSONResponse(status_code=401, content={"detail": "Invalid API key"})
        return await call_next(request)


# --- Routers ------------------------------------------------------------------

app.include_router(health.router)
app.include_router(scoring.router)
app.include_router(retrain.router)
