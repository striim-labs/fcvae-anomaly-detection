"""
Step 5: FastAPI REST Scoring Service

Single-file FastAPI app that loads the Penny_All FCVAE model at startup
and provides scoring, health, and model info endpoints.

Usage:
    uv run python code/5_streaming_app.py
    # Then: curl http://localhost:8000/health
    # Score: curl -X POST http://localhost:8000/score -H "Content-Type: application/json" \
    #        -d '{"values": [10,12,8,15,20,25,30,35,40,38,32,28,22,18,15,12,10,8,6,5,4,3,2,1]}'
"""
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Pickle compatibility for old module paths
import src.model
import src.scorer
sys.modules["app.fcvae_model"] = src.model
sys.modules["app.fcvae_scorer"] = src.scorer
sys.modules["app.attention"] = src.model
sys.modules["fcvae_model"] = src.model
sys.modules["fcvae_scorer"] = src.scorer
sys.modules["attention"] = src.model

from fastapi import FastAPI, HTTPException
from src.model import FCVAE, FCVAEConfig
from src.scorer import FCVAEScorer
from src.schemas import ScoreRequest, ScoreResponse, HealthResponse, ModelInfoResponse
from src.utils import auto_device

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
model: FCVAE = None
scorer: FCVAEScorer = None
scaler = None
device: torch.device = None
model_config: FCVAEConfig = None

app = FastAPI(
    title="FCVAE Penny Anomaly Detector",
    description="Anomaly detection for penny transactions using Frequency-enhanced Conditional VAE",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
def load_model():
    """Load Penny_All model artifacts at startup."""
    global model, scorer, scaler, device, model_config

    model_dir = Path(os.environ.get("FCVAE_MODEL_PATH", str(PROJECT_ROOT / "models" / "fcvae" / "Penny_All")))
    device_pref = os.environ.get("FCVAE_DEVICE", None)

    device = auto_device(device_pref)
    logger.info(f"Device: {device}")
    logger.info(f"Loading model from: {model_dir}")

    # Load model
    checkpoint = torch.load(model_dir / "model.pt", map_location=device, weights_only=False)

    if "config" in checkpoint or "model_config" in checkpoint:
        model_config = checkpoint.get("config") or checkpoint.get("model_config")
        model = FCVAE(model_config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model_config = FCVAEConfig()
        model = FCVAE(model_config).to(device)
        model.load_state_dict(checkpoint)

    model.eval()
    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Load scorer
    scorer = FCVAEScorer.load(model_dir / "scorer.pkl")
    logger.info(f"Scorer loaded: threshold={scorer.last_point_threshold}")

    # Load scaler
    with open(model_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    logger.info(f"Scaler loaded: mean={scaler.mean_[0]:.2f}, std={scaler.scale_[0]:.2f}")

    logger.info("Penny_All model ready for scoring")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest):
    """Score a 24-hour window of penny transaction counts.

    Returns anomaly decision based on last-point NLL score.
    Lower NLL = more anomalous. score < threshold => anomaly.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Normalize input using fitted scaler
    values = np.array(request.values, dtype=np.float32)
    normalized = scaler.transform(values.reshape(-1, 1)).flatten()

    # Score
    window_score, point_scores = scorer.score_window(model, normalized, device)

    # Last-point decision
    last_point_score = float(point_scores[-1])
    threshold = scorer.last_point_threshold or scorer.point_threshold
    is_anomaly = last_point_score < threshold

    scored_timestamp = None
    if request.timestamps:
        scored_timestamp = request.timestamps[-1]

    return ScoreResponse(
        is_anomaly=is_anomaly,
        last_point_score=last_point_score,
        threshold=threshold,
        scored_timestamp=scored_timestamp,
        all_point_scores=[float(s) for s in point_scores],
    )


@app.get("/health", response_model=HealthResponse)
def health():
    """Service health check."""
    is_loaded = model is not None and scorer is not None and scaler is not None
    return HealthResponse(
        status="healthy" if is_loaded else "unhealthy",
        model_loaded=is_loaded,
        threshold=scorer.last_point_threshold if scorer else None,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    """Model metadata and configuration."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfoResponse(
        model_config_info={
            "window": model_config.window,
            "latent_dim": model_config.latent_dim,
            "condition_emb_dim": model_config.condition_emb_dim,
            "d_model": model_config.d_model,
            "n_head": model_config.n_head,
            "kernel_size": model_config.kernel_size,
            "hidden_dims": list(model_config.hidden_dims),
            "parameters": sum(p.numel() for p in model.parameters()),
        },
        scorer_config=scorer.get_stats() if scorer else {},
        threshold=scorer.last_point_threshold if scorer else None,
        last_point_threshold=scorer.last_point_threshold if scorer else None,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
