"""
Step 5: FastAPI REST Scoring Service (Multi-Model)

Loads all 5 FCVAE models (4 combo + Penny_All) at startup and routes
scoring requests by the `combo` field in each request.

Usage:
    uv run python code/5_streaming_app.py
    # Then: curl http://localhost:8000/health
    # Score: curl -X POST http://localhost:8000/score -H "Content-Type: application/json" \
    #        -d '{"values": [10,12,8,15,20,25,30,35,40,38,32,28,22,18,15,12,10,8,6,5,4,3,2,1], "combo": "Penny_All"}'
"""
import logging
import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict

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
from src.schemas import ScoreRequest, ScoreResponse, HealthResponse, ModelInfoResponse, VALID_COMBOS
from src.utils import auto_device

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model directory mapping
# ---------------------------------------------------------------------------
MODEL_DIRS: Dict[str, str] = {
    "Penny_All": "Penny_All",
    "Accel_CMP": "Accel_CMP",
    "Accel_nopin": "Accel_nopin",
    "Star_CMP": "Star_CMP",
    "Star_nopin": "Star_nopin",
}

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
loaded_models: Dict[str, Dict] = {}
device: torch.device = None

app = FastAPI(
    title="FCVAE Anomaly Detector",
    description="Multi-model anomaly detection using Frequency-enhanced Conditional VAE",
    version="2.0.0",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_model_entry(combo_name: str) -> Dict:
    """Look up a loaded model entry by combo name, raising 404 if missing."""
    if combo_name not in loaded_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{combo_name}' not loaded. Available: {list(loaded_models.keys())}",
        )
    return loaded_models[combo_name]


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
def load_models():
    """Load all 5 FCVAE model artifacts at startup."""
    global device

    models_root = Path(
        os.environ.get("FCVAE_MODELS_ROOT", str(PROJECT_ROOT / "models" / "fcvae"))
    )
    device_pref = os.environ.get("FCVAE_DEVICE", None)

    device = auto_device(device_pref)
    logger.info(f"Device: {device}")
    logger.info(f"Models root: {models_root}")

    for combo_name, dir_name in MODEL_DIRS.items():
        model_dir = models_root / dir_name

        if not model_dir.exists():
            logger.warning(f"Skipping {combo_name}: directory not found at {model_dir}")
            continue

        try:
            # Load model
            checkpoint = torch.load(
                model_dir / "model.pt", map_location=device, weights_only=False
            )

            if "config" in checkpoint or "model_config" in checkpoint:
                model_config = checkpoint.get("config") or checkpoint.get("model_config")
                model = FCVAE(model_config).to(device)
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model_config = FCVAEConfig()
                model = FCVAE(model_config).to(device)
                model.load_state_dict(checkpoint)

            model.eval()

            # Load scorer
            scorer = FCVAEScorer.load(model_dir / "scorer.pkl")

            # Load scaler
            with open(model_dir / "scaler.pkl", "rb") as f:
                scaler = pickle.load(f)

            loaded_models[combo_name] = {
                "model": model,
                "scorer": scorer,
                "scaler": scaler,
                "config": model_config,
            }

            logger.info(
                f"Loaded {combo_name}: "
                f"{sum(p.numel() for p in model.parameters()):,} params, "
                f"threshold={scorer.last_point_threshold}"
            )
        except Exception as e:
            logger.error(f"Failed to load {combo_name} from {model_dir}: {e}")

    logger.info(f"Models ready: {list(loaded_models.keys())}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/score", response_model=ScoreResponse)
@app.post("/v1/score", response_model=ScoreResponse, include_in_schema=False)
def score(request: ScoreRequest):
    """Score a 24-hour window of transaction counts.

    Routes to the model specified by request.combo (default: Penny_All).
    Returns anomaly decision based on last-point NLL score.
    Lower NLL = more anomalous. score < threshold => anomaly.
    """
    entry = _get_model_entry(request.combo)
    model = entry["model"]
    scorer = entry["scorer"]
    scaler = entry["scaler"]

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
        combo=request.combo,
        threshold=threshold,
        scored_timestamp=scored_timestamp,
        all_point_scores=[float(s) for s in point_scores],
    )


@app.get("/health", response_model=HealthResponse)
def health():
    """Service health check."""
    is_loaded = len(loaded_models) > 0
    return HealthResponse(
        status="healthy" if is_loaded else "unhealthy",
        models_loaded=list(loaded_models.keys()),
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info(combo: str = "Penny_All"):
    """Model metadata and configuration."""
    entry = _get_model_entry(combo)
    model = entry["model"]
    scorer = entry["scorer"]
    model_config = entry["config"]

    return ModelInfoResponse(
        model_config_info={
            "combo": combo,
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

    # Suppress noisy warnings that don't affect behavior
    warnings.filterwarnings("ignore", message=".*output with one or more elements was resized.*")
    logging.getLogger("uvicorn.error").setLevel(logging.ERROR)

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
