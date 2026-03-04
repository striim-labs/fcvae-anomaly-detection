"""Model registry lifecycle and shared dependencies for the FCVAE scoring API."""

import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

from pydantic_settings import BaseSettings, SettingsConfigDict

# Add project root to Python path so `from app.X` imports work
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.fcvae_streaming_detector import FCVAEStreamingDetector  # noqa: E402

logger = logging.getLogger(__name__)

# Mapping from API combo string keys to (network_type, txn_type) tuples
COMBO_KEY_MAP: Dict[str, Tuple[str, str]] = {
    "Accel_CMP": ("Accel", "CMP"),
    "Accel_nopin": ("Accel", "no-pin"),
    "Star_CMP": ("Star", "CMP"),
    "Star_nopin": ("Star", "no-pin"),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FCVAE_")

    model_path: str = "models/fcvae"
    model_version: str = "1.0.0"
    n_samples: int = 16
    score_mode: str = "single_pass"
    device: str = "cpu"
    api_key: str = ""
    log_level: str = "INFO"
    data_store_dir: str = "data/retrain"
    enable_data_store: bool = True


settings = Settings()


class ModelStore:
    """Holds per-combo detectors and startup metadata."""

    def __init__(self) -> None:
        self.detectors: Dict[str, FCVAEStreamingDetector] = {}
        self.oracle_thresholds: Dict[str, float] = {}
        self.start_time: float = time.time()
        # Version tracking (Phase E)
        self.retrain_counter: int = 0
        self.combo_versions: Dict[str, int] = {}
        self.last_retrain_time: Optional[str] = None

    def load_all(self, model_path: str, device: str, n_samples: int) -> None:
        """Load detectors for all four combos."""
        oracle_path = Path(model_path) / "oracle_thresholds.json"
        if oracle_path.exists():
            with open(oracle_path) as f:
                self.oracle_thresholds = json.load(f)
            logger.info(f"Loaded oracle thresholds: {self.oracle_thresholds}")

        for combo_key, combo_tuple in COMBO_KEY_MAP.items():
            oracle_thresh = self.oracle_thresholds.get(combo_key)
            try:
                detector = FCVAEStreamingDetector(
                    model_path=model_path,
                    combo=combo_tuple,
                    window_size=24,
                    min_samples=24,
                    device=device,
                    n_samples=n_samples,
                    decision_mode="last_point",
                    oracle_threshold=oracle_thresh,
                )
                if detector.is_ready:
                    self.detectors[combo_key] = detector
                    logger.info(f"Loaded detector for {combo_key}")
                else:
                    logger.warning(
                        f"Detector for {combo_key} failed to load: {detector._load_error}"
                    )
            except Exception:
                logger.exception(f"Failed to create detector for {combo_key}")

    def get_detector(self, combo_key: str) -> Optional[FCVAEStreamingDetector]:
        return self.detectors.get(combo_key)

    def is_combo_loaded(self, combo_key: str) -> bool:
        return combo_key in self.detectors

    @property
    def any_loaded(self) -> bool:
        return len(self.detectors) > 0

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def effective_version(self) -> str:
        """Version string reflecting retrain history: '{base}.{retrain_counter}'."""
        base = settings.model_version
        if self.retrain_counter > 0:
            return f"{base}.{self.retrain_counter}"
        return base

    def get_model_config_dict(self) -> dict:
        """Return the FCVAEConfig from the first loaded model as a dict."""
        for detector in self.detectors.values():
            if detector.model is not None:
                return asdict(detector.model.config)
        return {}

    def get_scorer_config_dict(self, combo_key: str) -> dict:
        """Return the scorer config for a combo as a dict."""
        detector = self.detectors.get(combo_key)
        if detector and detector.scorer:
            return asdict(detector.scorer.config)
        return {}

    def get_threshold(self, combo_key: str) -> Optional[float]:
        """Return the active threshold for a combo."""
        detector = self.detectors.get(combo_key)
        if detector and detector.scorer:
            return detector.scorer.last_point_threshold or detector.scorer.point_threshold
        return None

    def reload_combo(
        self, combo_key: str, model_path: str, device: str, n_samples: int,
    ) -> bool:
        """Hot-swap a combo's detector with a freshly loaded one.

        Creates a new FCVAEStreamingDetector from model_path/{combo_dirname}/,
        verifies it loaded successfully, then atomically swaps the dict entry
        (GIL-safe dict assignment).

        Returns True on success, False on failure.
        """
        combo_tuple = COMBO_KEY_MAP.get(combo_key)
        if combo_tuple is None:
            logger.error(f"reload_combo: unknown combo_key '{combo_key}'")
            return False

        try:
            new_detector = FCVAEStreamingDetector(
                model_path=model_path,
                combo=combo_tuple,
                window_size=24,
                min_samples=24,
                device=device,
                n_samples=n_samples,
                decision_mode="last_point",
            )
            if not new_detector.is_ready:
                logger.error(
                    f"reload_combo: new detector for {combo_key} failed to load: "
                    f"{new_detector._load_error}"
                )
                return False

            # Atomic swap — GIL-safe dict assignment
            self.detectors[combo_key] = new_detector

            # Increment version counters
            self.retrain_counter += 1
            self.combo_versions[combo_key] = self.combo_versions.get(combo_key, 0) + 1
            self.last_retrain_time = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(),
            )

            logger.info(f"reload_combo: hot-swapped detector for {combo_key}")
            return True

        except Exception:
            logger.exception(f"reload_combo: failed to create detector for {combo_key}")
            return False


# Singleton — populated during FastAPI lifespan
model_store = ModelStore()

# Data store for accumulating scored windows (used by retrain pipeline)
from api.data_store import WindowDataStore  # noqa: E402

data_store: Optional[WindowDataStore] = (
    WindowDataStore(data_dir=settings.data_store_dir)
    if settings.enable_data_store
    else None
)

# Model event log (Phase E — JSONL lifecycle events for Striim observability)
from api.model_events import ModelEventLog  # noqa: E402

model_event_log = ModelEventLog(
    log_path=str(Path(settings.data_store_dir) / "model_events.jsonl"),
)

# Retrain job manager singleton (lazy import to avoid circular deps)
from api.retrain_job import RetrainJobManager  # noqa: E402

retrain_job_manager = RetrainJobManager()
