"""
FCVAE Streaming Detector for Real-Time Anomaly Detection

Wraps the trained FCVAE (Frequency-enhanced Conditional VAE) model for use
in the Kafka/Spark streaming pipeline. Scores 24-hour sliding windows using
single-pass NLL scoring (no MCMC) for low latency.

Per the paper (Wang et al., WWW'24), the FCVAE is a POINT-LEVEL detector:
each 24-hour window exists to predict the LAST POINT only. The frequency
conditioning (GFM/LFM) explicitly masks position [-1], so only that position
is a genuine prediction — positions [0..22] are reconstruction scores with
the model having seen them in the conditioning signal.

Anomaly decision: last_point_score < threshold => anomaly for that hour.

Key differences from LSTMStreamingDetector:
- No DoW features needed (frequency conditioning is internal)
- Input shape: (1, 1, 24) instead of (24, 3)
- Score type: Negative log-likelihood (not Mahalanobis)
- Score direction: Lower = anomalous (inverted)
- Threshold comparison: < instead of >
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import torch

from base_detector import BaseDetector
from fcvae_model import FCVAE, FCVAEConfig
from fcvae_scorer import FCVAEScorer

# Pickle compatibility: models were saved with "app." prefixed module paths
# (e.g., app.fcvae_model.FCVAEConfig) but Docker runs flat from /app.
# Register current modules under their app.* aliases so pickle can find them.
import sys
import types
import fcvae_model as _fcvae_model
import fcvae_scorer as _fcvae_scorer
import fcvae_augment as _fcvae_augment
import attention as _attention

_app_module = types.ModuleType("app")
sys.modules["app"] = _app_module
for _name, _mod in [("fcvae_model", _fcvae_model), ("fcvae_scorer", _fcvae_scorer),
                     ("fcvae_augment", _fcvae_augment), ("attention", _attention)]:
    sys.modules[f"app.{_name}"] = _mod
    setattr(_app_module, _name, _mod)

logger = logging.getLogger(__name__)


class FCVAEStreamingDetector(BaseDetector):
    """
    Streaming anomaly detector using pre-trained FCVAE model.

    Per the paper, this is a POINT-LEVEL detector: each 24-hour window
    produces a single anomaly score for the LAST POINT (position [-1]).
    The frequency conditioning masks this position, making it a genuine
    prediction rather than reconstruction.

    This detector:
    1. Loads a pre-trained FCVAE model, scaler, and scorer
    2. Scores incoming 24-hour windows using single-pass NLL
    3. Uses last-point score for anomaly decision: score[-1] < threshold
    4. Returns the anomalous point with metadata

    Unlike LSTM detector:
    - No DoW feature injection (FCVAE uses frequency conditioning)
    - Inverted score semantics (lower score = more anomalous)

    Attributes:
        model: Trained FCVAE model
        scaler: Fitted StandardScaler for normalization
        scorer: Fitted FCVAEScorer with threshold
        window_size: Number of hours per detection window (24)
    """

    def __init__(
        self,
        model_path: str = "models/transactions_fcvae",
        scaler_path: Optional[str] = None,
        scorer_path: Optional[str] = None,
        combo: Optional[Tuple[str, str]] = None,
        window_size: int = 24,
        min_samples: int = 24,
        device: Optional[str] = None,
        n_samples: int = 16,
        decision_mode: str = "last_point",
        severity_margin: float = 0.5,
        oracle_threshold: Optional[float] = None,
    ):
        """
        Initialize the FCVAE streaming detector.

        Args:
            model_path: Base directory containing saved FCVAE models
            scaler_path: Path to scaler (auto-derived from model_path if None)
            scorer_path: Path to scorer (auto-derived from model_path if None)
            combo: Optional (network_type, txn_type) tuple to load specific combo
            window_size: Samples per detection window (default: 24 = 1 day)
            min_samples: Minimum samples before detection runs
            device: Device to use ('cuda', 'cpu', or None for auto)
            n_samples: Number of latent samples for single-pass scoring
            decision_mode: Decision mode - "last_point" (paper-faithful, default) or
                          legacy modes "count_only"/"severity"/"hybrid" for backward compat
            severity_margin: Score margin below threshold for severity criterion (legacy modes)
            oracle_threshold: If set, overrides last_point_threshold with this value
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.scorer_path = scorer_path
        self.combo = combo
        self.window_size = window_size
        self._min_samples = min_samples
        self.n_samples = n_samples
        self.decision_mode = decision_mode
        self.severity_margin = severity_margin
        self.oracle_threshold = oracle_threshold

        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Model components (loaded lazily)
        self.model: Optional[FCVAE] = None
        self.scaler = None
        self.scorer: Optional[FCVAEScorer] = None

        # State
        self._is_loaded = False
        self._load_error: Optional[str] = None

        # Try to load model artifacts
        self._load_artifacts()

        logger.info(f"Initialized FCVAEStreamingDetector")
        logger.info(f"  Window size: {self.window_size}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Model loaded: {self._is_loaded}")

    def _load_artifacts(self) -> None:
        """Load model, scaler, and scorer from disk."""
        try:
            model_dir = Path(self.model_path)

            # If combo is specified, load from combo subdirectory
            if self.combo is not None:
                combo_dirname = f"{self.combo[0]}_{self.combo[1].replace('-', '')}"
                model_dir = model_dir / combo_dirname

            if not model_dir.exists():
                self._load_error = f"Model directory not found: {model_dir}"
                logger.warning(self._load_error)
                return

            # Derive paths if not specified
            model_file = model_dir / "model.pt"
            scaler_file = Path(self.scaler_path) if self.scaler_path else model_dir / "scaler.pkl"
            scorer_file = Path(self.scorer_path) if self.scorer_path else model_dir / "scorer.pkl"

            if not model_file.exists():
                self._load_error = f"Model file not found: {model_file}"
                logger.warning(self._load_error)
                return

            if not scaler_file.exists():
                self._load_error = f"Scaler file not found: {scaler_file}"
                logger.warning(self._load_error)
                return

            if not scorer_file.exists():
                self._load_error = f"Scorer file not found: {scorer_file}"
                logger.warning(self._load_error)
                return

            # Load model
            logger.info(f"Loading FCVAE model from {model_file}")
            torch.serialization.add_safe_globals([FCVAEConfig])
            checkpoint = torch.load(
                model_file,
                map_location=self.device,
                weights_only=False
            )
            self.model = FCVAE(config=checkpoint["model_config"])
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()

            # Load scaler
            logger.info(f"Loading scaler from {scaler_file}")
            with open(scaler_file, "rb") as f:
                self.scaler = pickle.load(f)

            # Load scorer
            logger.info(f"Loading scorer from {scorer_file}")
            self.scorer = FCVAEScorer.load(str(scorer_file))

            self._is_loaded = True
            self._load_error = None

            logger.info("All FCVAE model artifacts loaded successfully")
            logger.info(f"  Model config: window={self.model.config.window}, "
                       f"latent_dim={self.model.config.latent_dim}")
            logger.info(f"  Point threshold: {self.scorer.point_threshold:.4f}")
            logger.info(f"  Decision mode: {self.decision_mode}")

            if self.oracle_threshold is not None:
                old_thresh = self.scorer.last_point_threshold
                self.scorer.last_point_threshold = self.oracle_threshold
                logger.info(
                    f"  Overriding last_point_threshold with oracle: "
                    f"{self.oracle_threshold:.4f} (was: {old_thresh})"
                )

        except Exception as e:
            self._load_error = f"Failed to load FCVAE model artifacts: {e}"
            logger.error(self._load_error)
            self._is_loaded = False

    def _compute_sequence_score_and_errors(
        self,
        sequence: np.ndarray
    ) -> Tuple[bool, float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute anomaly score and point-wise NLL for a sequence.

        Uses single-pass scoring (no MCMC) for low latency.

        Per the paper, the anomaly decision uses ONLY the last point's score
        (position [-1]), since the frequency conditioning masks that position —
        making it the only genuine prediction. Positions [0..22] are reconstruction
        scores where the model had access to the values via conditioning.

        IMPORTANT: FCVAE score semantics are INVERTED from LSTM-AE:
        - Lower (more negative) NLL = more anomalous
        - Threshold comparison: score < threshold => anomaly

        Args:
            sequence: Normalized sequence of shape (window_size,)

        Returns:
            Tuple of (is_anomaly, last_point_score, point_scores, point_predictions, reconstruction_errors)
            - last_point_score: NLL at position [-1] (the genuine prediction)
            - point_scores: Per-point NLL for all positions (for diagnostics)
            - point_predictions: Boolean array (all positions, for diagnostics)
            - reconstruction_errors: Absolute errors for localization
        """
        if self.model is None or self.scorer is None:
            seq_len = len(sequence)
            return False, 0.0, np.zeros(seq_len), np.zeros(seq_len, dtype=bool), np.zeros(seq_len)

        # Prepare tensor: (1, 1, W) for FCVAE
        x = torch.FloatTensor(sequence).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # MCMC mode 2 scoring (matches evaluation/simulation)
            _, point_scores = self.model.score_mcmc(x)
            point_scores = point_scores.squeeze().cpu().numpy()  # (B,1,W) -> (W,)

            # Get reconstruction for error computation
            mu_x, var_x = self.model.reconstruct(x)
            mu_x = mu_x.squeeze().cpu().numpy()  # (W,)
            reconstruction_errors = np.abs(sequence - mu_x)

        # Last-point score: the only genuine prediction (position [-1] is masked in conditioning)
        last_point_score = float(point_scores[-1])

        # Use last-point-specific threshold if available (calibrated on position [-1] distribution)
        threshold = self.scorer.last_point_threshold or self.scorer.point_threshold

        # Point-level predictions for all positions (diagnostics/localization)
        point_predictions = point_scores < self.scorer.point_threshold

        # Anomaly decision
        if self.decision_mode == "last_point":
            # Paper-faithful: only the last point's score determines anomaly
            is_anomaly = last_point_score < threshold
        elif self.decision_mode == "severity":
            # Legacy: k-of-W count OR severity on all positions
            k = self.scorer.config.hard_criterion_k
            count_criterion = int(np.sum(point_predictions)) >= k
            severity_threshold = self.scorer.point_threshold - self.severity_margin
            severity_criterion = bool(np.any(point_scores < severity_threshold))
            is_anomaly = count_criterion or severity_criterion
        else:
            # Legacy count_only: k-of-W count on all positions
            k = self.scorer.config.hard_criterion_k
            is_anomaly = int(np.sum(point_predictions)) >= k

        return is_anomaly, last_point_score, point_scores, point_predictions, reconstruction_errors

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in the provided data.

        Per the paper, the FCVAE is a point-level detector: each 24-hour window
        produces an anomaly score for the LAST POINT only (position [-1]).
        The decision is: last_point_score < threshold => anomaly for that hour.

        Args:
            df: DataFrame with 'timestamp' and 'value' columns

        Returns:
            DataFrame with anomalous records and anomaly scores
        """
        if not self._is_loaded:
            logger.warning(f"Model not loaded: {self._load_error}")
            return pd.DataFrame()

        if len(df) < self._min_samples:
            logger.debug(
                f"Not enough samples for FCVAE detection: "
                f"{len(df)} < {self._min_samples}"
            )
            return pd.DataFrame()

        # Extract values and normalize (no DoW features for FCVAE)
        values = df["value"].values.astype(float).reshape(-1, 1)
        values_normalized = self.scaler.transform(values).flatten()

        # Get the window
        window_values = values_normalized[-self.window_size:]

        is_anomaly, last_point_score, point_scores, point_predictions, reconstruction_errors = \
            self._compute_sequence_score_and_errors(window_values)

        # Log detailed scoring information
        logger.info(
            f"  FCVAE Scoring: raw_range=[{values.min():.0f}, {values.max():.0f}], "
            f"normalized_range=[{values_normalized.min():.4f}, {values_normalized.max():.4f}]"
        )

        threshold = self.scorer.last_point_threshold or self.scorer.point_threshold
        logger.info(
            f"  Last-point score: {last_point_score:.4f}, "
            f"threshold={threshold:.4f} "
            f"-> {'ANOMALY' if is_anomaly else 'NORMAL'} (mode={self.decision_mode})"
        )

        if is_anomaly:
            # The anomalous point is the last point in the window
            window_df = df.tail(self.window_size).copy().reset_index(drop=True)
            window_df["point_score"] = point_scores
            window_df["is_anomalous_point"] = point_predictions

            # Get timestamps for localization
            timestamps_list = window_df["timestamp"].astype(str).tolist()

            # Localize anomaly: find the contiguous region with most anomalous points
            localization = self._localize_anomaly(
                point_predictions, reconstruction_errors, timestamps_list
            )

            # The last point is the primary anomalous point
            last_point_row = window_df.iloc[[-1]].copy()
            last_point_row["anomaly_score"] = last_point_score

            logger.info(
                f"FCVAE detected anomaly at last point! "
                f"score={last_point_score:.4f} < threshold={self.scorer.point_threshold:.4f}"
            )

            # Add localization metadata
            last_point_row["localization_start"] = localization["anomaly_start"]
            last_point_row["localization_end"] = localization["anomaly_end"]
            last_point_row["scale_hours"] = localization["scale_hours"]
            last_point_row["contrast_ratio"] = localization["contrast_ratio"]

            logger.info(
                f"  Localized anomaly to {localization['scale_hours']}h window: "
                f"{localization['anomaly_start']} - {localization['anomaly_end']} "
                f"(contrast={localization['contrast_ratio']:.2f})"
            )

            return last_point_row[["timestamp", "value", "anomaly_score",
                                     "localization_start", "localization_end",
                                     "scale_hours", "contrast_ratio"]]

        return pd.DataFrame()

    def _localize_anomaly(
        self,
        point_predictions: np.ndarray,
        reconstruction_errors: np.ndarray,
        timestamps: List[str],
        scale_hours: int = 6
    ) -> Dict[str, Any]:
        """
        Localize anomaly to a contiguous window.

        Finds the scale_hours window with highest anomaly concentration.

        Args:
            point_predictions: Boolean array of per-point anomaly predictions
            reconstruction_errors: Reconstruction errors for scoring
            timestamps: List of timestamp strings
            scale_hours: Size of localization window

        Returns:
            Dict with localization info
        """
        window_size = len(point_predictions)

        if scale_hours >= window_size:
            return {
                "anomaly_start": timestamps[0],
                "anomaly_end": timestamps[-1],
                "anomaly_start_idx": 0,
                "anomaly_end_idx": window_size - 1,
                "scale_hours": window_size,
                "contrast_ratio": 1.0,
            }

        # Find window with most anomalous points
        best_start = 0
        best_count = 0
        best_error = 0

        for start in range(window_size - scale_hours + 1):
            end = start + scale_hours
            count = point_predictions[start:end].sum()
            error = reconstruction_errors[start:end].sum()

            if count > best_count or (count == best_count and error > best_error):
                best_count = count
                best_start = start
                best_error = error

        end_idx = best_start + scale_hours - 1

        # Compute contrast ratio
        inside_error = reconstruction_errors[best_start:end_idx + 1].mean()
        outside_mask = np.ones(window_size, dtype=bool)
        outside_mask[best_start:end_idx + 1] = False
        outside_error = reconstruction_errors[outside_mask].mean() if outside_mask.sum() > 0 else 1e-8

        contrast_ratio = inside_error / (outside_error + 1e-8)

        return {
            "anomaly_start": timestamps[best_start],
            "anomaly_end": timestamps[end_idx],
            "anomaly_start_idx": best_start,
            "anomaly_end_idx": end_idx,
            "scale_hours": scale_hours,
            "contrast_ratio": float(contrast_ratio),
        }

    def score_window_detailed(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Score a 24-hour window and return full per-point results.

        Unlike detect(), this always returns scoring info regardless of
        the anomaly decision.

        Args:
            df: DataFrame with 'timestamp' and 'value' columns

        Returns:
            Dict with per-point scores and last-point decision, or None if not ready.
        """
        if not self._is_loaded:
            logger.warning(f"Model not loaded: {self._load_error}")
            return None

        if len(df) < self._min_samples:
            return None

        values = df["value"].values.astype(float).reshape(-1, 1)
        values_normalized = self.scaler.transform(values).flatten()
        window_values = values_normalized[-self.window_size:]

        is_anomaly, last_point_score, point_scores, point_predictions, reconstruction_errors = \
            self._compute_sequence_score_and_errors(window_values)

        window_df = df.tail(self.window_size).copy().reset_index(drop=True)

        threshold = self.scorer.last_point_threshold or self.scorer.point_threshold
        return {
            "is_anomaly": bool(is_anomaly),
            "last_point_score": last_point_score,
            "point_scores": point_scores,
            "point_predictions": point_predictions,
            "num_anomalous_points": int(point_predictions.sum()),
            "point_threshold": float(threshold),
            "decision_mode": self.decision_mode,
            "timestamps": window_df["timestamp"].tolist(),
            "values": window_df["value"].tolist(),
        }

    def detect_window(
        self,
        timestamps: List[str],
        values: List[float]
    ) -> Optional[Dict[str, Any]]:
        """
        Detect anomaly in a specific window of data.

        Alternative interface for when data comes as separate lists.
        Uses last-point scoring per the paper.

        Args:
            timestamps: List of timestamp strings
            values: List of values

        Returns:
            Dict with anomaly info if detected, None otherwise
        """
        if not self._is_loaded:
            return None

        if len(values) < self.window_size:
            return None

        # Normalize
        values_array = np.array(values[-self.window_size:]).reshape(-1, 1)
        values_normalized = self.scaler.transform(values_array).flatten()

        # Compute scores
        is_anomaly, last_point_score, point_scores, point_predictions, reconstruction_errors = \
            self._compute_sequence_score_and_errors(values_normalized)

        if is_anomaly:
            window_timestamps = timestamps[-self.window_size:]
            localization = self._localize_anomaly(
                point_predictions, reconstruction_errors, window_timestamps
            )

            return {
                "is_anomaly": True,
                "last_point_score": last_point_score,
                "timestamp": timestamps[-1],
                "start_time": timestamps[-self.window_size],
                "end_time": timestamps[-1],
                "window_size": self.window_size,
                "threshold": float(self.scorer.point_threshold),
                "decision_mode": self.decision_mode,
                "localization": localization,
            }

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        stats = {
            "detector_type": "fcvae",
            "is_loaded": self._is_loaded,
            "window_size": self.window_size,
            "min_samples": self._min_samples,
            "device": str(self.device),
            "n_samples": self.n_samples,
            "decision_mode": self.decision_mode,
        }

        if self._is_loaded and self.scorer is not None:
            stats["point_threshold"] = float(self.scorer.point_threshold) if self.scorer.point_threshold else None
            stats["threshold_method"] = self.scorer.config.threshold_method
            stats["score_mode"] = self.scorer.config.score_mode

        if self._load_error:
            stats["load_error"] = self._load_error

        if self._is_loaded and self.model is not None:
            stats["model_config"] = {
                "window": self.model.config.window,
                "latent_dim": self.model.config.latent_dim,
                "condition_emb_dim": self.model.config.condition_emb_dim,
                "d_model": self.model.config.d_model,
                "n_head": self.model.config.n_head,
            }

        return stats

    def get_name(self) -> str:
        """Get the detector name for display purposes."""
        return "FCVAE (Frequency-enhanced Conditional VAE)"

    @property
    def is_ready(self) -> bool:
        """Check if the detector is ready to perform detection."""
        return self._is_loaded

    @property
    def min_samples_required(self) -> int:
        """Minimum number of samples required for detection."""
        return self._min_samples
