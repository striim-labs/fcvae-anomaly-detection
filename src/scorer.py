"""
FCVAE Anomaly Scorer

NLL-based scoring for FCVAE models.

Score semantics (INVERTED from typical anomaly detectors):
- Lower (more negative) NLL = more anomalous
- Threshold comparison: score < threshold => anomaly
- Penny oracle threshold: -21.3322
"""
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class FCVAEScorerConfig:
    """Configuration for FCVAE anomaly scorer."""
    threshold_method: str = "percentile"    # "percentile" or "f1_max"
    threshold_percentile: float = 5.0       # LOW percentile (anomalies are low scores)
    hard_criterion_k: int = 3               # Points below threshold to flag window
    score_mode: str = "single_pass"         # "single_pass" (fast) or "mcmc" (accurate)
    n_samples: int = 16                     # Latent samples for single-pass scoring

    # Decision logic for window-level predictions
    decision_mode: str = "severity"
    severity_margin: float = 0.5
    outlier_z_threshold: float = 3.0


class FCVAEScorer:
    """
    Computes anomaly scores using FCVAE negative log-likelihood.

    The anomaly score for each point is:
        score = -0.5 * (log(var_x) + (x - mu_x)^2 / var_x)

    IMPORTANT: Lower scores indicate anomalies (inverted from Mahalanobis).
    Threshold semantics: score < threshold => anomaly
    """

    def __init__(self, config: Optional[FCVAEScorerConfig] = None):
        self.config = config or FCVAEScorerConfig()

        self.point_threshold: Optional[float] = None
        self.last_point_threshold: Optional[float] = None
        self.window_threshold: Optional[float] = None

        self.normal_score_mean: Optional[float] = None
        self.normal_score_std: Optional[float] = None

        self.is_fitted: bool = False

        logger.info(f"Initialized FCVAEScorer with config: {self.config}")

    def fit(
        self,
        model: torch.nn.Module,
        val_loader: DataLoader,
        device: torch.device
    ) -> None:
        """Fit scorer by computing score statistics on normal validation data."""
        logger.info("Fitting FCVAEScorer on validation data...")

        point_scores, window_scores = self.score_batch(model, val_loader, device)

        all_point_scores = point_scores.flatten()
        self.normal_score_mean = float(np.mean(all_point_scores))
        self.normal_score_std = float(np.std(all_point_scores))

        self.is_fitted = True

        logger.info(
            f"FCVAEScorer fitted: mean={self.normal_score_mean:.4f}, "
            f"std={self.normal_score_std:.4f}"
        )

    def score_batch(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        device: torch.device
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Score all windows in a DataLoader.

        Returns:
            point_scores: (num_windows, window_size) per-point NLL scores
            window_scores: (num_windows,) mean NLL per window
        """
        model.eval()
        all_point_scores = []

        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(device)
                else:
                    x = batch.to(device)

                if self.config.score_mode == "mcmc":
                    _, scores = model.score_mcmc(x)
                    scores = scores.squeeze(1).cpu().numpy()
                else:
                    scores = model.score_single_pass(x, self.config.n_samples)
                    scores = scores.cpu().numpy()

                all_point_scores.append(scores)

        point_scores = np.concatenate(all_point_scores, axis=0)
        window_scores = np.mean(point_scores, axis=1)

        return point_scores, window_scores

    def score_window(
        self,
        model: torch.nn.Module,
        window: np.ndarray,
        device: torch.device
    ) -> Tuple[float, np.ndarray]:
        """Score a single window.

        Returns:
            Tuple of (window_score, point_scores)
        """
        model.eval()

        if window.ndim == 1:
            x = torch.FloatTensor(window).unsqueeze(0).unsqueeze(0)
        elif window.ndim == 2:
            x = torch.FloatTensor(window).unsqueeze(0)
        else:
            x = torch.FloatTensor(window)

        x = x.to(device)

        with torch.no_grad():
            if self.config.score_mode == "mcmc":
                _, scores = model.score_mcmc(x)
                point_scores = scores.squeeze().cpu().numpy()
            else:
                scores = model.score_single_pass(x, self.config.n_samples)
                point_scores = scores.squeeze().cpu().numpy()

        window_score = float(np.mean(point_scores))
        return window_score, point_scores

    def set_threshold(
        self,
        normal_scores: np.ndarray,
        method: Optional[str] = None,
        percentile: Optional[float] = None
    ) -> float:
        """Set point threshold from normal validation scores.

        For FCVAE, anomalies have LOWER scores, so we use a LOW percentile
        (e.g., 5th percentile). Points scoring BELOW this are anomalous.
        """
        method = method or self.config.threshold_method
        percentile = percentile if percentile is not None else self.config.threshold_percentile

        all_scores = normal_scores.flatten()

        if method == "percentile":
            self.point_threshold = float(np.percentile(all_scores, percentile))
            logger.info(
                f"Set point threshold using {percentile}th percentile: "
                f"{self.point_threshold:.4f}"
            )
        else:
            raise ValueError(f"Unknown threshold method: {method}")

        return self.point_threshold

    def set_last_point_threshold(
        self,
        point_scores: np.ndarray,
        method: Optional[str] = None,
        percentile: Optional[float] = None
    ) -> float:
        """Set threshold calibrated on position [-1] scores only (paper-faithful).

        Position [-1] is the only genuine prediction (masked in frequency conditioning).
        """
        method = method or self.config.threshold_method
        percentile = percentile if percentile is not None else self.config.threshold_percentile

        if point_scores.ndim == 2:
            last_point_scores = point_scores[:, -1]
        else:
            last_point_scores = point_scores

        if method == "percentile":
            self.last_point_threshold = float(np.percentile(last_point_scores, percentile))
            logger.info(
                f"Set last-point threshold using {percentile}th percentile: "
                f"{self.last_point_threshold:.4f} "
                f"(vs all-position threshold: {self.point_threshold})"
            )
        else:
            raise ValueError(f"Unknown threshold method: {method}")

        return self.last_point_threshold

    @staticmethod
    def find_optimal_last_point_threshold(
        normal_scores: np.ndarray,
        anomaly_scores: np.ndarray,
        method: str = "f1_max",
        beta: float = 1.0
    ) -> Tuple[float, Dict]:
        """Find optimal threshold using position [-1] scores only."""
        return FCVAEScorer.find_optimal_threshold(
            normal_scores.flatten(), anomaly_scores.flatten(), method, beta
        )

    def set_window_threshold(
        self,
        normal_window_scores: np.ndarray,
        method: Optional[str] = None,
        percentile: Optional[float] = None
    ) -> float:
        """Set window-level threshold from normal validation scores."""
        method = method or self.config.threshold_method
        percentile = percentile if percentile is not None else self.config.threshold_percentile

        if method == "percentile":
            self.window_threshold = float(np.percentile(normal_window_scores, percentile))
            logger.info(
                f"Set window threshold using {percentile}th percentile: "
                f"{self.window_threshold:.4f}"
            )
        else:
            raise ValueError(f"Unknown threshold method: {method}")

        return self.window_threshold

    @staticmethod
    def find_optimal_threshold(
        normal_scores: np.ndarray,
        anomaly_scores: np.ndarray,
        method: str = "f1_max",
        beta: float = 1.0
    ) -> Tuple[float, Dict]:
        """Find optimal threshold using labeled normal and anomaly scores.

        IMPORTANT: For FCVAE, the comparison is INVERTED:
        - Anomalies have LOWER scores
        - Prediction: score < threshold => anomaly

        Methods:
            - "midpoint": Midpoint between min(normal) and max(anomaly)
            - "f1_max": Search for threshold that maximizes F1 score
            - "youden": Maximize Youden's J statistic
        """
        min_normal = np.min(normal_scores)
        max_anomaly = np.max(anomaly_scores)
        gap = min_normal - max_anomaly

        metrics = {
            "method": method,
            "min_normal": float(min_normal),
            "max_anomaly": float(max_anomaly),
            "gap": float(gap),
            "separable": gap > 0,
        }

        if method == "midpoint":
            if gap > 0:
                threshold = (min_normal + max_anomaly) / 2
                metrics["threshold_source"] = "midpoint"
            else:
                threshold = float(np.percentile(normal_scores, 5))
                metrics["threshold_source"] = "fallback_percentile_5"
                logger.warning(
                    f"Distributions overlap (gap={gap:.2f}), "
                    f"falling back to 5th percentile: {threshold:.4f}"
                )

            logger.info(f"Optimal threshold (midpoint): {threshold:.4f}")
            return threshold, metrics

        elif method == "f1_max":
            all_scores = np.concatenate([normal_scores, anomaly_scores])
            labels = np.concatenate([
                np.zeros(len(normal_scores)),
                np.ones(len(anomaly_scores))
            ])

            candidates = np.percentile(all_scores, np.arange(1, 100, 0.5))

            best_f_beta = 0
            best_threshold = float(np.median(all_scores))
            best_metrics = {}

            for candidate in candidates:
                predictions = all_scores < candidate

                tp = np.sum(predictions & labels.astype(bool))
                fp = np.sum(predictions & ~labels.astype(bool))
                fn = np.sum(~predictions & labels.astype(bool))
                tn = np.sum(~predictions & ~labels.astype(bool))

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0

                if precision + recall > 0:
                    f_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
                else:
                    f_beta = 0

                if f_beta > best_f_beta:
                    best_f_beta = f_beta
                    best_threshold = float(candidate)
                    best_metrics = {
                        "precision": float(precision),
                        "recall": float(recall),
                        "tp": int(tp),
                        "fp": int(fp),
                        "fn": int(fn),
                        "tn": int(tn),
                    }

            metrics["f1"] = float(best_f_beta)
            metrics["beta"] = beta
            metrics.update(best_metrics)

            logger.info(
                f"Optimal threshold (F{beta:.1f} max): {best_threshold:.4f} "
                f"(F1={best_f_beta:.4f}, P={best_metrics.get('precision', 0):.3f}, "
                f"R={best_metrics.get('recall', 0):.3f})"
            )
            return best_threshold, metrics

        elif method == "youden":
            all_scores = np.concatenate([normal_scores, anomaly_scores])
            labels = np.concatenate([
                np.zeros(len(normal_scores)),
                np.ones(len(anomaly_scores))
            ])

            candidates = np.percentile(all_scores, np.arange(1, 100, 0.5))

            best_j = -1
            best_threshold = float(np.median(all_scores))

            for candidate in candidates:
                predictions = all_scores < candidate

                tp = np.sum(predictions & labels.astype(bool))
                fp = np.sum(predictions & ~labels.astype(bool))
                fn = np.sum(~predictions & labels.astype(bool))
                tn = np.sum(~predictions & ~labels.astype(bool))

                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                j = sensitivity + specificity - 1

                if j > best_j:
                    best_j = j
                    best_threshold = float(candidate)

            metrics["youden_j"] = float(best_j)
            logger.info(f"Optimal threshold (Youden): {best_threshold:.4f} (J={best_j:.4f})")
            return best_threshold, metrics

        else:
            raise ValueError(f"Unknown method: {method}")

    def predict_points(self, point_scores: np.ndarray) -> np.ndarray:
        """Predict which points are anomalous. INVERTED: score < threshold => anomaly."""
        if self.point_threshold is None:
            raise ValueError("Point threshold not set. Call set_threshold() first.")
        return point_scores < self.point_threshold

    def predict_windows_from_points(
        self,
        point_predictions: np.ndarray,
        point_scores: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Predict which windows are anomalous using configurable decision logic.

        Decision modes:
        - "count_only": >= k points below threshold
        - "k1": any single point triggers
        - "severity": k-count OR any point below (threshold - severity_margin)
        - "zscore": k-count OR any point with z-score < -outlier_z_threshold
        - "hybrid": k-count OR severity OR zscore
        """
        mode = self.config.decision_mode
        k = self.config.hard_criterion_k

        anomalous_point_counts = np.sum(point_predictions, axis=1)

        if mode == "k1":
            return anomalous_point_counts >= 1

        elif mode == "count_only":
            return anomalous_point_counts >= k

        elif mode == "severity":
            if point_scores is None:
                logger.warning("severity mode requires point_scores, falling back to count_only")
                return anomalous_point_counts >= k
            count_criterion = anomalous_point_counts >= k
            severe_threshold = self.point_threshold - self.config.severity_margin
            severity_criterion = np.any(point_scores < severe_threshold, axis=1)
            return count_criterion | severity_criterion

        elif mode == "zscore":
            if point_scores is None:
                logger.warning("zscore mode requires point_scores, falling back to count_only")
                return anomalous_point_counts >= k
            count_criterion = anomalous_point_counts >= k
            window_means = np.mean(point_scores, axis=1, keepdims=True)
            window_stds = np.std(point_scores, axis=1, keepdims=True)
            z_scores = (point_scores - window_means) / (window_stds + 1e-8)
            zscore_criterion = np.any(z_scores < -self.config.outlier_z_threshold, axis=1)
            return count_criterion | zscore_criterion

        elif mode == "hybrid":
            if point_scores is None:
                logger.warning("hybrid mode requires point_scores, falling back to count_only")
                return anomalous_point_counts >= k
            count_criterion = anomalous_point_counts >= k
            severe_threshold = self.point_threshold - self.config.severity_margin
            severity_criterion = np.any(point_scores < severe_threshold, axis=1)
            window_means = np.mean(point_scores, axis=1, keepdims=True)
            window_stds = np.std(point_scores, axis=1, keepdims=True)
            z_scores = (point_scores - window_means) / (window_stds + 1e-8)
            zscore_criterion = np.any(z_scores < -self.config.outlier_z_threshold, axis=1)
            return count_criterion | severity_criterion | zscore_criterion

        else:
            logger.warning(f"Unknown decision mode: {mode}, using count_only")
            return anomalous_point_counts >= k

    def predict_windows(self, window_scores: np.ndarray) -> np.ndarray:
        """Predict which windows are anomalous. INVERTED: score < threshold => anomaly."""
        if self.window_threshold is None:
            raise ValueError("Window threshold not set. Call set_window_threshold() first.")
        return window_scores < self.window_threshold

    def save(self, path: str) -> None:
        """Save scorer state to file."""
        path = Path(path)
        state = {
            "config": self.config,
            "point_threshold": self.point_threshold,
            "last_point_threshold": self.last_point_threshold,
            "window_threshold": self.window_threshold,
            "normal_score_mean": self.normal_score_mean,
            "normal_score_std": self.normal_score_std,
            "is_fitted": self.is_fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.debug(f"Saved FCVAEScorer to {path}")

    @classmethod
    def load(cls, path: str) -> "FCVAEScorer":
        """Load scorer from file."""
        path = Path(path)
        with open(path, "rb") as f:
            state = pickle.load(f)

        scorer = cls(config=state["config"])
        scorer.point_threshold = state["point_threshold"]
        scorer.last_point_threshold = state.get("last_point_threshold")
        scorer.window_threshold = state.get("window_threshold")
        scorer.normal_score_mean = state.get("normal_score_mean")
        scorer.normal_score_std = state.get("normal_score_std")
        scorer.is_fitted = state["is_fitted"]

        logger.debug(f"Loaded FCVAEScorer from {path}")
        return scorer

    def get_stats(self) -> Dict:
        """Get scorer statistics."""
        return {
            "config": {
                "threshold_method": self.config.threshold_method,
                "threshold_percentile": self.config.threshold_percentile,
                "hard_criterion_k": self.config.hard_criterion_k,
                "score_mode": self.config.score_mode,
                "n_samples": self.config.n_samples,
                "decision_mode": self.config.decision_mode,
                "severity_margin": self.config.severity_margin,
                "outlier_z_threshold": self.config.outlier_z_threshold,
            },
            "is_fitted": self.is_fitted,
            "point_threshold": self.point_threshold,
            "window_threshold": self.window_threshold,
            "normal_score_mean": self.normal_score_mean,
            "normal_score_std": self.normal_score_std,
        }
