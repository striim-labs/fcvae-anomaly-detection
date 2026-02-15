"""
Base Detector Interface for FCVAE Anomaly Detection

Provides an abstract interface for the FCVAE detector implementation,
ensuring consistent behavior in the streaming pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import pandas as pd


@dataclass
class DetectionResult:
    """
    Standard result format for anomaly detection.

    Attributes:
        anomalies: DataFrame with detected anomalies (timestamp, value, score)
        total_anomalies: Count of anomalies detected
        metadata: Additional detector-specific information
    """
    anomalies: pd.DataFrame
    total_anomalies: int
    metadata: Dict[str, Any]


class BaseDetector(ABC):
    """
    Abstract base class for anomaly detectors.

    All detector implementations must inherit from this class and
    implement the required methods to ensure consistent behavior
    across different detection algorithms.
    """

    @abstractmethod
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run anomaly detection on the provided data.

        Args:
            df: DataFrame with at least 'timestamp' and 'value' columns

        Returns:
            DataFrame of detected anomalies with columns:
                - timestamp: When the anomaly occurred
                - value: The anomalous value
                - anomaly_score: Confidence/severity score (lower = more anomalous for FCVAE)
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get detector statistics and configuration.

        Returns:
            Dictionary with detector-specific statistics
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the detector name for display purposes.

        Returns:
            Human-readable detector name
        """
        pass

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """
        Check if the detector is ready to perform detection.

        For FCVAE, this requires a pre-trained model to be loaded.

        Returns:
            True if detector can perform detection
        """
        pass

    @property
    @abstractmethod
    def min_samples_required(self) -> int:
        """
        Minimum number of samples required for detection.

        Returns:
            Minimum sample count needed before detection can run
        """
        pass


def create_detector(
    detector_type: str,
    model_path: str = "models/fcvae",
    combo: Optional[Tuple[str, str]] = None,
    window_size: int = 24,
    min_samples: int = 24,
    n_samples: int = 16,
    decision_mode: str = "severity",
    severity_margin: float = 0.5,
    oracle_threshold: Optional[float] = None,
    **kwargs
) -> BaseDetector:
    """
    Factory function to create an FCVAE detector.

    Args:
        detector_type: Must be "fcvae"
        model_path: Path to the FCVAE model directory
        combo: Optional (network_type, txn_type) tuple to load specific combo
        window_size: Samples per detection window (default: 24 = 1 day)
        min_samples: Minimum samples before detection runs
        n_samples: Number of latent samples for single-pass scoring
        decision_mode: Window-level decision mode ("count_only", "severity", "hybrid")
        severity_margin: Score margin below threshold for severity criterion
        oracle_threshold: If set, overrides last_point_threshold with this value
        **kwargs: Additional configuration (scaler_path, scorer_path, device)

    Returns:
        Configured FCVAEStreamingDetector instance

    Raises:
        ValueError: If detector_type is not "fcvae"
    """
    detector_type = detector_type.lower().strip()

    if detector_type != "fcvae":
        raise ValueError(
            f"Unknown detector type: {detector_type}. "
            f"This repository only supports 'fcvae' detector."
        )

    from fcvae_streaming_detector import FCVAEStreamingDetector

    return FCVAEStreamingDetector(
        model_path=model_path,
        scaler_path=kwargs.get("scaler_path"),
        scorer_path=kwargs.get("scorer_path"),
        combo=combo,
        window_size=window_size,
        min_samples=min_samples,
        n_samples=n_samples,
        device=kwargs.get("device"),
        decision_mode=decision_mode,
        severity_margin=severity_margin,
        oracle_threshold=oracle_threshold,
    )
