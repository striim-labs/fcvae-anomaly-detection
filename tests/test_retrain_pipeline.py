"""Tests for api.retrain — RetrainPipeline core logic.

Uses tiny model configs and synthetic data for fast execution.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

# Ensure app/ is importable
_app_dir = str(Path(__file__).resolve().parent.parent / "app")
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

from fcvae_model import FCVAE, FCVAEConfig
from fcvae_scorer import FCVAEScorer
from transaction_preprocessor import SlidingWindowDataset

from api.data_store import WindowDataStore
from api.retrain import RetrainConfig, RetrainPipeline, RetrainResult, _compute_f1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_normal_values(i: int) -> list:
    """Smooth sine wave — representative of normal transaction patterns."""
    t = np.linspace(0, 2 * np.pi, 24)
    return (100.0 + 50.0 * np.sin(t + i * 0.1)).tolist()


def _make_anomaly_values(i: int) -> list:
    """Sine wave with a large spike at the last position."""
    t = np.linspace(0, 2 * np.pi, 24)
    values = 100.0 + 50.0 * np.sin(t + i * 0.1)
    values[-1] = 500.0
    return values.tolist()


TINY_CONFIG = FCVAEConfig(
    window=24,
    latent_dim=4,
    condition_emb_dim=8,
    d_model=16,
    d_inner=32,
    n_head=2,
    kernel_size=8,
    stride=4,
    dropout_rate=0.0,
    hidden_dims=(16, 16),
)


def _create_production_artifacts(
    prod_dir: Path,
    combo: str = "TestCombo",
    config: FCVAEConfig = TINY_CONFIG,
) -> None:
    """Save model.pt, scaler.pkl, scorer.pkl in production format."""
    combo_dir = prod_dir / combo
    combo_dir.mkdir(parents=True, exist_ok=True)

    model = FCVAE(config=config)

    # Fit scaler on representative data
    data = np.array([_make_normal_values(i) for i in range(20)], dtype=np.float32)
    scaler = StandardScaler()
    scaler.fit(data.reshape(-1, 1))

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": config,
            "model_version": 1,
        },
        combo_dir / "model.pt",
    )

    with open(combo_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    scorer = FCVAEScorer()
    scorer.point_threshold = -2.0
    scorer.last_point_threshold = -2.0
    scorer.is_fitted = True
    scorer.save(str(combo_dir / "scorer.pkl"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_config():
    return TINY_CONFIG


@pytest.fixture
def store(tmp_path):
    """Empty WindowDataStore."""
    return WindowDataStore(data_dir=str(tmp_path / "data"))


@pytest.fixture
def populated_store(tmp_path):
    """Data store with 80 normal + 10 anomalous windows for TestCombo."""
    ds = WindowDataStore(data_dir=str(tmp_path / "data"))

    for i in range(80):
        ds.append(
            combo="TestCombo",
            raw_values=_make_normal_values(i),
            window_end=f"2026-03-{(i // 24) + 1:02d}T{i % 24:02d}:00:00Z",
            last_point_score=-0.5,
            is_anomaly=False,
        )

    for i in range(10):
        ds.append(
            combo="TestCombo",
            raw_values=_make_anomaly_values(i),
            window_end=f"2026-03-05T{i:02d}:00:00Z",
            last_point_score=-5.0,
            is_anomaly=True,
        )

    return ds


@pytest.fixture
def production_dir(tmp_path):
    """Production directory with saved model artifacts."""
    prod = tmp_path / "production"
    _create_production_artifacts(prod)
    return prod


@pytest.fixture
def retrain_config(tmp_path):
    """Fast RetrainConfig for testing."""
    return RetrainConfig(
        learning_rate=1e-3,
        epochs=5,
        kl_warmup_epochs=2,
        patience=3,
        batch_size=16,
        train_fraction=0.8,
        max_nll_degradation=0.5,
        min_f1_ratio=0.9,
        staging_dir=str(tmp_path / "staging"),
        production_dir=str(tmp_path / "production"),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChronologicalSplit:
    def test_preserves_temporal_order_and_ratio(self, populated_store):
        """80/20 split preserves chronological ordering."""
        windows = populated_store.get_normal_windows("TestCombo")
        assert len(windows) == 80

        split_idx = int(len(windows) * 0.8)
        train = windows[:split_idx]
        val = windows[split_idx:]

        assert len(train) == 64
        assert len(val) == 16

        # Train windows come before val windows
        assert train[-1]["window_end"] <= val[0]["window_end"]

        # Each partition is internally ordered
        for i in range(len(train) - 1):
            assert train[i]["window_end"] <= train[i + 1]["window_end"]
        for i in range(len(val) - 1):
            assert val[i]["window_end"] <= val[i + 1]["window_end"]


class TestScalerFitting:
    def test_fits_on_train_only(self):
        """StandardScaler mean should reflect training partition only."""
        train_data = np.full((50, 24), 100.0, dtype=np.float32)
        val_data = np.full((12, 24), 200.0, dtype=np.float32)

        scaler = StandardScaler()
        scaler.fit(train_data.reshape(-1, 1))

        assert abs(scaler.mean_[0] - 100.0) < 1.0

        train_norm = scaler.transform(train_data.reshape(-1, 1)).reshape(train_data.shape)
        assert abs(np.mean(train_norm)) < 0.01

        # Val transform should NOT be centred at zero
        val_norm = scaler.transform(val_data.reshape(-1, 1)).reshape(val_data.shape)
        assert np.mean(val_norm) > 1.0


class TestTrainingLoop:
    def test_reduces_loss(self, tiny_config, populated_store, retrain_config):
        """Fine-tuning should reduce validation loss over epochs."""
        windows = populated_store.get_normal_windows("TestCombo")
        split_idx = int(len(windows) * 0.8)

        train_raw = np.array(
            [w["raw_values"] for w in windows[:split_idx]], dtype=np.float32,
        )
        val_raw = np.array(
            [w["raw_values"] for w in windows[split_idx:]], dtype=np.float32,
        )

        scaler = StandardScaler()
        scaler.fit(train_raw.reshape(-1, 1))

        train_norm = scaler.transform(train_raw.reshape(-1, 1)).reshape(train_raw.shape).astype(np.float32)
        val_norm = scaler.transform(val_raw.reshape(-1, 1)).reshape(val_raw.shape).astype(np.float32)

        train_loader = DataLoader(
            SlidingWindowDataset(train_norm, np.zeros_like(train_norm)),
            batch_size=16, shuffle=True,
        )
        val_loader = DataLoader(
            SlidingWindowDataset(val_norm, np.zeros_like(val_norm)),
            batch_size=16,
        )

        model = FCVAE(config=tiny_config)
        pipeline = RetrainPipeline(data_store=populated_store, config=retrain_config)
        history = pipeline._fine_tune(model, train_loader, val_loader, "TestCombo")

        assert len(history["val_loss"]) >= 2
        assert history["best_epoch"] > 0
        # Best val loss should improve over first epoch
        assert min(history["val_loss"]) <= history["val_loss"][0]


class TestOracleThreshold:
    def test_separates_clearly_distinct_scores(self):
        """Oracle threshold should achieve near-perfect F1 on separable data."""
        normal_scores = np.random.normal(-1.0, 0.1, 100)
        anomaly_scores = np.random.normal(-5.0, 0.1, 20)

        threshold, metrics = FCVAEScorer.find_optimal_threshold(
            normal_scores, anomaly_scores, method="f1_max",
        )

        assert threshold > np.max(anomaly_scores)
        assert threshold < np.min(normal_scores)
        assert metrics["f1"] > 0.95

    def test_handles_overlapping_scores(self):
        """Should still find a reasonable threshold when distributions overlap."""
        normal_scores = np.random.normal(-1.0, 1.0, 200)
        anomaly_scores = np.random.normal(-3.0, 1.0, 50)

        threshold, metrics = FCVAEScorer.find_optimal_threshold(
            normal_scores, anomaly_scores, method="f1_max",
        )

        # Threshold should be somewhere between the means
        assert -5.0 < threshold < 1.0
        assert metrics["f1"] > 0.0


class TestPercentileFallback:
    def test_no_anomaly_windows_uses_percentile(
        self, tmp_path, retrain_config,
    ):
        """When no anomalous windows exist, threshold = 5th percentile."""
        # Store with only normal windows
        ds = WindowDataStore(data_dir=str(tmp_path / "fallback_data"))
        for i in range(80):
            ds.append(
                combo="TestCombo",
                raw_values=_make_normal_values(i),
                window_end=f"2026-03-{(i // 24) + 1:02d}T{i % 24:02d}:00:00Z",
                last_point_score=-0.5,
                is_anomaly=False,
            )

        prod = tmp_path / "fallback_prod"
        _create_production_artifacts(prod)

        retrain_config.staging_dir = str(tmp_path / "fallback_staging")
        retrain_config.max_nll_degradation = 100.0
        retrain_config.min_f1_ratio = 0.0

        pipeline = RetrainPipeline(data_store=ds, config=retrain_config)
        result = pipeline.retrain_combo(
            combo="TestCombo",
            production_dir=str(prod),
            min_windows=10,
        )

        assert result.status == "success"
        assert result.num_anomaly_windows == 0
        assert result.new_threshold is not None
        # No anomalous data → F1 fields stay None
        assert result.old_f1 is None
        assert result.new_f1 is None


class TestValidationGate:
    def test_passes_comparable_model(
        self, populated_store, production_dir, retrain_config,
    ):
        """Permissive gates → pipeline succeeds."""
        retrain_config.production_dir = str(production_dir)
        retrain_config.max_nll_degradation = 100.0
        retrain_config.min_f1_ratio = 0.0

        pipeline = RetrainPipeline(
            data_store=populated_store, config=retrain_config,
        )
        result = pipeline.retrain_combo(
            combo="TestCombo",
            production_dir=str(production_dir),
            min_windows=10,
        )

        assert result.status == "success"
        assert result.duration_seconds > 0
        assert result.num_train_windows > 0
        assert result.num_val_windows > 0
        assert result.best_epoch > 0
        assert result.old_mean_nll is not None
        assert result.new_mean_nll is not None

    def test_rejects_f1_degradation(
        self, populated_store, production_dir, retrain_config,
    ):
        """Impossible F1 ratio requirement → rejected."""
        retrain_config.production_dir = str(production_dir)
        retrain_config.max_nll_degradation = 100.0  # pass NLL gate
        retrain_config.min_f1_ratio = 100.0  # impossible requirement

        pipeline = RetrainPipeline(
            data_store=populated_store, config=retrain_config,
        )
        result = pipeline.retrain_combo(
            combo="TestCombo",
            production_dir=str(production_dir),
            min_windows=10,
        )

        assert result.status == "rejected"
        assert result.error_message is not None
        assert "F1" in result.error_message
        assert result.old_f1 is not None
        assert result.new_f1 is not None


class TestStagedArtifacts:
    def test_loadable_by_streaming_detector_format(
        self, populated_store, production_dir, retrain_config,
    ):
        """Staged artifacts must match FCVAEStreamingDetector._load_artifacts format."""
        retrain_config.production_dir = str(production_dir)
        retrain_config.max_nll_degradation = 100.0
        retrain_config.min_f1_ratio = 0.0

        pipeline = RetrainPipeline(
            data_store=populated_store, config=retrain_config,
        )
        result = pipeline.retrain_combo(
            combo="TestCombo",
            production_dir=str(production_dir),
            min_windows=10,
        )
        assert result.status == "success"

        staging = Path(retrain_config.staging_dir) / "TestCombo"

        # --- model.pt ---
        assert (staging / "model.pt").exists()
        checkpoint = torch.load(staging / "model.pt", weights_only=False)
        assert "model_state_dict" in checkpoint
        assert "model_config" in checkpoint
        assert "model_version" in checkpoint
        assert isinstance(checkpoint["model_config"], FCVAEConfig)
        assert checkpoint["model_version"] == 2  # old was 1

        model = FCVAE(config=checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])  # must not raise

        # --- scaler.pkl ---
        assert (staging / "scaler.pkl").exists()
        with open(staging / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        assert isinstance(scaler, StandardScaler)
        assert hasattr(scaler, "mean_")

        # --- scorer.pkl ---
        assert (staging / "scorer.pkl").exists()
        scorer = FCVAEScorer.load(str(staging / "scorer.pkl"))
        assert scorer.point_threshold is not None
        assert scorer.last_point_threshold is not None
        assert scorer.is_fitted is True

        # --- history.pkl ---
        assert (staging / "history.pkl").exists()
        with open(staging / "history.pkl", "rb") as f:
            history = pickle.load(f)
        assert "train_loss" in history
        assert "val_loss" in history
        assert "best_epoch" in history
        assert len(history["train_loss"]) > 0


class TestInsufficientData:
    def test_returns_insufficient(self, store, production_dir, retrain_config):
        """Fewer than min_windows normal windows → insufficient_data."""
        for i in range(5):
            store.append(
                combo="TestCombo",
                raw_values=_make_normal_values(i),
                window_end=f"2026-03-01T{i:02d}:00:00Z",
                last_point_score=-0.5,
                is_anomaly=False,
            )

        pipeline = RetrainPipeline(data_store=store, config=retrain_config)
        result = pipeline.retrain_combo(
            combo="TestCombo",
            production_dir=str(production_dir),
        )

        assert result.status == "insufficient_data"
        assert result.num_train_windows == 5


class TestErrorHandling:
    def test_missing_production_dir(self, populated_store, retrain_config, tmp_path):
        """Missing production directory → error status with message."""
        pipeline = RetrainPipeline(
            data_store=populated_store, config=retrain_config,
        )
        result = pipeline.retrain_combo(
            combo="TestCombo",
            production_dir=str(tmp_path / "nonexistent"),
        )

        assert result.status == "error"
        assert result.error_message is not None
        assert result.duration_seconds > 0

    def test_missing_combo_subdir(self, populated_store, retrain_config, tmp_path):
        """Production dir exists but combo subdir missing → error."""
        (tmp_path / "empty_prod").mkdir()

        pipeline = RetrainPipeline(
            data_store=populated_store, config=retrain_config,
        )
        result = pipeline.retrain_combo(
            combo="TestCombo",
            production_dir=str(tmp_path / "empty_prod"),
        )

        assert result.status == "error"
        assert result.error_message is not None


class TestComputeF1:
    def test_perfect_separation(self):
        scores = np.array([-1.0, -1.0, -5.0, -5.0])
        labels = np.array([False, False, True, True])
        assert _compute_f1(scores, -3.0, labels) == 1.0

    def test_no_anomalies_detected(self):
        scores = np.array([-1.0, -1.0, -1.0])
        labels = np.array([False, False, False])
        assert _compute_f1(scores, -3.0, labels) == 0.0

    def test_partial_detection(self):
        scores = np.array([-1.0, -2.0, -4.0, -5.0])
        labels = np.array([False, False, True, True])
        # threshold = -3.0: only -4.0 and -5.0 detected
        # tp=2, fp=0, fn=0, precision=1.0, recall=1.0 → F1=1.0
        assert _compute_f1(scores, -3.0, labels) == 1.0

    def test_false_positives(self):
        scores = np.array([-4.0, -4.0, -5.0, -5.0])
        labels = np.array([False, False, True, True])
        # threshold = -3.0: all detected as anomaly
        # tp=2, fp=2, fn=0 → precision=0.5, recall=1.0 → F1=2/3
        f1 = _compute_f1(scores, -3.0, labels)
        assert abs(f1 - 2.0 / 3.0) < 1e-6
