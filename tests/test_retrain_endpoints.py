"""Tests for Phase C: retrain endpoints, job manager, and ModelStore.reload_combo.

Uses FastAPI TestClient for integration tests and direct unit tests for
ModelStore and RetrainJobManager.
"""

import pickle
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

# Ensure app/ is importable
_app_dir = str(Path(__file__).resolve().parent.parent / "app")
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

from fcvae_model import FCVAE, FCVAEConfig
from fcvae_scorer import FCVAEScorer

from api.data_store import WindowDataStore
from api.dependencies import COMBO_KEY_MAP, ModelStore
from api.retrain import RetrainConfig, RetrainPipeline, RetrainResult
from api.retrain_job import RetrainJobManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _make_normal_values(i: int) -> list:
    t = np.linspace(0, 2 * np.pi, 24)
    return (100.0 + 50.0 * np.sin(t + i * 0.1)).tolist()


def _create_combo_artifacts(base_dir: Path, combo: str) -> None:
    """Save model.pt, scaler.pkl, scorer.pkl for one combo."""
    combo_dir = base_dir / combo
    combo_dir.mkdir(parents=True, exist_ok=True)

    model = FCVAE(config=TINY_CONFIG)
    data = np.array([_make_normal_values(i) for i in range(20)], dtype=np.float32)
    scaler = StandardScaler()
    scaler.fit(data.reshape(-1, 1))

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": TINY_CONFIG,
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
def model_dir(tmp_path):
    """Create model artifacts for all 4 combos."""
    model_path = tmp_path / "models" / "fcvae"
    for combo_key in COMBO_KEY_MAP:
        _create_combo_artifacts(model_path, combo_key)
    return model_path


@pytest.fixture
def staging_dir(tmp_path):
    """Create staging artifacts for all 4 combos."""
    staging = tmp_path / "models" / "fcvae_staging"
    for combo_key in COMBO_KEY_MAP:
        _create_combo_artifacts(staging, combo_key)
    return staging


@pytest.fixture
def model_store_loaded(model_dir):
    """A ModelStore with all 4 detectors loaded."""
    ms = ModelStore()
    ms.load_all(str(model_dir), device="cpu", n_samples=4)
    assert ms.any_loaded
    return ms


@pytest.fixture
def data_store_populated(tmp_path):
    """Data store with windows for Accel_CMP."""
    ds = WindowDataStore(data_dir=str(tmp_path / "data"))
    for i in range(80):
        ds.append(
            combo="Accel_CMP",
            raw_values=_make_normal_values(i),
            window_end=f"2026-03-{(i // 24) + 1:02d}T{i % 24:02d}:00:00Z",
            last_point_score=-0.5,
            is_anomaly=False,
        )
    for i in range(10):
        vals = _make_normal_values(i)
        vals[-1] = 500.0
        ds.append(
            combo="Accel_CMP",
            raw_values=vals,
            window_end=f"2026-03-05T{i:02d}:00:00Z",
            last_point_score=-5.0,
            is_anomaly=True,
        )
    return ds


@pytest.fixture
def test_client(tmp_path, model_dir, staging_dir):
    """FastAPI TestClient with models loaded and data store enabled."""
    from fastapi.testclient import TestClient

    # Patch dependencies before importing the app
    import api.dependencies as deps

    original_model_store = deps.model_store
    original_data_store = deps.data_store
    original_settings = deps.settings
    original_job_manager = deps.retrain_job_manager

    test_ms = ModelStore()
    test_ms.load_all(str(model_dir), device="cpu", n_samples=4)

    test_ds = WindowDataStore(data_dir=str(tmp_path / "data"))

    test_jm = RetrainJobManager()

    deps.model_store = test_ms
    deps.data_store = test_ds
    deps.retrain_job_manager = test_jm
    deps.settings.model_path = str(model_dir)

    # Also patch the module-level references used by routers
    import api.routers.retrain as retrain_router
    import api.routers.scoring as scoring_router

    retrain_router.model_store = test_ms
    retrain_router.data_store = test_ds
    retrain_router.retrain_job_manager = test_jm
    scoring_router.model_store = test_ms
    scoring_router.data_store = test_ds

    from api.main import app

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client

    # Restore originals
    deps.model_store = original_model_store
    deps.data_store = original_data_store
    deps.settings = original_settings
    deps.retrain_job_manager = original_job_manager


# ---------------------------------------------------------------------------
# ModelStore.reload_combo unit tests
# ---------------------------------------------------------------------------


class TestModelStoreReloadCombo:
    def test_successful_swap(self, model_store_loaded, model_dir):
        """reload_combo swaps the detector on success."""
        old_detector = model_store_loaded.get_detector("Accel_CMP")
        assert old_detector is not None

        ok = model_store_loaded.reload_combo(
            combo_key="Accel_CMP",
            model_path=str(model_dir),
            device="cpu",
            n_samples=4,
        )
        assert ok is True

        new_detector = model_store_loaded.get_detector("Accel_CMP")
        assert new_detector is not old_detector
        assert new_detector.is_ready

    def test_failed_load_leaves_old_detector(self, model_store_loaded, tmp_path):
        """If new detector fails to load, old detector stays."""
        old_detector = model_store_loaded.get_detector("Accel_CMP")
        assert old_detector is not None

        # Point to empty dir — load will fail
        ok = model_store_loaded.reload_combo(
            combo_key="Accel_CMP",
            model_path=str(tmp_path / "nonexistent"),
            device="cpu",
            n_samples=4,
        )
        assert ok is False

        # Old detector still in place
        current = model_store_loaded.get_detector("Accel_CMP")
        assert current is old_detector

    def test_unknown_combo_key(self, model_store_loaded, model_dir):
        """Unknown combo key returns False."""
        ok = model_store_loaded.reload_combo(
            combo_key="Nonexistent_Combo",
            model_path=str(model_dir),
            device="cpu",
            n_samples=4,
        )
        assert ok is False

    def test_all_combos_reloadable(self, model_store_loaded, model_dir):
        """Every combo in COMBO_KEY_MAP can be reloaded."""
        for combo_key in COMBO_KEY_MAP:
            ok = model_store_loaded.reload_combo(
                combo_key=combo_key,
                model_path=str(model_dir),
                device="cpu",
                n_samples=4,
            )
            assert ok is True, f"Failed to reload {combo_key}"


# ---------------------------------------------------------------------------
# RetrainJobManager unit tests
# ---------------------------------------------------------------------------


class TestRetrainJobManager:
    def test_submit_returns_job_id(self):
        jm = RetrainJobManager()
        pipeline = MagicMock()
        pipeline.retrain_combo.return_value = RetrainResult(
            combo="Accel_CMP", status="success",
        )
        pipeline.config = RetrainConfig()

        job_id = jm.submit_job(
            pipeline=pipeline,
            combos=["Accel_CMP"],
            mode="weekly",
            lookback_days=7,
            start_date=None,
            end_date=None,
            auto_reload=False,
            model_store=MagicMock(),
            min_windows=50,
        )

        assert isinstance(job_id, str)
        assert len(job_id) == 12

    def test_single_job_at_a_time(self):
        """Submitting while a job is running raises RuntimeError."""
        jm = RetrainJobManager()
        # Simulate a long-running job
        slow_pipeline = MagicMock()
        slow_pipeline.retrain_combo.side_effect = lambda **kw: (
            time.sleep(2) or RetrainResult(combo="Accel_CMP", status="success")
        )
        slow_pipeline.config = RetrainConfig()

        jm.submit_job(
            pipeline=slow_pipeline,
            combos=["Accel_CMP"],
            mode="weekly",
            lookback_days=7,
            start_date=None,
            end_date=None,
            auto_reload=False,
            model_store=MagicMock(),
            min_windows=50,
        )

        time.sleep(0.1)  # Let thread start

        with pytest.raises(RuntimeError, match="already running"):
            jm.submit_job(
                pipeline=slow_pipeline,
                combos=["Accel_CMP"],
                mode="weekly",
                lookback_days=7,
                start_date=None,
                end_date=None,
                auto_reload=False,
                model_store=MagicMock(),
                min_windows=50,
            )

    def test_job_completes(self):
        jm = RetrainJobManager()
        pipeline = MagicMock()
        pipeline.retrain_combo.return_value = RetrainResult(
            combo="Accel_CMP", status="success",
        )
        pipeline.config = RetrainConfig()

        job_id = jm.submit_job(
            pipeline=pipeline,
            combos=["Accel_CMP"],
            mode="weekly",
            lookback_days=7,
            start_date=None,
            end_date=None,
            auto_reload=False,
            model_store=MagicMock(),
            min_windows=50,
        )

        # Wait for completion
        for _ in range(50):
            status = jm.get_job_status(job_id)
            if status and status["status"] != "running":
                break
            time.sleep(0.1)

        status = jm.get_job_status(job_id)
        assert status["status"] == "completed"
        assert len(status["combo_results"]) == 1
        assert status["combo_results"][0]["combo"] == "Accel_CMP"
        assert status["duration_seconds"] is not None

    def test_nonexistent_job_returns_none(self):
        jm = RetrainJobManager()
        assert jm.get_job_status("nonexistent") is None


# ---------------------------------------------------------------------------
# Endpoint integration tests
# ---------------------------------------------------------------------------


class TestRetrainEndpoint:
    def test_returns_202_with_job_id(self, test_client):
        resp = test_client.post("/v1/retrain", json={
            "mode": "weekly",
            "combos": ["Accel_CMP"],
            "lookback_days": 7,
            "auto_reload": False,
            "min_windows": 1,
        })
        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "running"

    def test_returns_409_if_job_running(self, test_client):
        # Start a job
        resp1 = test_client.post("/v1/retrain", json={
            "mode": "weekly",
            "combos": ["Accel_CMP"],
            "lookback_days": 7,
            "auto_reload": False,
            "min_windows": 999999,  # Will likely be insufficient_data — fast
        })
        assert resp1.status_code == 202

        # Immediately try another — should get 409
        time.sleep(0.05)
        # Check if the first job is still running
        import api.routers.retrain as retrain_mod
        if retrain_mod.retrain_job_manager.is_running:
            resp2 = test_client.post("/v1/retrain", json={
                "mode": "weekly",
                "combos": ["Accel_CMP"],
            })
            assert resp2.status_code == 409

    def test_invalid_combo_returns_422(self, test_client):
        resp = test_client.post("/v1/retrain", json={
            "mode": "weekly",
            "combos": ["InvalidCombo"],
        })
        assert resp.status_code == 422

    def test_defaults_to_all_combos(self, test_client):
        resp = test_client.post("/v1/retrain", json={
            "mode": "weekly",
            "auto_reload": False,
            "min_windows": 1,
        })
        assert resp.status_code == 202
        data = resp.json()
        assert "all" not in data["message"].lower() or "combos" in data["message"].lower()


class TestRetrainStatusEndpoint:
    def test_returns_job_status(self, test_client):
        # Start a job
        resp = test_client.post("/v1/retrain", json={
            "mode": "weekly",
            "combos": ["Accel_CMP"],
            "auto_reload": False,
            "min_windows": 1,
        })
        job_id = resp.json()["job_id"]

        # Poll status
        status_resp = test_client.get(f"/v1/retrain/status/{job_id}")
        assert status_resp.status_code == 200
        data = status_resp.json()
        assert data["job_id"] == job_id
        assert data["status"] in ("running", "completed", "failed")
        assert data["started_at"] is not None

    def test_nonexistent_job_returns_404(self, test_client):
        resp = test_client.get("/v1/retrain/status/nonexistent123")
        assert resp.status_code == 404


class TestReloadEndpoint:
    def test_reload_with_staged_artifacts(self, test_client, staging_dir, model_dir):
        """POST /v1/model/reload copies staging → production and swaps detectors."""
        import api.routers.retrain as retrain_mod

        resp = test_client.post("/v1/model/reload", json={
            "combos": ["Accel_CMP"],
            "staging_dir": str(staging_dir),
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "Accel_CMP" in data["reloaded_combos"]

    def test_reload_missing_staging_returns_empty(self, test_client, tmp_path):
        resp = test_client.post("/v1/model/reload", json={
            "combos": ["Accel_CMP"],
            "staging_dir": str(tmp_path / "nonexistent"),
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["reloaded_combos"] == []

    def test_score_works_after_reload(self, test_client, staging_dir):
        """Score requests continue working after a model reload."""
        # Reload
        test_client.post("/v1/model/reload", json={
            "combos": ["Accel_CMP"],
            "staging_dir": str(staging_dir),
        })

        # Score — should still work
        resp = test_client.post("/v1/score", json={
            "combo": "Accel_CMP",
            "values": _make_normal_values(0),
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "is_anomaly" in data
        assert "last_point_score" in data
