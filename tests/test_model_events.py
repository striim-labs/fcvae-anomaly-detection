"""Tests for Phase E: model versioning, event logging, and Striim integration.

Covers ModelEventLog (JSONL append/read), ModelStore version tracking
(effective_version, combo_versions, retrain_counter), and event wiring
through the retrain job manager.
"""

import json
import pickle
import sys
import threading
import time
from pathlib import Path
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
from api.model_events import ModelEventLog
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
def event_log(tmp_path):
    """A ModelEventLog writing to a temp directory."""
    return ModelEventLog(log_path=str(tmp_path / "events.jsonl"))


@pytest.fixture
def model_dir(tmp_path):
    model_path = tmp_path / "models" / "fcvae"
    for combo_key in COMBO_KEY_MAP:
        _create_combo_artifacts(model_path, combo_key)
    return model_path


@pytest.fixture
def model_store_loaded(model_dir):
    ms = ModelStore()
    ms.load_all(str(model_dir), device="cpu", n_samples=4)
    assert ms.any_loaded
    return ms


# ===========================================================================
# ModelEventLog Tests
# ===========================================================================


class TestModelEventLog:
    """Tests for the JSONL event log."""

    def test_log_event_creates_file(self, event_log):
        event_log.log_event("retrain_started")
        assert Path(event_log.path).exists()

    def test_log_event_appends_valid_json(self, event_log):
        event_log.log_event("retrain_started", combo="Accel_CMP")
        event_log.log_event("retrain_completed", combo="Accel_CMP", details={"f1": 0.95})

        with open(event_log.path) as f:
            lines = f.readlines()
        assert len(lines) == 2

        first = json.loads(lines[0])
        assert first["event_type"] == "retrain_started"
        assert first["combo"] == "Accel_CMP"
        assert "timestamp" in first

        second = json.loads(lines[1])
        assert second["event_type"] == "retrain_completed"
        assert second["details"]["f1"] == 0.95

    def test_log_event_no_combo(self, event_log):
        event_log.log_event("retrain_started", details={"job_id": "abc123"})

        events = event_log.get_recent_events()
        assert len(events) == 1
        assert "combo" not in events[0]
        assert events[0]["details"]["job_id"] == "abc123"

    def test_get_recent_events_empty(self, event_log):
        assert event_log.get_recent_events() == []

    def test_get_recent_events_limit(self, event_log):
        for i in range(10):
            event_log.log_event("test_event", details={"i": i})

        events = event_log.get_recent_events(limit=3)
        assert len(events) == 3
        # Should be the last 3
        assert events[0]["details"]["i"] == 7
        assert events[2]["details"]["i"] == 9

    def test_thread_safety(self, event_log):
        """Multiple threads can write concurrently without corruption."""
        n_threads = 4
        n_per_thread = 25

        def writer(thread_id):
            for i in range(n_per_thread):
                event_log.log_event("thread_event", details={"t": thread_id, "i": i})

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        events = event_log.get_recent_events(limit=200)
        assert len(events) == n_threads * n_per_thread

    def test_corrupted_line_skipped(self, event_log):
        """Malformed JSONL lines are skipped gracefully."""
        with open(event_log.path, "w") as f:
            f.write('{"event_type":"good","timestamp":"2026-01-01T00:00:00Z"}\n')
            f.write("this is not json\n")
            f.write('{"event_type":"also_good","timestamp":"2026-01-01T00:01:00Z"}\n')

        events = event_log.get_recent_events()
        assert len(events) == 2
        assert events[0]["event_type"] == "good"
        assert events[1]["event_type"] == "also_good"

    def test_parent_dir_created(self, tmp_path):
        deep_path = str(tmp_path / "a" / "b" / "c" / "events.jsonl")
        log = ModelEventLog(log_path=deep_path)
        log.log_event("test")
        assert Path(deep_path).exists()


# ===========================================================================
# ModelStore Version Tracking Tests
# ===========================================================================


class TestModelStoreVersioning:
    """Tests for version counters on ModelStore."""

    def test_initial_state(self):
        ms = ModelStore()
        assert ms.retrain_counter == 0
        assert ms.combo_versions == {}
        assert ms.last_retrain_time is None

    def test_effective_version_no_retrains(self, model_store_loaded):
        with patch("api.dependencies.settings") as mock_settings:
            mock_settings.model_version = "1.0.0"
            assert model_store_loaded.effective_version == "1.0.0"

    def test_effective_version_after_retrain(self, model_store_loaded, model_dir):
        # Simulate a successful reload
        ok = model_store_loaded.reload_combo(
            combo_key="Accel_CMP",
            model_path=str(model_dir),
            device="cpu",
            n_samples=4,
        )
        assert ok

        with patch("api.dependencies.settings") as mock_settings:
            mock_settings.model_version = "1.0.0"
            assert model_store_loaded.effective_version == "1.0.0.1"

    def test_retrain_counter_increments(self, model_store_loaded, model_dir):
        assert model_store_loaded.retrain_counter == 0

        model_store_loaded.reload_combo("Accel_CMP", str(model_dir), "cpu", 4)
        assert model_store_loaded.retrain_counter == 1

        model_store_loaded.reload_combo("Star_CMP", str(model_dir), "cpu", 4)
        assert model_store_loaded.retrain_counter == 2

    def test_combo_versions_tracked(self, model_store_loaded, model_dir):
        model_store_loaded.reload_combo("Accel_CMP", str(model_dir), "cpu", 4)
        model_store_loaded.reload_combo("Accel_CMP", str(model_dir), "cpu", 4)
        model_store_loaded.reload_combo("Star_CMP", str(model_dir), "cpu", 4)

        assert model_store_loaded.combo_versions["Accel_CMP"] == 2
        assert model_store_loaded.combo_versions["Star_CMP"] == 1

    def test_last_retrain_time_set(self, model_store_loaded, model_dir):
        assert model_store_loaded.last_retrain_time is None
        model_store_loaded.reload_combo("Accel_CMP", str(model_dir), "cpu", 4)
        assert model_store_loaded.last_retrain_time is not None
        assert "T" in model_store_loaded.last_retrain_time  # ISO format

    def test_failed_reload_does_not_increment(self, model_store_loaded, tmp_path):
        """A failed reload should not change version counters."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        initial_counter = model_store_loaded.retrain_counter
        ok = model_store_loaded.reload_combo("Accel_CMP", str(empty_dir), "cpu", 4)
        assert not ok
        assert model_store_loaded.retrain_counter == initial_counter


# ===========================================================================
# Event Wiring Through Job Manager Tests
# ===========================================================================


class TestJobManagerEventWiring:
    """Tests that the job manager logs events via ModelEventLog."""

    def test_retrain_started_event(self, event_log):
        """submit_job logs a retrain_started event."""
        mgr = RetrainJobManager()
        pipeline = MagicMock()
        ms = MagicMock()

        # Pipeline returns insufficient_data so the job completes quickly
        result = RetrainResult(combo="Accel_CMP", status="insufficient_data")
        pipeline.retrain_combo.return_value = result

        job_id = mgr.submit_job(
            pipeline=pipeline,
            combos=["Accel_CMP"],
            mode="weekly",
            lookback_days=7,
            start_date=None,
            end_date=None,
            auto_reload=True,
            model_store=ms,
            min_windows=50,
            event_log=event_log,
        )

        # Wait for background thread
        time.sleep(0.5)

        events = event_log.get_recent_events()
        started = [e for e in events if e["event_type"] == "retrain_started"]
        assert len(started) == 1
        assert started[0]["details"]["job_id"] == job_id
        assert started[0]["details"]["combos"] == ["Accel_CMP"]

    def test_retrain_completed_event(self, event_log):
        """Successful retrain logs retrain_completed event."""
        mgr = RetrainJobManager()
        pipeline = MagicMock()
        ms = MagicMock()

        result = RetrainResult(
            combo="Accel_CMP",
            status="success",
            new_f1=0.95,
            old_f1=0.90,
            new_threshold=-2.5,
            duration_seconds=10.0,
        )
        pipeline.retrain_combo.return_value = result

        mgr.submit_job(
            pipeline=pipeline,
            combos=["Accel_CMP"],
            mode="weekly",
            lookback_days=7,
            start_date=None,
            end_date=None,
            auto_reload=True,
            model_store=ms,
            min_windows=50,
            event_log=event_log,
        )

        time.sleep(0.5)

        events = event_log.get_recent_events()
        completed = [e for e in events if e["event_type"] == "retrain_completed"]
        assert len(completed) == 1
        assert completed[0]["combo"] == "Accel_CMP"
        assert completed[0]["details"]["new_f1"] == 0.95

    def test_retrain_rejected_event(self, event_log):
        """Rejected retrain logs retrain_rejected event."""
        mgr = RetrainJobManager()
        pipeline = MagicMock()
        ms = MagicMock()

        result = RetrainResult(
            combo="Accel_CMP",
            status="rejected",
            error_message="F1 degraded below threshold",
            new_f1=0.60,
            old_f1=0.90,
        )
        pipeline.retrain_combo.return_value = result

        mgr.submit_job(
            pipeline=pipeline,
            combos=["Accel_CMP"],
            mode="weekly",
            lookback_days=7,
            start_date=None,
            end_date=None,
            auto_reload=True,
            model_store=ms,
            min_windows=50,
            event_log=event_log,
        )

        time.sleep(0.5)

        events = event_log.get_recent_events()
        rejected = [e for e in events if e["event_type"] == "retrain_rejected"]
        assert len(rejected) == 1
        assert rejected[0]["details"]["reason"] == "F1 degraded below threshold"

    def test_no_event_log_does_not_crash(self):
        """Passing event_log=None should not raise."""
        mgr = RetrainJobManager()
        pipeline = MagicMock()
        ms = MagicMock()

        result = RetrainResult(combo="Accel_CMP", status="insufficient_data")
        pipeline.retrain_combo.return_value = result

        job_id = mgr.submit_job(
            pipeline=pipeline,
            combos=["Accel_CMP"],
            mode="weekly",
            lookback_days=7,
            start_date=None,
            end_date=None,
            auto_reload=True,
            model_store=ms,
            min_windows=50,
            event_log=None,
        )

        time.sleep(0.5)
        job = mgr.get_job_status(job_id)
        assert job["status"] in ("completed", "failed")

    def test_model_reloaded_event_on_promote(self, event_log, tmp_path):
        """_promote_and_reload logs model_reloaded on success."""
        # Staging and production must be separate dirs to avoid copytree self-copy
        staging_path = tmp_path / "staging"
        production_path = tmp_path / "production"
        _create_combo_artifacts(staging_path, "Accel_CMP")
        _create_combo_artifacts(production_path, "Accel_CMP")

        ms = ModelStore()
        ms.load_all(str(production_path), device="cpu", n_samples=4)

        config = MagicMock()
        config.staging_dir = str(staging_path)
        config.production_dir = str(production_path)

        RetrainJobManager._promote_and_reload(
            combo="Accel_CMP",
            config=config,
            model_store=ms,
            event_log=event_log,
        )

        events = event_log.get_recent_events()
        reloaded = [e for e in events if e["event_type"] == "model_reloaded"]
        assert len(reloaded) == 1
        assert reloaded[0]["combo"] == "Accel_CMP"

    def test_validation_failed_event_on_bad_promote(self, event_log, tmp_path):
        """_promote_and_reload logs validation_failed when detector fails to load."""
        ms = ModelStore()
        # Load with real artifacts first
        model_path = tmp_path / "models"
        _create_combo_artifacts(model_path, "Accel_CMP")
        ms.load_all(str(model_path), device="cpu", n_samples=4)

        # Create staging with corrupt artifacts
        staging = tmp_path / "staging"
        combo_staging = staging / "Accel_CMP"
        combo_staging.mkdir(parents=True)
        (combo_staging / "model.pt").write_text("corrupt")
        (combo_staging / "scaler.pkl").write_text("corrupt")
        (combo_staging / "scorer.pkl").write_text("corrupt")

        config = MagicMock()
        config.staging_dir = str(staging)
        config.production_dir = str(model_path)

        RetrainJobManager._promote_and_reload(
            combo="Accel_CMP",
            config=config,
            model_store=ms,
            event_log=event_log,
        )

        events = event_log.get_recent_events()
        failed = [e for e in events if e["event_type"] == "validation_failed"]
        assert len(failed) == 1
        assert failed[0]["combo"] == "Accel_CMP"
