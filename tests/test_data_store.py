"""Tests for api.data_store.WindowDataStore."""

import json
import os
import tempfile
import threading
from datetime import datetime, timezone

import pytest

from api.data_store import WindowDataStore


@pytest.fixture
def store(tmp_path):
    """Create a WindowDataStore backed by a temporary directory."""
    return WindowDataStore(data_dir=str(tmp_path))


@pytest.fixture
def sample_values():
    """24 hourly transaction counts (synthetic normal pattern)."""
    return [100.0 + i * 10.0 for i in range(24)]


class TestAppend:
    def test_append_creates_row(self, store, sample_values):
        store.append(
            combo="Accel_CMP",
            raw_values=sample_values,
            window_end="2026-03-01T12:00:00Z",
            last_point_score=-0.5,
            is_anomaly=False,
        )
        stats = store.get_stats()
        assert stats["total_windows"] == 1
        assert stats["combos"]["Accel_CMP"]["total"] == 1
        assert stats["combos"]["Accel_CMP"]["normal"] == 1

    def test_append_anomalous_window(self, store, sample_values):
        store.append(
            combo="Star_CMP",
            raw_values=sample_values,
            window_end="2026-03-01T12:00:00Z",
            last_point_score=-3.0,
            is_anomaly=True,
        )
        stats = store.get_stats()
        assert stats["combos"]["Star_CMP"]["anomalous"] == 1
        assert stats["combos"]["Star_CMP"]["normal"] == 0

    def test_append_stores_raw_values_correctly(self, store, sample_values):
        store.append(
            combo="Accel_CMP",
            raw_values=sample_values,
            window_end="2026-03-01T12:00:00Z",
            last_point_score=-0.5,
            is_anomaly=False,
        )
        windows = store.get_normal_windows("Accel_CMP")
        assert len(windows) == 1
        assert windows[0]["raw_values"] == sample_values
        assert len(windows[0]["raw_values"]) == 24

    def test_append_multiple_combos(self, store, sample_values):
        for combo in ["Accel_CMP", "Accel_nopin", "Star_CMP", "Star_nopin"]:
            store.append(
                combo=combo,
                raw_values=sample_values,
                window_end="2026-03-01T12:00:00Z",
                last_point_score=-0.5,
                is_anomaly=False,
            )
        stats = store.get_stats()
        assert stats["total_windows"] == 4
        assert len(stats["combos"]) == 4


class TestGetNormalWindows:
    def test_excludes_anomalous(self, store, sample_values):
        store.append("Accel_CMP", sample_values, "2026-03-01T10:00:00Z", -0.5, False)
        store.append("Accel_CMP", sample_values, "2026-03-01T11:00:00Z", -3.0, True)
        store.append("Accel_CMP", sample_values, "2026-03-01T12:00:00Z", -0.4, False)

        windows = store.get_normal_windows("Accel_CMP")
        assert len(windows) == 2
        # All returned windows should be non-anomalous
        for w in windows:
            assert w["last_point_score"] > -1.0  # our normal scores

    def test_chronological_order(self, store, sample_values):
        timestamps = [
            "2026-03-01T12:00:00Z",
            "2026-03-01T10:00:00Z",
            "2026-03-01T11:00:00Z",
        ]
        for ts in timestamps:
            store.append("Accel_CMP", sample_values, ts, -0.5, False)

        windows = store.get_normal_windows("Accel_CMP")
        assert len(windows) == 3
        # Should be sorted by window_end ASC
        assert windows[0]["window_end"] == "2026-03-01T10:00:00Z"
        assert windows[1]["window_end"] == "2026-03-01T11:00:00Z"
        assert windows[2]["window_end"] == "2026-03-01T12:00:00Z"

    def test_filter_by_date_range(self, store, sample_values):
        store.append("Accel_CMP", sample_values, "2026-02-28T12:00:00Z", -0.5, False)
        store.append("Accel_CMP", sample_values, "2026-03-01T12:00:00Z", -0.5, False)
        store.append("Accel_CMP", sample_values, "2026-03-02T12:00:00Z", -0.5, False)

        windows = store.get_normal_windows(
            "Accel_CMP",
            start_date="2026-03-01T00:00:00Z",
            end_date="2026-03-01T23:59:59Z",
        )
        assert len(windows) == 1
        assert windows[0]["window_end"] == "2026-03-01T12:00:00Z"

    def test_filter_by_combo(self, store, sample_values):
        store.append("Accel_CMP", sample_values, "2026-03-01T12:00:00Z", -0.5, False)
        store.append("Star_CMP", sample_values, "2026-03-01T12:00:00Z", -0.5, False)

        windows = store.get_normal_windows("Accel_CMP")
        assert len(windows) == 1
        windows = store.get_normal_windows("Star_CMP")
        assert len(windows) == 1
        windows = store.get_normal_windows("Accel_nopin")
        assert len(windows) == 0

    def test_limit(self, store, sample_values):
        for i in range(10):
            store.append("Accel_CMP", sample_values, f"2026-03-01T{i:02d}:00:00Z", -0.5, False)

        windows = store.get_normal_windows("Accel_CMP", limit=3)
        assert len(windows) == 3

    def test_empty_store_returns_empty(self, store):
        windows = store.get_normal_windows("Accel_CMP")
        assert windows == []


class TestGetAllWindows:
    def test_includes_both_normal_and_anomalous(self, store, sample_values):
        store.append("Accel_CMP", sample_values, "2026-03-01T10:00:00Z", -0.5, False)
        store.append("Accel_CMP", sample_values, "2026-03-01T11:00:00Z", -3.0, True)
        store.append("Accel_CMP", sample_values, "2026-03-01T12:00:00Z", -0.4, False)

        windows = store.get_all_windows("Accel_CMP")
        assert len(windows) == 3

    def test_includes_is_anomaly_label(self, store, sample_values):
        store.append("Accel_CMP", sample_values, "2026-03-01T10:00:00Z", -0.5, False)
        store.append("Accel_CMP", sample_values, "2026-03-01T11:00:00Z", -3.0, True)

        windows = store.get_all_windows("Accel_CMP")
        assert windows[0]["is_anomaly"] is False
        assert windows[1]["is_anomaly"] is True

    def test_is_anomaly_is_bool(self, store, sample_values):
        store.append("Accel_CMP", sample_values, "2026-03-01T10:00:00Z", -0.5, False)
        store.append("Accel_CMP", sample_values, "2026-03-01T11:00:00Z", -3.0, True)

        windows = store.get_all_windows("Accel_CMP")
        for w in windows:
            assert isinstance(w["is_anomaly"], bool)

    def test_chronological_order(self, store, sample_values):
        store.append("Accel_CMP", sample_values, "2026-03-01T12:00:00Z", -0.5, False)
        store.append("Accel_CMP", sample_values, "2026-03-01T10:00:00Z", -3.0, True)
        store.append("Accel_CMP", sample_values, "2026-03-01T11:00:00Z", -0.4, False)

        windows = store.get_all_windows("Accel_CMP")
        assert windows[0]["window_end"] == "2026-03-01T10:00:00Z"
        assert windows[1]["window_end"] == "2026-03-01T11:00:00Z"
        assert windows[2]["window_end"] == "2026-03-01T12:00:00Z"

    def test_filter_by_date_range(self, store, sample_values):
        store.append("Accel_CMP", sample_values, "2026-02-28T12:00:00Z", -0.5, False)
        store.append("Accel_CMP", sample_values, "2026-03-01T12:00:00Z", -3.0, True)
        store.append("Accel_CMP", sample_values, "2026-03-02T12:00:00Z", -0.4, False)

        windows = store.get_all_windows(
            "Accel_CMP",
            start_date="2026-03-01T00:00:00Z",
            end_date="2026-03-01T23:59:59Z",
        )
        assert len(windows) == 1
        assert windows[0]["is_anomaly"] is True

    def test_filter_by_combo(self, store, sample_values):
        store.append("Accel_CMP", sample_values, "2026-03-01T10:00:00Z", -0.5, False)
        store.append("Star_CMP", sample_values, "2026-03-01T11:00:00Z", -3.0, True)

        windows = store.get_all_windows("Accel_CMP")
        assert len(windows) == 1
        windows = store.get_all_windows("Star_CMP")
        assert len(windows) == 1

    def test_limit(self, store, sample_values):
        for i in range(10):
            store.append("Accel_CMP", sample_values, f"2026-03-01T{i:02d}:00:00Z", -0.5, i % 3 == 0)

        windows = store.get_all_windows("Accel_CMP", limit=3)
        assert len(windows) == 3

    def test_empty_store_returns_empty(self, store):
        windows = store.get_all_windows("Accel_CMP")
        assert windows == []

    def test_returned_dict_has_all_keys(self, store, sample_values):
        store.append("Accel_CMP", sample_values, "2026-03-01T10:00:00Z", -0.5, False)

        windows = store.get_all_windows("Accel_CMP")
        assert len(windows) == 1
        w = windows[0]
        assert set(w.keys()) == {"raw_values", "window_end", "last_point_score", "is_anomaly"}
        assert w["raw_values"] == sample_values
        assert w["window_end"] == "2026-03-01T10:00:00Z"
        assert w["last_point_score"] == -0.5
        assert w["is_anomaly"] is False


class TestCountNormalWindows:
    def test_count_basic(self, store, sample_values):
        for i in range(5):
            store.append("Accel_CMP", sample_values, f"2026-03-01T{i:02d}:00:00Z", -0.5, False)
        store.append("Accel_CMP", sample_values, "2026-03-01T05:00:00Z", -3.0, True)

        assert store.count_normal_windows("Accel_CMP") == 5

    def test_count_with_date_range(self, store, sample_values):
        store.append("Accel_CMP", sample_values, "2026-02-28T12:00:00Z", -0.5, False)
        store.append("Accel_CMP", sample_values, "2026-03-01T12:00:00Z", -0.5, False)
        store.append("Accel_CMP", sample_values, "2026-03-02T12:00:00Z", -0.5, False)

        count = store.count_normal_windows(
            "Accel_CMP",
            start_date="2026-03-01T00:00:00Z",
        )
        assert count == 2

    def test_count_empty(self, store):
        assert store.count_normal_windows("Accel_CMP") == 0


class TestGetStats:
    def test_empty_stats(self, store):
        stats = store.get_stats()
        assert stats["total_windows"] == 0
        assert stats["combos"] == {}

    def test_stats_multiple_combos(self, store, sample_values):
        store.append("Accel_CMP", sample_values, "2026-03-01T10:00:00Z", -0.5, False)
        store.append("Accel_CMP", sample_values, "2026-03-01T11:00:00Z", -3.0, True)
        store.append("Star_CMP", sample_values, "2026-03-01T12:00:00Z", -0.5, False)

        stats = store.get_stats()
        assert stats["total_windows"] == 3
        assert stats["combos"]["Accel_CMP"]["total"] == 2
        assert stats["combos"]["Accel_CMP"]["normal"] == 1
        assert stats["combos"]["Accel_CMP"]["anomalous"] == 1
        assert stats["combos"]["Star_CMP"]["total"] == 1


class TestConcurrency:
    def test_concurrent_appends(self, store, sample_values):
        """Test that concurrent appends from multiple threads are safe."""
        errors = []
        num_threads = 4
        writes_per_thread = 25

        def writer(combo):
            try:
                for i in range(writes_per_thread):
                    store.append(
                        combo=combo,
                        raw_values=sample_values,
                        window_end=f"2026-03-01T{i:02d}:00:00Z",
                        last_point_score=-0.5,
                        is_anomaly=False,
                    )
            except Exception as e:
                errors.append(e)

        combos = ["Accel_CMP", "Accel_nopin", "Star_CMP", "Star_nopin"]
        threads = [threading.Thread(target=writer, args=(c,)) for c in combos[:num_threads]]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent append errors: {errors}"
        stats = store.get_stats()
        assert stats["total_windows"] == num_threads * writes_per_thread


class TestPersistence:
    def test_data_survives_new_instance(self, tmp_path, sample_values):
        """Test that data persists across WindowDataStore instances."""
        store1 = WindowDataStore(data_dir=str(tmp_path))
        store1.append("Accel_CMP", sample_values, "2026-03-01T12:00:00Z", -0.5, False)
        store1.close()

        store2 = WindowDataStore(data_dir=str(tmp_path))
        windows = store2.get_normal_windows("Accel_CMP")
        assert len(windows) == 1
        assert windows[0]["raw_values"] == sample_values
        store2.close()

    def test_db_file_created(self, tmp_path):
        store = WindowDataStore(data_dir=str(tmp_path))
        assert (tmp_path / "scored_windows.db").exists()
        store.close()
