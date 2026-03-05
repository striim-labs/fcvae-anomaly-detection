"""
Append-only SQLite store for scored windows.

Captures every scored 24-hour window from /v1/score as a side-effect,
providing training data for the retrain pipeline. Stores raw (pre-normalization)
values so that the StandardScaler can be refit during retraining.

Thread-safe via threading.local() for per-thread SQLite connections
and a threading.Lock for write serialization.
"""
import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS scored_windows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    combo TEXT NOT NULL,
    window_end TEXT NOT NULL,
    raw_values TEXT NOT NULL,
    last_point_score REAL NOT NULL,
    is_anomaly INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
"""

CREATE_INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_combo_window_end ON scored_windows (combo, window_end);",
    "CREATE INDEX IF NOT EXISTS idx_combo_anomaly ON scored_windows (combo, is_anomaly);",
]


class WindowDataStore:
    """
    Append-only SQLite store for scored windows.

    The database file lives at {data_dir}/scored_windows.db.
    Each record contains the combo key, 24 raw hourly transaction counts,
    the window end timestamp, the last-point NLL score, and the anomaly decision.
    """

    def __init__(self, data_dir: str = "data/retrain"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "scored_windows.db"
        self._local = threading.local()
        self._write_lock = threading.Lock()
        # Initialize schema on the main thread
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a thread-local SQLite connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path), check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_schema(self) -> None:
        """Create the table and indexes if they don't exist."""
        conn = self._get_connection()
        conn.execute(CREATE_TABLE_SQL)
        for idx_sql in CREATE_INDEX_SQL:
            conn.execute(idx_sql)
        conn.commit()

    def append(
        self,
        combo: str,
        raw_values: List[float],
        window_end: Optional[str],
        last_point_score: float,
        is_anomaly: bool,
    ) -> None:
        """
        Append a scored window record.

        Args:
            combo: Combo key (e.g., "Accel_CMP")
            raw_values: 24 raw (pre-normalization) hourly transaction counts
            window_end: ISO timestamp of the window's last hour
            last_point_score: NLL score at position [-1]
            is_anomaly: Whether the window was classified as anomalous
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        values_json = json.dumps(raw_values)
        with self._write_lock:
            conn = self._get_connection()
            conn.execute(
                """
                INSERT INTO scored_windows
                    (combo, window_end, raw_values, last_point_score, is_anomaly, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    combo,
                    window_end or now,
                    values_json,
                    last_point_score,
                    1 if is_anomaly else 0,
                    now,
                ),
            )
            conn.commit()

    def get_normal_windows(
        self,
        combo: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        lookback_days: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve normal (non-anomalous) windows for a combo.

        Windows are returned in chronological order by window_end.
        If lookback_days is provided, it overrides start_date with (now - lookback_days).

        Args:
            combo: Combo key
            start_date: ISO start date (inclusive)
            end_date: ISO end date (inclusive)
            lookback_days: Alternative to start_date — look back N days from now
            limit: Max number of windows to return

        Returns:
            List of dicts with keys: raw_values, window_end, last_point_score
        """
        conn = self._get_connection()

        query = "SELECT raw_values, window_end, last_point_score FROM scored_windows WHERE combo = ? AND is_anomaly = 0"
        params: list = [combo]

        if lookback_days is not None:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            query += " AND created_at >= ?"
            params.append(cutoff)
        else:
            if start_date is not None:
                query += " AND window_end >= ?"
                params.append(start_date)
            if end_date is not None:
                query += " AND window_end <= ?"
                params.append(end_date)

        query += " ORDER BY window_end ASC"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [
            {
                "raw_values": json.loads(row["raw_values"]),
                "window_end": row["window_end"],
                "last_point_score": row["last_point_score"],
            }
            for row in rows
        ]

    def get_all_windows(
        self,
        combo: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        lookback_days: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all windows (normal and anomalous) for a combo.

        Same date-range filtering as get_normal_windows() but without the
        is_anomaly = 0 filter. Includes is_anomaly label in each returned dict.

        Returns:
            List of dicts with keys: raw_values, window_end, last_point_score, is_anomaly
        """
        conn = self._get_connection()

        query = "SELECT raw_values, window_end, last_point_score, is_anomaly FROM scored_windows WHERE combo = ?"
        params: list = [combo]

        if lookback_days is not None:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            query += " AND created_at >= ?"
            params.append(cutoff)
        else:
            if start_date is not None:
                query += " AND window_end >= ?"
                params.append(start_date)
            if end_date is not None:
                query += " AND window_end <= ?"
                params.append(end_date)

        query += " ORDER BY window_end ASC"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [
            {
                "raw_values": json.loads(row["raw_values"]),
                "window_end": row["window_end"],
                "last_point_score": row["last_point_score"],
                "is_anomaly": bool(row["is_anomaly"]),
            }
            for row in rows
        ]

    def count_normal_windows(
        self,
        combo: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        lookback_days: Optional[int] = None,
    ) -> int:
        """Count available normal windows for a combo in a date range."""
        conn = self._get_connection()

        query = "SELECT COUNT(*) as cnt FROM scored_windows WHERE combo = ? AND is_anomaly = 0"
        params: list = [combo]

        if lookback_days is not None:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            query += " AND created_at >= ?"
            params.append(cutoff)
        else:
            if start_date is not None:
                query += " AND window_end >= ?"
                params.append(start_date)
            if end_date is not None:
                query += " AND window_end <= ?"
                params.append(end_date)

        row = conn.execute(query, params).fetchone()
        return row["cnt"] if row else 0

    def get_stats(self) -> Dict[str, Any]:
        """Return per-combo window counts and date ranges."""
        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT
                combo,
                COUNT(*) as total,
                SUM(CASE WHEN is_anomaly = 0 THEN 1 ELSE 0 END) as normal,
                SUM(CASE WHEN is_anomaly = 1 THEN 1 ELSE 0 END) as anomalous,
                MIN(window_end) as earliest,
                MAX(window_end) as latest
            FROM scored_windows
            GROUP BY combo
            """
        ).fetchall()

        stats: Dict[str, Any] = {"combos": {}}
        for row in rows:
            stats["combos"][row["combo"]] = {
                "total": row["total"],
                "normal": row["normal"],
                "anomalous": row["anomalous"],
                "earliest": row["earliest"],
                "latest": row["latest"],
            }

        total_row = conn.execute("SELECT COUNT(*) as cnt FROM scored_windows").fetchone()
        stats["total_windows"] = total_row["cnt"] if total_row else 0
        return stats

    def close(self) -> None:
        """Close the thread-local connection if open."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None
