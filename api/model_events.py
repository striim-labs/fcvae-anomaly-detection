"""Append-only JSONL event log for model lifecycle events.

Records retrain starts, completions, rejections, reloads, and validation
failures. The JSONL file can be tailed by a Striim FileReader for
real-time observability.
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ModelEventLog:
    """Thread-safe, append-only JSONL event log.

    Each line is a self-contained JSON object with fields:
        timestamp, event_type, combo (optional), details (optional).

    Event types:
        retrain_started     — a retrain job was submitted
        retrain_completed   — a combo retrained successfully and was promoted
        retrain_rejected    — a combo's retrain was rejected by the validation gate
        model_reloaded      — a combo's detector was hot-swapped (manual or auto)
        validation_failed   — a combo's new detector failed to load after promotion
    """

    def __init__(self, log_path: str = "data/retrain/model_events.jsonl") -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def path(self) -> str:
        return str(self._path)

    def log_event(
        self,
        event_type: str,
        combo: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a single event to the JSONL file."""
        record: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event_type": event_type,
        }
        if combo is not None:
            record["combo"] = combo
        if details:
            record["details"] = details

        line = json.dumps(record, separators=(",", ":"))

        with self._lock:
            try:
                with open(self._path, "a") as f:
                    f.write(line + "\n")
            except Exception:
                logger.warning("Failed to write model event", exc_info=True)

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Read the last N events from the log file."""
        if not self._path.exists():
            return []

        events: List[Dict[str, Any]] = []
        try:
            with open(self._path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception:
            logger.warning("Failed to read model events", exc_info=True)
            return []

        return events[-limit:]
