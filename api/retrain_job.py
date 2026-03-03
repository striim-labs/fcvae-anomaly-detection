"""Background job manager for FCVAE retrain pipeline.

Runs retrain jobs in a background thread. Single-job-at-a-time:
rejects new submissions if a job is already running.
"""

import logging
import shutil
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from api.dependencies import ModelStore
    from api.retrain import RetrainPipeline

logger = logging.getLogger(__name__)


class RetrainJobManager:
    """In-memory background job manager. One job at a time."""

    def __init__(self) -> None:
        self._jobs: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._running_job_id: Optional[str] = None

    def submit_job(
        self,
        pipeline: "RetrainPipeline",
        combos: List[str],
        mode: str,
        lookback_days: int,
        start_date: Optional[str],
        end_date: Optional[str],
        auto_reload: bool,
        model_store: "ModelStore",
        min_windows: int,
    ) -> str:
        """Submit a retrain job. Returns job_id.

        Raises RuntimeError if a job is already running.
        """
        with self._lock:
            if self._running_job_id is not None:
                raise RuntimeError("A retrain job is already running")

            job_id = uuid.uuid4().hex[:12]
            self._running_job_id = job_id
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "running",
                "combo_results": [],
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "completed_at": None,
                "duration_seconds": None,
                "mode": mode,
                "combos": combos,
                "auto_reload": auto_reload,
            }

        thread = threading.Thread(
            target=self._run_job,
            args=(
                job_id, pipeline, combos, mode, lookback_days,
                start_date, end_date, auto_reload, model_store, min_windows,
            ),
            daemon=True,
        )
        thread.start()
        return job_id

    def get_job_status(self, job_id: str) -> Optional[dict]:
        """Return job status dict or None if not found."""
        return self._jobs.get(job_id)

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running_job_id is not None

    def _run_job(
        self,
        job_id: str,
        pipeline: "RetrainPipeline",
        combos: List[str],
        mode: str,
        lookback_days: int,
        start_date: Optional[str],
        end_date: Optional[str],
        auto_reload: bool,
        model_store: "ModelStore",
        min_windows: int,
    ) -> None:
        """Execute retrain for each combo in sequence."""
        start_time = time.perf_counter()
        job = self._jobs[job_id]
        overall_status = "completed"

        try:
            for combo in combos:
                logger.info(f"[job={job_id}] Retraining {combo}")
                result = pipeline.retrain_combo(
                    combo=combo,
                    lookback_days=lookback_days if mode == "weekly" else None,
                    start_date=start_date if mode == "manual" else None,
                    end_date=end_date if mode == "manual" else None,
                    min_windows=min_windows,
                )

                result_dict = asdict(result)
                job["combo_results"].append(result_dict)

                if result.status == "success" and auto_reload:
                    self._promote_and_reload(
                        combo, pipeline.config, model_store,
                    )

                if result.status == "error":
                    overall_status = "failed"

        except Exception:
            logger.exception(f"[job={job_id}] Retrain job failed")
            overall_status = "failed"
        finally:
            elapsed = time.perf_counter() - start_time
            with self._lock:
                job["status"] = overall_status
                job["completed_at"] = time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(),
                )
                job["duration_seconds"] = round(elapsed, 2)
                self._running_job_id = None
            logger.info(
                f"[job={job_id}] Retrain job {overall_status} in {elapsed:.1f}s"
            )

    @staticmethod
    def _promote_and_reload(
        combo: str,
        config: "RetrainConfig",  # noqa: F821
        model_store: "ModelStore",
    ) -> None:
        """Copy staged artifacts to production dir and hot-swap the detector."""
        staging = Path(config.staging_dir) / combo
        production = Path(config.production_dir) / combo

        if not staging.exists():
            logger.warning(f"Staging dir does not exist: {staging}")
            return

        shutil.copytree(str(staging), str(production), dirs_exist_ok=True)
        logger.info(f"Promoted {combo} artifacts: {staging} → {production}")

        ok = model_store.reload_combo(
            combo_key=combo,
            model_path=config.production_dir,
            device="cpu",
            n_samples=16,
        )
        if ok:
            logger.info(f"Hot-swapped detector for {combo}")
        else:
            logger.error(f"Failed to hot-swap detector for {combo}")
