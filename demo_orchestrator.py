#!/usr/bin/env python3
"""
Striim FCVAE Lifecycle Demo Orchestrator

Drives the full anomaly detection lifecycle through the Striim pipelines:
  1. Generates synthetic weekly transaction data as CSV
  2. Drops CSV into Striim FileReader's watched directory
  3. Waits for scored output, parses and summarizes results
  4. Triggers retrain via the FCVAE scoring API
  5. Monitors retrain events captured by Striim's RetrainMonitor
  6. Hot-swaps the model and continues scoring

Usage:
    python3 demo_orchestrator.py
    python3 demo_orchestrator.py --combo Star_CMP --iterations 2
    python3 demo_orchestrator.py --no-color --delay 5

Prerequisites:
    - Striim Platform running with fcvae.FCVAE and fcvae.RetrainMonitor deployed
    - FCVAE scoring API running at localhost:8000
    - Python packages: numpy, pandas, requests
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is required. Install with: pip install numpy")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("ERROR: requests is required. Install with: pip install requests")
    sys.exit(1)

# ---------------------------------------------------------------------------
# ANSI colors
# ---------------------------------------------------------------------------
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Combo definitions (from generate_transactions.py in the FCVAE repo)
# ---------------------------------------------------------------------------
COMBO_MAP = {
    "Accel_CMP": ("Accel", "CMP"),
    "Accel_nopin": ("Accel", "no-pin"),
    "Star_CMP": ("Star", "CMP"),
    "Star_nopin": ("Star", "no-pin"),
}

# Intraday frequency parameters per combo (from generate_transactions.py)
COMBO_FREQ_PARAMS = {
    ("Accel", "CMP"): {
        "base_rate": 1765,
        "amplitude": 1230,
        "period_h": 24,
        "phase": 0.0,
    },
    ("Accel", "no-pin"): {
        "base_rate": 1370,
        "amplitude": 878,
        "period_h": 12,
        "phase": np.pi / 3,
    },
    ("Star", "CMP"): {
        "base_rate": 2000,
        "amplitude": 1500,
        "period_h": 24,
        "phase": np.pi,
    },
    ("Star", "no-pin"): {
        "base_rate": 1252,
        "amplitude": 878,
        "period_h": 8,
        "phase": np.pi / 2,
    },
}

# Per-combo day-of-week multipliers
#                               Mon   Tue   Wed   Thu   Fri   Sat   Sun
DOW_MULTIPLIERS = {
    ("Accel", "CMP"):    [1.00, 1.05, 1.15, 1.10, 0.95, 0.60, 0.55],
    ("Accel", "no-pin"): [0.90, 1.00, 1.05, 1.10, 1.15, 0.70, 0.65],
    ("Star",  "CMP"):    [1.05, 1.00, 0.95, 1.00, 1.10, 0.75, 0.70],
    ("Star",  "no-pin"): [0.85, 0.95, 1.00, 1.05, 1.00, 0.80, 0.90],
}

# Variability config (from generate_transactions.py)
VARIABILITY_CONFIG = {
    "amplitude_jitter_pct": 0.15,
    "phase_jitter_hours": 1.0,
    "baseline_drift_pct_per_day": 0.002,
    "hourly_noise_std": 0.08,
    "micro_anomaly_prob": 0.05,
    "micro_anomaly_magnitude": 0.15,
}

# Anomaly configs per iteration (mirrors demo_e2e.py WEEKLY_ANOMALIES)
WEEKLY_ANOMALIES = [
    [{"day_offset": 2, "type": "spike", "hours": [10, 11, 12], "multiplier": 3.5}],
    [{"day_offset": 3, "type": "dip", "hours": [10, 11, 12], "multiplier": 0.02}],
    [{"day_offset": 1, "type": "ramp", "hours": [9, 10, 11, 12],
      "start_mult": 2.0, "end_mult": 4.0}],
    [
        {"day_offset": 2, "type": "spike", "hours": [10, 11, 12], "multiplier": 3.0},
        {"day_offset": 5, "type": "dip", "hours": [10, 11, 12], "multiplier": 0.02},
    ],
]

ANOMALY_DESCRIPTIONS = [
    "spike (x3.5) at hours 10-12 on day 3",
    "near-outage (x0.02) at hours 10-12 on day 4",
    "gradual ramp (2x to 4x) at hours 9-12 on day 2",
    "spike (x3.0) hours 10-12 day 3 + outage (x0.02) hours 10-12 day 6",
]


# ---------------------------------------------------------------------------
# Synthetic data generation (using real rate functions from the FCVAE repo)
# ---------------------------------------------------------------------------

def generate_day_params(
    day_index: int,
    combo: tuple,
    rng: np.random.Generator,
) -> Dict:
    """Generate per-day variability parameters (matches generate_transactions.py)."""
    cfg = VARIABILITY_CONFIG

    amplitude_mult = 1.0 + rng.uniform(
        -cfg["amplitude_jitter_pct"], cfg["amplitude_jitter_pct"]
    )
    phase_shift = rng.uniform(
        -cfg["phase_jitter_hours"], cfg["phase_jitter_hours"]
    ) * (2 * np.pi / 24)
    drift_step = rng.uniform(
        -cfg["baseline_drift_pct_per_day"], cfg["baseline_drift_pct_per_day"]
    )
    baseline_drift = drift_step * day_index

    hourly_noise = 1.0 + rng.normal(0, cfg["hourly_noise_std"], size=24)
    hourly_noise = np.clip(hourly_noise, 0.7, 1.3)

    micro_anomaly_mask = rng.random(24) < cfg["micro_anomaly_prob"]
    micro_anomaly_mults = 1.0 + rng.uniform(
        -cfg["micro_anomaly_magnitude"], cfg["micro_anomaly_magnitude"], size=24,
    )

    return {
        "amplitude_mult": amplitude_mult,
        "phase_shift": phase_shift,
        "baseline_drift": baseline_drift,
        "hourly_noise": hourly_noise,
        "micro_anomaly_mask": micro_anomaly_mask,
        "micro_anomaly_mults": micro_anomaly_mults,
    }


def rate_function_v2(
    t: np.ndarray,
    params: dict,
    dow_multiplier: float,
    day_params: Dict,
) -> np.ndarray:
    """Compute instantaneous transaction rate (matches generate_transactions.py)."""
    amplitude_jittered = params["amplitude"] * day_params["amplitude_mult"]
    phase_jittered = params["phase"] + day_params["phase_shift"]
    base_jittered = params["base_rate"] * (1 + day_params["baseline_drift"])

    scaled_base = base_jittered * dow_multiplier
    rate = scaled_base + amplitude_jittered * np.sin(
        2 * np.pi * t / params["period_h"] + phase_jittered
    )

    hours = np.floor(t).astype(int) % 24
    rate = rate * day_params["hourly_noise"][hours]

    for h in range(24):
        if day_params["micro_anomaly_mask"][h]:
            mask = hours == h
            rate[mask] = rate[mask] * day_params["micro_anomaly_mults"][h]

    return np.maximum(rate, 0.0)


def inject_anomaly(
    day_params: Dict,
    anomaly_config: Dict,
    combo: tuple = None,
) -> List[int]:
    """Inject anomaly into day_params by modifying hourly_noise in place."""
    atype = anomaly_config["type"]
    hours = anomaly_config.get("hours", [])
    anomaly_hours = list(hours)

    if atype == "spike":
        mult = anomaly_config["multiplier"]
        for h in hours:
            day_params["hourly_noise"][h] *= mult
    elif atype == "dip":
        mult = anomaly_config["multiplier"]
        for h in hours:
            day_params["hourly_noise"][h] *= mult
    elif atype == "ramp":
        start_mult = anomaly_config["start_mult"]
        end_mult = anomaly_config["end_mult"]
        n = len(hours)
        for i, h in enumerate(hours):
            frac = i / max(n - 1, 1)
            m = start_mult + frac * (end_mult - start_mult)
            day_params["hourly_noise"][h] *= m

    return anomaly_hours


def generate_combo_timestamps(
    day_offset: int,
    params: dict,
    dow_multiplier: float,
    day_params: Dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate transaction timestamps using inhomogeneous Poisson thinning.

    This matches generate_transactions.py's approach for realistic data.
    """
    base_max = params["base_rate"] * (1 + abs(day_params["baseline_drift"]))
    amp_max = params["amplitude"] * day_params["amplitude_mult"]
    noise_max = max(day_params["hourly_noise"])
    micro_max = max(day_params["micro_anomaly_mults"])

    lambda_max = (base_max * dow_multiplier + amp_max) * noise_max * micro_max

    if lambda_max <= 0:
        return np.array([])

    expected_candidates = int(lambda_max * 24 * 1.2)
    inter_arrivals = rng.exponential(1.0 / lambda_max, size=expected_candidates)
    candidate_times_hours = np.cumsum(inter_arrivals)
    candidate_times_hours = candidate_times_hours[candidate_times_hours < 24.0]

    if len(candidate_times_hours) == 0:
        return np.array([])

    acceptance_prob = rate_function_v2(
        candidate_times_hours, params, dow_multiplier, day_params
    ) / lambda_max
    uniform_draws = rng.random(len(candidate_times_hours))
    accepted = candidate_times_hours[uniform_draws < acceptance_prob]

    return accepted  # fractional hours [0, 24)


def generate_week_csv(
    combo_key: str,
    week_start: datetime,
    anomaly_configs: List[Dict],
    seed: int,
    day_index_offset: int = 60,
    all_combos: bool = True,
) -> Tuple[List[Dict], List[int]]:
    """Generate one week of synthetic transaction CSV rows using real rate functions.

    Returns (rows, anomaly_hour_indices).
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []
    anomaly_hour_indices: List[int] = []

    combos_to_gen = list(COMBO_MAP.keys()) if all_combos else [combo_key]

    for combo in combos_to_gen:
        combo_tuple = COMBO_MAP[combo]
        params = COMBO_FREQ_PARAMS[combo_tuple]
        multipliers = DOW_MULTIPLIERS[combo_tuple]
        network, txn_type = combo_tuple

        for day_offset in range(7):
            abs_day = day_index_offset + day_offset
            day_date = week_start + timedelta(days=day_offset)
            dow = abs_day % 7  # Consistent with START_DATE being a Monday
            dow_mult = multipliers[dow]

            # Generate day-specific variability (matches real generator)
            day_params = generate_day_params(abs_day, combo_tuple, rng)

            # Inject anomalies for this combo/day
            is_anomaly_hours: set = set()
            if combo == combo_key:
                for ac in anomaly_configs:
                    if ac["day_offset"] == day_offset:
                        anom_hours = inject_anomaly(day_params, ac, combo=combo_tuple)
                        is_anomaly_hours.update(anom_hours)
                        anomaly_hour_indices.extend(
                            [day_offset * 24 + h for h in anom_hours]
                        )

            # Generate timestamps using Poisson thinning
            accepted_hours = generate_combo_timestamps(
                day_offset, params, dow_mult, day_params, rng,
            )

            # Convert to individual transaction rows
            for frac_hour in accepted_hours:
                hour_int = int(frac_hour)
                minute = int((frac_hour - hour_int) * 60)
                second = int(((frac_hour - hour_int) * 60 - minute) * 60)
                ts = day_date.replace(
                    hour=hour_int, minute=minute, second=second,
                )
                is_anom = 1 if hour_int in is_anomaly_hours else 0
                rows.append({
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "network_type": network,
                    "transaction_type": txn_type,
                    "is_anomaly": is_anom,
                })

    rows.sort(key=lambda r: r["timestamp"])
    return rows, anomaly_hour_indices


def write_csv(rows: List[Dict], path: str, header: bool = True) -> int:
    """Write transaction rows to CSV. Returns row count."""
    fieldnames = ["timestamp", "network_type", "transaction_type", "is_anomaly"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if header:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def append_csv(rows: List[Dict], path: str) -> int:
    """Append transaction rows to an existing CSV (no header). Returns row count."""
    fieldnames = ["timestamp", "network_type", "transaction_type", "is_anomaly"]
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(rows)
    return len(rows)


# ---------------------------------------------------------------------------
# Scored output parsing
# ---------------------------------------------------------------------------

def read_scored_file(filepath: str) -> List[Dict]:
    """Read a single scored_output file.

    Striim's JSONFormatter writes JSON arrays, not JSONL.
    Each file looks like: [ {record1}, {record2}, ... ]
    """
    try:
        with open(filepath) as f:
            content = f.read().strip()
        if not content:
            return []
        data = json.loads(content)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        return []
    except json.JSONDecodeError:
        # File may be partially written; try to salvage what we can
        # by finding the last complete record
        try:
            with open(filepath) as f:
                content = f.read().strip()
            # Try adding a closing bracket if it's incomplete
            if content.startswith("[") and not content.endswith("]"):
                # Find the last complete object (ends with "}")
                last_brace = content.rfind("}")
                if last_brace > 0:
                    content = content[:last_brace + 1] + "]"
                    return json.loads(content)
        except Exception:
            pass
        return []
    except Exception:
        return []


def count_scored_records(output_dir: str) -> int:
    """Count total scored records across all scored_output* files."""
    total = 0
    output_path = Path(output_dir)
    for f in sorted(output_path.glob("scored_output*")):
        total += len(read_scored_file(str(f)))
    return total


def read_all_scored_records(output_dir: str) -> List[Dict]:
    """Read all scored records from all scored_output* files."""
    all_records: List[Dict] = []
    output_path = Path(output_dir)
    for f in sorted(output_path.glob("scored_output*")):
        all_records.extend(read_scored_file(str(f)))
    return all_records


def wait_for_scored_output(
    output_dir: str,
    baseline_count: int,
    min_new_records: int = 10,
    timeout: float = 180.0,
    poll_interval: float = 3.0,
    settle_time: float = 10.0,
) -> List[Dict]:
    """Wait for Striim's FileWriter to produce new scored output.

    Each scored_output.XX file is a JSON array of records.
    Waits until at least min_new_records appear beyond the baseline_count,
    then waits an additional settle_time with no new records.

    Returns ALL records (caller slices by baseline_count).
    """
    start = time.time()
    last_count = baseline_count
    last_change_time = start

    while time.time() - start < timeout:
        current_count = count_scored_records(output_dir)
        new_count = current_count - baseline_count

        if current_count > last_count:
            last_count = current_count
            last_change_time = time.time()

        # We have enough records AND output has settled
        if new_count >= min_new_records:
            time_since_change = time.time() - last_change_time
            if time_since_change >= settle_time:
                break

        time.sleep(poll_interval)

    all_records = read_all_scored_records(output_dir)
    return all_records[baseline_count:]


def summarize_scores(
    scores: List[Dict],
    combo_key: str,
) -> Dict[str, Any]:
    """Summarize scoring results for a given combo."""
    combo_scores = [s for s in scores if s.get("combo_key") == combo_key]
    if not combo_scores:
        return {"total": 0, "detected": 0, "threshold": None}

    detected = sum(1 for s in combo_scores if s.get("is_anomaly") == "true")
    thresholds = [float(s["threshold"]) for s in combo_scores if "threshold" in s]
    avg_threshold = sum(thresholds) / len(thresholds) if thresholds else None

    anomaly_scores_vals = [
        float(s["anomaly_score"]) for s in combo_scores if "anomaly_score" in s
    ]

    return {
        "total": len(combo_scores),
        "detected": detected,
        "threshold": avg_threshold,
        "scores": anomaly_scores_vals,
    }


# ---------------------------------------------------------------------------
# API interaction
# ---------------------------------------------------------------------------

def check_api_health(api_url: str) -> bool:
    """Check if the FCVAE scoring API is healthy."""
    try:
        r = requests.get(f"{api_url}/health", timeout=5)
        data = r.json()
        return data.get("status") == "healthy"
    except Exception:
        return False


def trigger_retrain(
    api_url: str,
    combo_key: str,
    mode: str = "weekly",
    lookback_days: int = 9999,
    auto_reload: bool = True,
    min_windows: int = 50,
) -> Optional[Dict]:
    """Trigger a background retrain job.

    Endpoint: POST /v1/retrain  (returns 202 with job_id)
    Request body follows api.schemas.RetrainRequest:
        combos:        list of combo keys (None = all 4)
        mode:          "weekly" | "manual"
        lookback_days: how far back in the data store to look (default 7)
        auto_reload:   if True, the job manager auto-promotes staged
                       artifacts and hot-swaps the detector on success,
                       and writes the full event lifecycle to model_events.jsonl
        min_windows:   minimum normal windows required per combo (default 50)
    """
    url = f"{api_url}/v1/retrain"
    payload = {
        "combos": [combo_key],
        "mode": mode,
        "lookback_days": lookback_days,
        "auto_reload": auto_reload,
        "min_windows": min_windows,
    }

    try:
        r = requests.post(url, json=payload, timeout=30)
        if r.status_code == 202:
            return r.json()
        elif r.status_code == 409:
            # A retrain job is already running
            return {"error": "conflict", "detail": "A retrain job is already running"}
        elif r.status_code == 503:
            return {"error": "unavailable", "detail": "Data store is disabled"}
        else:
            return {"error": f"http_{r.status_code}", "detail": r.text}
    except requests.ConnectionError as e:
        return {"error": "connection", "detail": str(e)}
    except Exception as e:
        return {"error": "exception", "detail": str(e)}


def poll_retrain_status(
    api_url: str,
    job_id: str,
    timeout: float = 300.0,
    poll_interval: float = 3.0,
) -> Optional[Dict]:
    """Poll for retrain job completion.

    Endpoint: GET /v1/retrain/status/{job_id}
    Response follows api.schemas.RetrainStatusResponse:
        job_id, status, combo_results, started_at, completed_at, duration_seconds
    where combo_results is a list of per-combo dicts with:
        status, combo, duration_seconds, error_message,
        old_f1, new_f1, old_threshold, new_threshold,
        old_mean_nll, new_mean_nll, num_train_windows, num_test_windows
    """
    url = f"{api_url}/v1/retrain/status/{job_id}"

    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                status = data.get("status", "")
                if status in ("completed", "failed"):
                    return data
                # Still "running" -- keep polling
            elif r.status_code == 404:
                return {"error": "not_found", "detail": f"Job '{job_id}' not found"}
        except Exception:
            pass
        time.sleep(poll_interval)

    return None


def reload_model(
    api_url: str,
    combo_key: str,
    staging_dir: str = "models/fcvae_staging",
) -> Optional[Dict]:
    """Manually reload models from staged artifacts.

    Endpoint: POST /v1/model/reload
    Request body follows api.schemas.ReloadRequest:
        combos:      list of combo keys to reload
        staging_dir: path to staged artifacts

    NOTE: If auto_reload=True was set on the retrain call, the
    RetrainJobManager already handles promotion and hot-swap
    automatically. This endpoint is for manual overrides only.
    """
    url = f"{api_url}/v1/model/reload"
    payload = {
        "combos": [combo_key],
        "staging_dir": staging_dir,
    }

    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code == 200:
            return r.json()
        else:
            return {"error": f"http_{r.status_code}", "detail": r.text}
    except Exception as e:
        return {"error": "exception", "detail": str(e)}


# ---------------------------------------------------------------------------
# Striim log monitoring
# ---------------------------------------------------------------------------

def get_retrain_events_from_log(
    log_path: str,
    since_line: int = 0,
) -> Tuple[List[Dict], int]:
    """Parse RetrainEventParser events from the Striim server log.

    Returns (events, last_line_number).
    """
    events: List[Dict] = []
    current_line = 0

    try:
        with open(log_path) as f:
            for i, line in enumerate(f):
                current_line = i
                if i < since_line:
                    continue
                if "Parsed event:" in line:
                    # Extract: type=..., time=..., combo=..., job_id=...
                    parts = line.split("Parsed event:")[1].strip()
                    event = {}
                    for kv in parts.split(","):
                        kv = kv.strip()
                        if "=" in kv:
                            k, v = kv.split("=", 1)
                            event[k.strip()] = v.strip()
                    if event:
                        # Try to extract a timestamp from the log line itself
                        try:
                            log_ts = line[:19]  # typical log timestamp
                            event["log_time"] = log_ts
                        except Exception:
                            pass
                        events.append(event)
    except FileNotFoundError:
        pass

    return events, current_line


def get_retrain_events_from_jsonl(jsonl_path: str) -> List[Dict]:
    """Read model events from the JSONL file written by model_events.py."""
    events = []
    try:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        pass
    return events


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def banner(text: str) -> None:
    width = 60
    print()
    print(f"{BOLD}{'=' * width}{RESET}")
    print(f"{BOLD}  {text}{RESET}")
    print(f"{BOLD}{'=' * width}{RESET}")


def stage_header(title: str) -> None:
    print()
    sep = "-" * 55
    print(f"{BOLD}{sep}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{sep}{RESET}")
    print()


def print_event_timeline(events: List[Dict]) -> None:
    """Print a formatted timeline of retrain events."""
    if not events:
        print(f"  {DIM}(no events captured){RESET}")
        return

    for ev in events:
        ts = ev.get("log_time", ev.get("timestamp", ""))[:19]
        etype = ev.get("type", ev.get("event_type", ""))
        combo = ev.get("combo", "")
        job = ev.get("job_id", "")

        color = DIM
        if "started" in etype:
            color = CYAN
        elif "completed" in etype or "reloaded" in etype:
            color = GREEN
        elif "rejected" in etype or "failed" in etype:
            color = YELLOW

        job_str = f"  job={job}" if job else ""
        print(f"  {color}{ts}  {etype:<22} {combo:<15}{job_str}{RESET}")


# ---------------------------------------------------------------------------
# Main demo loop
# ---------------------------------------------------------------------------

def run_demo(args: argparse.Namespace) -> None:
    combo_key = args.combo
    n_iterations = min(args.iterations, len(WEEKLY_ANOMALIES))
    api_url = args.api_url.rstrip("/")
    data_dir = args.striim_data_dir
    retrain_dir = args.retrain_dir
    log_path = args.striim_log
    delay = args.delay

    banner("FCVAE Lifecycle Demo (Striim Pipeline)")
    print(f"  Combo:       {combo_key}")
    print(f"  Iterations:  {n_iterations} weekly retrain cycles")
    print(f"  API:         {api_url}")
    print(f"  Striim data: {data_dir}")
    print(f"  Log:         {log_path}")

    # --- Preflight checks ---
    print()
    print(f"{BOLD}Preflight checks...{RESET}")

    if not check_api_health(api_url):
        print(f"  {RED}[FAIL] Scoring API not healthy at {api_url}{RESET}")
        print(f"  Start the API and try again.")
        return
    print(f"  {GREEN}[OK]{RESET} Scoring API healthy")

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    print(f"  {GREEN}[OK]{RESET} Data directory: {data_dir}")

    os.makedirs(retrain_dir, exist_ok=True)
    print(f"  {GREEN}[OK]{RESET} Retrain directory: {retrain_dir}")

    if os.path.isfile(log_path):
        print(f"  {GREEN}[OK]{RESET} Striim log found")
    else:
        print(f"  {YELLOW}[WARN]{RESET} Striim log not found at {log_path}")
        print(f"         Event monitoring will rely on JSONL file only")

    # Track log position for incremental reading
    log_line_pos = 0
    if os.path.isfile(log_path):
        with open(log_path) as f:
            log_line_pos = sum(1 for _ in f)

    # Clean old scored output so we start fresh
    output_path = Path(data_dir)
    old_files = list(output_path.glob("scored_output*"))
    if old_files:
        print(f"  {YELLOW}[CLEAN]{RESET} Removing {len(old_files)} old scored_output files")
        for f in old_files:
            f.unlink()

    # Also clean old CSV so FileReader doesn't skip it
    old_csv = output_path / "synthetic_transactions_phase2.csv"
    if old_csv.exists():
        old_csv.unlink()
        print(f"  {YELLOW}[CLEAN]{RESET} Removed old CSV (FileReader will see fresh file)")

    # Clean old retrain events -- truncate instead of deleting (preserves symlinks)
    old_jsonl = Path(retrain_dir) / "model_events.jsonl"
    if old_jsonl.exists() or old_jsonl.is_symlink():
        # Resolve symlink to truncate the actual file
        real_path = old_jsonl.resolve()
        try:
            with open(real_path, "w") as f:
                pass  # truncate to 0 bytes
            print(f"  {YELLOW}[CLEAN]{RESET} Truncated model_events.jsonl")
        except Exception as e:
            print(f"  {YELLOW}[WARN]{RESET} Could not truncate {real_path}: {e}")

    # Baseline: should be 0 after cleanup
    initial_score_count = count_scored_records(data_dir)
    print(f"  Scored output baseline: {initial_score_count} lines")

    # Week start date: 60 days from a reference date (matching demo_e2e.py)
    ref_start = datetime(2025, 1, 1)
    existing_days = 60

    iteration_results: List[Dict] = []

    for it in range(n_iterations):
        week_day_offset = existing_days + it * 7
        week_start = ref_start + timedelta(days=week_day_offset)
        week_end = week_start + timedelta(days=6)
        seed = 42 + 1000 + it * 100

        stage_header(
            f"ITERATION {it + 1}/{n_iterations}: "
            f"Week of {week_start.strftime('%b %d')} - "
            f"{week_end.strftime('%b %d, %Y')}"
        )

        anomaly_configs = WEEKLY_ANOMALIES[it]

        # ---- GENERATE ----
        print(f"  {CYAN}[GENERATE]{RESET}  Generating synthetic data (seed={seed})...")
        rows, anomaly_hours = generate_week_csv(
            combo_key, week_start, anomaly_configs, seed, all_combos=True,
        )
        n_anomaly_hours = len(set(anomaly_hours))
        print(f"             {len(rows)} transaction rows, {n_anomaly_hours} anomaly hours")
        print(f"             Anomaly: {ANOMALY_DESCRIPTIONS[it]}")
        print()

        # Write CSV -- FileReader tracks files by name + byte position.
        # On iteration 1: write a fresh file (header + rows).
        # On subsequent iterations: APPEND rows to the same file (no header).
        # FileReader will pick up only the new bytes from its tracked position.
        csv_filename = "synthetic_transactions_phase2.csv"
        csv_path = os.path.join(data_dir, csv_filename)

        if it == 0:
            # First iteration: clean file with header
            if os.path.exists(csv_path):
                os.remove(csv_path)
                print(f"             Removed old CSV, waiting for FileReader to notice...")
                time.sleep(2)
            n_written = write_csv(rows, csv_path, header=True)
            print(f"  {CYAN}[INJECT]{RESET}   Wrote {n_written} rows (fresh file) to {csv_path}")
        else:
            # Subsequent iterations: append rows WITHOUT header to the same file.
            # FileReader will read from its last known byte position, picking up
            # only the new data.
            n_written = append_csv(rows, csv_path)
            print(f"  {CYAN}[INJECT]{RESET}   Appended {n_written} rows to {csv_path}")

        print(f"             Striim FileReader should pick this up...")
        print()

        # ---- SCORING ----
        print(f"  {CYAN}[SCORING]{RESET}  Waiting for Striim to process...")

        # The FCVAE pipeline produces ~145 windows per combo (168 hours - 23),
        # times 4 combos = ~580 scored lines. But we only need to detect
        # that output appeared and stabilized.
        #
        # Iteration 1 needs a much longer timeout because Striim must:
        #   - Parse the entire ~1M row CSV through FileReader + DSVParser
        #   - Fill HourlyWindow (1hr jumping) across all simulated hours
        #   - Accumulate 24 hourly counts per combo in DailyWindow
        #   - Call the scoring API and flush FileWriter
        # Subsequent iterations are faster because windows carry state
        # and FileReader only reads newly appended bytes.
        score_start = time.time()

        # Adaptive timeouts: generous for iter 1, tighter for subsequent
        poll_timeout = 600 if it == 0 else 180
        settle_timeout = 180 if it == 0 else 120

        # Give Striim time to notice the new file and start processing
        print(f"             Waiting {delay}s for FileReader to pick up file...")
        if it == 0:
            print(f"             (iteration 1: using {poll_timeout}s poll + "
                  f"{settle_timeout}s settle timeout)")
        time.sleep(delay)

        # Now poll for output with progress
        last_reported = initial_score_count
        poll_start = time.time()
        while time.time() - poll_start < poll_timeout:
            current = count_scored_records(data_dir)
            if current > last_reported:
                new = current - initial_score_count
                print(f"             ... {new} scored lines so far (total: {current})")
                last_reported = current
            if current > initial_score_count:
                # Some output appeared; now wait for it to settle
                break
            time.sleep(3)

        # Wait for output to stabilize (no new lines for 10s)
        new_scores = wait_for_scored_output(
            data_dir,
            baseline_count=initial_score_count,
            min_new_records=1,
            timeout=settle_timeout,
            poll_interval=3,
            settle_time=10,
        )

        # Update baseline for next iteration
        initial_score_count = count_scored_records(data_dir)
        score_elapsed = time.time() - score_start

        summary = summarize_scores(new_scores, combo_key)
        total = summary["total"]
        detected = summary["detected"]
        threshold = summary["threshold"]

        total_all_combos = len(new_scores)
        print(f"             Done: {total_all_combos} total lines "
              f"({total} for {combo_key}) in {score_elapsed:.1f}s")
        if detected > 0:
            print(f"             {GREEN}Detected: {detected} anomalies{RESET}")
        else:
            print(f"             Detected: 0 anomalies")
        if threshold is not None:
            print(f"             Threshold: {threshold:.4f}")
        if total == 0 and total_all_combos > 0:
            print(f"             {YELLOW}Note: {total_all_combos} lines scored but none matched "
                  f"combo_key={combo_key}. Check the CSV combo field names.{RESET}")
        elif total == 0:
            print(f"             {YELLOW}No scored output appeared. Check:{RESET}")
            print(f"               - Is fcvae.FCVAE running? (check Striim UI)")
            print(f"               - Is the scoring API at {api_url} healthy?")
            print(f"               - Check: grep 'FCVAEScoreCaller\\|ERROR' {log_path} | tail -20")
        print()

        # ---- RETRAIN ----
        print(f"  {CYAN}[RETRAIN]{RESET}  Triggering retrain via POST /v1/retrain ...")

        retrain_response = trigger_retrain(
            api_url, combo_key,
            mode="weekly",
            lookback_days=9999,
            auto_reload=True,  # job manager handles promote + hot-swap + event logging
            min_windows=50,
        )
        retrain_status = "unavailable"
        retrain_duration = 0.0
        job_id = ""
        new_threshold = None

        if retrain_response and "error" not in retrain_response:
            job_id = retrain_response.get("job_id", "")
            print(f"             Job ID: {job_id}")
            print(f"             Status: {retrain_response.get('status', '')} "
                  f"-- {retrain_response.get('message', '')}")

            if job_id:
                print(f"             Polling GET /v1/retrain/status/{job_id} ...")
                result = poll_retrain_status(api_url, job_id, timeout=300)
                if result and "error" not in result:
                    retrain_status = result.get("status", "unknown")
                    retrain_duration = result.get("duration_seconds", 0) or 0

                    # combo_results is a list of per-combo result dicts
                    combo_results = result.get("combo_results", [])
                    combo_result = None
                    for cr in combo_results:
                        if cr.get("combo") == combo_key:
                            combo_result = cr
                            break
                    # If only one combo, just use the first result
                    if combo_result is None and len(combo_results) == 1:
                        combo_result = combo_results[0]

                    if retrain_status == "completed" and combo_result:
                        cr_status = combo_result.get("status", "")
                        if cr_status == "success":
                            print(f"             {GREEN}Retrain SUCCESS ({retrain_duration:.1f}s){RESET}")
                            # Print validation metrics
                            old_nll = combo_result.get("old_mean_nll")
                            new_nll = combo_result.get("new_mean_nll")
                            if old_nll is not None and new_nll is not None:
                                delta = new_nll - old_nll
                                print(f"             NLL: old={old_nll:.4f} new={new_nll:.4f} delta={delta:+.4f}")
                            old_f1 = combo_result.get("old_f1")
                            new_f1 = combo_result.get("new_f1")
                            if old_f1 is not None and new_f1 is not None:
                                ratio = new_f1 / old_f1 if old_f1 > 0 else float("inf")
                                print(f"             F1:  old={old_f1:.4f} new={new_f1:.4f} (x{ratio:.2f})")
                            old_t = combo_result.get("old_threshold")
                            new_t = combo_result.get("new_threshold")
                            new_threshold = new_t
                            if old_t is not None and new_t is not None:
                                print(f"             Threshold: {old_t:.4f} -> {new_t:.4f}")
                            n_train = combo_result.get("num_train_windows")
                            n_test = combo_result.get("num_test_windows")
                            if n_train is not None:
                                print(f"             Windows: {n_train} train, {n_test} test")
                            print(f"             auto_reload=True: model promoted + hot-swapped automatically")
                        elif cr_status == "rejected":
                            retrain_status = "rejected"
                            print(f"             {YELLOW}Retrain REJECTED{RESET}")
                            msg = combo_result.get("error_message", "")
                            if msg:
                                print(f"             Reason: {msg}")
                            old_f1 = combo_result.get("old_f1")
                            new_f1 = combo_result.get("new_f1")
                            if old_f1 is not None and new_f1 is not None:
                                print(f"             F1:  old={old_f1:.4f} new={new_f1:.4f}")
                        elif cr_status == "insufficient_data":
                            retrain_status = "insufficient_data"
                            print(f"             {YELLOW}Insufficient data{RESET} "
                                  f"({combo_result.get('num_train_windows', '?')} windows)")
                        else:
                            print(f"             Combo status: {cr_status}")
                    elif retrain_status == "failed":
                        print(f"             {RED}Retrain FAILED{RESET}")
                    else:
                        print(f"             Job status: {retrain_status} ({retrain_duration:.1f}s)")
                elif result and "error" in result:
                    print(f"             {YELLOW}Error: {result.get('detail', '')}{RESET}")
                    retrain_status = "error"
                else:
                    print(f"             {YELLOW}Timeout waiting for retrain result{RESET}")
                    retrain_status = "timeout"
        elif retrain_response and "error" in retrain_response:
            err = retrain_response["error"]
            detail = retrain_response.get("detail", "")
            if err == "conflict":
                print(f"             {YELLOW}Conflict: {detail}{RESET}")
                retrain_status = "conflict"
            elif err == "unavailable":
                print(f"             {YELLOW}Data store disabled: {detail}{RESET}")
                retrain_status = "data_store_disabled"
            else:
                print(f"             {YELLOW}Error ({err}): {detail}{RESET}")
                retrain_status = err
        else:
            print(f"             {YELLOW}No response from retrain endpoint{RESET}")

        print()

        # ---- MONITOR ----
        print(f"  {CYAN}[MONITOR]{RESET} Checking RetrainMonitor events...")
        time.sleep(delay)  # Wait for events to propagate through Striim

        # Check JSONL file (written by model_events.py via RetrainJobManager)
        jsonl_path = os.path.join(retrain_dir, "model_events.jsonl")
        jsonl_events = get_retrain_events_from_jsonl(jsonl_path)

        # Check Striim server log (RetrainEventParser OP output)
        log_events, log_line_pos = get_retrain_events_from_log(log_path, log_line_pos)

        if log_events:
            print(f"             Striim RetrainMonitor captured {len(log_events)} events:")
            print_event_timeline(log_events)
        elif jsonl_events:
            print(f"             JSONL has {len(jsonl_events)} events (check Striim UI for parsed view)")
            for ev in jsonl_events[-6:]:
                ts = ev.get("timestamp", "")[:19]
                etype = ev.get("event_type", "")
                combo = ev.get("combo", "")
                details = ev.get("details", {})
                job_str = f"  job={details.get('job_id', '')}" if details.get("job_id") else ""
                print(f"             {DIM}{ts}  {etype:<22} {combo}{job_str}{RESET}")
        else:
            print(f"             {DIM}(no events captured yet){RESET}")

        print()

        # ---- HOT-SWAP ----
        # With auto_reload=True, the RetrainJobManager already handled promotion
        # and hot-swap. We just report the outcome.
        if retrain_status == "completed":
            print(f"  {CYAN}[HOT-SWAP]{RESET} {GREEN}Automatic: model promoted + reloaded (auto_reload=True){RESET}")
            print(f"             Now scoring with v{it + 2}")
        elif retrain_status == "rejected":
            print(f"  {CYAN}[HOT-SWAP]{RESET} {YELLOW}Skipped (retrain was rejected){RESET}")
        elif retrain_status == "insufficient_data":
            print(f"  {CYAN}[HOT-SWAP]{RESET} {YELLOW}Skipped (insufficient data){RESET}")
        else:
            print(f"  {CYAN}[HOT-SWAP]{RESET} {DIM}Skipped (retrain status: {retrain_status}){RESET}")

        print()

        # ---- Track results ----
        iteration_results.append({
            "week": it + 1,
            "start": week_start.strftime("%b %d"),
            "windows": total,
            "detected": detected,
            "threshold": new_threshold if new_threshold is not None else threshold,
            "retrain_status": retrain_status,
            "retrain_duration": retrain_duration,
        })

        # Brief pause between iterations
        if it < n_iterations - 1:
            print()
            print(f"  {DIM}Pausing {delay}s before next iteration...{RESET}")
            time.sleep(delay)

    # ---- SUMMARY ----
    banner("Demo Summary")
    print(f"  Combo: {combo_key} | Iterations: {n_iterations}")
    print()

    hdr = (
        f"  {'Iter':>4}  {'Week':>10}  {'Windows':>7}  "
        f"{'Detect':>6}  {'Threshold':>10}  {'Status':>12}"
    )
    print(hdr)
    sep = "-" * (len(hdr) - 2)
    print(f"  {sep}")

    for r in iteration_results:
        thresh_str = f"{r['threshold']:.4f}" if r["threshold"] is not None else "       -"
        status = r["retrain_status"]
        color = GREEN if status in ("completed", "success") else YELLOW
        print(
            f"  {r['week']:>4}  {r['start']:>10}  {r['windows']:>7}  "
            f"{r['detected']:>6}  {thresh_str:>10}  "
            f"{color}{status:>12}{RESET}"
        )

    n_success = sum(
        1 for r in iteration_results
        if r["retrain_status"] in ("completed", "success")
    )
    total_duration = sum(r["retrain_duration"] for r in iteration_results)

    print()
    print(f"  Retrains: {n_success} success out of {n_iterations}")
    print(f"  Total retrain time: {total_duration:.1f}s")
    print(f"  Model versions: 1 -> {1 + n_success}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Striim FCVAE Lifecycle Demo Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 demo_orchestrator.py
    python3 demo_orchestrator.py --combo Star_CMP --iterations 2
    python3 demo_orchestrator.py --api-url http://localhost:8000 --delay 5
        """,
    )
    parser.add_argument(
        "--combo", default="Accel_CMP",
        choices=list(COMBO_MAP.keys()),
        help="Combo key to demo (default: Accel_CMP)",
    )
    parser.add_argument(
        "--iterations", type=int, default=4,
        help="Number of weekly retrain cycles (default: 4, max: 4)",
    )
    parser.add_argument(
        "--api-url", default="http://localhost:8000",
        help="FCVAE scoring API base URL",
    )
    parser.add_argument(
        "--striim-data-dir", default="/tmp/fcvae_test",
        help="Directory Striim FileReader watches",
    )
    parser.add_argument(
        "--striim-log", default="/opt/Striim/logs/striim.server.log",
        help="Striim server log path",
    )
    parser.add_argument(
        "--retrain-dir", default="/tmp/fcvae_test/retrain",
        help="Directory for model_events.jsonl",
    )
    parser.add_argument(
        "--delay", type=float, default=3.0,
        help="Seconds to wait between stages (default: 3.0)",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable ANSI colors",
    )

    args = parser.parse_args()

    if args.no_color:
        global RED, GREEN, YELLOW, CYAN, BOLD, DIM, RESET
        RED = GREEN = YELLOW = CYAN = BOLD = DIM = RESET = ""

    run_demo(args)


if __name__ == "__main__":
    main()