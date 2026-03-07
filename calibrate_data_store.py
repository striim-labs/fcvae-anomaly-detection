#!/usr/bin/env python3
"""
Pre-demo calibration: seed the API's data store with test-split windows
so the retrain pipeline has labeled data to work with.

This is the equivalent of demo_e2e.py's Stage 2, but done via the API
so the data store is populated on the server side.

Run this ONCE before running demo_orchestrator.py.

Usage:
    python3 calibrate_data_store.py
    python3 calibrate_data_store.py --combo Accel_CMP --api-url http://localhost:8000
    python3 calibrate_data_store.py --all-combos --data-path data/synthetic_transactions.csv
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from typing import List

COMBO_MAP = {
    "Accel_CMP": ("Accel", "CMP"),
    "Accel_nopin": ("Accel", "no-pin"),
    "Star_CMP": ("Star", "CMP"),
    "Star_nopin": ("Star", "no-pin"),
}

WINDOW_SIZE = 24

# Oracle thresholds derived from demo_e2e.py --pretrained --iterations 0
# for each combo. These are the F1-max last-point NLL thresholds computed
# on the test split. Convention: score < threshold => anomaly.
ORACLE_THRESHOLDS = {
    "Accel_CMP": -1.1483,     # P=1.00  R=0.86  F1=0.92
    "Accel_nopin": -0.9271,   # P=1.00  R=1.00  F1=1.00
    "Star_CMP": -1.4831,      # P=1.00  R=1.00  F1=1.00
    "Star_nopin": -0.6914,    # P=0.88  R=1.00  F1=0.93
}

BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


def load_and_aggregate(file_path: str, combo_key: str, split: str) -> pd.DataFrame:
    """Load CSV, filter by combo + split, aggregate to hourly counts."""
    combo_tuple = COMBO_MAP[combo_key]
    network, txn_type = combo_tuple

    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    if "split" in df.columns:
        df = df[df["split"] == split]
    df = df[(df["network_type"] == network) & (df["transaction_type"] == txn_type)]

    if df.empty:
        return pd.DataFrame()

    df["hour_bucket"] = df["timestamp"].dt.floor("h")
    hourly = (
        df.groupby("hour_bucket")
        .agg(value=("timestamp", "size"), is_anomaly=("is_anomaly", "max"))
        .reset_index()
    )

    full_hours = pd.date_range(
        start=hourly["hour_bucket"].min(),
        end=hourly["hour_bucket"].max(),
        freq="h",
    )
    full_df = pd.DataFrame({"hour_bucket": full_hours})
    hourly = full_df.merge(hourly, on="hour_bucket", how="left")
    hourly["value"] = hourly["value"].fillna(0).astype(int)
    hourly["is_anomaly"] = hourly["is_anomaly"].fillna(0).astype(int)
    hourly = hourly.rename(columns={"hour_bucket": "timestamp"})
    return hourly


def seed_combo(api_url: str, combo_key: str, data_path: str, split: str = "test"):
    """Score all test windows for a combo via the API to seed the data store."""
    print(f"\n  {CYAN}[{combo_key}]{RESET} Loading {split} split...")

    hourly = load_and_aggregate(data_path, combo_key, split)
    if hourly.empty:
        print(f"    {YELLOW}No data for {combo_key} split={split}{RESET}")
        return

    values = hourly["value"].values.astype(float)
    timestamps = hourly["timestamp"].tolist()
    anomalies = hourly["is_anomaly"].values

    n_windows = len(values) - WINDOW_SIZE + 1
    n_anomaly = int(anomalies.sum())

    print(f"    {len(hourly)} hourly records, {n_windows} windows, {n_anomaly} anomaly hours")

    # Score each window via the API (this also accumulates to the data store)
    scored = 0
    detected = 0
    errors = 0

    for i in range(n_windows):
        window_values = values[i : i + WINDOW_SIZE].tolist()
        window_timestamps = [
            ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            for ts in timestamps[i : i + WINDOW_SIZE]
        ]

        try:
            r = requests.post(
                f"{api_url}/v1/score",
                json={
                    "combo": combo_key,
                    "values": window_values,
                    "timestamps": window_timestamps,
                },
                timeout=10,
            )
            if r.status_code == 200:
                result = r.json()
                scored += 1
                if result.get("is_anomaly"):
                    detected += 1
            else:
                errors += 1
        except Exception:
            errors += 1

    print(f"    Scored {scored} windows, {detected} anomalies detected, {errors} errors")
    print(f"    {GREEN}Data store seeded for {combo_key}{RESET}")


def main():
    parser = argparse.ArgumentParser(description="Pre-demo data store calibration")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--combo", default=None, help="Single combo to seed (default: all)")
    parser.add_argument("--all-combos", action="store_true", help="Seed all 4 combos")
    parser.add_argument(
        "--data-path",
        default="data/synthetic_transactions.csv",
        help="Path to synthetic transactions CSV with splits",
    )
    parser.add_argument("--split", default="test", help="Data split to use")
    parser.add_argument(
        "--oracle-thresholds",
        default="models/fcvae/oracle_thresholds.json",
        help="Path to oracle_thresholds.json (default: models/fcvae/oracle_thresholds.json)",
    )
    parser.add_argument(
        "--skip-calibration", action="store_true",
        help="Seed data store only, don't update oracle thresholds",
    )
    args = parser.parse_args()

    # Health check
    try:
        r = requests.get(f"{args.api_url}/health", timeout=5)
        health = r.json()
        print(f"{BOLD}API healthy: {health.get('status')}{RESET}")
        print(f"  Models: {health.get('models_loaded')}")
    except Exception as e:
        print(f"API not reachable at {args.api_url}: {e}")
        sys.exit(1)

    combos = list(COMBO_MAP.keys()) if (args.all_combos or args.combo is None) else [args.combo]

    # Write oracle thresholds before seeding so the API uses them
    if not args.skip_calibration:
        thresholds_path = Path(args.oracle_thresholds)
        print(f"\n{BOLD}Writing oracle thresholds to {thresholds_path}{RESET}")

        # Load existing file to preserve any extra keys
        if thresholds_path.exists():
            with open(thresholds_path) as f:
                existing = json.load(f)
        else:
            existing = {}

        for combo in combos:
            old_val = existing.get(combo)
            new_val = ORACLE_THRESHOLDS[combo]
            existing[combo] = new_val
            old_str = f"{old_val:.4f}" if old_val is not None else "none"
            print(f"  {combo}: {old_str} -> {new_val:.4f}")

        thresholds_path.parent.mkdir(parents=True, exist_ok=True)
        with open(thresholds_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"  {GREEN}Wrote {thresholds_path}{RESET}")

        # Reload models so the API picks up the new thresholds
        print(f"  Reloading models via POST /v1/model/reload ...")
        try:
            r = requests.post(
                f"{args.api_url}/v1/model/reload",
                json={"combos": combos},
                timeout=15,
            )
            if r.status_code == 200:
                result = r.json()
                reloaded = result.get("reloaded_combos", [])
                print(f"  {GREEN}Reloaded: {reloaded}{RESET}")
            else:
                print(f"  {YELLOW}Reload returned {r.status_code}: {r.text}{RESET}")
        except Exception as e:
            print(f"  {YELLOW}Reload failed: {e}{RESET}")
            print(f"  Restart the API to pick up the new thresholds.")

    # Seed data store
    print(f"\n{BOLD}Seeding data store with {args.split} split for: {combos}{RESET}")

    for combo in combos:
        seed_combo(args.api_url, combo, args.data_path, args.split)

    # Check model info after seeding
    try:
        r = requests.get(f"{args.api_url}/v1/model/info", timeout=5)
        info = r.json()
        print(f"\n{BOLD}Model thresholds after calibration:{RESET}")
        for combo_key, combo_info in info.get("combos", {}).items():
            thresh = combo_info.get("threshold")
            print(f"  {combo_key}: threshold = {thresh}")
    except Exception:
        pass

    print(f"\n{GREEN}Done. Data store is seeded and thresholds are calibrated. "
          f"Run demo_orchestrator.py next.{RESET}")


if __name__ == "__main__":
    main()