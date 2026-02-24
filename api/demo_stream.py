#!/usr/bin/env python3
"""
Streaming demo for the FCVAE Scoring API.

Loads test-split data, aggregates to hourly counts (same as the Kafka producer),
then slides a 24-hour window one hour at a time — calling POST /v1/score for
each new hour. Prints a live table of results with anomaly highlighting.

Usage:
    # Start the API first:
    #   PYTHONPATH=app:$PYTHONPATH uvicorn api.main:app --port 8000
    #
    # Then run the demo:
    python -m api.demo_stream                          # defaults
    python -m api.demo_stream --combo Star_CMP         # different combo
    python -m api.demo_stream --delay 0.2              # faster
    python -m api.demo_stream --delay 0 --no-color     # full speed, plain output
"""

import argparse
import sys
import time

import pandas as pd
import requests

# ── ANSI colors ──────────────────────────────────────────────────────────────

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def load_and_aggregate(
    file_path: str, combo: str, split: str = "test"
) -> pd.DataFrame:
    """Load CSV, filter by combo + split, aggregate to hourly counts.

    Mirrors the logic in producer/producer.py exactly.
    """
    parts = combo.split("_", 1)
    network = parts[0]
    txn_type = "no-pin" if "nopin" in parts[1].lower() else parts[1]

    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    if "split" in df.columns:
        df = df[df["split"] == split]
    df = df[(df["network_type"] == network) & (df["transaction_type"] == txn_type)]

    if df.empty:
        print(f"No data for combo={combo} split={split}", file=sys.stderr)
        sys.exit(1)

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


def score_window(
    api_url: str, combo: str, timestamps: list, values: list
) -> dict:
    """Call POST /v1/score and return the parsed response."""
    resp = requests.post(
        f"{api_url}/v1/score",
        json={
            "combo": combo,
            "values": values,
            "timestamps": [t.isoformat() for t in timestamps],
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Streaming demo for FCVAE Scoring API")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--combo", default="Accel_nopin", help="Combo key")
    parser.add_argument("--data-path", default="data/synthetic_transactions.csv")
    parser.add_argument("--split", default="test", help="Data split to use")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between requests")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    parser.add_argument("--window-size", type=int, default=24)
    args = parser.parse_args()

    # Disable colors if requested
    if args.no_color:
        global RED, YELLOW, GREEN, DIM, BOLD, RESET
        RED = YELLOW = GREEN = DIM = BOLD = RESET = ""

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"Loading {args.split} data for {args.combo}...")
    hourly = load_and_aggregate(args.data_path, args.combo, args.split)
    total_hours = len(hourly)
    print(f"  {total_hours} hourly records "
          f"({hourly['timestamp'].iloc[0]} to {hourly['timestamp'].iloc[-1]})")
    print(f"  Ground-truth anomaly hours: {hourly['is_anomaly'].sum()}")
    print()

    # ── Check API health ──────────────────────────────────────────────────
    try:
        health = requests.get(f"{args.api_url}/health", timeout=5).json()
        if not health.get("models_loaded", {}).get(args.combo, False):
            print(f"WARNING: Model for {args.combo} not loaded on the API.", file=sys.stderr)
    except requests.ConnectionError:
        print(f"Cannot reach API at {args.api_url}. Is it running?", file=sys.stderr)
        sys.exit(1)

    # ── Sliding-window loop ───────────────────────────────────────────────
    w = args.window_size
    n_windows = total_hours - w + 1
    anomalies_detected = 0

    header = (
        f"{'Hour':>5}  {'Timestamp':>19}  {'Value':>7}  {'Score':>9}  "
        f"{'Threshold':>10}  {'Anomaly?':>8}  {'Ground Truth':>12}"
    )
    print(f"{BOLD}{header}{RESET}")
    print("─" * len(header))

    for i in range(n_windows):
        window = hourly.iloc[i : i + w]
        timestamps = window["timestamp"].tolist()
        values = window["value"].tolist()
        ground_truth = int(window["is_anomaly"].iloc[-1])

        result = score_window(args.api_url, args.combo, timestamps, values)

        is_anom = result["is_anomaly"]
        score = result["last_point_score"]
        thresh = result["threshold"]
        scored_ts = result["scored_timestamp"]

        if is_anom:
            anomalies_detected += 1

        # Format output
        hour_idx = i + w - 1
        val = values[-1]

        if is_anom:
            tag = f"{RED}{BOLD}  YES{RESET}"
        else:
            tag = f"{GREEN}   no{RESET}"

        gt_str = f"{YELLOW}  ANOMALY{RESET}" if ground_truth else f"{DIM}  normal{RESET}"

        line = (
            f"{hour_idx:>5}  {scored_ts:>19}  {val:>7}  {score:>9.4f}  "
            f"{thresh:>10.4f}  {tag:>8}  {gt_str:>12}"
        )
        print(line)

        if args.delay > 0:
            time.sleep(args.delay)

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print(f"{BOLD}Summary{RESET}")
    print(f"  Windows scored:       {n_windows}")
    print(f"  Anomalies detected:   {anomalies_detected}")
    print(f"  Ground-truth anomaly hours: {hourly['is_anomaly'].sum()}")


if __name__ == "__main__":
    main()
