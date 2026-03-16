#!/usr/bin/env python3
"""
compare_fcvae_forseer.py

Compare FCVAE and Forseer anomaly detection results on the penny transaction
dataset. Both models scored the same synthetic_transactions.csv data through
the Striim FCVAE_PENNY pipeline. This script:

1. Parses FCVAE scored output (JSON arrays with combo_key, is_anomaly, anomaly_score, threshold, window_end)
2. Parses Forseer scored output (JSON arrays with a forecast string containing predictedValue, actualValue, isAnomaly, currentTimestamp, etc.)
3. Loads ground-truth penny_is_anomaly labels from the source CSV
4. Aligns all three by hour and computes precision, recall, F1 for each model

Usage:
    python compare_fcvae_forseer.py \
        --fcvae-dir data/comparison \
        --forseer-dir data/comparison \
        --csv-path data/synthetic_transactions.csv \
        --output-dir plots/comparison
"""

import argparse
import json
import re
import glob
import os
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _load_striim_json(fpath: str) -> list:
    """Load a Striim JSONFormatter output file.

    Striim writes each flush as a separate JSON array. A single file may
    contain one array, multiple concatenated arrays (][), or may have
    trailing commas or other quirks. This function handles all cases.
    """
    with open(fpath, "r") as f:
        content = f.read().strip()

    if not content:
        return []

    # First, try parsing as a single JSON array
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        return [data]
    except json.JSONDecodeError:
        pass

    # Handle concatenated arrays: "][" -> "],[" then wrap in outer array
    content = re.sub(r"\]\s*\[", "],[", content)
    try:
        data = json.loads("[" + content + "]") if not content.startswith("[") else json.loads(content)
        # Flatten if we got a list of lists
        if data and isinstance(data[0], list):
            return [item for sublist in data for item in sublist]
        return data
    except json.JSONDecodeError:
        pass

    # Last resort: extract individual JSON objects with regex
    objects = []
    for m in re.finditer(r"\{[^{}]+\}", content):
        try:
            objects.append(json.loads(m.group()))
        except json.JSONDecodeError:
            continue
    return objects


def parse_fcvae_output(directory: str) -> pd.DataFrame:
    """Parse all penny_scored_output.* files into a DataFrame."""
    records = []
    files = sorted(glob.glob(os.path.join(directory, "penny_scored_output.*")))
    for fpath in files:
        data = _load_striim_json(fpath)
        for entry in data:
            if entry.get("combo_key") != "Penny_All":
                continue
            # Skip incomplete entries (e.g., truncated writes at shutdown)
            if not all(k in entry for k in ("window_end", "is_anomaly", "anomaly_score", "threshold")):
                continue
            # Parse window_end timestamp: "2025/01/06 23:08:54.000"
            ts_str = entry["window_end"].replace("\\/", "/")
            ts = datetime.strptime(ts_str[:19], "%Y/%m/%d %H:%M:%S")
            records.append({
                "timestamp": ts,
                "hour": ts.replace(minute=0, second=0, microsecond=0),
                "fcvae_is_anomaly": entry["is_anomaly"] == "true",
                "fcvae_score": float(entry["anomaly_score"]),
                "fcvae_threshold": float(entry["threshold"]),
            })
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def parse_forseer_field(forecast_str: str) -> dict:
    """Parse Forseer's Java toString() format into a dict.

    Example input:
    "[ predictedValue: 188.0, actualValue: 188.0, ..., isAnomaly: false, currentTimestamp: 2025-01-06T00:00:05.000-08:00, ...]"
    """
    # Strip outer brackets
    s = forecast_str.strip()
    if s.startswith("["):
        s = s[1:]
    if s.endswith("]"):
        s = s[:-1]
    s = s.strip()

    result = {}

    # Extract isAnomaly (boolean)
    m = re.search(r"isAnomaly:\s*(true|false)", s)
    if m:
        result["isAnomaly"] = m.group(1) == "true"

    # Extract predictedValue
    m = re.search(r"predictedValue:\s*([\d.E+-]+)", s)
    if m:
        try:
            result["predictedValue"] = float(m.group(1))
        except ValueError:
            result["predictedValue"] = None

    # Extract actualValue
    m = re.search(r"actualValue:\s*([\d.E+-]+)", s)
    if m:
        try:
            result["actualValue"] = float(m.group(1))
        except ValueError:
            result["actualValue"] = None

    # Extract adjustedSymmetricPercentageError
    m = re.search(r"adjustedSymmetricPercentageError:\s*([\d.E+-]+)", s)
    if m:
        try:
            result["percentageError"] = float(m.group(1))
        except ValueError:
            result["percentageError"] = None

    # Extract percentageErrorBound
    m = re.search(r"percentageErrorBound:\s*([\d.E+-]+)", s)
    if m:
        try:
            result["errorBound"] = float(m.group(1))
        except ValueError:
            result["errorBound"] = None

    # Extract currentTimestamp
    m = re.search(r"currentTimestamp:\s*(\S+)", s)
    if m:
        ts_str = m.group(1).rstrip(",")
        try:
            # Parse ISO format with timezone: 2025-01-06T00:00:05.000-08:00
            # Remove the last colon in timezone for strptime compatibility
            if re.search(r"[+-]\d{2}:\d{2}$", ts_str):
                ts_str = ts_str[:-3] + ts_str[-2:]
            ts = datetime.strptime(ts_str[:24], "%Y-%m-%dT%H:%M:%S.%f")
            result["timestamp"] = ts
        except (ValueError, IndexError):
            try:
                ts = datetime.strptime(ts_str[:19], "%Y-%m-%dT%H:%M:%S")
                result["timestamp"] = ts
            except ValueError:
                result["timestamp"] = None

    return result


def parse_forseer_output(directory: str) -> pd.DataFrame:
    """Parse all forseer_scored_output.* files into a DataFrame."""
    records = []
    files = sorted(glob.glob(os.path.join(directory, "forseer_scored_output.*")))
    for fpath in files:
        data = _load_striim_json(fpath)
        for entry in data:
            forecast_str = entry.get("forecast", "")
            parsed = parse_forseer_field(forecast_str)
            if parsed.get("timestamp") is None:
                continue
            ts = parsed["timestamp"]
            records.append({
                "timestamp": ts,
                "hour": ts.replace(minute=0, second=0, microsecond=0),
                "forseer_is_anomaly": parsed.get("isAnomaly", False),
                "forseer_predicted": parsed.get("predictedValue"),
                "forseer_actual": parsed.get("actualValue"),
                "forseer_pct_error": parsed.get("percentageError"),
                "forseer_error_bound": parsed.get("errorBound"),
            })
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_ground_truth(csv_path: str) -> pd.DataFrame:
    """Load ground-truth penny anomaly labels from the source CSV.

    Aggregates to hourly penny counts and labels. An hour is anomalous if any
    penny transaction in that hour has penny_is_anomaly=1.
    """
    print(f"Loading ground truth from {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    # Filter to penny transactions
    penny = df[df["amount"] < 1.00].copy()

    # Floor to hour
    penny["hour"] = penny["timestamp"].dt.floor("h")

    # Aggregate: count per hour, max anomaly label per hour
    label_col = "penny_is_anomaly" if "penny_is_anomaly" in penny.columns else "is_anomaly"
    hourly = penny.groupby("hour").agg(
        penny_count=("amount", "count"),
        is_anomaly=(label_col, "max"),
    ).reset_index()

    hourly["is_anomaly"] = hourly["is_anomaly"].astype(bool)
    return hourly


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute precision, recall, F1 from boolean arrays."""
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_flagged": tp + fp,
        "total_anomalous": tp + fn,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare FCVAE vs Forseer penny anomaly detection")
    parser.add_argument("--fcvae-dir", default="data/comparison",
                        help="Directory containing penny_scored_output.* files")
    parser.add_argument("--forseer-dir", default="data/comparison",
                        help="Directory containing forseer_scored_output.* files")
    parser.add_argument("--csv-path", default="data/synthetic_transactions.csv",
                        help="Path to source CSV with penny_is_anomaly labels")
    parser.add_argument("--output-dir", default="plots/comparison",
                        help="Directory for output plots and reports")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse outputs
    print("Parsing FCVAE output...")
    fcvae_df = parse_fcvae_output(args.fcvae_dir)
    print(f"  FCVAE: {len(fcvae_df)} scored windows, {fcvae_df['fcvae_is_anomaly'].sum()} anomalies flagged")

    print("Parsing Forseer output...")
    forseer_df = parse_forseer_output(args.forseer_dir)
    print(f"  Forseer: {len(forseer_df)} scored windows, {forseer_df['forseer_is_anomaly'].sum()} anomalies flagged")

    # Load ground truth
    gt_df = load_ground_truth(args.csv_path)
    print(f"  Ground truth: {len(gt_df)} hours, {gt_df['is_anomaly'].sum()} anomalous hours")

    # Align by hour
    # FCVAE: window_end is the last point of the 24-hour window.
    # The FCVAE decision applies to the hour of window_end (last-point prediction).
    # If multiple windows end in the same hour, take the max (any anomaly = anomaly).
    if not fcvae_df.empty:
        fcvae_hourly = fcvae_df.groupby("hour").agg(
            fcvae_is_anomaly=("fcvae_is_anomaly", "max"),
            fcvae_score=("fcvae_score", "min"),  # most anomalous score (most negative)
        ).reset_index()
    else:
        fcvae_hourly = pd.DataFrame(columns=["hour", "fcvae_is_anomaly", "fcvae_score"])

    # Forseer: currentTimestamp is the hour being evaluated.
    if not forseer_df.empty:
        forseer_hourly = forseer_df.groupby("hour").agg(
            forseer_is_anomaly=("forseer_is_anomaly", "max"),
            forseer_predicted=("forseer_predicted", "first"),
            forseer_actual=("forseer_actual", "first"),
            forseer_pct_error=("forseer_pct_error", "first"),
        ).reset_index()
    else:
        forseer_hourly = pd.DataFrame(columns=["hour", "forseer_is_anomaly", "forseer_predicted",
                                                 "forseer_actual", "forseer_pct_error"])

    # Merge all three on hour
    merged = gt_df.merge(fcvae_hourly, on="hour", how="left")
    merged = merged.merge(forseer_hourly, on="hour", how="left")

    # Fill NaN (hours where a model had no output) with False (not anomalous)
    merged["fcvae_is_anomaly"] = merged["fcvae_is_anomaly"].fillna(False).astype(bool)
    merged["forseer_is_anomaly"] = merged["forseer_is_anomaly"].fillna(False).astype(bool)

    print(f"\n  Merged dataset: {len(merged)} hours")
    print(f"  Hours with FCVAE output: {merged['fcvae_score'].notna().sum()}")
    print(f"  Hours with Forseer output: {merged['forseer_predicted'].notna().sum()}")

    # Compute metrics
    y_true = merged["is_anomaly"].values

    fcvae_metrics = compute_metrics(y_true, merged["fcvae_is_anomaly"].values)
    forseer_metrics = compute_metrics(y_true, merged["forseer_is_anomaly"].values)

    # Print results
    print("\n" + "=" * 70)
    print("COMPARISON: FCVAE vs Forseer on Penny Transaction Anomaly Detection")
    print("=" * 70)

    print(f"\nGround truth: {int(y_true.sum())} anomalous hours out of {len(y_true)} total")

    print(f"\n{'Metric':<25} {'FCVAE':>12} {'Forseer':>12}")
    print("-" * 50)
    for key in ["TP", "FP", "FN", "TN", "precision", "recall", "f1", "total_flagged"]:
        fv = fcvae_metrics[key]
        sv = forseer_metrics[key]
        if isinstance(fv, float):
            print(f"{key:<25} {fv:>12.4f} {sv:>12.4f}")
        else:
            print(f"{key:<25} {fv:>12d} {sv:>12d}")

    # Detailed breakdown: which anomalous hours did each model catch?
    anomalous_hours = merged[merged["is_anomaly"]].copy()
    if not anomalous_hours.empty:
        print(f"\n{'Hour':<22} {'GT':>4} {'FCVAE':>6} {'Forseer':>8} {'Penny Count':>12}")
        print("-" * 55)
        for _, row in anomalous_hours.iterrows():
            hour_str = row["hour"].strftime("%Y-%m-%d %H:00")
            gt_flag = "YES" if row["is_anomaly"] else "no"
            fcvae_flag = "YES" if row["fcvae_is_anomaly"] else "no"
            forseer_flag = "YES" if row["forseer_is_anomaly"] else "no"
            count = int(row["penny_count"]) if pd.notna(row["penny_count"]) else "?"
            print(f"{hour_str:<22} {gt_flag:>4} {fcvae_flag:>6} {forseer_flag:>8} {count:>12}")

    # False positive analysis for Forseer
    forseer_fps = merged[merged["forseer_is_anomaly"] & ~merged["is_anomaly"]]
    if not forseer_fps.empty:
        print(f"\nForseer false positives: {len(forseer_fps)} hours")
        print(f"  Date range: {forseer_fps['hour'].min()} to {forseer_fps['hour'].max()}")

        # Breakdown by week
        forseer_fps_copy = forseer_fps.copy()
        forseer_fps_copy["week"] = forseer_fps_copy["hour"].dt.isocalendar().week
        weekly_fps = forseer_fps_copy.groupby("week").size()
        print(f"  By week: {dict(weekly_fps)}")

    # Save report
    report_path = os.path.join(args.output_dir, "comparison_report.txt")
    with open(report_path, "w") as f:
        f.write("FCVAE vs Forseer: Penny Transaction Anomaly Detection Comparison\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Ground truth anomalous hours: {int(y_true.sum())}\n")
        f.write(f"Total hours evaluated: {len(y_true)}\n\n")

        f.write(f"{'Metric':<25} {'FCVAE':>12} {'Forseer':>12}\n")
        f.write("-" * 50 + "\n")
        for key in ["TP", "FP", "FN", "TN", "precision", "recall", "f1", "total_flagged"]:
            fv = fcvae_metrics[key]
            sv = forseer_metrics[key]
            if isinstance(fv, float):
                f.write(f"{key:<25} {fv:>12.4f} {sv:>12.4f}\n")
            else:
                f.write(f"{key:<25} {fv:>12d} {sv:>12d}\n")

    print(f"\nReport saved to {report_path}")

    # Save merged data for further analysis
    merged_path = os.path.join(args.output_dir, "merged_hourly.csv")
    merged.to_csv(merged_path, index=False)
    print(f"Merged hourly data saved to {merged_path}")


if __name__ == "__main__":
    main()