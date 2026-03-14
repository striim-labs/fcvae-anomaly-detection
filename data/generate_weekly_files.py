"""
Generate Weekly Demo Files for FCVAE Striim Pipeline
====================================================
Produces 4 separate CSV files (week_1.csv through week_4.csv), each containing
7 days of synthetic transaction data. Drop them into /tmp/fcvae_test/ one at a
time to drive the scoring pipeline and retrain trigger.

Uses the same generation logic as generate_transactions.py (v2) with consistent
distributions across all weeks. Anomaly injection is optional (--inject-anomalies)
and targets weeks 3 and 4 with spike, dip, ramp, and compound patterns.

Usage:
    # Normal data only (all 4 weeks clean)
    python generate_weekly_files.py --output-dir /tmp/fcvae_demo_files

    # With anomalies injected in weeks 3 and 4
    python generate_weekly_files.py --output-dir /tmp/fcvae_demo_files --inject-anomalies

Then during the demo, copy files one at a time:
    cp /tmp/fcvae_demo_files/week_1.csv /tmp/fcvae_test/
    # wait for scoring + retrain to complete
    cp /tmp/fcvae_demo_files/week_2.csv /tmp/fcvae_test/
    # ... etc
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Import the generation functions from the existing module.
# Adjust this import path if generate_transactions.py is not on PYTHONPATH.
from generate_transactions import (
    COMBO_KEYS,
    COMBO_FREQ_PARAMS,
    DOW_MULTIPLIERS,
    VARIABILITY_CONFIG,
    generate_day_params,
    generate_combo_timestamps_v2,
    generate_amounts,
    inject_anomaly,
    apply_penny_amounts,
    PENNY_CONFIG,
)

# -- Configuration ------------------------------------------------------------

NUM_WEEKS = 4
DAYS_PER_WEEK = 7
START_DATE = datetime(2025, 3, 3)  # a Monday -- week 1 starts here
BASE_SEED = 7777                   # different from training seed (42)


# -- Per-week anomaly schedules ------------------------------------------------
#
# Each entry is a dict consumed by inject_anomaly() from generate_transactions.py.
# "day_in_week" (0-6, Mon-Sun) controls which day inside that week gets the
# anomaly.  All other fields follow the existing anomaly config schema:
#   type        : spike | dip | ramp | duration
#   hours       : list of affected hours
#   multiplier  : rate multiplier (spike/dip)
#   start_mult / end_mult : for ramp
#   peak_shift_hours      : for duration
#   combo_hours : optional per-combo hour override (for dip)
#
# Weeks 1-2 are always clean.  Weeks 3-4 get anomalies when --inject-anomalies
# is passed.

WEEKLY_ANOMALIES: Dict[int, List[dict]] = {
    # Week 3 (index 2): one spike on Wednesday, one dip on Friday
    2: [
        {
            "day_in_week": 2,   # Wednesday
            "type": "spike",
            "hours": [10, 11, 12],
            "multiplier": 3.0,
        },
        {
            "day_in_week": 4,   # Friday
            "type": "dip",
            "hours": [14],
            "multiplier": 0.1,
            "combo_hours": {
                ("Star", "CMP"): [18],
                ("Star", "no-pin"): [8],
            },
        },
    ],
    # Week 4 (index 3): ramp on Tuesday, compound spike+dip on Thursday
    3: [
        {
            "day_in_week": 1,   # Tuesday
            "type": "ramp",
            "hours": [8, 9, 10, 11],
            "start_mult": 1.5,
            "end_mult": 2.5,
        },
        {
            "day_in_week": 3,   # Thursday -- spike in the morning
            "type": "spike",
            "hours": [9, 10, 11],
            "multiplier": 2.5,
        },
        {
            "day_in_week": 3,   # Thursday -- dip in the afternoon (compound)
            "type": "dip",
            "hours": [15, 16],
            "multiplier": 0.15,
        },
    ],
}


# -- Per-week penny anomaly schedules ------------------------------------------
#
# Penny anomalies multiply the penny fraction during specified hours, producing
# a spike in sub-dollar transaction counts without changing total volume.
# These are injected only when --inject-penny-anomalies is passed.
# Each entry has:
#   day_in_week        : 0-6 (Mon-Sun)
#   hours              : list of affected hours
#   penny_spike_factor : multiplier on the penny fraction for those hours

PENNY_WEEKLY_ANOMALIES: Dict[int, List[dict]] = {
    # Week 3 (index 2): mild carding probe on Monday
    2: [
        {
            "day_in_week": 0,   # Monday
            "hours": list(range(2, 8)),
            "penny_spike_factor": 5.0,
        },
    ],
    # Week 4 (index 3): moderate probe Wednesday, severe probe Saturday
    3: [
        {
            "day_in_week": 2,   # Wednesday
            "hours": list(range(10, 14)),
            "penny_spike_factor": 10.0,
        },
        {
            "day_in_week": 5,   # Saturday
            "hours": list(range(0, 4)),
            "penny_spike_factor": 25.0,
        },
    ],
}


def _build_penny_anomaly_lookup(
    week_index: int,
) -> Dict[int, Dict[int, float]]:
    """
    Build a mapping from day-in-week (0-6) to {hour: spike_factor} for penny
    anomalies in the given week.

    Returns:
        Dict mapping day_in_week -> {hour: spike_factor}
    """
    configs = PENNY_WEEKLY_ANOMALIES.get(week_index, [])
    lookup: Dict[int, Dict[int, float]] = {}
    for cfg in configs:
        d = cfg["day_in_week"]
        spike_hours = {h: cfg["penny_spike_factor"] for h in cfg["hours"]}
        if d in lookup:
            lookup[d].update(spike_hours)
        else:
            lookup[d] = spike_hours
    return lookup


def _build_anomaly_lookup(
    week_index: int,
) -> Dict[int, List[dict]]:
    """
    Build a mapping from day-in-week (0-6) to a list of anomaly configs
    for the given week.

    Returns:
        Dict mapping day_in_week -> [anomaly_config, ...]
    """
    configs = WEEKLY_ANOMALIES.get(week_index, [])
    lookup: Dict[int, List[dict]] = {}
    for cfg in configs:
        d = cfg["day_in_week"]
        lookup.setdefault(d, []).append(cfg)
    return lookup


def generate_one_week(
    week_index: int,
    start_date: datetime,
    seed: int,
    inject_anomalies: bool = False,
    include_penny: bool = False,
    inject_penny_anomalies: bool = False,
) -> pd.DataFrame:
    """
    Generate 7 days of transaction data for all combos.

    Args:
        week_index:              0-based week number (used to offset dates)
        start_date:              The Monday that week 0 begins on
        seed:                    Random seed for this week
        inject_anomalies:        If True, apply anomaly configs from WEEKLY_ANOMALIES
        include_penny:           If True, impose penny amount distribution
        inject_penny_anomalies:  If True, apply penny spike configs from
                                 PENNY_WEEKLY_ANOMALIES

    Returns:
        DataFrame with columns: timestamp, network_type, transaction_type,
                                amount, is_anomaly
    """
    rng = np.random.default_rng(seed)

    day_offset_base = week_index * DAYS_PER_WEEK
    anomaly_lookup = _build_anomaly_lookup(week_index) if inject_anomalies else {}
    penny_anomaly_lookup = (
        _build_penny_anomaly_lookup(week_index)
        if inject_penny_anomalies else {}
    )

    all_records = []

    for combo in COMBO_KEYS:
        params = COMBO_FREQ_PARAMS[combo]
        multipliers = DOW_MULTIPLIERS[combo]
        network, txn_type = combo

        for day_in_week in range(DAYS_PER_WEEK):
            absolute_day = day_offset_base + day_in_week
            dow = absolute_day % 7
            dow_mult = multipliers[dow]

            day_params = generate_day_params(absolute_day, combo, rng)

            # Inject anomalies for this day (if any)
            anomaly_hours: List[int] = []
            if day_in_week in anomaly_lookup:
                for anom_cfg in anomaly_lookup[day_in_week]:
                    hours_affected = inject_anomaly(day_params, anom_cfg, combo=combo)
                    anomaly_hours.extend(hours_affected)
                # Deduplicate while preserving order
                seen = set()
                deduped = []
                for h in anomaly_hours:
                    if h not in seen:
                        seen.add(h)
                        deduped.append(h)
                anomaly_hours = deduped

            ts = generate_combo_timestamps_v2(
                absolute_day, params, dow_mult, day_params, rng
            )

            if len(ts) == 0:
                continue

            amounts = generate_amounts(len(ts), combo, rng)

            # Apply penny amount distribution if enabled
            if include_penny:
                penny_spike_hours = penny_anomaly_lookup.get(day_in_week)
                amounts = apply_penny_amounts(
                    ts, amounts, absolute_day, rng,
                    penny_spike_hours=penny_spike_hours,
                )

            day_start_seconds = absolute_day * 24 * 3600
            for i, t in enumerate(ts):
                hour = int((t - day_start_seconds) / 3600)
                is_anomaly = 1 if hour in anomaly_hours else 0

                all_records.append({
                    "timestamp_seconds": t,
                    "network_type": network,
                    "transaction_type": txn_type,
                    "amount": amounts[i],
                    "is_anomaly": is_anomaly,
                })

    df = pd.DataFrame(all_records)

    base_ts = pd.Timestamp(start_date)
    df["timestamp"] = base_ts + pd.to_timedelta(df["timestamp_seconds"], unit="s")
    df.drop(columns="timestamp_seconds", inplace=True)

    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = df[["timestamp", "network_type", "transaction_type", "amount", "is_anomaly"]]

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate 4 weekly CSV files for FCVAE Striim demo"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="data/demo_weeks",
        help="Directory to write the weekly CSV files"
    )
    parser.add_argument(
        "--num-weeks", "-n",
        type=int,
        default=NUM_WEEKS,
        help=f"Number of weekly files to generate (default: {NUM_WEEKS})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=BASE_SEED,
        help=f"Base random seed (default: {BASE_SEED})"
    )
    parser.add_argument(
        "--inject-anomalies",
        action="store_true",
        default=False,
        help="Inject anomalies in weeks 3-4 (spike, dip, ramp, compound)"
    )
    parser.add_argument(
        "--include-penny",
        action="store_true",
        default=False,
        help="Include penny transaction amount distribution (~2%% sub-dollar)"
    )
    parser.add_argument(
        "--inject-penny-anomalies",
        action="store_true",
        default=False,
        help="Inject penny-rate spike anomalies in weeks 3-4 (requires --include-penny)"
    )
    args = parser.parse_args()

    # --inject-penny-anomalies implies --include-penny
    if args.inject_penny_anomalies:
        args.include_penny = True

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("WEEKLY DEMO FILE GENERATOR")
    print("=" * 60)
    print(f"\n  Weeks:            {args.num_weeks}")
    print(f"  Start date:       {START_DATE.strftime('%Y-%m-%d')} (Monday)")
    print(f"  Output dir:       {args.output_dir}")
    print(f"  Base seed:        {args.seed}")
    print(f"  Inject anomalies: {args.inject_anomalies}")
    print(f"  Include penny:    {args.include_penny}")
    print(f"  Penny anomalies:  {args.inject_penny_anomalies}")

    if args.inject_anomalies:
        print("\n  Anomaly schedule:")
        for wk_idx in sorted(WEEKLY_ANOMALIES.keys()):
            if wk_idx >= args.num_weeks:
                continue
            wk_num = wk_idx + 1
            for cfg in WEEKLY_ANOMALIES[wk_idx]:
                day_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][cfg["day_in_week"]]
                print(f"    Week {wk_num}, {day_name}: {cfg['type']}"
                      f" (hours {cfg.get('hours', 'N/A')})")

    if args.inject_penny_anomalies:
        print("\n  Penny anomaly schedule:")
        for wk_idx in sorted(PENNY_WEEKLY_ANOMALIES.keys()):
            if wk_idx >= args.num_weeks:
                continue
            wk_num = wk_idx + 1
            for cfg in PENNY_WEEKLY_ANOMALIES[wk_idx]:
                day_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][cfg["day_in_week"]]
                print(f"    Week {wk_num}, {day_name}: {cfg['penny_spike_factor']}x penny spike"
                      f" (hours {cfg['hours'][0]}-{cfg['hours'][-1]})")

    total_anomaly_rows = 0

    for week in range(args.num_weeks):
        week_seed = args.seed + week * 1000
        df = generate_one_week(week, START_DATE, week_seed,
                               inject_anomalies=args.inject_anomalies,
                               include_penny=args.include_penny,
                               inject_penny_anomalies=args.inject_penny_anomalies)

        filename = f"week_{week + 1}.csv"
        filepath = os.path.join(args.output_dir, filename)
        df.to_csv(filepath, index=False)

        date_min = df["timestamp"].min().strftime("%Y-%m-%d")
        date_max = df["timestamp"].max().strftime("%Y-%m-%d")
        n_anomaly = int(df["is_anomaly"].sum())
        total_anomaly_rows += n_anomaly

        anomaly_label = f"  ({n_anomaly:,} anomaly rows)" if n_anomaly > 0 else ""
        print(f"\n  {filename}: {len(df):>10,} rows  |  {date_min} to {date_max}{anomaly_label}")

        combo_counts = df.groupby(["network_type", "transaction_type"]).size()
        for (net, txn), count in combo_counts.items():
            anom_count = int(
                df[(df["network_type"] == net) &
                   (df["transaction_type"] == txn) &
                   (df["is_anomaly"] == 1)].shape[0]
            )
            anom_str = f"  ({anom_count} anomaly)" if anom_count > 0 else ""
            print(f"    {net}/{txn}: {count:,}{anom_str}")

        if args.include_penny:
            n_penny = int((df["amount"] < 1.00).sum())
            print(f"    Penny txns: {n_penny:,} ({n_penny/len(df)*100:.2f}%)")

    print(f"\n{'=' * 60}")
    if total_anomaly_rows > 0:
        print(f"  Total anomaly transactions: {total_anomaly_rows:,}")
    print("DONE")
    print("=" * 60)
    print(f"\nDemo usage:")
    print(f"  1. Start Striim app with wildcard 'week_*.csv'")
    print(f"  2. cp {args.output_dir}/week_1.csv /tmp/fcvae_test/")
    print(f"  3. Wait for scoring + retrain trigger to fire")
    print(f"  4. cp {args.output_dir}/week_2.csv /tmp/fcvae_test/")
    print(f"  5. Repeat for week_3.csv and week_4.csv")


if __name__ == "__main__":
    main()
