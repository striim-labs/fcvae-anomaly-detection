#!/usr/bin/env python3
"""
End-to-end demo of the FCVAE anomaly detection continual learning cycle.

Demonstrates the full pipeline in a single local run:
  Stage 1: Train a model on historical data (train + val splits)
  Stage 2: Derive oracle threshold from test split, seed data store
  Iterations 1-4: Generate synthetic week, score, retrain, hot-swap

No FastAPI server required — directly imports app/ and api/ components.

Usage:
    python scripts/demo_e2e.py                           # full demo (~10 min)
    python scripts/demo_e2e.py --pretrained --delay 0.05  # fast demo (~3 min)
    python scripts/demo_e2e.py --combo Star_CMP            # different combo
"""

import argparse
import logging
import pickle
import shutil
import sys
import tempfile
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Suppress sklearn version warnings (models may be saved with a different version)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Suppress all library loggers before any imports trigger them.
logging.basicConfig(level=logging.WARNING, format="%(message)s")
_root = logging.getLogger()
for _h in _root.handlers:
    _h.setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = str(PROJECT_ROOT / "app")
DATA_DIR = str(PROJECT_ROOT / "data")
for _p in [APP_DIR, str(PROJECT_ROOT), DATA_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fcvae_model import FCVAE, FCVAEConfig  # noqa: E402
from fcvae_scorer import FCVAEScorer  # noqa: E402
from fcvae_augment import AugmentConfig, batch_augment  # noqa: E402
from transaction_preprocessor import SlidingWindowDataset  # noqa: E402
from fcvae_streaming_detector import FCVAEStreamingDetector  # noqa: E402

from api.data_store import WindowDataStore  # noqa: E402
from api.retrain import RetrainConfig, RetrainPipeline, RetrainResult  # noqa: E402

from generate_transactions import (  # noqa: E402
    COMBO_FREQ_PARAMS,
    DOW_MULTIPLIERS,
    START_DATE as GEN_START_DATE,
    SEED as GEN_SEED,
    generate_day_params,
    rate_function_v2,
    inject_anomaly,
)

# ── ANSI colors ───────────────────────────────────────────────────────────────

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


# ── Constants ─────────────────────────────────────────────────────────────────

COMBO_MAP = {
    "Accel_CMP": ("Accel", "CMP"),
    "Accel_nopin": ("Accel", "no-pin"),
    "Star_CMP": ("Star", "CMP"),
    "Star_nopin": ("Star", "no-pin"),
}

# Anomaly configs for each weekly retrain iteration.
# Each list entry is a week; each dict within specifies one injected anomaly.
# Anomalies are placed at high-traffic hours (8-12) for reliable detection.
WEEKLY_ANOMALIES = [
    # Week 1: 3-hour spike on day 3 — volume surge
    [{"day_offset": 2, "type": "spike", "hours": [10, 11, 12], "multiplier": 3.5}],
    # Week 2: 3-hour near-outage on day 4 — traffic drops to ~2%
    [{"day_offset": 3, "type": "dip", "hours": [10, 11, 12], "multiplier": 0.02}],
    # Week 3: 4-hour gradual ramp on day 2 — building from 2x to 4x
    [{"day_offset": 1, "type": "ramp", "hours": [9, 10, 11, 12],
      "start_mult": 2.0, "end_mult": 4.0}],
    # Week 4: spike on day 3 + dip on day 6
    [
        {"day_offset": 2, "type": "spike", "hours": [10, 11, 12], "multiplier": 3.0},
        {"day_offset": 5, "type": "dip", "hours": [10, 11, 12], "multiplier": 0.02},
    ],
]

ANOMALY_DESCRIPTIONS = [
    "spike (\u00d73.5) at hours 10\u201312 on day 3",
    "near-outage (\u00d70.02) at hours 10\u201312 on day 4",
    "gradual ramp (2\u00d7\u21924\u00d7) at hours 9\u201312 on day 2",
    "spike (\u00d73.0) hours 10\u201312 day 3 + outage (\u00d70.02) hours 10\u201312 day 6",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def banner(text: str) -> None:
    width = 60
    print()
    print(f"{BOLD}{'=' * width}{RESET}")
    print(f"{BOLD}  {text}{RESET}")
    print(f"{BOLD}{'=' * width}{RESET}")


def stage_header(title: str) -> None:
    print()
    sep = "\u2500" * 55
    print(f"{BOLD}{sep}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{sep}{RESET}")
    print()


def scoring_table_header() -> None:
    hdr = (
        f"{'Win':>5}  {'Timestamp':>19}  {'Value':>7}  "
        f"{'Score':>9}  {'Thresh':>9}  {'Anomaly?':>8}  {'Truth':>8}"
    )
    print(f"  {BOLD}{hdr}{RESET}")
    sep = "\u2500" * len(hdr)
    print(f"  {sep}")


def scoring_table_row(
    win_idx: int,
    timestamp: str,
    value: float,
    score: float,
    threshold: float,
    is_anomaly: bool,
    ground_truth: bool,
) -> None:
    anom_str = f"{RED}{BOLD}  YES{RESET}" if is_anomaly else f"{GREEN}   no{RESET}"
    gt_str = f"{YELLOW} ANOMALY{RESET}" if ground_truth else f"{DIM}  normal{RESET}"
    print(
        f"  {win_idx:>5}  {timestamp:>19}  {value:>7.0f}  "
        f"{score:>9.4f}  {threshold:>9.4f}  {anom_str}  {gt_str}"
    )


def load_and_aggregate(
    file_path: str, combo_key: str, split: str,
) -> pd.DataFrame:
    """Load CSV, filter by combo + split, aggregate to hourly counts.

    Returns DataFrame with columns: timestamp, value, is_anomaly.
    """
    combo_tuple = COMBO_MAP[combo_key]
    network, txn_type = combo_tuple

    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    if "split" in df.columns:
        df = df[df["split"] == split]
    df = df[
        (df["network_type"] == network) & (df["transaction_type"] == txn_type)
    ]

    if df.empty:
        print(f"No data for combo={combo_key} split={split}", file=sys.stderr)
        sys.exit(1)

    df["hour_bucket"] = df["timestamp"].dt.floor("h")
    hourly = (
        df.groupby("hour_bucket")
        .agg(value=("timestamp", "size"), is_anomaly=("is_anomaly", "max"))
        .reset_index()
    )

    # Fill gaps with 0
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


# ── Synthetic Data Generation ─────────────────────────────────────────────────

def generate_week_hourly(
    combo_key: str,
    week_start_date: datetime,
    day_index_offset: int,
    anomaly_configs: List[Dict],
    seed: int,
) -> pd.DataFrame:
    """Generate one week of hourly transaction counts for a combo.

    Uses the same sinusoidal patterns, day-of-week multipliers, and
    variability from generate_transactions.py. Counts are drawn from
    Poisson(rate) where rate comes from the combo's rate function.
    """
    combo_tuple = COMBO_MAP[combo_key]
    params = COMBO_FREQ_PARAMS[combo_tuple]
    multipliers = DOW_MULTIPLIERS[combo_tuple]
    rng = np.random.default_rng(seed)

    records: List[Dict] = []
    for day_offset in range(7):
        abs_day = day_index_offset + day_offset
        dow = abs_day % 7
        dow_mult = multipliers[dow]
        day_params = generate_day_params(abs_day, combo_tuple, rng)

        # Inject anomalies for this day (may modify day_params in-place)
        anomaly_hours: List[int] = []
        for ac in anomaly_configs:
            if ac["day_offset"] == day_offset:
                anomaly_hours.extend(inject_anomaly(day_params, ac, combo=combo_tuple))

        # Generate hourly counts using the rate function
        for hour in range(24):
            rate = rate_function_v2(
                np.array([hour + 0.5]), params, dow_mult, day_params
            )[0]
            count = int(rng.poisson(max(rate, 0)))
            ts = pd.Timestamp(week_start_date) + timedelta(days=day_offset, hours=hour)
            is_anomaly = 1 if hour in anomaly_hours else 0
            records.append({
                "timestamp": ts,
                "value": count,
                "is_anomaly": is_anomaly,
            })

    return pd.DataFrame(records)


# ── Stage 1: Initial Training ────────────────────────────────────────────────

def stage1_train(
    data_path: str,
    combo_key: str,
    workspace: Path,
    epochs: int,
) -> Path:
    """Train an FCVAE model from scratch on train+val splits.

    No threshold calibration here — oracle threshold is derived in Stage 2.
    Returns the path to the model directory within the workspace.
    """
    stage_header("STAGE 1: Initial Model Training")

    model_dir = workspace / "models" / combo_key
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and aggregate train + val data ─────────────────────────
    print("Loading and aggregating data...")
    train_hourly = load_and_aggregate(data_path, combo_key, "train")
    val_hourly = load_and_aggregate(data_path, combo_key, "val")

    print(
        f"  Train: {len(train_hourly)} hourly records "
        f"({train_hourly['timestamp'].iloc[0].strftime('%b %d')} \u2013 "
        f"{train_hourly['timestamp'].iloc[-1].strftime('%b %d, %Y')})"
    )
    print(
        f"  Val:   {len(val_hourly)} hourly records "
        f"({val_hourly['timestamp'].iloc[0].strftime('%b %d')} \u2013 "
        f"{val_hourly['timestamp'].iloc[-1].strftime('%b %d, %Y')})"
    )
    print()

    # ── Create sliding windows ──────────────────────────────────────
    window_size = 24

    def make_windows(hourly: pd.DataFrame):
        values = hourly["value"].values.astype(np.float32)
        n = len(values) - window_size + 1
        windows = np.array(
            [values[i : i + window_size] for i in range(n)], dtype=np.float32
        )
        return windows

    train_windows = make_windows(train_hourly)
    val_windows = make_windows(val_hourly)

    print(f"Creating sliding windows (size={window_size}, stride=1)...")
    print(f"  Train: {len(train_windows)} | Val: {len(val_windows)}")
    print()

    # ── Fit StandardScaler on training data ─────────────────────────
    scaler = StandardScaler()
    scaler.fit(train_windows.reshape(-1, 1))

    print(f"StandardScaler: mean={scaler.mean_[0]:.1f}, std={scaler.scale_[0]:.1f}")
    print()

    # Normalize
    train_norm = (
        scaler.transform(train_windows.reshape(-1, 1))
        .reshape(train_windows.shape)
        .astype(np.float32)
    )
    val_norm = (
        scaler.transform(val_windows.reshape(-1, 1))
        .reshape(val_windows.shape)
        .astype(np.float32)
    )

    # ── DataLoaders ─────────────────────────────────────────────────
    train_dataset = SlidingWindowDataset(train_norm, np.zeros_like(train_norm))
    val_dataset = SlidingWindowDataset(val_norm, np.zeros_like(val_norm))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # ── Train model ─────────────────────────────────────────────────
    model_config = FCVAEConfig(window=window_size, latent_dim=8)
    model = FCVAE(config=model_config)
    device = torch.device("cpu")
    model.to(device)

    augment_config = AugmentConfig()
    lr = 5e-4
    kl_warmup = 10

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0
    patience = 5

    print(f"Training FCVAE ({epochs} epochs, lr={lr:.0e})...")

    for epoch in range(epochs):
        kl_weight = min(1.0, (epoch + 1) / kl_warmup) if kl_warmup > 0 else 1.0

        # Train
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            x, y, z = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            x, y, z = batch_augment(x, y, z, augment_config)
            optimizer.zero_grad()
            _, _, _, _, _, loss = model(x, mode="train", kl_weight=kl_weight)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 2.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(n_batches, 1)

        # Validate
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                _, _, _, _, _, loss = model(x, mode="valid")
                val_loss += loss.item()
                n_val += 1
        val_loss /= max(n_val, 1)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        marker = ""
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_counter = 0
            marker = f"  {YELLOW}\u2605 best{RESET}"
        else:
            patience_counter += 1

        print(
            f"  Epoch {epoch + 1:>3}/{epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"\u03b2={kl_weight:.2f}  lr={current_lr:.1e}{marker}"
        )

        if patience_counter >= patience:
            print(f"  {DIM}Early stopping at epoch {epoch + 1} (patience={patience}){RESET}")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    model.eval()

    print()
    print(f"  Best: epoch {best_epoch}, val_loss={best_val_loss:.4f}")
    print()

    # ── Save artifacts (placeholder scorer — real one comes from Stage 2) ──
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.config,
            "model_version": 1,
        },
        model_dir / "model.pt",
    )
    with open(model_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    scorer = FCVAEScorer()
    scorer.is_fitted = True
    scorer.point_threshold = -999.0
    scorer.last_point_threshold = -999.0
    scorer.save(str(model_dir / "scorer.pkl"))

    print(f"{GREEN}\u2713 Model trained and saved to {model_dir.relative_to(workspace)}/{RESET}")
    return model_dir


def stage1_pretrained(combo_key: str, workspace: Path) -> Path:
    """Copy pre-trained models to workspace instead of training."""
    stage_header("STAGE 1: Loading Pre-trained Model (skipping training)")

    src = PROJECT_ROOT / "models" / "fcvae" / combo_key
    dst = workspace / "models" / combo_key
    dst.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        print(f"{RED}Pre-trained model not found at {src}{RESET}")
        sys.exit(1)

    for artifact in ["model.pt", "scaler.pkl", "scorer.pkl"]:
        src_file = src / artifact
        if src_file.exists():
            shutil.copy2(src_file, dst / artifact)
            print(f"  Copied {artifact}")
        else:
            print(f"  {YELLOW}Warning: {artifact} not found{RESET}")

    print()
    print(f"{GREEN}\u2713 Pre-trained model loaded to {dst.relative_to(workspace)}/{RESET}")
    return dst


# ── Stage 2: Oracle Threshold Calibration ──────────────────────────────────────

def stage2_oracle(
    data_path: str,
    combo_key: str,
    workspace: Path,
    data_store: WindowDataStore,
) -> float:
    """Score test data, derive oracle threshold, seed data store.

    Returns the oracle (last-point) threshold.
    """
    stage_header("STAGE 2: Oracle Threshold Calibration (test split)")

    model_dir = workspace / "models" / combo_key
    device = torch.device("cpu")

    # ── Load test data ──────────────────────────────────────────────
    test_hourly = load_and_aggregate(data_path, combo_key, "test")
    total_hours = len(test_hourly)
    window_size = 24
    n_windows = total_hours - window_size + 1
    anomaly_hours = int(test_hourly["is_anomaly"].sum())

    print(f"Test split: {total_hours} hourly records \u2192 {n_windows} windows")
    print(f"Ground-truth anomaly hours: {anomaly_hours}")
    print()

    # ── Load model and scaler ───────────────────────────────────────
    checkpoint = torch.load(
        model_dir / "model.pt", map_location=device, weights_only=False,
    )
    model = FCVAE(config=checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    with open(model_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # ── Create sliding windows ──────────────────────────────────────
    values = test_hourly["value"].values.astype(np.float32)
    anomalies = test_hourly["is_anomaly"].values.astype(np.float32)

    windows = np.array(
        [values[i : i + window_size] for i in range(n_windows)], dtype=np.float32
    )
    labels = np.array(
        [anomalies[i : i + window_size] for i in range(n_windows)], dtype=np.float32
    )

    # ── Normalize and score ─────────────────────────────────────────
    windows_norm = (
        scaler.transform(windows.reshape(-1, 1))
        .reshape(windows.shape)
        .astype(np.float32)
    )
    tensor = torch.FloatTensor(windows_norm).unsqueeze(1).to(device)
    with torch.no_grad():
        scores = model.score_single_pass(tensor).cpu().numpy()  # (N, 24)

    # Last-point scores and labels
    lp_scores = scores[:, -1]
    lp_labels = labels[:, -1].astype(bool)

    normal_lp = lp_scores[~lp_labels]
    anomaly_lp = lp_scores[lp_labels]

    print("Scoring test windows...")
    print(
        f"  Normal LP scores:  n={len(normal_lp)}, "
        f"mean={np.mean(normal_lp):.4f}, std={np.std(normal_lp):.4f}"
    )
    if len(anomaly_lp) > 0:
        print(
            f"  Anomaly LP scores: n={len(anomaly_lp)}, "
            f"mean={np.mean(anomaly_lp):.4f}, std={np.std(anomaly_lp):.4f}"
        )
    print()

    # ── Compute oracle threshold ────────────────────────────────────
    scorer = FCVAEScorer()
    if len(anomaly_lp) > 0:
        lp_threshold, metrics = FCVAEScorer.find_optimal_threshold(
            normal_lp, anomaly_lp, method="f1_max",
        )
        scorer.last_point_threshold = lp_threshold
        scorer.point_threshold = lp_threshold
        f1 = metrics.get("f1", 0)
        precision = metrics.get("precision", 0)
        recall = metrics.get("recall", 0)
        print(f"Oracle threshold (F1-max): {BOLD}{lp_threshold:.4f}{RESET}")
        print(f"  P={precision:.2f}  R={recall:.2f}  F1={f1:.2f}")
    else:
        fallback = float(np.percentile(normal_lp, 5))
        scorer.last_point_threshold = fallback
        scorer.point_threshold = fallback
        print(f"Percentile fallback threshold: {fallback:.4f}")

    scorer.is_fitted = True
    scorer.save(str(model_dir / "scorer.pkl"))

    # ── Seed data store with test windows (ground-truth labels) ─────
    print()
    print(f"Seeding data store with {n_windows} test windows...")
    timestamps = test_hourly["timestamp"].tolist()
    for i in range(n_windows):
        w_values = values[i : i + window_size].tolist()
        w_end = timestamps[i + window_size - 1]
        w_end_str = (
            w_end.isoformat() if hasattr(w_end, "isoformat") else str(w_end)
        )
        ground_truth = bool(lp_labels[i])
        data_store.append(
            combo=combo_key,
            raw_values=w_values,
            window_end=w_end_str,
            last_point_score=float(lp_scores[i]),
            is_anomaly=ground_truth,
        )

    store_stats = data_store.get_stats()
    combo_store = store_stats.get("combos", {}).get(combo_key, {})
    print(
        f"  {CYAN}[Data Store]{RESET} "
        f"{combo_store.get('total', 0)} windows "
        f"({combo_store.get('normal', 0)} normal, "
        f"{combo_store.get('anomalous', 0)} anomalous)"
    )

    print()
    print(f"{GREEN}\u2713 Oracle threshold saved{RESET}")

    return scorer.last_point_threshold


# ── Scoring Loop ──────────────────────────────────────────────────────────────

def score_windows(
    detector: FCVAEStreamingDetector,
    data_store: WindowDataStore,
    combo_key: str,
    hourly_data: pd.DataFrame,
    delay: float,
    window_size: int = 24,
) -> Dict:
    """Score all windows in hourly data, accumulate to data store.

    Ground-truth labels (from hourly_data['is_anomaly']) are stored
    in the data store so the retrain pipeline can compute oracle thresholds.

    Returns stats dict with counts and confusion matrix components.
    """
    n_windows = len(hourly_data) - window_size + 1
    if n_windows <= 0:
        return {
            "total": 0, "detected": 0, "ground_truth": 0,
            "tp": 0, "fp": 0, "fn": 0, "tn": 0,
        }

    scoring_table_header()

    tp = fp = fn = tn = 0
    total_detected = 0
    total_gt = 0

    for i in range(n_windows):
        window = hourly_data.iloc[i : i + window_size]
        timestamps = window["timestamp"].tolist()
        values = window["value"].tolist()
        ground_truth = bool(window["is_anomaly"].iloc[-1])

        # Build DataFrame for detector
        score_df = pd.DataFrame({
            "timestamp": timestamps,
            "value": [float(v) for v in values],
        })

        result = detector.score_window_detailed(score_df)
        if result is None:
            continue

        is_anomaly = result["is_anomaly"]
        last_point_score = result["last_point_score"]
        threshold = result["point_threshold"]

        # Accumulate to data store with ground-truth label
        scored_ts = (
            timestamps[-1].isoformat()
            if hasattr(timestamps[-1], "isoformat")
            else str(timestamps[-1])
        )
        try:
            data_store.append(
                combo=combo_key,
                raw_values=values,
                window_end=scored_ts,
                last_point_score=last_point_score,
                is_anomaly=ground_truth,
            )
        except Exception:
            pass

        # Track metrics
        if is_anomaly:
            total_detected += 1
        if ground_truth:
            total_gt += 1

        if is_anomaly and ground_truth:
            tp += 1
        elif is_anomaly and not ground_truth:
            fp += 1
        elif not is_anomaly and ground_truth:
            fn += 1
        else:
            tn += 1

        ts_str = str(timestamps[-1])[:19]
        scoring_table_row(
            i + 1, ts_str, values[-1],
            last_point_score, threshold, is_anomaly, ground_truth,
        )

        if delay > 0:
            time.sleep(delay)

    return {
        "total": n_windows,
        "detected": total_detected,
        "ground_truth": total_gt,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ── Retrain + Hot-Swap ────────────────────────────────────────────────────────

def retrain_and_swap(
    data_store: WindowDataStore,
    combo_key: str,
    workspace: Path,
    old_detector: FCVAEStreamingDetector,
    iteration: int,
) -> Tuple[FCVAEStreamingDetector, Optional[RetrainResult]]:
    """Run the retrain pipeline on accumulated data and hot-swap if successful.

    Returns (new_or_old_detector, retrain_result).
    """
    production_dir = str(workspace / "models")
    staging_dir = str(workspace / "staging")

    config = RetrainConfig(
        staging_dir=staging_dir,
        production_dir=production_dir,
        min_f1_ratio=0.7,
    )

    pipeline = RetrainPipeline(
        data_store=data_store,
        config=config,
        device="cpu",
    )

    # Show accumulation stats
    stats = data_store.get_stats()
    combo_stats = stats.get("combos", {}).get(combo_key, {})
    normal_count = combo_stats.get("normal", 0)
    anomaly_count = combo_stats.get("anomalous", 0)

    print(f"  Triggering retrain...")
    print(f"    {normal_count} normal + {anomaly_count} anomalous windows in data store")

    # Configure verbose logging for the retrain pipeline
    retrain_logger = logging.getLogger("api.retrain")
    old_level = retrain_logger.level
    old_propagate = retrain_logger.propagate
    retrain_logger.setLevel(logging.INFO)
    retrain_logger.propagate = False

    class DemoHandler(logging.Handler):
        def emit(self, record):
            print(f"    {DIM}{self.format(record)}{RESET}")

    handler = DemoHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    retrain_logger.addHandler(handler)

    try:
        result = pipeline.retrain_combo(
            combo=combo_key,
            lookback_days=9999,
            min_windows=30,
        )
    finally:
        retrain_logger.removeHandler(handler)
        retrain_logger.setLevel(old_level)
        retrain_logger.propagate = old_propagate

    print()

    if result.status == "success":
        print(f"  {GREEN}{BOLD}\u2713 Retrain SUCCESS{RESET} \u2014 {result.duration_seconds:.1f}s")

        # Show validation metrics
        if result.old_mean_nll is not None and result.new_mean_nll is not None:
            nll_diff = result.new_mean_nll - result.old_mean_nll
            print(
                f"    NLL: old={result.old_mean_nll:.4f} new={result.new_mean_nll:.4f} "
                f"\u0394={nll_diff:+.4f} {GREEN}\u2713{RESET}"
            )
        if result.old_f1 is not None and result.new_f1 is not None:
            f1_ratio = result.new_f1 / result.old_f1 if result.old_f1 > 0 else float("inf")
            print(
                f"    F1:  old={result.old_f1:.4f} new={result.new_f1:.4f} "
                f"\u00d7{f1_ratio:.2f} {GREEN}\u2713{RESET}"
            )
        if result.old_threshold is not None and result.new_threshold is not None:
            print(
                f"    Threshold: {result.old_threshold:.4f} \u2192 {result.new_threshold:.4f}"
            )
        print()

        # Hot-swap: promote staged artifacts and create new detector
        staging_path = Path(staging_dir) / combo_key
        production_path = workspace / "models" / combo_key
        shutil.copytree(str(staging_path), str(production_path), dirs_exist_ok=True)

        combo_tuple = COMBO_MAP[combo_key]
        new_detector = FCVAEStreamingDetector(
            model_path=str(workspace / "models"),
            combo=combo_tuple,
            window_size=24,
            min_samples=24,
            device="cpu",
            n_samples=16,
            decision_mode="last_point",
        )

        if new_detector.is_ready:
            print(f"  {GREEN}\u2713 Hot-swap \u2192 v{iteration + 1}{RESET}")
            return new_detector, result
        else:
            print(f"  {RED}\u2717 New detector failed to load, keeping old{RESET}")
            return old_detector, result

    elif result.status == "rejected":
        print(f"  {YELLOW}{BOLD}\u2717 Retrain REJECTED{RESET}: {result.error_message}")
        return old_detector, result
    elif result.status == "insufficient_data":
        print(
            f"  {YELLOW}\u2717 Insufficient data{RESET} "
            f"({result.num_train_windows} windows)"
        )
        return old_detector, result
    else:
        print(f"  {RED}\u2717 Error{RESET}: {result.error_message}")
        return old_detector, result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end demo of FCVAE continual learning cycle"
    )
    parser.add_argument(
        "--combo", default="Accel_CMP",
        choices=list(COMBO_MAP.keys()),
        help="Combo key to demo (default: Accel_CMP)",
    )
    parser.add_argument(
        "--data-path",
        default=str(PROJECT_ROOT / "data" / "synthetic_transactions.csv"),
        help="Path to synthetic transactions CSV",
    )
    parser.add_argument(
        "--delay", type=float, default=0.1,
        help="Seconds between scoring windows (default: 0.1)",
    )
    parser.add_argument(
        "--epochs", type=int, default=15,
        help="Initial training epochs (default: 15)",
    )
    parser.add_argument(
        "--iterations", type=int, default=4,
        help="Number of weekly retrain iterations (default: 4, max: 4)",
    )
    parser.add_argument(
        "--pretrained", action="store_true",
        help="Skip training, use pre-trained models from models/fcvae/",
    )
    parser.add_argument(
        "--workspace", type=str, default=None,
        help="Demo workspace directory (default: auto temp dir)",
    )
    parser.add_argument(
        "--keep-workspace", action="store_true",
        help="Don't clean up workspace on exit",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable ANSI colors",
    )
    args = parser.parse_args()

    # Disable colors if requested
    if args.no_color:
        global RED, GREEN, YELLOW, CYAN, BOLD, DIM, RESET
        RED = GREEN = YELLOW = CYAN = BOLD = DIM = RESET = ""

    # Create workspace
    if args.workspace:
        workspace = Path(args.workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        workspace = Path(tempfile.mkdtemp(prefix="fcvae_demo_"))
        cleanup = not args.keep_workspace

    try:
        _run_demo(args, workspace)
    finally:
        if cleanup:
            shutil.rmtree(workspace, ignore_errors=True)
            print(f"\n{DIM}Workspace cleaned up.{RESET}")
        else:
            print(f"\n{DIM}Workspace retained at: {workspace}{RESET}")


def _run_demo(args, workspace: Path):
    combo_key = args.combo
    n_iterations = min(args.iterations, len(WEEKLY_ANOMALIES))

    banner("FCVAE Anomaly Detection \u2014 End-to-End Demo")
    print(f"  Combo:      {combo_key}")
    print(f"  Workspace:  {workspace}")
    if args.pretrained:
        print(f"  Mode:       Pre-trained (skipping initial training)")
    else:
        print(f"  Mode:       Training from scratch ({args.epochs} epochs)")
    print(f"  Iterations: {n_iterations} weekly retrain cycles")

    # ── STAGE 1: Train or load model ────────────────────────────────
    if args.pretrained:
        model_dir = stage1_pretrained(combo_key, workspace)
    else:
        model_dir = stage1_train(args.data_path, combo_key, workspace, args.epochs)

    # ── Data store ──────────────────────────────────────────────────
    data_store = WindowDataStore(data_dir=str(workspace / "data"))

    # ── STAGE 2: Oracle threshold calibration ───────────────────────
    oracle_thresh = stage2_oracle(
        args.data_path, combo_key, workspace, data_store,
    )

    # ── Create detector with oracle threshold ───────────────────────
    combo_tuple = COMBO_MAP[combo_key]
    detector = FCVAEStreamingDetector(
        model_path=str(workspace / "models"),
        combo=combo_tuple,
        window_size=24,
        min_samples=24,
        device="cpu",
        n_samples=16,
        decision_mode="last_point",
    )

    if not detector.is_ready:
        print(f"{RED}Failed to load detector! Check model artifacts.{RESET}")
        return

    # ── Iteration tracking ──────────────────────────────────────────
    iteration_results: List[Dict] = []

    # The existing CSV covers 60 days starting from GEN_START_DATE.
    # New weeks start from day 60 onwards.
    existing_days = 60

    # ── ITERATIONS ──────────────────────────────────────────────────
    for it in range(n_iterations):
        week_day_offset = existing_days + it * 7
        week_start = GEN_START_DATE + timedelta(days=week_day_offset)
        week_end = week_start + timedelta(days=6)

        stage_header(
            f"ITERATION {it + 1}/{n_iterations}: "
            f"Week of {week_start.strftime('%b %d')} \u2013 "
            f"{week_end.strftime('%b %d, %Y')}"
        )

        # ── Generate synthetic data ─────────────────────────────────
        seed = GEN_SEED + 1000 + it * 100
        anomaly_configs = WEEKLY_ANOMALIES[it]

        print(f"Generating synthetic data (seed={seed})...")
        week_hourly = generate_week_hourly(
            combo_key, week_start, week_day_offset, anomaly_configs, seed,
        )

        week_anomaly_hours = int(week_hourly["is_anomaly"].sum())
        n_windows = len(week_hourly) - 24 + 1

        print(f"  {len(week_hourly)} hourly records | {week_anomaly_hours} anomaly hours")
        print(f"  Anomaly: {ANOMALY_DESCRIPTIONS[it]}")
        print()

        # ── Score windows ───────────────────────────────────────────
        print(f"Scoring {n_windows} windows...")
        print()

        stats = score_windows(
            detector, data_store, combo_key, week_hourly,
            delay=args.delay,
        )

        # Accumulation summary
        store_stats = data_store.get_stats()
        combo_store = store_stats.get("combos", {}).get(combo_key, {})
        print()
        print(
            f"  {CYAN}[Accumulation]{RESET} "
            f"{stats['total']} windows scored | "
            f"{stats['detected']} detected | "
            f"{stats['ground_truth']} ground truth"
        )
        print(
            f"  {CYAN}[Data Store]{RESET} "
            f"{combo_store.get('total', 0)} total windows "
            f"({combo_store.get('normal', 0)} normal, "
            f"{combo_store.get('anomalous', 0)} anomalous)"
        )
        print()

        # ── Retrain + hot-swap ──────────────────────────────────────
        detector, retrain_result = retrain_and_swap(
            data_store, combo_key, workspace, detector, it + 1,
        )

        # ── Track results ───────────────────────────────────────────
        p = (
            stats["tp"] / (stats["tp"] + stats["fp"])
            if (stats["tp"] + stats["fp"]) > 0 else 0
        )
        r = (
            stats["tp"] / (stats["tp"] + stats["fn"])
            if (stats["tp"] + stats["fn"]) > 0 else 0
        )
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        iteration_results.append({
            "week": it + 1,
            "start": week_start.strftime("%b %d"),
            "windows": stats["total"],
            "detected": stats["detected"],
            "ground_truth": stats["ground_truth"],
            "precision": p,
            "recall": r,
            "f1": f1,
            "threshold": (
                retrain_result.new_threshold
                if retrain_result and retrain_result.status == "success"
                else None
            ),
            "retrain_status": retrain_result.status if retrain_result else "N/A",
            "retrain_duration": (
                retrain_result.duration_seconds if retrain_result else 0
            ),
        })

    # ── Summary ─────────────────────────────────────────────────────
    banner("Demo Summary")

    print(f"  Combo: {combo_key}")
    print(f"  Initial oracle threshold: {oracle_thresh:.4f}")
    print()

    # Summary table
    hdr = (
        f"  {'Iter':>4}  {'Week':>10}  {'Windows':>7}  "
        f"{'Detect':>6}  {'Truth':>5}  {'F1':>6}  "
        f"{'Threshold':>10}  {'Status':>8}"
    )
    print(hdr)
    sep = "\u2500" * (len(hdr) - 2)
    print(f"  {sep}")

    for r in iteration_results:
        thresh_str = f"{r['threshold']:.4f}" if r["threshold"] is not None else "       \u2013"
        status_color = GREEN if r["retrain_status"] == "success" else YELLOW
        print(
            f"  {r['week']:>4}  {r['start']:>10}  {r['windows']:>7}  "
            f"{r['detected']:>6}  {r['ground_truth']:>5}  {r['f1']:>6.2f}  "
            f"{thresh_str:>10}  {status_color}{r['retrain_status']:>8}{RESET}"
        )

    n_success = sum(1 for r in iteration_results if r["retrain_status"] == "success")
    n_rejected = sum(1 for r in iteration_results if r["retrain_status"] == "rejected")
    total_duration = sum(r["retrain_duration"] for r in iteration_results)

    print()
    print(
        f"  Retrains: {n_success} success, {n_rejected} rejected "
        f"out of {n_iterations}"
    )
    print(f"  Total retrain time: {total_duration:.1f}s")
    print(f"  Model versions: 1 \u2192 {1 + n_success}")


if __name__ == "__main__":
    main()
