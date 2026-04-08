"""
Penny Transaction Data Preprocessing

Procedural functions for loading, aggregating, windowing, and normalizing
penny transaction data for FCVAE anomaly detection.

Pipeline: load_penny_data() -> create_sliding_windows() -> create_splits()
          -> normalize() -> create_dataloaders()
"""
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

SAMPLES_PER_DAY = 24
WINDOW_SIZE = 24


class SlidingWindowDataset(Dataset):
    """PyTorch Dataset for FCVAE sliding windows.

    Each sample is a sliding window of hourly transaction counts,
    shaped as (1, window_size) for FCVAE input. No DoW features
    (FCVAE uses frequency conditioning instead).
    """

    def __init__(self, windows: np.ndarray, labels: np.ndarray):
        """
        Args:
            windows: Array of shape (num_windows, window_size)
            labels: Array of shape (num_windows, window_size) - anomaly labels
        """
        self.windows = torch.FloatTensor(windows).unsqueeze(1)  # (N, 1, W)
        self.labels = torch.FloatTensor(labels)  # (N, W)
        self.missing = torch.zeros_like(self.labels)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.labels[idx], self.missing[idx]


def load_penny_data(csv_path: Path) -> pd.DataFrame:
    """Load transaction CSV, filter to penny transactions, aggregate to hourly counts.

    Penny transactions are those with amount < $1.00, pooled across all
    network types and transaction types.

    Args:
        csv_path: Path to synthetic_transactions.csv

    Returns:
        DataFrame with columns: hour_bucket, count, is_anomaly, split (if available)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    logger.info(f"Loading transaction data from {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    logger.info(f"Loaded {len(df):,} transactions")

    # Bin timestamps to hour
    df["hour_bucket"] = df["timestamp"].dt.floor("h")

    # Ensure anomaly columns exist
    if "is_anomaly" not in df.columns:
        df["is_anomaly"] = 0

    has_split_col = "split" in df.columns

    # Create complete hourly index
    data_start = df["timestamp"].min().floor("h")
    data_end = df["timestamp"].max().floor("h")
    full_hours = pd.date_range(start=data_start, end=data_end, freq="h")
    logger.info(f"Date range: {data_start} to {data_end} ({len(full_hours)} hours, {len(full_hours) // 24} days)")

    # Filter to penny transactions
    total_before = len(df)
    df = df[df["amount"] < 1.00]
    logger.info(
        f"Penny filter: {len(df):,} / {total_before:,} transactions "
        f"({len(df) / total_before * 100:.2f}%)"
    )

    # Use penny-specific anomaly labels if available
    anomaly_col = "penny_is_anomaly" if "penny_is_anomaly" in df.columns else "is_anomaly"
    if anomaly_col == "penny_is_anomaly":
        logger.info("Using penny_is_anomaly column (penny-specific labels)")
    else:
        logger.info("Using is_anomaly column")

    # Aggregate by hour (pool all combos)
    agg_dict = {"timestamp": "count", anomaly_col: "max"}
    if has_split_col:
        agg_dict["split"] = "first"

    counts = df.groupby("hour_bucket").agg(agg_dict).reset_index()

    if has_split_col:
        counts.columns = ["hour_bucket", "count", "is_anomaly", "split"]
    else:
        counts.columns = ["hour_bucket", "count", "is_anomaly"]

    # Reindex to full hour range, fill missing with 0
    hourly_df = pd.DataFrame({"hour_bucket": full_hours})
    hourly_df = hourly_df.merge(counts, on="hour_bucket", how="left")
    hourly_df["count"] = hourly_df["count"].fillna(0).astype(int)
    hourly_df["is_anomaly"] = hourly_df["is_anomaly"].fillna(0).astype(int)

    if has_split_col:
        hourly_df["split"] = hourly_df["split"].ffill().bfill()

    total_txns = hourly_df["count"].sum()
    mean_hourly = hourly_df["count"].mean()
    anomaly_hours = hourly_df["is_anomaly"].sum()
    logger.info(
        f"Penny hourly: {len(hourly_df)} hours, {total_txns:,} total transactions, "
        f"{mean_hourly:.1f} mean/hour, {anomaly_hours} anomaly hours"
    )

    return hourly_df


def create_sliding_windows(
    hourly_df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create overlapping sliding windows from hourly count series.

    Args:
        hourly_df: DataFrame from load_penny_data() with count and is_anomaly columns
        window_size: Window size in hours (default 24)
        stride: Step size between windows (default 1 hour)

    Returns:
        windows: (num_windows, window_size) hourly counts
        labels: (num_windows, window_size) binary anomaly labels
        timestamps: (num_windows,) starting hour index for each window
    """
    values = hourly_df["count"].values.astype(np.float32)
    anomalies = hourly_df["is_anomaly"].values.astype(np.float32)
    total_hours = len(values)

    num_windows = (total_hours - window_size) // stride + 1

    if num_windows <= 0:
        logger.warning(f"Not enough data for sliding windows "
                       f"(need {window_size} hours, have {total_hours})")
        return np.array([]), np.array([]), np.array([])

    windows = np.zeros((num_windows, window_size), dtype=np.float32)
    labels = np.zeros((num_windows, window_size), dtype=np.float32)
    timestamps = np.zeros(num_windows, dtype=np.int32)

    for i in range(num_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        windows[i] = values[start_idx:end_idx]
        labels[i] = anomalies[start_idx:end_idx]
        timestamps[i] = start_idx

    anomaly_windows = (labels.sum(axis=1) > 0).sum()
    logger.info(
        f"Created {num_windows} sliding windows "
        f"(W={window_size}, stride={stride}, {anomaly_windows} with anomalies)"
    )

    return windows, labels, timestamps


def create_splits(
    windows: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    hourly_df: pd.DataFrame,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Split windows into train/val/test using CSV split column.

    If the CSV has a 'split' column, windows are assigned based on
    the split of their starting hour. Otherwise falls back to day-based splits.

    Args:
        windows: (N, W) array from create_sliding_windows
        labels: (N, W) array from create_sliding_windows
        timestamps: (N,) array of starting hour indices
        hourly_df: DataFrame from load_penny_data (must have 'split' column)

    Returns:
        Dict with keys: train, val, test. Each value is (windows, labels).
    """
    if len(windows) == 0:
        empty = (np.array([]), np.array([]))
        return {"train": empty, "val": empty, "test": empty}

    if "split" in hourly_df.columns:
        hourly_splits = hourly_df["split"].values

        window_splits = np.array([hourly_splits[ts] for ts in timestamps])

        train_mask = window_splits == "train"
        val_mask = window_splits == "val"
        test_mask = window_splits == "test"

        splits = {
            "train": (windows[train_mask], labels[train_mask]),
            "val": (windows[val_mask], labels[val_mask]),
            "test": (windows[test_mask], labels[test_mask]),
        }

        val_anomaly = (labels[val_mask].sum(axis=1) > 0).sum() if val_mask.sum() > 0 else 0
        test_anomaly = (labels[test_mask].sum(axis=1) > 0).sum() if test_mask.sum() > 0 else 0

        logger.info(
            f"CSV-based splits: "
            f"train={train_mask.sum()}, "
            f"val={val_mask.sum()} ({val_anomaly} anomaly windows), "
            f"test={test_mask.sum()} ({test_anomaly} anomaly windows)"
        )
    else:
        # Day-based fallback: 42 train / 8 val / 10 test (for 60-day dataset)
        hours_per_day = SAMPLES_PER_DAY
        train_end_hour = 42 * hours_per_day
        val_end_hour = 50 * hours_per_day

        train_mask = timestamps < train_end_hour
        val_mask = (timestamps >= train_end_hour) & (timestamps < val_end_hour)
        test_mask = timestamps >= val_end_hour

        splits = {
            "train": (windows[train_mask], labels[train_mask]),
            "val": (windows[val_mask], labels[val_mask]),
            "test": (windows[test_mask], labels[test_mask]),
        }

        logger.info(
            f"Day-based splits: "
            f"train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}"
        )

    return splits


def normalize(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    fit_on: str = "train"
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], StandardScaler]:
    """Normalize sliding windows using StandardScaler.

    Scaler is fit only on the specified split to prevent data leakage.

    Args:
        splits: Dict from create_splits()
        fit_on: Which split to fit the scaler on (default: "train")

    Returns:
        Tuple of (normalized_splits, scaler)
    """
    if fit_on not in splits or len(splits[fit_on][0]) == 0:
        raise ValueError(f"Cannot fit scaler: '{fit_on}' split is empty")

    train_windows = splits[fit_on][0]
    train_flat = train_windows.flatten().reshape(-1, 1)

    scaler = StandardScaler()
    scaler.fit(train_flat)

    logger.info(f"Scaler: mean={scaler.mean_[0]:.2f}, std={scaler.scale_[0]:.2f}")

    normalized_splits = {}
    for name, (windows, labels) in splits.items():
        if len(windows) == 0:
            normalized_splits[name] = (windows, labels)
            continue

        flat = windows.flatten().reshape(-1, 1)
        normalized = scaler.transform(flat)
        normalized_splits[name] = (normalized.reshape(windows.shape), labels)

    return normalized_splits, scaler


def create_dataloaders(
    normalized_splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    batch_size: int = 64,
    shuffle_train: bool = True
) -> Dict[str, Optional[DataLoader]]:
    """Create PyTorch DataLoaders for FCVAE sliding windows.

    Args:
        normalized_splits: Dict from normalize()
        batch_size: Batch size
        shuffle_train: Whether to shuffle training data

    Returns:
        Dict with DataLoader for each split (None for empty splits)
    """
    dataloaders = {}

    for name, (windows, labels) in normalized_splits.items():
        if len(windows) == 0:
            dataloaders[name] = None
            continue

        dataset = SlidingWindowDataset(windows, labels)

        loader = DataLoader(
            dataset,
            batch_size=min(batch_size, len(windows)),
            shuffle=(shuffle_train and name == "train"),
            drop_last=False,
        )

        dataloaders[name] = loader

    return dataloaders
