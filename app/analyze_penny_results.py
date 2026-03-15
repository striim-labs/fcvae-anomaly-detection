"""
Penny Transaction Anomaly Detection — Diagnostic Analysis

Loads the trained Penny_All FCVAE model and test data, then produces
per-severity breakdowns, score distributions, threshold sensitivity
curves, temporal heatmaps, raw count analysis, and window composition
diagnostics to identify why F1 fell short of the 0.85 target.

Usage:
    cd app/
    python analyze_penny_results.py \
        --model-dir ../models/fcvae/Penny_All \
        --data-path ../data/synthetic_transactions.csv
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from transaction_config import TransactionPreprocessorConfig
from transaction_preprocessor import TransactionPreprocessor
from fcvae_model import FCVAE, FCVAEConfig
from fcvae_scorer import FCVAEScorer, FCVAEScorerConfig

logger = logging.getLogger(__name__)

# Known anomaly injection schedule from generate_transactions.py TEST_PENNY_ANOMALIES
# Offsets are relative to the first test day (day 51, i.e. index 50 in 0-based)
# The test split starts at day 51 (train=42 days, val=8 days, test starts day 51)
PENNY_ANOMALIES = [
    {
        "label": "5x (mild)",
        "day_offset": 2,   # Day 53
        "hours": list(range(2, 8)),
        "spike_factor": 5.0,
    },
    {
        "label": "10x (moderate)",
        "day_offset": 4,   # Day 55
        "hours": list(range(10, 14)),
        "spike_factor": 10.0,
    },
    {
        "label": "25x (severe)",
        "day_offset": 7,   # Day 58
        "hours": list(range(20, 24)),
        "spike_factor": 25.0,
    },
]

# Test split starts at day 51 (0-indexed: day 50)
TEST_START_DAY = 50
TRAIN_DAYS = 42
VAL_DAYS = 8


def load_model_and_scorer(
    model_dir: str, device: torch.device
) -> Tuple[FCVAE, FCVAEScorer, object]:
    """Load trained Penny_All model, scorer, and scaler from disk."""
    model_dir = Path(model_dir)

    # Load model
    torch.serialization.add_safe_globals([FCVAEConfig])
    checkpoint = torch.load(
        model_dir / "model.pt", map_location=device, weights_only=False
    )
    model = FCVAE(config=checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load scorer
    scorer = FCVAEScorer.load(model_dir / "scorer.pkl")

    # Load scaler
    with open(model_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scorer, scaler


def prepare_test_data(
    data_path: str, window_size: int = 24, stride: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object]:
    """
    Load and prepare penny test data.

    Returns:
        test_windows: (N, W) raw count windows
        test_labels: (N, W) binary anomaly labels
        test_timestamps: (N,) starting hour index for each window
        hourly_df_values: full hourly penny count array
        preprocessor: fitted TransactionPreprocessor
    """
    config = TransactionPreprocessorConfig()
    preprocessor = TransactionPreprocessor(config=config)
    preprocessor.load_and_aggregate(data_path, aggregation_mode="penny")

    combo = ("Penny", "All")
    splits = preprocessor.create_sliding_splits(combo, window_size, stride)
    test_windows, test_labels = splits["test"]

    # Get timestamps for test windows
    sw_data = preprocessor.combo_sliding_windows[combo]
    timestamps = sw_data["timestamps"]

    # Get test mask
    hourly_splits = preprocessor.combo_hour_splits[combo].values
    window_splits = np.array([hourly_splits[ts] for ts in timestamps])
    test_mask = window_splits == "test"
    test_timestamps = timestamps[test_mask]

    hourly_df = preprocessor.combo_hourly[combo]

    return test_windows, test_labels, test_timestamps, hourly_df, preprocessor


def score_windows(
    model: FCVAE,
    scorer: FCVAEScorer,
    scaler,
    test_windows: np.ndarray,
    device: torch.device,
    score_mode: str = "single_pass",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize and score test windows.

    Args:
        score_mode: "single_pass" (matches training calibration) or "mcmc"

    Returns:
        point_scores: (N, W) per-point NLL scores
        last_point_scores: (N,) last-point NLL scores
    """
    # Normalize using the fitted scaler
    flat = test_windows.flatten().reshape(-1, 1)
    normalized = scaler.transform(flat).reshape(test_windows.shape)

    model.eval()
    all_scores = []
    batch_size = 64

    with torch.no_grad():
        for i in range(0, len(normalized), batch_size):
            batch = normalized[i : i + batch_size]
            x = torch.FloatTensor(batch).unsqueeze(1).to(device)  # (B, 1, W)
            if score_mode == "mcmc":
                _, scores = model.score_mcmc(x)
                scores = scores.squeeze(1).cpu().numpy()  # (B, W)
            else:
                scores = model.score_single_pass(x, n_samples=16)
                scores = scores.cpu().numpy()  # (B, W)
            all_scores.append(scores)

    point_scores = np.concatenate(all_scores, axis=0)
    last_point_scores = point_scores[:, -1]

    return point_scores, last_point_scores


def classify_windows_by_severity(
    test_labels: np.ndarray,
    test_timestamps: np.ndarray,
    hourly_df,
) -> Dict[str, np.ndarray]:
    """
    Classify each test window by which anomaly severity it belongs to.

    Returns:
        Dict mapping severity label -> boolean mask over test windows
    """
    hourly_anomalies = hourly_df["is_anomaly"].values
    hourly_hours = hourly_df["hour_bucket"].values

    # Build hour-to-anomaly-info mapping from the known injection schedule
    # Test split starts at hour TEST_START_DAY * 24
    test_start_hour = TEST_START_DAY * 24

    severity_hour_sets = {}
    for anom in PENNY_ANOMALIES:
        # Absolute day = TEST_START_DAY + day_offset (0-indexed day 50 + offset)
        abs_day = TEST_START_DAY + anom["day_offset"]
        abs_hours = set(abs_day * 24 + h for h in anom["hours"])
        severity_hour_sets[anom["label"]] = abs_hours

    # For each test window, check if its last point falls in an anomaly region
    # and which severity
    n_windows = len(test_timestamps)
    masks = {"normal": np.ones(n_windows, dtype=bool)}

    for label in [a["label"] for a in PENNY_ANOMALIES]:
        masks[label] = np.zeros(n_windows, dtype=bool)

    for i, ts in enumerate(test_timestamps):
        window_hours = set(range(ts, ts + 24))
        for label, hour_set in severity_hour_sets.items():
            if window_hours & hour_set:  # any overlap
                masks[label][i] = True
                masks["normal"][i] = False

    return masks


def analysis_1_per_severity_breakdown(
    last_point_scores: np.ndarray,
    window_labels: np.ndarray,
    severity_masks: Dict[str, np.ndarray],
    oracle_threshold: float,
) -> Dict:
    """Section 1: Per-severity TP/FN/recall breakdown."""
    print("\n" + "=" * 70)
    print("1. PER-SEVERITY BREAKDOWN")
    print("=" * 70)

    window_preds = (last_point_scores < oracle_threshold).astype(int)

    results = {}
    print(
        f"\n{'Severity':<20} {'Total':>6} {'Anom':>6} {'TP':>6} {'FN':>6} "
        f"{'Recall':>8} {'Mean NLL':>10} {'Min NLL':>10}"
    )
    print("-" * 78)

    for label in ["normal"] + [a["label"] for a in PENNY_ANOMALIES]:
        mask = severity_masks[label]
        n_total = mask.sum()
        if n_total == 0:
            continue

        wl = window_labels[mask]
        wp = window_preds[mask]
        scores = last_point_scores[mask]

        n_anom = wl.sum()
        tp = int(((wp == 1) & (wl == 1)).sum())
        fn = int(((wp == 0) & (wl == 1)).sum())
        fp = int(((wp == 1) & (wl == 0)).sum())
        tn = int(((wp == 0) & (wl == 0)).sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

        results[label] = {
            "total": int(n_total),
            "anomalous": int(n_anom),
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "recall": recall,
            "mean_nll": float(scores.mean()),
            "min_nll": float(scores.min()),
        }

        print(
            f"{label:<20} {n_total:>6} {n_anom:>6} {tp:>6} {fn:>6} "
            f"{recall:>8.3f} {scores.mean():>10.4f} {scores.min():>10.4f}"
        )

    # Overall stats
    tp_all = int(((window_preds == 1) & (window_labels == 1)).sum())
    fp_all = int(((window_preds == 1) & (window_labels == 0)).sum())
    fn_all = int(((window_preds == 0) & (window_labels == 1)).sum())
    tn_all = int(((window_preds == 0) & (window_labels == 0)).sum())
    prec = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0
    rec = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    print("-" * 78)
    print(
        f"{'OVERALL':<20} {len(window_labels):>6} {window_labels.sum():>6} "
        f"{tp_all:>6} {fn_all:>6} {rec:>8.3f}"
    )
    print(f"\n  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    print(f"  TP={tp_all}  FP={fp_all}  FN={fn_all}  TN={tn_all}")
    print(f"  Oracle threshold: {oracle_threshold:.4f}")

    return results


def analysis_2_score_distributions(
    last_point_scores: np.ndarray,
    severity_masks: Dict[str, np.ndarray],
    oracle_threshold: float,
    output_dir: Path,
):
    """Section 2: NLL score distribution histograms by severity."""
    print("\n" + "=" * 70)
    print("2. NLL SCORE DISTRIBUTIONS")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {"normal": "#2196F3", "5x (mild)": "#FF9800", "10x (moderate)": "#F44336", "25x (severe)": "#9C27B0"}
    stats = {}

    for label in ["normal", "5x (mild)", "10x (moderate)", "25x (severe)"]:
        mask = severity_masks[label]
        if mask.sum() == 0:
            continue
        scores = last_point_scores[mask]
        stats[label] = {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "min": float(scores.min()),
            "max": float(scores.max()),
            "median": float(np.median(scores)),
            "count": int(mask.sum()),
        }

        ax.hist(
            scores,
            bins=50,
            alpha=0.5,
            label=f"{label} (n={mask.sum()}, μ={scores.mean():.2f})",
            color=colors[label],
            density=True,
        )

        print(
            f"  {label:<20} n={mask.sum():>5}  mean={scores.mean():>8.4f}  "
            f"std={scores.std():>7.4f}  min={scores.min():>8.4f}  max={scores.max():>8.4f}"
        )

    ax.axvline(oracle_threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold={oracle_threshold:.4f}")
    ax.set_xlabel("Last-Point NLL Score")
    ax.set_ylabel("Density")
    ax.set_title("NLL Score Distribution by Anomaly Severity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "score_distributions.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {output_dir / 'score_distributions.png'}")

    # Quantify overlap: fraction of anomaly scores above threshold (would be missed)
    print("\n  Overlap analysis (fraction of anomaly windows ABOVE threshold = missed):")
    for label in ["5x (mild)", "10x (moderate)", "25x (severe)"]:
        mask = severity_masks[label]
        if mask.sum() == 0:
            continue
        scores = last_point_scores[mask]
        frac_above = (scores >= oracle_threshold).mean()
        print(f"    {label}: {frac_above:.1%} above threshold (missed)")

    return stats


def analysis_3_threshold_sensitivity(
    last_point_scores: np.ndarray,
    window_labels: np.ndarray,
    oracle_threshold: float,
    output_dir: Path,
):
    """Section 3: Precision-recall tradeoff and F1 vs threshold."""
    print("\n" + "=" * 70)
    print("3. THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 70)

    # Sweep thresholds
    candidates = np.percentile(last_point_scores, np.linspace(0.5, 99.5, 500))
    candidates = np.unique(candidates)

    precisions, recalls, f1s = [], [], []
    for t in candidates:
        preds = (last_point_scores < t).astype(int)
        tp = ((preds == 1) & (window_labels == 1)).sum()
        fp = ((preds == 1) & (window_labels == 0)).sum()
        fn = ((preds == 0) & (window_labels == 1)).sum()

        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.array(f1s)

    best_idx = np.argmax(f1s)
    best_threshold = candidates[best_idx]
    best_f1 = f1s[best_idx]

    print(f"  Best F1={best_f1:.4f} at threshold={best_threshold:.4f}")
    print(f"  Current oracle threshold={oracle_threshold:.4f}")

    # Find thresholds for target recall levels
    for target_recall in [0.80, 0.85, 0.90, 0.95]:
        viable = np.where(recalls >= target_recall)[0]
        if len(viable) > 0:
            # Among viable, pick the one with best precision
            best_viable = viable[np.argmax(precisions[viable])]
            print(
                f"  Recall >= {target_recall:.0%}: threshold={candidates[best_viable]:.4f}, "
                f"P={precisions[best_viable]:.4f}, R={recalls[best_viable]:.4f}, "
                f"F1={f1s[best_viable]:.4f}"
            )
        else:
            print(f"  Recall >= {target_recall:.0%}: not achievable")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # F1 vs threshold
    ax1.plot(candidates, f1s, "b-", linewidth=1.5, label="F1")
    ax1.plot(candidates, precisions, "g--", linewidth=1, alpha=0.7, label="Precision")
    ax1.plot(candidates, recalls, "r--", linewidth=1, alpha=0.7, label="Recall")
    ax1.axvline(oracle_threshold, color="orange", linestyle=":", linewidth=2, label=f"Oracle={oracle_threshold:.4f}")
    ax1.axvline(best_threshold, color="purple", linestyle=":", linewidth=1, label=f"Best F1={best_f1:.4f}")
    ax1.set_xlabel("Threshold (NLL)")
    ax1.set_ylabel("Score")
    ax1.set_title("F1 / Precision / Recall vs Threshold")
    ax1.legend(fontsize=8)

    # P-R curve
    ax2.plot(recalls, precisions, "b-", linewidth=1.5)
    # Mark current operating point
    oracle_preds = (last_point_scores < oracle_threshold).astype(int)
    oracle_tp = ((oracle_preds == 1) & (window_labels == 1)).sum()
    oracle_fp = ((oracle_preds == 1) & (window_labels == 0)).sum()
    oracle_fn = ((oracle_preds == 0) & (window_labels == 1)).sum()
    oracle_p = oracle_tp / (oracle_tp + oracle_fp) if (oracle_tp + oracle_fp) > 0 else 0
    oracle_r = oracle_tp / (oracle_tp + oracle_fn) if (oracle_tp + oracle_fn) > 0 else 0
    ax2.plot(oracle_r, oracle_p, "ro", markersize=10, label=f"Oracle ({oracle_r:.2f}, {oracle_p:.2f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curve.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {output_dir / 'precision_recall_curve.png'}")


def analysis_4_temporal_heatmap(
    last_point_scores: np.ndarray,
    window_labels: np.ndarray,
    test_timestamps: np.ndarray,
    oracle_threshold: float,
    output_dir: Path,
):
    """Section 4: Temporal heatmap of scores across the test period."""
    print("\n" + "=" * 70)
    print("4. TEMPORAL HEATMAP")
    print("=" * 70)

    window_preds = (last_point_scores < oracle_threshold).astype(int)

    # Map timestamps to days (relative to test start)
    test_start = test_timestamps[0]
    relative_hours = test_timestamps - test_start
    relative_days = relative_hours / 24.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    # Top: NLL scores over time
    colors = []
    for wl, wp in zip(window_labels, window_preds):
        if wl == 1 and wp == 1:
            colors.append("#4CAF50")   # TP = green
        elif wl == 1 and wp == 0:
            colors.append("#F44336")   # FN = red
        elif wl == 0 and wp == 1:
            colors.append("#FF9800")   # FP = orange
        else:
            colors.append("#90CAF9")   # TN = light blue

    ax1.scatter(relative_days, last_point_scores, c=colors, s=8, alpha=0.7)
    ax1.axhline(oracle_threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold={oracle_threshold:.4f}")
    ax1.set_ylabel("Last-Point NLL Score")
    ax1.set_title("Test Period: NLL Scores Over Time")

    # Add anomaly window shading
    for anom in PENNY_ANOMALIES:
        abs_day = TEST_START_DAY + anom["day_offset"]
        rel_day_start = (abs_day * 24 + anom["hours"][0] - test_start) / 24.0
        rel_day_end = (abs_day * 24 + anom["hours"][-1] + 1 - test_start) / 24.0
        ax1.axvspan(rel_day_start, rel_day_end, alpha=0.15, color="red")
        ax1.text(
            (rel_day_start + rel_day_end) / 2,
            ax1.get_ylim()[1] if ax1.get_ylim()[1] != ax1.get_ylim()[0] else last_point_scores.max(),
            anom["label"],
            ha="center",
            va="top",
            fontsize=8,
            fontweight="bold",
        )

    legend_patches = [
        mpatches.Patch(color="#4CAF50", label="TP"),
        mpatches.Patch(color="#F44336", label="FN"),
        mpatches.Patch(color="#FF9800", label="FP"),
        mpatches.Patch(color="#90CAF9", label="TN"),
    ]
    ax1.legend(handles=legend_patches, loc="lower left", fontsize=8)

    # Bottom: ground truth and predictions
    ax2.fill_between(relative_days, 0, window_labels, alpha=0.3, color="red", label="Ground Truth")
    ax2.scatter(relative_days, window_preds * 0.5, c=window_preds, cmap="RdYlGn_r", s=5, alpha=0.7)
    ax2.set_xlabel("Days into Test Period")
    ax2.set_ylabel("Anomaly")
    ax2.set_title("Ground Truth vs Predictions")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "temporal_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'temporal_heatmap.png'}")

    # Print per-anomaly-window detail
    for anom in PENNY_ANOMALIES:
        abs_day = TEST_START_DAY + anom["day_offset"]
        anom_hours = set(abs_day * 24 + h for h in anom["hours"])

        print(f"\n  {anom['label']} (Day {abs_day + 1}, hours {anom['hours']}):")
        hit_count = 0
        miss_count = 0
        for i, ts in enumerate(test_timestamps):
            last_hour = ts + 23  # last point in window
            if last_hour in anom_hours:
                status = "TP" if window_preds[i] == 1 else "FN"
                if status == "TP":
                    hit_count += 1
                else:
                    miss_count += 1
                print(f"    Hour {last_hour % 24:02d}: NLL={last_point_scores[i]:.4f} -> {status}")
        print(f"    Summary: {hit_count} TP, {miss_count} FN")


def analysis_5_raw_counts(
    test_windows: np.ndarray,
    test_labels: np.ndarray,
    test_timestamps: np.ndarray,
    scaler,
    hourly_df,
    output_dir: Path,
):
    """Section 5: Raw count analysis and signal-to-noise ratio."""
    print("\n" + "=" * 70)
    print("5. RAW COUNT ANALYSIS & SIGNAL-TO-NOISE RATIO")
    print("=" * 70)

    hourly_counts = hourly_df["count"].values
    hourly_anomalies = hourly_df["is_anomaly"].values

    # Get test period
    test_start = test_timestamps[0]
    test_end = test_timestamps[-1] + 24
    test_counts = hourly_counts[test_start:test_end]
    test_anom_flags = hourly_anomalies[test_start:test_end]

    # Normal hours stats
    normal_mask = test_anom_flags == 0
    normal_counts = test_counts[normal_mask]
    normal_mean = normal_counts.mean()
    normal_std = normal_counts.std()

    print(f"  Normal hours: n={normal_mask.sum()}, mean={normal_mean:.1f}, std={normal_std:.1f}")

    # Per-severity SNR
    print(f"\n  {'Severity':<20} {'Mean Count':>12} {'SNR':>8} {'Peak Count':>12}")
    print("  " + "-" * 55)

    for anom in PENNY_ANOMALIES:
        abs_day = TEST_START_DAY + anom["day_offset"]
        anom_indices = [abs_day * 24 + h for h in anom["hours"]]
        anom_counts = hourly_counts[anom_indices]
        anom_mean = anom_counts.mean()
        snr = (anom_mean - normal_mean) / normal_std if normal_std > 0 else float("inf")
        print(
            f"  {anom['label']:<20} {anom_mean:>12.1f} {snr:>8.2f} {anom_counts.max():>12.0f}"
        )

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    hours = np.arange(len(test_counts))
    ax1.plot(hours, test_counts, "b-", linewidth=0.8, alpha=0.8)
    ax1.fill_between(
        hours, 0, test_counts,
        where=test_anom_flags.astype(bool),
        alpha=0.3, color="red", label="Anomaly hours",
    )
    ax1.axhline(normal_mean, color="green", linestyle="--", alpha=0.5, label=f"Normal mean={normal_mean:.0f}")
    ax1.axhline(normal_mean + 2 * normal_std, color="orange", linestyle=":", alpha=0.5, label=f"+2σ={normal_mean + 2*normal_std:.0f}")
    ax1.set_ylabel("Hourly Penny Count")
    ax1.set_title("Raw Penny Transaction Counts (Test Period)")
    ax1.legend(fontsize=8)

    # Normalized counts
    test_counts_norm = scaler.transform(test_counts.reshape(-1, 1)).flatten()
    ax2.plot(hours, test_counts_norm, "b-", linewidth=0.8, alpha=0.8)
    ax2.fill_between(
        hours, test_counts_norm.min(), test_counts_norm,
        where=test_anom_flags.astype(bool),
        alpha=0.3, color="red",
    )
    ax2.axhline(0, color="green", linestyle="--", alpha=0.5, label="Scaler mean")
    ax2.set_ylabel("Normalized Count")
    ax2.set_xlabel("Hours into Test Period")
    ax2.set_title("Normalized Penny Counts (Test Period)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "raw_counts.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {output_dir / 'raw_counts.png'}")


def analysis_6_window_composition(
    test_labels: np.ndarray,
    last_point_scores: np.ndarray,
    severity_masks: Dict[str, np.ndarray],
    oracle_threshold: float,
    output_dir: Path,
):
    """Section 6: Window composition — edge effects from partial anomaly overlap."""
    print("\n" + "=" * 70)
    print("6. WINDOW COMPOSITION ANALYSIS (EDGE EFFECTS)")
    print("=" * 70)

    # For each window that overlaps any anomaly, compute fraction of anomalous points
    any_anomaly_mask = ~severity_masks["normal"]
    anomaly_windows_labels = test_labels[any_anomaly_mask]
    anomaly_windows_scores = last_point_scores[any_anomaly_mask]

    fractions = anomaly_windows_labels.sum(axis=1) / anomaly_windows_labels.shape[1]
    preds = (anomaly_windows_scores < oracle_threshold).astype(int)
    ground_truth = (anomaly_windows_labels.sum(axis=1) > 0).astype(int)

    # Bin by fraction
    bins = [0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.01]
    bin_labels = ["0-10%", "10-20%", "20-30%", "30-50%", "50-75%", "75-100%"]

    print(f"\n  {'Anom Frac':<12} {'Count':>6} {'TP':>6} {'FN':>6} {'Recall':>8}")
    print("  " + "-" * 42)

    for i in range(len(bins) - 1):
        mask = (fractions >= bins[i]) & (fractions < bins[i + 1])
        n = mask.sum()
        if n == 0:
            continue
        tp = ((preds[mask] == 1) & (ground_truth[mask] == 1)).sum()
        fn = ((preds[mask] == 0) & (ground_truth[mask] == 1)).sum()
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        print(f"  {bin_labels[i]:<12} {n:>6} {tp:>6} {fn:>6} {recall:>8.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))

    tp_mask = (preds == 1) & (ground_truth == 1)
    fn_mask = (preds == 0) & (ground_truth == 1)

    ax.scatter(fractions[tp_mask], anomaly_windows_scores[tp_mask], c="#4CAF50", s=30, alpha=0.6, label="TP")
    ax.scatter(fractions[fn_mask], anomaly_windows_scores[fn_mask], c="#F44336", s=30, alpha=0.6, label="FN")
    ax.axhline(oracle_threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold={oracle_threshold:.4f}")
    ax.set_xlabel("Fraction of Anomalous Points in Window")
    ax.set_ylabel("Last-Point NLL Score")
    ax.set_title("Detection Success vs Window Anomaly Composition")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "window_composition.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {output_dir / 'window_composition.png'}")


def analysis_7_summary(
    severity_results: Dict,
    oracle_threshold: float,
):
    """Section 7: Summary and recommendations."""
    print("\n" + "=" * 70)
    print("7. SUMMARY & RECOMMENDATIONS")
    print("=" * 70)

    print("\n  Per-Severity Recap:")
    print(f"  {'Severity':<20} {'Recall':>8} {'FN':>6} {'Verdict':<30}")
    print("  " + "-" * 66)

    for label in [a["label"] for a in PENNY_ANOMALIES]:
        r = severity_results.get(label, {})
        recall = r.get("recall", 0)
        fn = r.get("fn", 0)
        if recall >= 0.90:
            verdict = "Good"
        elif recall >= 0.70:
            verdict = "Moderate — room for improvement"
        elif recall >= 0.50:
            verdict = "Poor — significant miss rate"
        else:
            verdict = "Critical — mostly undetected"
        print(f"  {label:<20} {recall:>8.3f} {fn:>6} {verdict:<30}")

    print("\n  Potential Improvements:")
    print("  " + "-" * 60)

    # Check if mild spike is the main culprit
    mild_r = severity_results.get("5x (mild)", {})
    if mild_r.get("recall", 1.0) < 0.70:
        print("  [1] MILD SPIKE DETECTION is the primary bottleneck.")
        print("      Options:")
        print("      a) Increase mild spike factor from 5x to 7-8x (data side)")
        print("      b) Apply log-transform: log(1+count) before normalization")
        print("         This compresses high counts and amplifies low-count deviations")
        print("      c) Lower the threshold (trades precision for recall)")
        print("      d) Use RobustScaler (median/IQR) instead of StandardScaler")

    moderate_r = severity_results.get("10x (moderate)", {})
    if moderate_r.get("recall", 1.0) < 0.90:
        print("  [2] MODERATE SPIKE (10x) also has gaps — likely edge-effect windows.")
        print("      Consider window-level decision mode changes (k=1 instead of k=3)")

    severe_r = severity_results.get("25x (severe)", {})
    if severe_r.get("recall", 1.0) < 0.95:
        print("  [3] SEVERE SPIKE (25x) should be near-perfect. Investigate if")
        print("      edge windows with only 1-2 anomalous points are dragging recall down.")

    # General recommendations
    print("\n  General recommendations:")
    print("  - If edge-effect windows dominate FN: switch to last-point-only evaluation")
    print("    (only consider a window anomalous if its LAST point is in an anomaly region)")
    print("  - If score distributions overlap: try log-transform preprocessing")
    print("  - If threshold is too conservative: the oracle threshold should already")
    print("    maximize F1, so the issue is likely in the model/data, not the threshold")


def main():
    parser = argparse.ArgumentParser(description="Analyze Penny_All FCVAE results")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="../models/fcvae/Penny_All",
        help="Path to Penny_All model directory",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="../data/synthetic_transactions.csv",
        help="Path to synthetic transactions CSV",
    )
    parser.add_argument(
        "--oracle-threshold",
        type=float,
        default=-2.0658,
        help="Oracle threshold for Penny_All",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--score-mode",
        type=str,
        choices=["single_pass", "mcmc"],
        default="single_pass",
        help="Scoring mode (should match training calibration)",
    )
    parser.add_argument(
        "--last-point-labels",
        action="store_true",
        default=True,
        help="Use last-point-only window labels (default: True)",
    )
    parser.add_argument(
        "--any-point-labels",
        action="store_true",
        default=False,
        help="Use any-point window labels (window is anomalous if ANY point is)",
    )
    args = parser.parse_args()
    if args.any_point_labels:
        args.last_point_labels = False

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("\nLoading model from", args.model_dir)
    model, scorer, scaler = load_model_and_scorer(args.model_dir, device)
    print(f"  Scaler: mean={scaler.mean_[0]:.2f}, std={scaler.scale_[0]:.2f}")
    print(f"  Scorer thresholds: point={scorer.point_threshold}, last_point={scorer.last_point_threshold}")

    # Load and prepare data
    print("\nLoading test data from", args.data_path)
    test_windows, test_labels, test_timestamps, hourly_df, preprocessor = prepare_test_data(
        args.data_path, args.window_size, args.stride
    )
    print(f"  Test windows: {test_windows.shape}")
    print(f"  Anomalous windows (any-point): {(test_labels.sum(axis=1) > 0).sum()}")
    print(f"  Anomalous windows (last-point): {(test_labels[:, -1] > 0).sum()}")
    print(f"  Label mode: {'last-point' if args.last_point_labels else 'any-point'}")

    # Score
    print(f"\nScoring test windows ({args.score_mode})...")
    point_scores, last_point_scores = score_windows(
        model, scorer, scaler, test_windows, device, score_mode=args.score_mode
    )
    print(f"  Score range: [{last_point_scores.min():.4f}, {last_point_scores.max():.4f}]")

    # Classify by severity
    severity_masks = classify_windows_by_severity(test_labels, test_timestamps, hourly_df)
    for label, mask in severity_masks.items():
        print(f"  {label}: {mask.sum()} windows")

    # Output directory
    output_dir = Path("../plots/penny")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute window-level labels based on selected mode
    if args.last_point_labels:
        window_labels = test_labels[:, -1].astype(int)
    else:
        window_labels = (test_labels.sum(axis=1) > 0).astype(int)

    # Run all analyses
    severity_results = analysis_1_per_severity_breakdown(
        last_point_scores, window_labels, severity_masks, args.oracle_threshold
    )
    analysis_2_score_distributions(
        last_point_scores, severity_masks, args.oracle_threshold, output_dir
    )
    analysis_3_threshold_sensitivity(
        last_point_scores, window_labels, args.oracle_threshold, output_dir
    )
    analysis_4_temporal_heatmap(
        last_point_scores, window_labels, test_timestamps, args.oracle_threshold, output_dir
    )
    analysis_5_raw_counts(
        test_windows, test_labels, test_timestamps, scaler, hourly_df, output_dir
    )
    analysis_6_window_composition(
        test_labels, last_point_scores, severity_masks, args.oracle_threshold, output_dir
    )
    analysis_7_summary(severity_results, args.oracle_threshold)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"Plots saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
