"""
Step 2: Evaluate FCVAE Model (Penny and Combo Modes)

Load trained model(s), score test set, compute F1/precision/recall,
plot NLL distributions, reconstruction plots, per-hour heatmaps.

Defaults to models/fcvae/initial/Penny_All (baseline from step 1).
Use --model-dir models/fcvae/best/Penny_All for the grid-sweep winner
from step 4, or --model-dir models/fcvae/Penny_All for the prebuilt
reference.

Modes:
    --mode penny   Evaluate Penny_All model only (default)
    --mode combo   Evaluate all 4 combo models

Usage:
    uv run python code/2_evaluate_model.py
    uv run python code/2_evaluate_model.py --model-dir models/fcvae/best/Penny_All
    uv run python code/2_evaluate_model.py --model-dir models/fcvae/Penny_All
    uv run python code/2_evaluate_model.py --mode combo
    uv run python code/2_evaluate_model.py --no-plots
"""
import argparse
import logging
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

# Suppress PyTorch FFT resize deprecation warnings
warnings.filterwarnings("ignore", message=".*output with one or more elements was resized.*")

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Pickle compatibility for old module paths
import types
import src.model
import src.scorer
sys.modules["app"] = types.ModuleType("app")
sys.modules["app.fcvae_model"] = src.model
sys.modules["app.fcvae_scorer"] = src.scorer
sys.modules["app.attention"] = src.model
sys.modules["fcvae_model"] = src.model
sys.modules["fcvae_scorer"] = src.scorer
sys.modules["attention"] = src.model

from src.model import FCVAE, FCVAEConfig
from src.scorer import FCVAEScorer, FCVAEScorerConfig
from src.preprocess import (
    load_penny_data, load_combo_data, COMBO_KEYS,
    create_sliding_windows, create_splits, normalize, create_dataloaders,
)
from src.utils import auto_device

logger = logging.getLogger(__name__)

# Maps combo key tuples to directory-safe names
combo_dir_names = {
    ("Accel", "CMP"): "Accel_CMP",
    ("Accel", "no-pin"): "Accel_nopin",
    ("Star", "CMP"): "Star_CMP",
    ("Star", "no-pin"): "Star_nopin",
}

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ---------------------------------------------------------------------------
# Point-Adjusted F1 (Best F1 from DONUT/FCVAE paper)
# ---------------------------------------------------------------------------

def find_contiguous_segments(labels: np.ndarray):
    """Find contiguous segments of True/1 values in a binary array.

    Returns list of (start_idx, end_idx) tuples (inclusive on both ends).
    """
    segments = []
    in_segment = False
    start = 0

    for i in range(len(labels)):
        if labels[i] and not in_segment:
            start = i
            in_segment = True
        elif not labels[i] and in_segment:
            segments.append((start, i - 1))
            in_segment = False

    if in_segment:
        segments.append((start, len(labels) - 1))

    return segments


def point_adjusted_f1(predictions: np.ndarray, ground_truth: np.ndarray) -> dict:
    """Compute point-adjusted F1 score (Best F1 from DONUT/FCVAE paper).

    If ANY point within a contiguous anomaly segment is correctly detected,
    the ENTIRE segment is counted as a true positive. This is the standard
    evaluation metric in time series anomaly detection literature.
    """
    predictions = np.asarray(predictions, dtype=bool)
    ground_truth = np.asarray(ground_truth, dtype=bool)

    gt_segments = find_contiguous_segments(ground_truth)

    tp_segments = 0
    fn_segments = 0

    for seg_start, seg_end in gt_segments:
        if np.any(predictions[seg_start:seg_end + 1]):
            tp_segments += 1
        else:
            fn_segments += 1

    # Adjust predictions: detected segments have all points as TP
    adjusted_predictions = predictions.copy()
    for seg_start, seg_end in gt_segments:
        if np.any(predictions[seg_start:seg_end + 1]):
            adjusted_predictions[seg_start:seg_end + 1] = True

    tp = int(np.sum(adjusted_predictions & ground_truth))
    fp = int(np.sum(adjusted_predictions & ~ground_truth))
    fn = int(np.sum(~adjusted_predictions & ground_truth))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "tp_segments": tp_segments, "fn_segments": fn_segments,
        "total_segments": len(gt_segments),
    }

# Known anomaly injection schedule from generate_transactions.py
PENNY_ANOMALIES = [
    {"label": "5x (mild)", "day_offset": 2, "hours": list(range(2, 8)), "spike_factor": 5.0},
    {"label": "10x (moderate)", "day_offset": 4, "hours": list(range(10, 14)), "spike_factor": 10.0},
    {"label": "25x (severe)", "day_offset": 7, "hours": list(range(20, 24)), "spike_factor": 25.0},
]


def load_model_artifacts(model_dir: Path, device: torch.device):
    """Load pre-trained FCVAE model, scorer, and scaler."""
    # Load model
    checkpoint = torch.load(model_dir / "model.pt", map_location=device, weights_only=False)

    if "config" in checkpoint or "model_config" in checkpoint:
        config = checkpoint.get("config") or checkpoint.get("model_config")
        model = FCVAE(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        config = FCVAEConfig()
        model = FCVAE(config).to(device)
        model.load_state_dict(checkpoint)

    model.eval()

    # Load scorer
    scorer = FCVAEScorer.load(model_dir / "scorer.pkl")

    # Load scaler
    with open(model_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Load history if available
    history = None
    history_path = model_dir / "history.pkl"
    if history_path.exists():
        with open(history_path, "rb") as f:
            history = pickle.load(f)

    return model, scorer, scaler, config, history


def plot_score_distribution(
    normal_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    threshold: float,
    output_path: Path,
    title: str = "NLL Score Distribution",
):
    """Plot score distributions for normal vs anomalous windows."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.hist(normal_scores, bins=100, alpha=0.6, label="Normal", color="steelblue", density=True)
    if len(anomaly_scores) > 0:
        ax.hist(anomaly_scores, bins=50, alpha=0.6, label="Anomaly", color="crimson", density=True)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold: {threshold:.2f}")

    ax.set_xlabel("NLL Score (lower = more anomalous)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_reconstruction(
    model: FCVAE,
    windows: np.ndarray,
    scaler,
    device: torch.device,
    output_path: Path,
    num_windows: int = 5,
):
    """Plot reconstruction with confidence bands."""
    if not HAS_MATPLOTLIB:
        return

    model.eval()
    num_windows = min(num_windows, len(windows))

    fig, axes = plt.subplots(num_windows, 1, figsize=(14, 3 * num_windows))
    if num_windows == 1:
        axes = [axes]

    hours = np.arange(24)

    for idx, ax in enumerate(axes):
        window = windows[idx]
        x = torch.FloatTensor(window).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            mu_x, var_x = model.reconstruct(x)

        mu_np = mu_x.squeeze().cpu().numpy()
        std_np = np.sqrt(var_x.squeeze().cpu().numpy())

        # Inverse transform for display
        orig = scaler.inverse_transform(window.reshape(-1, 1)).flatten()
        mu_orig = scaler.inverse_transform(mu_np.reshape(-1, 1)).flatten()
        std_orig = std_np * scaler.scale_[0]

        ax.plot(hours, orig, "b-o", markersize=4, label="Original", linewidth=1.5)
        ax.plot(hours, mu_orig, "r--", label="Reconstruction (mu)", linewidth=1.5)
        ax.fill_between(hours, mu_orig - 2 * std_orig, mu_orig + 2 * std_orig,
                        alpha=0.2, color="red", label="+/-2 sigma")
        ax.set_ylabel("Count")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"Window {idx}")

    axes[-1].set_xlabel("Hour of Day")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_training_history(history: dict, output_path: Path):
    """Plot training loss curves."""
    if not HAS_MATPLOTLIB or history is None:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="Train", linewidth=1.5)
    ax1.plot(epochs, history["val_loss"], label="Validation", linewidth=1.5)
    if history.get("best_epoch"):
        ax1.axvline(history["best_epoch"], color="green", linestyle=":", label=f"Best: epoch {history['best_epoch']}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()

    if "kl_weights" in history:
        ax2.plot(epochs, history["kl_weights"], color="purple", linewidth=1.5)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("KL Weight")
        ax2.set_title("KL Annealing Schedule")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def evaluate_single(name, model_dir, hourly_df, scaler, device, output_dir, no_plots=False):
    """Evaluate a single trained FCVAE model.

    Args:
        name: Model name (e.g. "Penny_All", "Accel_CMP")
        model_dir: Path to model directory containing model.pt, scorer.pkl, scaler.pkl
        hourly_df: Hourly DataFrame for this model's data
        scaler: Fitted scaler (used if model scaler not available; overridden by saved scaler)
        device: torch device
        output_dir: Base output directory (plots saved to output_dir/name/)
    """
    print(f"\n{'=' * 60}")
    print(f"EVALUATING: {name}")
    print(f"{'=' * 60}")

    print(f"\nModel: {model_dir}")

    # Load model
    print("\n--- Loading Model ---")
    model, scorer, saved_scaler, config, history = load_model_artifacts(model_dir, device)
    # Use saved scaler from model dir
    scaler = saved_scaler

    print(f"  Config: window={config.window}, latent_dim={config.latent_dim}")
    print(f"  Last-point threshold: {scorer.last_point_threshold}")
    print(f"  Point threshold: {scorer.point_threshold}")

    # Preprocess data
    print("\n--- Loading Data ---")
    windows, labels, timestamps = create_sliding_windows(hourly_df)
    splits = create_splits(windows, labels, timestamps, hourly_df)

    # Normalize using the saved scaler
    normalized_splits = {}
    for split_name, (w, l) in splits.items():
        if len(w) == 0:
            normalized_splits[split_name] = (w, l)
            continue
        flat = w.flatten().reshape(-1, 1)
        normalized = scaler.transform(flat)
        normalized_splits[split_name] = (normalized.reshape(w.shape), l)

    loaders = create_dataloaders(normalized_splits, batch_size=64, shuffle_train=False)

    # Score test set
    print("\n--- Scoring Test Set ---")
    test_loader = loaders.get("test")
    if test_loader is None:
        print("  No test data available!")
        return

    test_point_scores, test_window_scores = scorer.score_batch(model, test_loader, device)

    # Collect test labels
    all_test_labels = []
    for batch in test_loader:
        _, labels_batch, _ = batch
        all_test_labels.append(labels_batch.numpy())
    all_test_labels = np.concatenate(all_test_labels)

    # Last-point metrics
    print("\n--- Last-Point Metrics ---")
    lp_scores = test_point_scores[:, -1]
    lp_labels = all_test_labels[:, -1].astype(int)

    threshold = scorer.last_point_threshold or scorer.point_threshold
    lp_preds = lp_scores < threshold

    tp = np.sum(lp_preds & lp_labels.astype(bool))
    fp = np.sum(lp_preds & ~lp_labels.astype(bool))
    fn = np.sum(~lp_preds & lp_labels.astype(bool))
    tn = np.sum(~lp_preds & ~lp_labels.astype(bool))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"  Threshold: {threshold:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # Point-adjusted F1 (paper metric)
    print("\n--- Point-Adjusted F1 (Paper Metric) ---")
    pa_metrics = point_adjusted_f1(lp_preds, lp_labels.astype(bool))
    print(f"  PA-Precision: {pa_metrics['precision']:.4f}")
    print(f"  PA-Recall:    {pa_metrics['recall']:.4f}")
    print(f"  PA-F1:        {pa_metrics['f1']:.4f}")
    print(f"  Segments: {pa_metrics['tp_segments']}/{pa_metrics['total_segments']} detected, "
          f"{pa_metrics['fn_segments']} missed")

    # All-position metrics
    print("\n--- All-Position Metrics ---")
    flat_scores = test_point_scores.flatten()
    flat_labels = all_test_labels.flatten().astype(int)

    normal_scores = flat_scores[flat_labels == 0]
    anomaly_scores = flat_scores[flat_labels == 1]

    print(f"  Normal points: {len(normal_scores)}, Anomaly points: {len(anomaly_scores)}")
    print(f"  Normal mean: {normal_scores.mean():.4f} +/- {normal_scores.std():.4f}")
    if len(anomaly_scores) > 0:
        print(f"  Anomaly mean: {anomaly_scores.mean():.4f} +/- {anomaly_scores.std():.4f}")

    # All-position point-adjusted F1
    all_threshold = scorer.point_threshold or threshold
    all_preds = flat_scores < all_threshold
    all_labels_bool = flat_labels.astype(bool)

    all_tp = np.sum(all_preds & all_labels_bool)
    all_fp = np.sum(all_preds & ~all_labels_bool)
    all_fn = np.sum(~all_preds & all_labels_bool)
    all_prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    all_rec = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    all_f1 = 2 * all_prec * all_rec / (all_prec + all_rec) if (all_prec + all_rec) > 0 else 0
    print(f"\n  All-position F1: {all_f1:.4f} (P={all_prec:.4f}, R={all_rec:.4f})")

    pa_all = point_adjusted_f1(all_preds, all_labels_bool)
    print(f"  All-position PA-F1: {pa_all['f1']:.4f} (P={pa_all['precision']:.4f}, R={pa_all['recall']:.4f})")
    print(f"  Segments: {pa_all['tp_segments']}/{pa_all['total_segments']} detected")

    if no_plots:
        print(f"\n  {name} EVALUATION COMPLETE (plots skipped)")
        return

    # Generate plots
    eval_output_dir = output_dir / name
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Generating Plots ---")

    plot_score_distribution(
        normal_scores=lp_scores[lp_labels == 0],
        anomaly_scores=lp_scores[lp_labels == 1],
        threshold=threshold,
        output_path=eval_output_dir / "score_distribution_lastpoint.png",
        title=f"Last-Point NLL Score Distribution ({name})",
    )

    plot_score_distribution(
        normal_scores=normal_scores,
        anomaly_scores=anomaly_scores,
        threshold=scorer.point_threshold or threshold,
        output_path=eval_output_dir / "score_distribution_all.png",
        title=f"All-Position NLL Score Distribution ({name})",
    )

    # Reconstruction plots (test normal + anomaly)
    test_norm_windows = normalized_splits["test"][0]
    if len(test_norm_windows) > 0:
        normal_mask = all_test_labels.sum(axis=1) == 0
        if normal_mask.sum() > 0:
            plot_reconstruction(
                model, test_norm_windows[normal_mask][:5], scaler, device,
                eval_output_dir / "reconstruction_normal.png"
            )

        anomaly_mask = all_test_labels.sum(axis=1) > 0
        if anomaly_mask.sum() > 0:
            plot_reconstruction(
                model, test_norm_windows[anomaly_mask][:5], scaler, device,
                eval_output_dir / "reconstruction_anomaly.png"
            )

    # Training history
    plot_training_history(history, eval_output_dir / "training_history.png")

    print(f"\n  {name} EVALUATION COMPLETE")
    print(f"  Plots saved to: {eval_output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Evaluate FCVAE model(s)")
    parser.add_argument("--mode", choices=["penny", "combo"], default="penny",
                        help="Evaluation mode: penny (Penny_All only) or combo (4 combo models)")
    parser.add_argument("--model-dir", type=str, default="models/fcvae/initial/Penny_All",
                        help="Model directory (default: initial/Penny_All from step 1)")
    parser.add_argument("--data-path", type=str, default="data/synthetic_transactions.csv")
    parser.add_argument("--output-dir", type=str, default="plots/penny_eval")
    parser.add_argument("--from-saved", action="store_true",
                        help="Load pre-computed scores instead of re-scoring")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    for mod in ["src.model", "src.scorer", "src.preprocess", "src.train", "src.training"]:
        logging.getLogger(mod).setLevel(logging.WARNING)

    device = auto_device(args.device)
    data_path = PROJECT_ROOT / args.data_path

    if args.mode == "penny":
        print("\n" + "=" * 60)
        print("FCVAE PENNY MODEL EVALUATION")
        print("=" * 60)

        print(f"\nDevice: {device}")

        model_dir = PROJECT_ROOT / args.model_dir
        output_dir = PROJECT_ROOT / args.output_dir

        hourly_df = load_penny_data(data_path)
        evaluate_single("Penny_All", model_dir, hourly_df, None, device, output_dir,
                        no_plots=args.no_plots)

    elif args.mode == "combo":
        print("\n" + "=" * 60)
        print("FCVAE COMBO MODEL EVALUATION")
        print("=" * 60)

        print(f"\nDevice: {device}")

        models_root = PROJECT_ROOT / "models" / "fcvae"
        output_dir = PROJECT_ROOT / "plots" / "combo_eval"

        combo_data = load_combo_data(data_path)

        for combo_key in COMBO_KEYS:
            dir_name = combo_dir_names[combo_key]
            model_dir = models_root / dir_name
            hourly_df = combo_data[combo_key]

            if not model_dir.exists():
                print(f"\nSkipping {dir_name}: model directory not found at {model_dir}")
                continue

            evaluate_single(dir_name, model_dir, hourly_df, None, device, output_dir,
                            no_plots=args.no_plots)

        print(f"\n{'=' * 60}")
        print("ALL COMBO EVALUATIONS COMPLETE")
        print(f"{'=' * 60}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
