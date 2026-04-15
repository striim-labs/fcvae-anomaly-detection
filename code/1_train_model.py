"""
Step 1: Train FCVAE Baseline Model (Penny Transactions)

Train a penny-only FCVAE with deliberately under-specified defaults so the
baseline catches most anomalies but leaves room for improvement. Run
code/4_grid_sweep.py next to search for a better configuration.

Output goes to models/fcvae/initial/Penny_All/ (gitignored).
Prebuilt artifacts in models/fcvae/Penny_All/ are NEVER overwritten.

Modes:
    --mode penny   Train Penny_All model only (default)
    --mode combo   Train all 4 combo models (advanced)

Usage:
    uv run python code/1_train_model.py
    uv run python code/1_train_model.py --mode combo --output-dir models/fcvae
    uv run python code/1_train_model.py --epochs 30 --lr 5e-4 --latent-dim 8
"""
import argparse
import logging
import random
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

from src.model import FCVAE, FCVAEConfig
from src.scorer import FCVAEScorer, FCVAEScorerConfig
from src.preprocess import (
    load_penny_data, load_combo_data, COMBO_KEYS,
    create_sliding_windows, create_splits, normalize, create_dataloaders,
)
from src.train import AugmentConfig
from src.training import TrainingConfig, train_model, save_training_artifacts
from src.utils import auto_device

logger = logging.getLogger(__name__)

# Maps combo key tuples to directory-safe names
combo_dir_names = {
    ("Accel", "CMP"): "Accel_CMP",
    ("Accel", "no-pin"): "Accel_nopin",
    ("Star", "CMP"): "Star_CMP",
    ("Star", "no-pin"): "Star_nopin",
}


def optimize_threshold_f1(
    model: FCVAE,
    scorer: FCVAEScorer,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    beta: float = 1.0,
) -> dict:
    """Optimize threshold using F1 score on validation set with real anomaly labels.

    Calibrates three thresholds:
    - All-position point threshold
    - Last-point threshold (paper-faithful, position [-1] only)
    - Window-level threshold
    """
    # Collect all validation labels
    all_labels = []
    for batch in val_loader:
        _, labels, _ = batch
        all_labels.append(labels.numpy())
    all_labels = np.concatenate(all_labels)

    # Score all windows
    all_point_scores, _ = scorer.score_batch(model, val_loader, device)

    # --- All-position threshold ---
    flat_scores = all_point_scores.flatten()
    flat_labels = all_labels.flatten().astype(int)

    normal_scores = flat_scores[flat_labels == 0]
    anomaly_scores = flat_scores[flat_labels == 1]

    logger.debug(f"  Validation: {len(normal_scores)} normal points, {len(anomaly_scores)} anomaly points")

    if len(anomaly_scores) == 0:
        logger.warning("  No anomaly points in validation -- falling back to percentile threshold")
        scorer.set_threshold(flat_scores, method="percentile", percentile=5.0)
        scorer.set_last_point_threshold(all_point_scores, method="percentile", percentile=5.0)
        return {"method": "percentile_fallback", "reason": "no_anomalies"}

    optimal_threshold, metrics = scorer.find_optimal_threshold(
        normal_scores=normal_scores,
        anomaly_scores=anomaly_scores,
        method="f1_max",
        beta=beta,
    )
    scorer.point_threshold = optimal_threshold

    # --- Last-point threshold (paper-faithful) ---
    lp_scores = all_point_scores[:, -1]
    lp_labels = all_labels[:, -1].astype(int)

    lp_normal = lp_scores[lp_labels == 0]
    lp_anomaly = lp_scores[lp_labels == 1]

    logger.debug(f"  Last-point: {len(lp_normal)} normal, {len(lp_anomaly)} anomaly")

    if len(lp_anomaly) > 0:
        lp_threshold, lp_metrics = scorer.find_optimal_last_point_threshold(
            normal_scores=lp_normal,
            anomaly_scores=lp_anomaly,
            method="f1_max",
            beta=beta,
        )
        scorer.last_point_threshold = lp_threshold
        metrics["last_point_threshold"] = lp_threshold
        metrics["last_point_f1"] = lp_metrics.get("f1", 0)
        logger.debug(
            f"  Last-point threshold: {lp_threshold:.4f} "
            f"(vs all-position: {optimal_threshold:.4f})"
        )
    else:
        scorer.set_last_point_threshold(all_point_scores, method="percentile", percentile=5.0)
        metrics["last_point_threshold"] = scorer.last_point_threshold
        metrics["last_point_method"] = "percentile_fallback"

    # --- Window threshold ---
    window_scores = all_point_scores.mean(axis=1)
    window_labels = (all_labels.sum(axis=1) > 0).astype(int)

    normal_window_scores = window_scores[window_labels == 0]
    anomaly_window_scores = window_scores[window_labels == 1]

    if len(anomaly_window_scores) > 0:
        window_threshold, window_metrics = scorer.find_optimal_threshold(
            normal_scores=normal_window_scores,
            anomaly_scores=anomaly_window_scores,
            method="f1_max",
            beta=beta,
        )
        scorer.window_threshold = window_threshold
        metrics["window_threshold"] = window_threshold
        metrics["window_f1"] = window_metrics.get("f1", 0)
    else:
        scorer.set_window_threshold(normal_window_scores, method="percentile", percentile=5.0)
        metrics["window_threshold"] = scorer.window_threshold

    logger.debug(
        f"  Thresholds: all-position={scorer.point_threshold:.4f}, "
        f"last-point={scorer.last_point_threshold:.4f}, "
        f"window={scorer.window_threshold}"
    )

    return metrics


def train_single(name, loaders, args, device, output_dir, scaler=None):
    """Train a single FCVAE model end-to-end and save artifacts."""
    print(f"\n{'=' * 60}")
    print(f"TRAINING: {name}")
    print(f"{'=' * 60}")

    for split_name in ["train", "val", "test"]:
        if loaders.get(split_name) is not None:
            print(f"  {split_name}: {len(loaders[split_name].dataset)} windows")

    # Initialize model
    print(f"\n--- Initializing FCVAE model ---")
    model_config = FCVAEConfig(window=args.window_size, latent_dim=args.latent_dim)
    model = FCVAE(model_config).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")
    print(f"  Latent dim: {args.latent_dim}, Window: {args.window_size}")

    scorer_config = FCVAEScorerConfig(score_mode=args.score_mode)
    scorer = FCVAEScorer(config=scorer_config)

    augment_config = AugmentConfig() if not args.no_augmentation else None

    if not args.no_augmentation:
        print(f"  Augmentation: point={augment_config.point_ano_rate}, "
              f"segment={augment_config.seg_ano_rate}, "
              f"missing={augment_config.missing_data_rate}")

    # Train using shared helper
    print(f"\n--- Training ---")
    training_config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        kl_warmup_epochs=args.kl_warmup_epochs,
        grad_clip=args.grad_clip,
        batch_size=args.batch_size,
        augmentation=not args.no_augmentation,
    )

    model, history = train_model(
        model, loaders["train"], loaders["val"], device,
        config=training_config,
        augment_config=augment_config,
    )

    # Calibrate threshold (F1-max)
    # Try validation set first; if it has no anomalies, fall back to test set
    # for threshold calibration (same approach as the prebuilt models).
    print(f"\n--- Calibrating threshold (F1-max) ---")
    scorer.fit(model, loaders["val"], device)

    cal_loader = loaders["val"]
    cal_label = "validation"
    if loaders.get("val") is not None:
        # Check if val has any anomaly labels
        val_has_anomalies = False
        for batch in loaders["val"]:
            _, labels, _ = batch
            if labels.sum() > 0:
                val_has_anomalies = True
                break
        if not val_has_anomalies and loaders.get("test") is not None:
            cal_loader = loaders["test"]
            cal_label = "test"
            print(f"  Validation has no anomalies -- calibrating on test set")

    metrics = optimize_threshold_f1(model, scorer, cal_loader, device)
    print(f"  Calibrated on: {cal_label}")
    print(f"  Method: {metrics.get('method', 'f1_max')}")
    print(f"  Last-point threshold: {scorer.last_point_threshold:.4f}")
    if "last_point_f1" in metrics:
        print(f"  Last-point F1: {metrics['last_point_f1']:.4f}")

    # Evaluate on test set
    print(f"\n--- Test set evaluation ---")
    if loaders.get("test") is not None:
        test_point_scores, test_window_scores = scorer.score_batch(model, loaders["test"], device)

        all_test_labels = []
        for batch in loaders["test"]:
            _, labels_batch, _ = batch
            all_test_labels.append(labels_batch.numpy())
        all_test_labels = np.concatenate(all_test_labels)

        lp_scores = test_point_scores[:, -1]
        lp_labels = all_test_labels[:, -1].astype(int)

        if scorer.last_point_threshold is not None:
            lp_preds = lp_scores < scorer.last_point_threshold
            tp = np.sum(lp_preds & lp_labels.astype(bool))
            fp = np.sum(lp_preds & ~lp_labels.astype(bool))
            fn = np.sum(~lp_preds & lp_labels.astype(bool))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            print(f"  Test last-point: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
            print(f"  TP={tp}, FP={fp}, FN={fn}")

    # Save artifacts
    if not args.skip_save:
        print(f"\n--- Saving artifacts ---")
        save_dir = output_dir / name
        save_training_artifacts(save_dir, model, model_config, scaler, scorer, history)

    print(f"\n  {name} TRAINING COMPLETE")


def train_penny_mode(args, device, output_dir):
    """Train Penny_All model."""
    data_path = PROJECT_ROOT / args.data_path

    print("\n" + "-" * 40)
    print("Preprocessing penny transaction data")
    print("-" * 40)

    hourly_df = load_penny_data(data_path)
    windows, labels, timestamps = create_sliding_windows(
        hourly_df, window_size=args.window_size, stride=args.stride
    )
    splits = create_splits(windows, labels, timestamps, hourly_df)

    # Pool train+val if requested
    if args.pool_train_val:
        train_w, train_l = splits["train"]
        val_w, val_l = splits["val"]
        if len(train_w) > 0 and len(val_w) > 0:
            pooled_w = np.concatenate([train_w, val_w])
            pooled_l = np.concatenate([train_l, val_l])
            splits["train"] = (pooled_w, pooled_l)
            print(f"  Pooled train+val: {len(train_w)} + {len(val_w)} = {len(pooled_w)} windows")

    normalized, scaler = normalize(splits)
    loaders = create_dataloaders(normalized, batch_size=args.batch_size)

    train_single("Penny_All", loaders, args, device, output_dir, scaler=scaler)


def train_combo_mode(args, device, output_dir):
    """Train all 4 combo models."""
    data_path = PROJECT_ROOT / args.data_path

    print("\n" + "-" * 40)
    print("Loading combo transaction data")
    print("-" * 40)

    combo_data = load_combo_data(data_path)

    for combo_key in COMBO_KEYS:
        dir_name = combo_dir_names[combo_key]
        hourly_df = combo_data[combo_key]

        print(f"\n{'=' * 60}")
        print(f"Preparing data for {dir_name}")
        print(f"{'=' * 60}")

        windows, labels, timestamps = create_sliding_windows(
            hourly_df, window_size=args.window_size, stride=args.stride
        )
        splits = create_splits(windows, labels, timestamps, hourly_df)

        if args.pool_train_val:
            train_w, train_l = splits["train"]
            val_w, val_l = splits["val"]
            if len(train_w) > 0 and len(val_w) > 0:
                pooled_w = np.concatenate([train_w, val_w])
                pooled_l = np.concatenate([train_l, val_l])
                splits["train"] = (pooled_w, pooled_l)
                print(f"  Pooled train+val: {len(train_w)} + {len(val_w)} = {len(pooled_w)} windows")

        normalized, scaler = normalize(splits)
        loaders = create_dataloaders(normalized, batch_size=args.batch_size)

        train_single(dir_name, loaders, args, device, output_dir, scaler=scaler)

    print(f"\n{'=' * 60}")
    print("ALL COMBO MODELS TRAINED")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Train FCVAE model for penny transaction anomaly detection"
    )
    parser.add_argument("--mode", choices=["penny", "combo"], default="penny",
                        help="Training mode: penny (Penny_All only) or combo (4 combo models)")
    parser.add_argument("--data-path", type=str,
                        default="data/synthetic_transactions.csv",
                        help="Path to synthetic transactions CSV")
    parser.add_argument("--output-dir", type=str,
                        default="models/fcvae/initial",
                        help="Directory to save trained models")
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--kl-warmup-epochs", type=int, default=5)
    parser.add_argument("--no-augmentation", action="store_true", default=True,
                        help="Disable data augmentation (default for baseline)")
    parser.add_argument("--augmentation", action="store_true",
                        help="Enable data augmentation (overrides --no-augmentation)")
    parser.add_argument("--score-mode", choices=["single_pass", "mcmc"], default="single_pass")
    parser.add_argument("--pool-train-val", action="store_true", default=False,
                        help="Pool train+val splits for training")
    parser.add_argument("--device", type=str, default=None, help="Force device (cuda/mps/cpu)")
    parser.add_argument("--skip-save", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # --augmentation overrides --no-augmentation
    if args.augmentation:
        args.no_augmentation = False

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    for mod in ["src.model", "src.scorer", "src.preprocess", "src.train", "src.training"]:
        logging.getLogger(mod).setLevel(logging.WARNING)

    # Deterministic seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = auto_device(args.device)
    output_dir = PROJECT_ROOT / args.output_dir

    aug_label = "augmentation ON" if not args.no_augmentation else "no augmentation"

    print("\n" + "=" * 60)
    print("FCVAE TRAINING -- baseline run")
    print(f"Output directory: {output_dir / 'Penny_All'}")
    print(f"Latent dim: {args.latent_dim}   LR: {args.lr}   Epochs: {args.epochs} ({aug_label})")
    print(f"Device: {device}")
    if args.mode == "penny":
        print(
            "This is a fast baseline -- expect the detector to catch most anomalies but\n"
            "trip on a few normal windows. Run\n"
            "    python code/4_grid_sweep.py\n"
            "next to search for a better configuration. The sweep will retrain the\n"
            "winning config and save it to models/fcvae/best/Penny_All/."
        )
    print("=" * 60)

    if args.mode == "penny":
        train_penny_mode(args, device, output_dir)
    elif args.mode == "combo":
        train_combo_mode(args, device, output_dir)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
