"""
Step 3: Train FCVAE Model for Penny Transactions

Full training pipeline: load data -> create windows -> train with KL annealing
+ augmentation -> calibrate threshold (F1-max) -> save artifacts.

Usage:
    uv run python code/3_train_model.py
    uv run python code/3_train_model.py --epochs 50 --lr 5e-4
    uv run python code/3_train_model.py --data-path data/synthetic_transactions.csv
"""
import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import FCVAE, FCVAEConfig
from src.scorer import FCVAEScorer, FCVAEScorerConfig
from src.preprocess import load_penny_data, create_sliding_windows, create_splits, normalize, create_dataloaders
from src.train import (
    AugmentConfig, compute_kl_weight, train_epoch, validate_epoch,
    EarlyStopping,
)
from src.utils import auto_device

logger = logging.getLogger(__name__)


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

    logger.info(f"  Validation: {len(normal_scores)} normal points, {len(anomaly_scores)} anomaly points")

    if len(anomaly_scores) == 0:
        logger.warning("  No anomaly points in validation — falling back to percentile threshold")
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

    logger.info(f"  Last-point: {len(lp_normal)} normal, {len(lp_anomaly)} anomaly")

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
        logger.info(
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

    logger.info(
        f"  Thresholds: all-position={scorer.point_threshold:.4f}, "
        f"last-point={scorer.last_point_threshold:.4f}, "
        f"window={scorer.window_threshold}"
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train FCVAE model for penny transaction anomaly detection"
    )
    parser.add_argument("--data-path", type=str,
                        default="data/synthetic_transactions.csv",
                        help="Path to synthetic transactions CSV")
    parser.add_argument("--output-dir", type=str,
                        default="models/fcvae",
                        help="Directory to save trained models")
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--kl-warmup-epochs", type=int, default=10)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--score-mode", choices=["single_pass", "mcmc"], default="single_pass")
    parser.add_argument("--pool-train-val", action="store_true", default=False,
                        help="Pool train+val splits for training")
    parser.add_argument("--device", type=str, default=None, help="Force device (cuda/mps/cpu)")
    parser.add_argument("--skip-save", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    print("\n" + "=" * 60)
    print("FCVAE PENNY TRANSACTION ANOMALY DETECTION")
    print("Training Pipeline")
    print("=" * 60)

    device = auto_device(args.device)
    print(f"\nDevice: {device}")

    # Resolve paths relative to project root
    data_path = PROJECT_ROOT / args.data_path
    output_dir = PROJECT_ROOT / args.output_dir

    # Step 1: Preprocess data
    print("\n" + "-" * 40)
    print("Step 1: Preprocessing penny transaction data")
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

    for split_name in ["train", "val", "test"]:
        if loaders.get(split_name) is not None:
            print(f"  {split_name}: {len(loaders[split_name].dataset)} windows")

    # Step 2: Initialize model
    print("\n" + "-" * 40)
    print("Step 2: Initializing FCVAE model")
    print("-" * 40)

    model_config = FCVAEConfig(window=args.window_size, latent_dim=args.latent_dim)
    model = FCVAE(model_config).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")
    print(f"  Latent dim: {args.latent_dim}, Window: {args.window_size}")

    scorer_config = FCVAEScorerConfig(score_mode=args.score_mode)
    scorer = FCVAEScorer(config=scorer_config)

    augment_config = AugmentConfig() if not args.no_augmentation else AugmentConfig(
        missing_data_rate=0.0, point_ano_rate=0.0, seg_ano_rate=0.0
    )

    if not args.no_augmentation:
        print(f"  Augmentation: point={augment_config.point_ano_rate}, "
              f"segment={augment_config.seg_ano_rate}, "
              f"missing={augment_config.missing_data_rate}")

    # Step 3: Train
    print("\n" + "-" * 40)
    print("Step 3: Training FCVAE model")
    print("-" * 40)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    early_stopping = EarlyStopping(patience=args.patience)

    history = {
        "train_loss": [],
        "val_loss": [],
        "learning_rates": [],
        "kl_weights": [],
        "best_epoch": 0,
    }

    for epoch in range(args.epochs):
        kl_weight = compute_kl_weight(epoch, args.kl_warmup_epochs)

        train_loss = train_epoch(
            model, loaders["train"], optimizer, kl_weight, device,
            augment_config=augment_config, grad_clip=args.grad_clip
        )

        val_loss = validate_epoch(model, loaders["val"], device)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rates"].append(optimizer.param_groups[0]["lr"])
        history["kl_weights"].append(kl_weight)

        should_stop = early_stopping.step(val_loss, model, epoch + 1)

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == args.epochs - 1:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch + 1:3d}: train={train_loss:.6f}, "
                f"val={val_loss:.6f}, lr={lr:.2e}, kl_w={kl_weight:.2f}"
            )

        if should_stop:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    early_stopping.restore_best(model, device)
    history["best_epoch"] = early_stopping.best_epoch
    print(f"  Best val loss: {early_stopping.best_loss:.6f} at epoch {early_stopping.best_epoch}")

    # Step 4: Calibrate threshold
    print("\n" + "-" * 40)
    print("Step 4: Calibrating threshold (F1-max)")
    print("-" * 40)

    scorer.fit(model, loaders["val"], device)

    if loaders.get("val") is not None:
        metrics = optimize_threshold_f1(model, scorer, loaders["val"], device)
        print(f"  Method: {metrics.get('method', 'f1_max')}")
        print(f"  Last-point threshold: {scorer.last_point_threshold:.4f}")
        if "last_point_f1" in metrics:
            print(f"  Last-point F1: {metrics['last_point_f1']:.4f}")

    # Step 5: Evaluate on test set
    print("\n" + "-" * 40)
    print("Step 5: Quick evaluation on test set")
    print("-" * 40)

    if loaders.get("test") is not None:
        test_point_scores, test_window_scores = scorer.score_batch(model, loaders["test"], device)

        # Collect test labels
        all_test_labels = []
        for batch in loaders["test"]:
            _, labels_batch, _ = batch
            all_test_labels.append(labels_batch.numpy())
        all_test_labels = np.concatenate(all_test_labels)

        # Last-point metrics
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

    # Step 6: Save artifacts
    if not args.skip_save:
        print("\n" + "-" * 40)
        print("Step 6: Saving artifacts")
        print("-" * 40)

        save_dir = output_dir / "Penny_All"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": model_config,
        }, save_dir / "model.pt")

        # Save scaler
        with open(save_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        # Save scorer
        scorer.save(save_dir / "scorer.pkl")

        # Save history
        with open(save_dir / "history.pkl", "wb") as f:
            pickle.dump(history, f)

        print(f"  Saved to: {save_dir}/")
        print(f"    - model.pt")
        print(f"    - scaler.pkl")
        print(f"    - scorer.pkl")
        print(f"    - history.pkl")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
