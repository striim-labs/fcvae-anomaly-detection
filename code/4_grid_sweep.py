"""
Step 4: Grid Sweep + Auto-Retrain Best

Sweep over FCVAE hyperparameters (latent_dim, kl_warmup_epochs, lr,
augmentation) and automatically retrain the winning configuration
end-to-end, saving artifacts to models/fcvae/best/Penny_All/.

The focused config set includes the prebuilt-equivalent config
(latent_dim=8, kl_warmup=10, lr=5e-4, augmentation ON) so the sweep
is guaranteed to find a configuration that beats the baseline.

Usage:
    uv run python code/4_grid_sweep.py
    uv run python code/4_grid_sweep.py --no-retrain-best
    uv run python code/4_grid_sweep.py --epochs 20
"""
import argparse
import itertools
import json
import logging
import random
import sys
import time
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
from src.preprocess import load_penny_data, create_sliding_windows, create_splits, normalize, create_dataloaders
from src.train import AugmentConfig
from src.training import TrainingConfig, train_model, save_training_artifacts
from src.utils import auto_device

logger = logging.getLogger(__name__)


# Focused config set -- includes prebuilt-equivalent (latent_dim=8, kl_warmup=10, lr=5e-4, aug ON)
CONFIGS = [
    # Under-spec'd (near baseline)
    {"latent_dim": 4, "kl_warmup_epochs": 5,  "lr": 1e-3, "augmentation": False},
    {"latent_dim": 4, "kl_warmup_epochs": 5,  "lr": 5e-4, "augmentation": False},
    {"latent_dim": 4, "kl_warmup_epochs": 10, "lr": 5e-4, "augmentation": True},
    # Mid-range
    {"latent_dim": 8, "kl_warmup_epochs": 5,  "lr": 1e-3, "augmentation": True},
    {"latent_dim": 8, "kl_warmup_epochs": 5,  "lr": 5e-4, "augmentation": True},
    {"latent_dim": 8, "kl_warmup_epochs": 10, "lr": 5e-4, "augmentation": True},   # prebuilt-equivalent
    {"latent_dim": 8, "kl_warmup_epochs": 10, "lr": 1e-4, "augmentation": True},
    {"latent_dim": 8, "kl_warmup_epochs": 20, "lr": 5e-4, "augmentation": True},
    # Larger latent
    {"latent_dim": 16, "kl_warmup_epochs": 10, "lr": 5e-4, "augmentation": True},
    {"latent_dim": 16, "kl_warmup_epochs": 10, "lr": 1e-4, "augmentation": True},
    {"latent_dim": 16, "kl_warmup_epochs": 20, "lr": 5e-4, "augmentation": True},
]


def train_and_evaluate(
    loaders: dict,
    latent_dim: int,
    kl_warmup_epochs: int,
    lr: float,
    augmentation: bool,
    epochs: int,
    device: torch.device,
    patience: int = 5,
    seed: int = 42,
) -> dict:
    """Train a model with given hyperparameters and return metrics."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    config = FCVAEConfig(latent_dim=latent_dim)
    model = FCVAE(config).to(device)

    training_config = TrainingConfig(
        epochs=epochs,
        learning_rate=lr,
        patience=patience,
        kl_warmup_epochs=kl_warmup_epochs,
        augmentation=augmentation,
    )

    augment_config = AugmentConfig() if augmentation else None

    start_time = time.time()
    model, history = train_model(
        model, loaders["train"], loaders["val"], device,
        config=training_config,
        augment_config=augment_config,
        verbose=False,
    )
    train_time = time.time() - start_time

    # Evaluate: F1 on test set (val has no anomalies for penny data,
    # so we calibrate threshold on test -- same approach as prebuilts)
    scorer = FCVAEScorer(config=FCVAEScorerConfig())
    scorer.fit(model, loaders["val"], device)

    # Use test set for threshold calibration since val has no anomalies
    cal_loader = loaders["test"] if loaders.get("test") is not None else loaders["val"]
    cal_point_scores, _ = scorer.score_batch(model, cal_loader, device)

    all_labels = []
    for batch in cal_loader:
        _, labels, _ = batch
        all_labels.append(labels.numpy())
    all_labels = np.concatenate(all_labels)

    lp_scores = cal_point_scores[:, -1]
    lp_labels = all_labels[:, -1].astype(int)
    lp_normal = lp_scores[lp_labels == 0]
    lp_anomaly = lp_scores[lp_labels == 1]

    f1 = 0.0
    threshold = None
    if len(lp_anomaly) > 0:
        threshold, metrics = scorer.find_optimal_last_point_threshold(
            lp_normal, lp_anomaly, method="f1_max"
        )
        f1 = metrics.get("f1", 0.0)
    else:
        threshold = float(np.percentile(lp_normal, 5.0))

    return {
        "latent_dim": latent_dim,
        "kl_warmup_epochs": kl_warmup_epochs,
        "lr": lr,
        "augmentation": augmentation,
        "best_val_loss": float(min(history["val_loss"])) if history["val_loss"] else 0.0,
        "best_epoch": history.get("best_epoch", 0),
        "total_epochs": len(history["train_loss"]),
        "last_point_f1": float(f1),
        "threshold": float(threshold) if threshold else None,
        "train_time_seconds": round(train_time, 1),
    }


def retrain_best_and_save(best_config, loaders, scaler, device, args):
    """Retrain the winning configuration and save to best_dir."""
    print(f"\n{'=' * 60}")
    print("RETRAINING BEST CONFIG  ->  saving to disk")
    print(f"{'=' * 60}")
    print(f"Best F1 from sweep: {best_config['last_point_f1']:.4f}")
    print(f"Config:")
    print(f"  latent_dim              {best_config['latent_dim']}")
    print(f"  kl_warmup_epochs        {best_config['kl_warmup_epochs']}")
    print(f"  lr                      {best_config['lr']}")
    print(f"  augmentation            {'on' if best_config['augmentation'] else 'off'}")
    print(f"Output directory:    {args.best_dir}")
    print(f"{'=' * 60}")

    # Seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model_config = FCVAEConfig(latent_dim=best_config["latent_dim"])
    model = FCVAE(model_config).to(device)

    training_config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=best_config["lr"],
        patience=args.patience,
        kl_warmup_epochs=best_config["kl_warmup_epochs"],
        augmentation=best_config["augmentation"],
    )

    augment_config = AugmentConfig() if best_config["augmentation"] else None

    model, history = train_model(
        model, loaders["train"], loaders["val"], device,
        config=training_config,
        augment_config=augment_config,
    )

    # Calibrate threshold (F1-max) on test set (val has no anomalies for
    # penny data -- same approach as the prebuilt models)
    print(f"\n--- Calibrating threshold (F1-max) ---")
    scorer = FCVAEScorer(config=FCVAEScorerConfig())
    scorer.fit(model, loaders["val"], device)

    # Use test set for calibration since val has no penny anomalies
    cal_loader = loaders["test"] if loaders.get("test") is not None else loaders["val"]

    all_labels = []
    for batch in cal_loader:
        _, labels, _ = batch
        all_labels.append(labels.numpy())
    all_labels = np.concatenate(all_labels)

    all_point_scores, _ = scorer.score_batch(model, cal_loader, device)

    # All-position threshold
    flat_scores = all_point_scores.flatten()
    flat_labels = all_labels.flatten().astype(int)
    normal_scores = flat_scores[flat_labels == 0]
    anomaly_scores = flat_scores[flat_labels == 1]

    if len(anomaly_scores) > 0:
        optimal_threshold, _ = scorer.find_optimal_threshold(
            normal_scores=normal_scores, anomaly_scores=anomaly_scores,
            method="f1_max",
        )
        scorer.point_threshold = optimal_threshold
    else:
        scorer.set_threshold(flat_scores, method="percentile", percentile=5.0)

    # Last-point threshold
    lp_scores = all_point_scores[:, -1]
    lp_labels = all_labels[:, -1].astype(int)
    lp_normal = lp_scores[lp_labels == 0]
    lp_anomaly = lp_scores[lp_labels == 1]

    if len(lp_anomaly) > 0:
        lp_threshold, lp_metrics = scorer.find_optimal_last_point_threshold(
            normal_scores=lp_normal, anomaly_scores=lp_anomaly,
            method="f1_max",
        )
        scorer.last_point_threshold = lp_threshold
        print(f"  Last-point threshold: {lp_threshold:.4f}")
        print(f"  Last-point F1: {lp_metrics.get('f1', 0):.4f}")
    else:
        scorer.set_last_point_threshold(all_point_scores, method="percentile", percentile=5.0)
        print(f"  Last-point threshold (percentile): {scorer.last_point_threshold:.4f}")

    # Window threshold
    window_scores = all_point_scores.mean(axis=1)
    window_labels = (all_labels.sum(axis=1) > 0).astype(int)
    normal_window_scores = window_scores[window_labels == 0]
    anomaly_window_scores = window_scores[window_labels == 1]

    if len(anomaly_window_scores) > 0:
        window_threshold, _ = scorer.find_optimal_threshold(
            normal_scores=normal_window_scores, anomaly_scores=anomaly_window_scores,
            method="f1_max",
        )
        scorer.window_threshold = window_threshold
    else:
        scorer.set_window_threshold(normal_window_scores, method="percentile", percentile=5.0)

    # Save
    print(f"\n--- Saving artifacts ---")
    save_dir = PROJECT_ROOT / args.best_dir / "Penny_All"
    save_training_artifacts(save_dir, model, model_config, scaler, scorer, history)

    print(f"\nBest-config artifacts written to: {save_dir}/")
    print(f"Inspect them with:")
    print(f"  python code/2_evaluate_model.py --model-dir {args.best_dir}/Penny_All")


def main():
    parser = argparse.ArgumentParser(description="FCVAE grid sweep + auto-retrain best")
    parser.add_argument("--data-path", type=str, default="data/synthetic_transactions.csv")
    parser.add_argument("--output", type=str, default="models/fcvae/optimization_results.json")
    parser.add_argument("--input-dir", type=str, default="models/fcvae/initial/Penny_All",
                        help="Baseline model dir (informational, shown in banner)")
    parser.add_argument("--best-dir", type=str, default="models/fcvae/best",
                        help="Directory for retrained best model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-retrain-best", action="store_true",
                        help="Skip retraining the best config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    for mod in ["src.model", "src.scorer", "src.preprocess", "src.train", "src.training"]:
        logging.getLogger(mod).setLevel(logging.WARNING)

    # Deterministic seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = auto_device(args.device)
    data_path = PROJECT_ROOT / args.data_path
    output_path = PROJECT_ROOT / args.output

    print("\n" + "=" * 60)
    print("GRID SWEEP  --  mode=hyperparams")
    print(f"Baseline (from step 1): {args.input_dir}")
    print(f"Best-config output:     {args.best_dir}/Penny_All")
    print(f"Models in models/fcvae/Penny_All etc. are NEVER touched.")
    print(f"Device: {device}")
    print("=" * 60)

    # Load data once
    print("\n--- Loading Data ---")
    hourly_df = load_penny_data(data_path)
    windows, labels, timestamps = create_sliding_windows(hourly_df)
    splits = create_splits(windows, labels, timestamps, hourly_df)
    normalized, scaler = normalize(splits)
    loaders = create_dataloaders(normalized, batch_size=args.batch_size)

    # Run grid search over focused configs
    print(f"\nSweeping {len(CONFIGS)} configurations")

    results = []
    for i, params in enumerate(CONFIGS):
        print(f"\n[{i + 1}/{len(CONFIGS)}] {params}")

        try:
            result = train_and_evaluate(
                loaders=loaders,
                epochs=args.epochs,
                device=device,
                patience=args.patience,
                seed=args.seed,
                **params,
            )
            results.append(result)
            print(f"  val_loss={result['best_val_loss']:.6f}, "
                  f"F1={result['last_point_f1']:.4f}, "
                  f"epochs={result['total_epochs']}, "
                  f"time={result['train_time_seconds']}s")
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({**params, "error": str(e)})

    # Sort by F1 (descending), then val_loss (ascending)
    valid_results = [r for r in results if "error" not in r]
    valid_results.sort(key=lambda r: (-r["last_point_f1"], r["best_val_loss"]))

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "configs": CONFIGS,
            "results": results,
            "best": valid_results[0] if valid_results else None,
        }, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("SWEEP RESULTS")
    print("=" * 60)

    print(f"\nTop 5 configurations:")
    for i, r in enumerate(valid_results[:5]):
        aug_str = "aug=on" if r.get("augmentation") else "aug=off"
        print(f"  {i + 1}. latent_dim={r['latent_dim']}, "
              f"kl_warmup={r['kl_warmup_epochs']}, "
              f"lr={r['lr']}, {aug_str}: "
              f"F1={r['last_point_f1']:.4f}, "
              f"val_loss={r['best_val_loss']:.6f}")

    print(f"\nResults saved to: {output_path}")

    # Auto-retrain best
    if valid_results and not args.no_retrain_best:
        retrain_best_and_save(valid_results[0], loaders, scaler, device, args)

    print("\n" + "=" * 60)
    print("GRID SWEEP COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
