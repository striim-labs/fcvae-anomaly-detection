"""
Step 6: Hyperparameter Optimization

Grid search over FCVAE hyperparameters: latent_dim, kl_warmup_epochs,
n_samples, learning rate. Saves results to a JSON file.

Usage:
    uv run python code/6_optimize.py
    uv run python code/6_optimize.py --epochs 20 --quick
"""
import argparse
import itertools
import json
import logging
import sys
import time
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


# Default search grid
GRID = {
    "latent_dim": [4, 8, 16],
    "kl_warmup_epochs": [5, 10, 20],
    "lr": [1e-4, 5e-4, 1e-3],
}

QUICK_GRID = {
    "latent_dim": [4, 8],
    "kl_warmup_epochs": [5, 10],
    "lr": [5e-4, 1e-3],
}


def train_and_evaluate(
    loaders: dict,
    latent_dim: int,
    kl_warmup_epochs: int,
    lr: float,
    epochs: int,
    device: torch.device,
    patience: int = 5,
) -> dict:
    """Train a model with given hyperparameters and return metrics."""
    config = FCVAEConfig(latent_dim=latent_dim)
    model = FCVAE(config).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    early_stopping = EarlyStopping(patience=patience)
    augment_config = AugmentConfig()

    start_time = time.time()

    for epoch in range(epochs):
        kl_weight = compute_kl_weight(epoch, kl_warmup_epochs)

        train_loss = train_epoch(
            model, loaders["train"], optimizer, kl_weight, device,
            augment_config=augment_config
        )
        val_loss = validate_epoch(model, loaders["val"], device)
        scheduler.step()

        if early_stopping.step(val_loss, model, epoch + 1):
            break

    early_stopping.restore_best(model, device)
    train_time = time.time() - start_time

    # Evaluate: F1 on validation last-point
    scorer = FCVAEScorer(config=FCVAEScorerConfig())
    scorer.fit(model, loaders["val"], device)

    val_point_scores, _ = scorer.score_batch(model, loaders["val"], device)

    all_labels = []
    for batch in loaders["val"]:
        _, labels, _ = batch
        all_labels.append(labels.numpy())
    all_labels = np.concatenate(all_labels)

    lp_scores = val_point_scores[:, -1]
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
        "best_val_loss": float(early_stopping.best_loss),
        "best_epoch": early_stopping.best_epoch,
        "total_epochs": epoch + 1,
        "last_point_f1": float(f1),
        "threshold": float(threshold) if threshold else None,
        "train_time_seconds": round(train_time, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="FCVAE hyperparameter optimization")
    parser.add_argument("--data-path", type=str, default="data/synthetic_transactions.csv")
    parser.add_argument("--output", type=str, default="models/fcvae/optimization_results.json")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--quick", action="store_true", help="Use smaller search grid")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    print("\n" + "=" * 60)
    print("FCVAE HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)

    device = auto_device(args.device)
    data_path = PROJECT_ROOT / args.data_path
    output_path = PROJECT_ROOT / args.output

    print(f"\nDevice: {device}")

    # Load data once
    print("\n--- Loading Data ---")
    hourly_df = load_penny_data(data_path)
    windows, labels, timestamps = create_sliding_windows(hourly_df)
    splits = create_splits(windows, labels, timestamps, hourly_df)
    normalized, scaler = normalize(splits)
    loaders = create_dataloaders(normalized, batch_size=args.batch_size)

    # Build parameter grid
    grid = QUICK_GRID if args.quick else GRID
    param_names = list(grid.keys())
    param_values = list(grid.values())
    combos = list(itertools.product(*param_values))

    print(f"\nGrid: {len(combos)} configurations")
    for name, values in grid.items():
        print(f"  {name}: {values}")

    # Run grid search
    results = []
    for i, combo in enumerate(combos):
        params = dict(zip(param_names, combo))
        print(f"\n[{i + 1}/{len(combos)}] {params}")

        try:
            result = train_and_evaluate(
                loaders=loaders,
                epochs=args.epochs,
                device=device,
                patience=args.patience,
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
        json.dump({"grid": grid, "results": results, "best": valid_results[0] if valid_results else None}, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)

    print(f"\nTop 5 configurations:")
    for i, r in enumerate(valid_results[:5]):
        print(f"  {i + 1}. latent_dim={r['latent_dim']}, "
              f"kl_warmup={r['kl_warmup_epochs']}, "
              f"lr={r['lr']}: "
              f"F1={r['last_point_f1']:.4f}, "
              f"val_loss={r['best_val_loss']:.6f}")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
