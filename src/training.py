"""
Shared Training Helper for FCVAE

Provides train_model() and save_training_artifacts() used by both
code/1_train_model.py and code/4_grid_sweep.py so they cannot drift.
"""
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.train import (
    AugmentConfig,
    EarlyStopping,
    compute_kl_weight,
    train_epoch,
    validate_epoch,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """All knobs for a single training run."""
    epochs: int = 30
    learning_rate: float = 5e-4
    patience: int = 5
    min_delta: float = 1e-6
    kl_warmup_epochs: int = 10
    grad_clip: float = 2.0
    batch_size: int = 64
    augmentation: bool = True


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
    augment_config: Optional[AugmentConfig] = None,
    verbose: bool = True,
) -> Tuple[torch.nn.Module, dict]:
    """Full training loop wrapping src.train helpers.

    Returns:
        (model, history) where model has best weights restored.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    early_stopping = EarlyStopping(patience=config.patience)

    if config.augmentation and augment_config is None:
        augment_config = AugmentConfig()
    elif not config.augmentation:
        augment_config = None

    history = {
        "train_loss": [],
        "val_loss": [],
        "learning_rates": [],
        "kl_weights": [],
        "best_epoch": 0,
    }

    for epoch in range(config.epochs):
        kl_weight = compute_kl_weight(epoch, config.kl_warmup_epochs)

        train_loss = train_epoch(
            model, train_loader, optimizer, kl_weight, device,
            augment_config=augment_config, grad_clip=config.grad_clip,
        )
        val_loss = validate_epoch(model, val_loader, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rates"].append(optimizer.param_groups[0]["lr"])
        history["kl_weights"].append(kl_weight)

        should_stop = early_stopping.step(val_loss, model, epoch + 1)

        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0 or epoch == config.epochs - 1):
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch + 1:3d}: train={train_loss:.6f}, "
                f"val={val_loss:.6f}, lr={lr:.2e}, kl_w={kl_weight:.2f}"
            )

        if should_stop:
            if verbose:
                print(f"  Early stopping at epoch {epoch + 1}")
            break

    early_stopping.restore_best(model, device)
    history["best_epoch"] = early_stopping.best_epoch

    if verbose:
        print(
            f"  Best val loss: {early_stopping.best_loss:.6f} "
            f"at epoch {early_stopping.best_epoch}"
        )

    return model, history


def save_training_artifacts(
    output_dir: Path,
    model: torch.nn.Module,
    model_config,
    scaler,
    scorer,
    history: dict,
) -> None:
    """Persist model.pt, scaler.pkl, scorer.pkl, history.pkl."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {"model_state_dict": model.state_dict(), "config": model_config},
        output_dir / "model.pt",
    )

    if scaler is not None:
        with open(output_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

    scorer.save(output_dir / "scorer.pkl")

    with open(output_dir / "history.pkl", "wb") as f:
        pickle.dump(history, f)

    print(f"  Saved to: {output_dir}/")
    print(f"    - model.pt")
    print(f"    - scaler.pkl")
    print(f"    - scorer.pkl")
    print(f"    - history.pkl")
