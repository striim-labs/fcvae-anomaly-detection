"""
Training Utilities for FCVAE

Includes KL annealing, training/validation epoch functions, early stopping,
and data augmentation (point anomalies, segment swaps, missing data injection).
"""
import logging
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Augmentation (from FCVAE/data_augment.py, Wang et al. WWW 2024)
# ---------------------------------------------------------------------------

@dataclass
class AugmentConfig:
    """Configuration for FCVAE data augmentation."""
    missing_data_rate: float = 0.01
    point_ano_rate: float = 0.05
    seg_ano_rate: float = 0.1


def missing_data_injection(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    rate: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly zero out points to simulate missing data."""
    miss_size = int(rate * x.shape[0] * x.shape[1] * x.shape[2])
    if miss_size == 0:
        return x, y, z

    row = torch.randint(low=0, high=x.shape[0], size=(miss_size,), device=x.device)
    col = torch.randint(low=0, high=x.shape[2], size=(miss_size,), device=x.device)
    x[row, :, col] = 0
    z[row, col] = 1
    return x, y, z


def point_ano(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    rate: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create augmented samples with point anomalies at the last timestep."""
    aug_size = int(rate * x.shape[0])
    if aug_size == 0:
        return x[:0], y[:0], z[:0]

    id_x = torch.randint(low=0, high=x.shape[0], size=(aug_size,), device=x.device)
    x_aug = x[id_x].clone()
    y_aug = y[id_x].clone()
    z_aug = z[id_x].clone()

    if x_aug.shape[1] == 1:
        half = aug_size // 2
        ano_noise1 = torch.randint(low=1, high=20, size=(half,), device=x.device)
        ano_noise2 = torch.randint(low=-20, high=-1, size=(aug_size - half,), device=x.device)
        ano_noise = torch.cat((ano_noise1, ano_noise2), dim=0).float() / 2

        x_aug[:, 0, -1] += ano_noise

        y_aug[:, -1] = torch.logical_or(
            y_aug[:, -1].bool(),
            torch.ones_like(y_aug[:, -1], dtype=torch.bool)
        ).to(y_aug.dtype)

    return x_aug, y_aug, z_aug


def seg_ano(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    rate: float,
    method: str = "swap"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create augmented samples by swapping tail segments between windows."""
    aug_size = int(rate * x.shape[0])
    if aug_size == 0 or x.shape[0] < 2:
        return x[:0], y[:0], z[:0]

    idx_1 = torch.arange(aug_size, device=x.device)
    idx_2 = torch.arange(aug_size, device=x.device)
    max_attempts = 100
    attempts = 0
    while torch.any(idx_1 == idx_2) and attempts < max_attempts:
        idx_1 = torch.randint(low=0, high=x.shape[0], size=(aug_size,), device=x.device)
        idx_2 = torch.randint(low=0, high=x.shape[0], size=(aug_size,), device=x.device)
        attempts += 1

    x_aug = x[idx_1].clone()
    y_aug = y[idx_1].clone()
    z_aug = z[idx_1].clone()

    window_size = x.shape[2]
    min_start = min(6, window_size // 4)
    max_start = max(min_start + 1, window_size - 6)

    time_start = torch.randint(
        low=min_start,
        high=max_start,
        size=(aug_size,),
        device=x.device
    )

    for i in range(len(idx_2)):
        if method == "swap":
            start = time_start[i].item()
            x_aug[i, :, start:] = x[idx_2[i], :, start:]
            y_aug[i, start:] = torch.logical_or(
                y_aug[i, start:].bool(),
                torch.ones_like(y_aug[i, start:], dtype=torch.bool)
            ).to(y_aug.dtype)

    return x_aug, y_aug, z_aug


def batch_augment(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    config: AugmentConfig
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply all augmentations to a batch, concatenating augmented samples.

    Order: point anomalies -> segment swaps -> missing data injection.
    """
    if config.point_ano_rate > 0:
        x_a, y_a, z_a = point_ano(x, y, z, config.point_ano_rate)
        if x_a.shape[0] > 0:
            x = torch.cat((x, x_a), dim=0)
            y = torch.cat((y, y_a), dim=0)
            z = torch.cat((z, z_a), dim=0)

    if config.seg_ano_rate > 0:
        x_a, y_a, z_a = seg_ano(x, y, z, config.seg_ano_rate, method="swap")
        if x_a.shape[0] > 0:
            x = torch.cat((x, x_a), dim=0)
            y = torch.cat((y, y_a), dim=0)
            z = torch.cat((z, z_a), dim=0)

    x, y, z = missing_data_injection(x, y, z, config.missing_data_rate)

    return x, y, z


# ---------------------------------------------------------------------------
# Training Utilities
# ---------------------------------------------------------------------------

def compute_kl_weight(epoch: int, warmup_epochs: int) -> float:
    """Compute KL divergence weight for annealing.

    Linear warmup from 0 to 1 over warmup_epochs.
    """
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, (epoch + 1) / warmup_epochs)


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    kl_weight: float,
    device: torch.device,
    augment_config: Optional[AugmentConfig] = None,
    grad_clip: float = 2.0,
) -> float:
    """Run one training epoch.

    Args:
        model: FCVAE model
        loader: Training DataLoader
        optimizer: Optimizer
        kl_weight: Current KL annealing weight
        device: Device
        augment_config: If provided, apply data augmentation
        grad_clip: Gradient clipping max value (0 to disable)

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x, y, z = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        else:
            x = batch.to(device)
            y = torch.zeros(x.shape[0], x.shape[2], device=device)
            z = torch.zeros_like(y)

        if augment_config is not None:
            x, y, z = batch_augment(x, y, z, augment_config)

        optimizer.zero_grad()
        mu_x, var_x, rec_x, mu, var, loss = model(x, mode="train", kl_weight=kl_weight)

        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Run one validation epoch.

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)

            mu_x, var_x, rec_x, mu, var, loss = model(x, mode="valid")
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


class EarlyStopping:
    """Track validation loss and stop when no improvement for `patience` epochs."""

    def __init__(self, patience: int = 5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.best_epoch = 0
        self._best_state = None

    def step(self, val_loss: float, model: torch.nn.Module, epoch: int) -> bool:
        """Update with new validation loss.

        Args:
            val_loss: Current epoch validation loss
            model: Model to snapshot if this is the best epoch
            epoch: Current epoch number (1-based)

        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - 1e-6:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self._best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def restore_best(self, model: torch.nn.Module, device: torch.device) -> None:
        """Restore model to best checkpoint."""
        if self._best_state is not None:
            model.load_state_dict(self._best_state)
            model.to(device)
            logger.info(
                f"Restored best model from epoch {self.best_epoch} "
                f"(val_loss={self.best_loss:.6f})"
            )
