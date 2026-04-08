"""Shared utilities for FCVAE anomaly detection."""

import torch


def auto_device(preference: str = None) -> torch.device:
    """Select the best available device.

    Priority: CUDA > MPS > CPU, unless a preference is specified.

    Args:
        preference: Force a specific device ("cuda", "mps", "cpu").

    Returns:
        torch.device for model placement.
    """
    if preference:
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
