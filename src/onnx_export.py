"""
ONNX Export for FCVAE Models

Provides a deterministic inference wrapper and export utility for converting
trained FCVAE models to ONNX format.

The wrapper replaces reparameterization sampling with z = mu (posterior mean),
collapsing the n_samples averaging loop to a single deterministic forward pass.
"""
import logging
from pathlib import Path

import torch
import torch.nn as nn

from src.model import FCVAE

logger = logging.getLogger(__name__)


class FCVAEInferenceWrapper(nn.Module):
    """Deterministic inference-only wrapper for ONNX export.

    Replaces reparameterization sampling with z = mu (posterior mean),
    collapsing the n_samples averaging loop to a single forward pass.
    """

    def __init__(self, model: FCVAE):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic forward pass returning per-point NLL scores.

        Args:
            x: Input tensor (B, 1, W)

        Returns:
            nll: Per-point negative log-likelihood (B, W).
                 Lower (more negative) = more anomalous.
        """
        condition = self.model.get_condition(x)
        mu, var = self.model.encode(torch.cat((x, condition), dim=2))
        z = mu  # deterministic — no reparameterization sampling
        mu_x, var_x = self.model.decode(
            torch.cat((z, condition.squeeze(1)), dim=1)
        )
        nll = -0.5 * (torch.log(var_x) + (x - mu_x) ** 2 / var_x)
        return nll.squeeze(1)


def export_to_onnx(
    model: FCVAE,
    out_path: Path,
    window: int = 24,
    opset_version: int = 18,
) -> Path:
    """Export an FCVAE model to ONNX format.

    Args:
        model: Trained FCVAE model.
        out_path: Destination path for the .onnx file.
        window: Input window size (fixed at export time).
        opset_version: ONNX opset version (>=17 required for DFT/FFT).

    Returns:
        The path to the exported .onnx file.
    """
    model.eval()
    wrapper = FCVAEInferenceWrapper(model).eval()
    dummy = torch.randn(1, 1, window)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting FCVAE to ONNX: {out_path} (opset {opset_version})")

    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out_path),
        input_names=["input"],
        output_names=["nll"],
        dynamic_axes={"input": {0: "batch"}, "nll": {0: "batch"}},
        opset_version=opset_version,
    )

    logger.info(f"ONNX export complete: {out_path}")
    return out_path
