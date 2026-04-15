"""
Step 5: Export FCVAE Models to ONNX

Exports trained FCVAE models to ONNX format using a deterministic inference
wrapper (z = mu, no sampling) and validates that ONNX Runtime produces scores
matching PyTorch within tolerance.

For each model, the script produces:
  - model.onnx          ONNX graph (opset 18, dynamic batch, fixed window=24)
  - model_config.json    Scaler params, thresholds, and ONNX metadata for
                         downstream consumers (e.g., JVM ONNX Runtime scorer)

Usage:
    uv sync --extra onnx
    uv run python code/5_export_onnx.py                      # Export all 5 prebuilt models
    uv run python code/5_export_onnx.py --model Penny_All    # Export one model
    uv run python code/5_export_onnx.py --include-best       # Also export best from step 4
    uv run python code/5_export_onnx.py --skip-parity        # Fast export, no ORT comparison
"""
import argparse
import json
import logging
import pickle
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import torch

# Suppress PyTorch FFT resize deprecation warnings and ONNX export noise
warnings.filterwarnings("ignore", message=".*output with one or more elements was resized.*")
warnings.filterwarnings("ignore", message=".*dynamic_axes.*is not recommended.*")
warnings.filterwarnings("ignore", message=".*torchvision is not installed.*")
warnings.filterwarnings("ignore", message=".*isinstance.*LeafSpec.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Pickle compatibility for old module paths
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
from src.scorer import FCVAEScorer
from src.onnx_export import FCVAEInferenceWrapper, export_to_onnx

logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models" / "fcvae"
PREBUILT_NAMES = ["Penny_All", "Accel_CMP", "Accel_nopin", "Star_CMP", "Star_nopin"]


def load_model(model_dir: Path, device: torch.device):
    """Load a trained FCVAE model from checkpoint."""
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
    return model, config


def load_test_windows(model_dir: Path, n_windows: int = 100):
    """Load normalized test windows for parity testing.

    Falls back to random synthetic windows if data loading fails.
    """
    try:
        from src.preprocess import (
            load_penny_data, load_combo_data, COMBO_KEYS,
            create_sliding_windows, create_splits,
        )

        model_name = model_dir.name
        data_path = PROJECT_ROOT / "data" / "synthetic_transactions.csv"

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        with open(model_dir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        if model_name == "Penny_All":
            hourly_df = load_penny_data(data_path)
        else:
            combo_keys = {
                "Accel_CMP": ("Accel", "CMP"),
                "Accel_nopin": ("Accel", "no-pin"),
                "Star_CMP": ("Star", "CMP"),
                "Star_nopin": ("Star", "no-pin"),
            }
            key = combo_keys.get(model_name)
            if key is None:
                raise ValueError(f"Unknown model name: {model_name}")
            all_combos = load_combo_data(data_path)
            hourly_df = all_combos[key]

        windows, labels, timestamps = create_sliding_windows(hourly_df)
        splits = create_splits(windows, labels, timestamps, hourly_df)

        test_windows, test_labels = splits["test"]
        flat = test_windows.flatten().reshape(-1, 1)
        normalized = scaler.transform(flat).reshape(test_windows.shape)

        if len(normalized) > n_windows:
            idx = np.random.default_rng(42).choice(len(normalized), n_windows, replace=False)
            return normalized[idx], test_labels[idx]
        return normalized, test_labels

    except Exception as e:
        logger.warning(f"Could not load real data for {model_dir.name}: {e}")
        logger.warning("Falling back to random synthetic windows for parity test")
        return np.random.default_rng(42).standard_normal((n_windows, 24)), None


def export_companion_json(model_dir: Path, model_name: str, config, onnx_path: Path):
    """Write model_config.json with scaler params, thresholds, and ONNX metadata."""
    result = {
        "model_name": model_name,
        "onnx": {
            "opset_version": 18,
            "window_size": config.window,
            "input_name": "input",
            "output_name": "nll",
            "input_shape": ["batch", 1, config.window],
            "output_shape": ["batch", config.window],
        },
    }

    # Scaler params
    scaler_path = model_dir / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        result["scaler"] = {
            "type": "StandardScaler",
            "mean": float(scaler.mean_[0]),
            "scale": float(scaler.scale_[0]),
        }

    # Scorer thresholds and stats
    scorer_path = model_dir / "scorer.pkl"
    if scorer_path.exists():
        scorer = FCVAEScorer.load(scorer_path)
        result["thresholds"] = {
            "last_point_threshold": scorer.last_point_threshold,
            "point_threshold": scorer.point_threshold,
            "window_threshold": scorer.window_threshold,
        }
        result["scorer"] = {
            "normal_score_mean": scorer.normal_score_mean,
            "normal_score_std": scorer.normal_score_std,
        }

    json_path = model_dir / "model_config.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Wrote {json_path}")


def run_parity(
    model: FCVAE,
    onnx_path: Path,
    test_windows: np.ndarray,
    test_labels,
    scorer,
    device: torch.device,
) -> dict:
    """Compare PyTorch wrapper output against ONNX Runtime output."""
    import onnxruntime as ort

    wrapper = FCVAEInferenceWrapper(model).eval()

    # PyTorch
    x_torch = torch.FloatTensor(test_windows).unsqueeze(1).to(device)
    with torch.no_grad():
        pt_scores = wrapper(x_torch).cpu().numpy()

    # ONNX Runtime
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x_np = test_windows[:, np.newaxis, :].astype(np.float32)
    ort_scores = sess.run(["nll"], {"input": x_np})[0]

    abs_diff = np.abs(pt_scores - ort_scores)
    rel_diff = abs_diff / (np.abs(pt_scores) + 1e-8)

    metrics = {
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "max_rel_diff": float(rel_diff.max()),
        "mean_rel_diff": float(rel_diff.mean()),
        "num_windows": len(test_windows),
    }

    if scorer is not None:
        threshold = scorer.last_point_threshold or scorer.point_threshold
        if threshold is not None:
            pt_lp = pt_scores[:, -1]
            ort_lp = ort_scores[:, -1]
            pt_decisions = pt_lp < threshold
            ort_decisions = ort_lp < threshold
            agreement = np.mean(pt_decisions == ort_decisions)
            metrics["threshold"] = float(threshold)
            metrics["decision_agreement"] = float(agreement)
            metrics["pt_anomalies"] = int(pt_decisions.sum())
            metrics["ort_anomalies"] = int(ort_decisions.sum())

            if test_labels is not None:
                lp_labels = test_labels[:, -1].astype(bool)
                metrics["ground_truth_anomalies"] = int(lp_labels.sum())

    return metrics


def export_and_validate(model_name: str, model_dir: Path, device: torch.device,
                        skip_parity: bool = False) -> bool:
    """Export a single model to ONNX, write companion JSON, and run parity."""
    if not (model_dir / "model.pt").exists():
        print(f"  SKIP: {model_dir / 'model.pt'} not found")
        return True

    print(f"\n{'=' * 60}")
    print(f"  {model_name}  ({model_dir})")
    print(f"{'=' * 60}")

    # Load model
    print("  Loading PyTorch model...")
    model, config = load_model(model_dir, device)
    print(f"  Config: window={config.window}, latent_dim={config.latent_dim}")

    # Load scorer
    scorer = None
    scorer_path = model_dir / "scorer.pkl"
    if scorer_path.exists():
        scorer = FCVAEScorer.load(scorer_path)
        threshold = scorer.last_point_threshold or scorer.point_threshold
        print(f"  Threshold: {threshold}")

    # Export ONNX
    onnx_path = model_dir / "model.onnx"
    print(f"  Exporting to {onnx_path}...")
    export_to_onnx(model, onnx_path, window=config.window)

    # Validate ONNX file
    import onnx
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("  ONNX checker: PASSED")

    # Write companion JSON
    export_companion_json(model_dir, model_name, config, onnx_path)

    if skip_parity:
        print(f"\n  {model_name}: EXPORTED (parity skipped)")
        return True

    # Load test data and run parity
    print("  Loading test windows...")
    test_windows, test_labels = load_test_windows(model_dir)
    print(f"  Test windows: {len(test_windows)}")

    print("  Running parity harness...")
    metrics = run_parity(model, onnx_path, test_windows, test_labels, scorer, device)

    # Report
    print(f"\n  --- Parity Results ---")
    print(f"  Max absolute diff:  {metrics['max_abs_diff']:.2e}")
    print(f"  Mean absolute diff: {metrics['mean_abs_diff']:.2e}")
    print(f"  Max relative diff:  {metrics['max_rel_diff']:.2e}")
    print(f"  Mean relative diff: {metrics['mean_rel_diff']:.2e}")

    passed = True

    if "decision_agreement" in metrics:
        agreement_pct = metrics["decision_agreement"] * 100
        print(f"  Decision agreement: {agreement_pct:.1f}%")
        print(f"  PT anomalies: {metrics['pt_anomalies']}, ORT anomalies: {metrics['ort_anomalies']}")
        if "ground_truth_anomalies" in metrics:
            print(f"  Ground truth anomalies: {metrics['ground_truth_anomalies']}")
        if metrics["decision_agreement"] < 0.999:
            print("  WARNING: Decision agreement below 99.9%")
            passed = False

    if metrics["max_abs_diff"] > 1.0:
        print(f"  WARNING: Max absolute diff {metrics['max_abs_diff']:.2e} exceeds 1.0")
        passed = False
    elif metrics["max_abs_diff"] > 1e-3:
        print(f"  NOTE: Max absolute diff {metrics['max_abs_diff']:.2e} > 1e-3 (expected for FFT ops)")

    if metrics["mean_abs_diff"] > 1e-2:
        print(f"  WARNING: Mean absolute diff {metrics['mean_abs_diff']:.2e} exceeds 1e-2")
        passed = False

    status = "PASSED" if passed else "FAILED"
    print(f"\n  {model_name}: {status}")
    return passed


def main():
    parser = argparse.ArgumentParser(description="Export FCVAE models to ONNX and validate parity")
    parser.add_argument("--model", type=str, default=None,
                        help="Export a single model (e.g. Penny_All). Default: all 5 prebuilt.")
    parser.add_argument("--include-best", action="store_true",
                        help="Also export models/fcvae/best/Penny_All if it exists")
    parser.add_argument("--include-initial", action="store_true",
                        help="Also export models/fcvae/initial/Penny_All if it exists")
    parser.add_argument("--skip-parity", action="store_true",
                        help="Skip ORT parity validation (fast export only)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for PyTorch inference (cpu recommended for parity)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    # Suppress noisy ONNX optimizer, torch.onnx, and torchvision registration logs
    for mod in ["src.model", "src.scorer", "src.preprocess", "src.train", "src.training",
                "src.onnx_export", "onnxscript", "torch.onnx", "onnx",
                "torch.onnx._internal", "torch._C._distributed_c10d"]:
        logging.getLogger(mod).setLevel(logging.WARNING)

    device = torch.device(args.device)

    # Build list of (name, dir) pairs to export
    targets = []

    if args.model:
        # Single model: could be a prebuilt name or a path
        model_path = MODELS_DIR / args.model
        if model_path.exists():
            targets.append((args.model, model_path))
        else:
            # Try as a relative path from project root
            alt_path = PROJECT_ROOT / args.model
            if alt_path.exists():
                targets.append((alt_path.name, alt_path))
            else:
                print(f"ERROR: Model directory not found: {model_path}")
                sys.exit(1)
    else:
        # All 5 prebuilt models
        for name in PREBUILT_NAMES:
            model_dir = MODELS_DIR / name
            if model_dir.exists():
                targets.append((name, model_dir))

    if args.include_best:
        best_dir = MODELS_DIR / "best" / "Penny_All"
        if best_dir.exists() and (best_dir / "model.pt").exists():
            targets.append(("best/Penny_All", best_dir))
        else:
            print("  Note: models/fcvae/best/Penny_All not found, skipping --include-best")

    if args.include_initial:
        initial_dir = MODELS_DIR / "initial" / "Penny_All"
        if initial_dir.exists() and (initial_dir / "model.pt").exists():
            targets.append(("initial/Penny_All", initial_dir))
        else:
            print("  Note: models/fcvae/initial/Penny_All not found, skipping --include-initial")

    print("FCVAE ONNX Export & Parity Harness")
    print(f"Device: {device}")
    print(f"Models: {[name for name, _ in targets]}")
    if args.skip_parity:
        print("Parity validation: SKIPPED")

    results = {}
    for name, model_dir in targets:
        passed = export_and_validate(name, model_dir, device, skip_parity=args.skip_parity)
        results[name] = passed

    # Summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
