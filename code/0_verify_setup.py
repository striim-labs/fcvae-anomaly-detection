"""
Step 0: Verify Environment Setup

Checks that all required packages are installed, data files exist,
and model artifacts are present.

Usage:
    uv run python code/0_verify_setup.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_packages():
    """Verify required Python packages are importable."""
    packages = {
        "torch": "PyTorch",
        "numpy": "NumPy",
        "pandas": "Pandas",
        "sklearn": "scikit-learn",
        "scipy": "SciPy",
        "fastapi": "FastAPI",
        "uvicorn": "Uvicorn",
        "pydantic": "Pydantic",
        "matplotlib": "Matplotlib",
        "plotly": "Plotly",
        "tqdm": "tqdm",
        "rich": "Rich",
    }

    results = {}
    for module, name in packages.items():
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "unknown")
            results[name] = ("OK", version)
        except ImportError:
            results[name] = ("MISSING", None)

    return results


def check_data():
    """Verify data files exist."""
    data_dir = PROJECT_ROOT / "data"
    files = {
        "synthetic_transactions.csv": data_dir / "synthetic_transactions.csv",
        "generate_transactions.py": data_dir / "generate_transactions.py",
    }

    results = {}
    for name, path in files.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            results[name] = ("OK", f"{size_mb:.1f} MB")
        else:
            results[name] = ("MISSING", None)

    return results


def check_model_artifacts():
    """Verify pre-trained model artifacts exist."""
    model_dir = PROJECT_ROOT / "models" / "fcvae" / "Penny_All"
    files = {
        "model.pt": model_dir / "model.pt",
        "scaler.pkl": model_dir / "scaler.pkl",
        "scorer.pkl": model_dir / "scorer.pkl",
    }

    results = {}
    for name, path in files.items():
        if path.exists():
            size_kb = path.stat().st_size / 1024
            results[name] = ("OK", f"{size_kb:.1f} KB")
        else:
            results[name] = ("MISSING", None)

    return results


def check_src_modules():
    """Verify src/ modules are importable."""
    sys.path.insert(0, str(PROJECT_ROOT))
    modules = ["src.model", "src.scorer", "src.preprocess", "src.train", "src.schemas", "src.utils"]

    results = {}
    for mod_name in modules:
        try:
            __import__(mod_name)
            results[mod_name] = ("OK", None)
        except Exception as e:
            results[mod_name] = ("ERROR", str(e))

    return results


def main():
    print("=" * 60)
    print("FCVAE Penny Anomaly Detection — Setup Verification")
    print("=" * 60)
    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Python: {sys.version}")

    all_ok = True

    # Packages
    print("\n--- Python Packages ---")
    pkg_results = check_packages()
    for name, (status, version) in pkg_results.items():
        if status == "OK":
            print(f"  [OK] {name} {version}")
        else:
            print(f"  [MISSING] {name}")
            all_ok = False

    # Data
    print("\n--- Data Files ---")
    data_results = check_data()
    for name, (status, info) in data_results.items():
        if status == "OK":
            print(f"  [OK] data/{name} ({info})")
        else:
            print(f"  [MISSING] data/{name}")
            all_ok = False

    # Model artifacts
    print("\n--- Model Artifacts (Penny_All) ---")
    model_results = check_model_artifacts()
    for name, (status, info) in model_results.items():
        if status == "OK":
            print(f"  [OK] models/fcvae/Penny_All/{name} ({info})")
        else:
            print(f"  [MISSING] models/fcvae/Penny_All/{name}")
            all_ok = False

    # src modules
    print("\n--- Source Modules ---")
    src_results = check_src_modules()
    for name, (status, info) in src_results.items():
        if status == "OK":
            print(f"  [OK] {name}")
        else:
            print(f"  [ERROR] {name}: {info}")
            all_ok = False

    # Device
    print("\n--- Compute Device ---")
    try:
        from src.utils import auto_device
        device = auto_device()
        print(f"  Selected device: {device}")
    except Exception as e:
        print(f"  Error detecting device: {e}")

    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("All checks passed! Ready to run.")
        print("\nNext steps:")
        print("  uv run jupyter notebook code/   # Explore notebooks")
        print("  uv run python code/3_train_model.py  # Train model")
        print("  uv run python code/5_streaming_app.py  # Start API")
    else:
        print("Some checks failed. Install missing dependencies with: uv sync")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
