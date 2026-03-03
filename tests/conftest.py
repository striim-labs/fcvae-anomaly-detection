"""Shared test configuration.

Ensures app/ modules are importable and pre-registers them under the
'app.*' namespace before fcvae_streaming_detector.py creates a synthetic
'app' module (for pickle compatibility) that would shadow them.
"""

import sys
from pathlib import Path

# Add app/ to sys.path
_app_dir = str(Path(__file__).resolve().parent.parent / "app")
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

# Pre-import modules that use `from app.xxx import ...` so they are
# already in sys.modules before the synthetic 'app' module is created
# by fcvae_streaming_detector.py.
import transaction_config  # noqa: F401
import transaction_preprocessor  # noqa: F401
