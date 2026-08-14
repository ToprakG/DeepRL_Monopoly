"""The submitted policy: two gradient-boosted models over engineered features.

The modules in this package import each other by bare name (``import feature_lib``,
``from act_lib import ...``) because that is how they were written and trained, and
renaming imports in a package whose feature layout is pinned to a fitted model is a
good way to serve a model under the wrong contract. So this directory goes on
``sys.path`` and its modules load as top-level names. ``agent.py`` at the repository
root does that insertion; importing this package performs it too, so either entry
works.

Nothing here is imported at package-import time beyond the path setup -- LightGBM and
the ~4.3 MB of boosters load on the first decision, not on import.
"""
from __future__ import annotations

import os
import sys

# One core. The match sandbox is memory-capped and the per-decision matrices are
# small enough that a thread pool costs more than it returns. Set before numpy or
# LightGBM is imported anywhere, or it has no effect.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

MODELS = os.path.join(_HERE, "models")

__all__ = ["MODELS"]
