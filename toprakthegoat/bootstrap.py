"""Load engine submodules without executing ``monopoly_game_engine.__init__``.

That package init imports torch. The tournament venv does not install torch,
and a raise on import is a strike. Submodules (``env``, ``state``, ``actions``,
``constants``) do not need it.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path


def install_engine_namespace(root: Path | None = None) -> None:
    name = "monopoly_game_engine"
    existing = sys.modules.get(name)
    if existing is not None:
        if getattr(existing, "_goat_dummy", False):
            return
        if hasattr(existing, "MonopolyEnv"):
            return
    here = Path(root) if root is not None else Path(__file__).resolve().parent.parent
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(here / name)]
    pkg.__package__ = name
    pkg.__file__ = str(here / name / "__init__.py")
    pkg._goat_dummy = True
    sys.modules[name] = pkg
