"""EXPO: a net-worth-gradient heuristic policy for the ppo-plus-v2 simulator."""

import sys
import types
from pathlib import Path


def _locate_engine() -> Path | None:
    """
    Find the ppo-plus-v2 engine in either supported layout.

    The package may sit *inside* the DeepRL_Monopoly checkout, or beside it
    as a sibling directory during development. Both are supported so the
    same code runs from a clone and from a working copy.
    """
    here = Path(__file__).resolve().parent
    for candidate in (here.parent, here.parent / "DeepRL_Monopoly"):
        if (candidate / "monopoly_game_engine").is_dir():
            return candidate
    return None


_ENGINE_ROOT = _locate_engine()
if _ENGINE_ROOT is not None and str(_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENGINE_ROOT))

# monopoly_game_engine/__init__.py eagerly imports torch so it can re-export
# the PPO/DDQN trainers. The *rules* half of the engine -- env, state,
# actions, constants, agents_fixed -- has no such dependency, and EXPO only
# ever touches the rules. Bind the package to its directory without running
# that heavy __init__ so the heuristic runs on a plain numpy install.
if _ENGINE_ROOT is not None and "monopoly_game_engine" not in sys.modules:
    try:
        import torch  # noqa: F401
    except ImportError:
        _shim = types.ModuleType("monopoly_game_engine")
        _shim.__path__ = [str(_ENGINE_ROOT / "monopoly_game_engine")]
        _shim.__doc__ = "Rules-only view of monopoly_game_engine (no torch)."
        sys.modules["monopoly_game_engine"] = _shim

from .agent import POLICY_ID, ExpoHeuristicAgent  # noqa: E402

__all__ = ["ExpoHeuristicAgent", "POLICY_ID"]
