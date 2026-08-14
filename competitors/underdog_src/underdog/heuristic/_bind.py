"""Bind the vendored `engine/` to the name `monopoly_game_engine`, and nothing else.

WHY THIS FILE EXISTS AND WHY IT MUST BE IMPORTED FIRST
The organizer's own code imports the simulator as `monopoly_game_engine`. This package
ships its own pinned copy under `engine/`. If both names end up loaded you get TWO
distinct `MonopolyEnv` classes in one process, and every `isinstance` check in the
organizer's harness silently takes the wrong branch -- a failure that produces plausible
wrong numbers rather than an error.

So: import this before anything that touches the simulator. `heuristic/__init__.py` does
it for you, so `import heuristic` is enough.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

#: the folder that contains `engine/` -- i.e. Kuzeys_heuristic/
ROOT = Path(__file__).resolve().parents[2]
ENGINE_DIR = ROOT / "engine"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_ENGINE_SUBMODULES = (
    "actions", "constants", "state", "env", "networks",
    "agents_fixed", "agent_ppo", "agent_ddqn", "train",
)


def bind_engine_alias(*, force: bool = False) -> bool:
    """Make `import monopoly_game_engine` resolve to our `engine/`. Idempotent."""
    existing = sys.modules.get("monopoly_game_engine")
    if existing is not None and not force:
        return str(ENGINE_DIR) in str(getattr(existing, "__file__", ""))
    engine_pkg = importlib.import_module("engine")
    sys.modules["monopoly_game_engine"] = engine_pkg
    for name in _ENGINE_SUBMODULES:
        try:
            module = importlib.import_module(f"engine.{name}")
        except Exception:      # optional submodules (torch-dependent ones) may not load
            continue
        sys.modules[f"monopoly_game_engine.{name}"] = module
    return True


def engine_is_ours() -> bool:
    """True when exactly one engine is loaded and it is ours."""
    module = sys.modules.get("monopoly_game_engine")
    if module is None:
        return False
    return str(ENGINE_DIR) in str(getattr(module, "__file__", ""))


def unwrap(game: Any) -> Any:
    """The raw `MonopolyEnv`, whether we were handed one or a wrapper around one.

    Duck-typed on purpose so this package never has to import the organizer's bench.
    """
    return getattr(game, "env", game)


_ALIAS_BOUND = bind_engine_alias()
