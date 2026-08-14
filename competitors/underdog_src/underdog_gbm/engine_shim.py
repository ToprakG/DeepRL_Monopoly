"""Engine binding: works with either package name for the game engine."""
import sys

try:  # the engine as distributed
    import monopoly_game_engine as _eng
except ImportError:  # a checkout that exposes it as `engine`
    import engine as _eng  # noqa: F401
    _eng = sys.modules["engine"]

sys.modules.setdefault("engine", _eng)
for _sub in ("actions", "constants", "env", "state"):
    _m = __import__(f"{_eng.__name__}.{_sub}", fromlist=[_sub])
    sys.modules.setdefault(f"engine.{_sub}", _m)
