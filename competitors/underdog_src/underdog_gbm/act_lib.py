"""Action encoding: the 21 hand-built features that describe one candidate action.

``_STATIC`` is the per-action lookup table -- section, target square, colour
group -- built once at import for all ``ACTION_SPACE_SIZE`` actions.
``action_features`` turns a list of legal action ids into the matrix Model A
consumes alongside the 300-float observation. ``v2_lib`` extends this with 15
more columns; see ``action_features_v2``.

Threads are pinned to one before numpy or LightGBM load. The match runs agents
in a memory-limited sandbox, and a per-decision matrix this small is slower
across a thread pool than on one core.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np  # noqa: E402

import engine_shim  # noqa: E402,F401
from engine.actions import ACTION_SPACE_SIZE, OFFSETS  # noqa: E402
from engine.constants import PROPERTIES, PROPERTY_IDS, REAL_ESTATE_IDS  # noqa: E402





NUM_PLAYERS = 4
OBS_DIM = 300
ACT_DIM = 21
FEAT_DIM = OBS_DIM + ACT_DIM
N_NEG = 7          # sampled negatives per decision in the TRAIN split
MAX_CAND = 160     # cap on candidates scored per held-out decision

_SEC = sorted(OFFSETS.items(), key=lambda kv: kv[1])
SEC_NAMES = [k for k, _ in _SEC]
SEC_STARTS = np.array([v for _, v in _SEC], dtype=np.int64)
SEC_ENDS = np.array(list(SEC_STARTS[1:]) + [ACTION_SPACE_SIZE], dtype=np.int64)
N_SEC = len(SEC_NAMES)
PROP_SECTIONS = {"mortgage", "unmortgage", "sell_prop"}
RE_SECTIONS = {"improve_house", "improve_hotel", "sell_house", "sell_hotel"}
TRADE_SECTIONS = {"buy_trade", "sell_trade", "exch_trade"}
COLORS = sorted({PROPERTIES[p]["color"] for p in PROPERTY_IDS})
COLOR_IDX = {c: i for i, c in enumerate(COLORS)}


def _action_static(a: int) -> tuple[int, int, int]:
    """(section index, local index, board position or -1)."""
    s = int(np.searchsorted(SEC_STARTS, a, side="right") - 1)
    local = a - int(SEC_STARTS[s])
    name = SEC_NAMES[s]
    pos = -1
    if name in PROP_SECTIONS:
        pos = PROPERTY_IDS[local % len(PROPERTY_IDS)]
    elif name in RE_SECTIONS:
        pos = REAL_ESTATE_IDS[local % len(REAL_ESTATE_IDS)]
    elif name in ("buy_trade", "sell_trade"):
        pos = PROPERTY_IDS[(local // 3) % len(PROPERTY_IDS)]
    elif name == "exch_trade":
        pos = PROPERTY_IDS[(local // 27) % len(PROPERTY_IDS)]
    return s, local, pos


# precompute the static part once for the whole action space
_STATIC = np.zeros((ACTION_SPACE_SIZE, 4), dtype=np.int32)
for _a in range(ACTION_SPACE_SIZE):
    _s, _l, _p = _action_static(_a)
    _STATIC[_a] = (_s, _l, _p, COLOR_IDX[PROPERTIES[_p]["color"]] if _p >= 0 else -1)
_SEC_SIZE = (SEC_ENDS - SEC_STARTS).astype(np.float32)


def action_features(env, pid: int, acts: np.ndarray) -> np.ndarray:
    """(len(acts), ACT_DIM) float32 action features, some state-dependent."""
    n = acts.size
    out = np.zeros((n, ACT_DIM), dtype=np.float32)
    st = _STATIC[acts]
    sec, local, pos, col = st[:, 0], st[:, 1], st[:, 2], st[:, 3]
    out[np.arange(n), sec] = 1.0                                  # 0..11 section one-hot
    out[:, 12] = local / _SEC_SIZE[sec]                            # 12 local index
    out[:, 13] = np.where(pos >= 0, pos / 39.0, -1.0)              # 13 board position
    out[:, 14] = np.where(col >= 0, col / max(1, len(COLORS) - 1), -1.0)  # 14 colour group
    for i in range(n):
        p = int(pos[i])
        if p < 0:
            out[i, 15:20] = -1.0
            continue
        prop = env.properties.get(p)
        if prop is None:
            out[i, 15:20] = -1.0
            continue
        out[i, 15] = prop.price / 400.0
        out[i, 16] = 1.0 if prop.owner == pid else 0.0
        out[i, 17] = 1.0 if (prop.owner is not None and prop.owner != pid) else 0.0
        out[i, 18] = 1.0 if prop.mortgaged else 0.0
        out[i, 19] = prop.houses / 5.0
    out[:, 20] = np.isin(sec, [i for i, nm in enumerate(SEC_NAMES) if nm in TRADE_SECTIONS]).astype(np.float32)
    return out


