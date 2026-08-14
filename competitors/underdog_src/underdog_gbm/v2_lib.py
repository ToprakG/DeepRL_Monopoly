"""ACTION FEATURES v2 — let the tree see what the move actually does.

(tree track, r5d, 2026-08-12.  Written because `census_bc_run1.json` proved v1 blind.)

WHAT THE CENSUS FOUND
---------------------
Measured over 81,483 of the tree's own decisions against PANEL-STRONGISH, how often it
picks the exact move the champion would:

    binary        49.1% of decisions    97.9%   <- near perfect
    buy_trade     15.7%                 75.9%
    auction        4.4%                 72.9%
    exch_trade    18.5%                  5.9%   <- near random
    unmortgage     9.6%                  3.5%   <- near random
    mortgage       1.5%                 19.7%

Roughly 30% of all decisions are made almost at random.  And it is not a learning
failure -- four rounds of DAgger raised on-policy agreement 70.7 -> 78.6% and moved the
win rate not at all.  You cannot learn a distinction that is not in the input.

WHY v1 IS BLIND, EXACTLY
------------------------
`engine/actions.py:118` decodes an exchange as THREE fields:

    exch_trade(player=<which opponent>, offer=<deed I give>, req=<deed I want>)

`gbdt_collect.action_features` derives its board position as
`PROPERTY_IDS[(local // 27) % 28]`, which is the **offered** deed and only that.  The
requested deed and the counterparty never reach the model.  The tree is asked to rank
2,268 exchanges while unable to see what it gets back -- 5.9% is what that looks like.

`buy_trade`/`sell_trade` lose their **cash level** the same way (feature 12 smears
player, property and price into one scalar), which is why they land at ~76%: right deed,
wrong price.

`unmortgage` and `mortgage` are a different blindness.  Their features say
`mortgaged=1, owner=me, houses=0` and little else, so every candidate looks alike.  The
champion decides by whether the deed's COLOUR GROUP is still winnable -- its war-chest
rule is precisely "mortgage deeds that can never become a monopoly and keep them
mortgaged".  Group ownership is in the 300 observation dims, but the model would have to
form a cross-product between an action's colour and a state one-hot to use it, which is
exactly what an axis-aligned tree cannot do in one split.

WHAT v2 ADDS (15 features, 21 -> 36; FEAT_DIM 321 -> 336)
----------------------------------------------------------
Per candidate action, all of it already available at the call site:

  the OTHER side of a trade      requested deed's position, colour, price, houses,
                                 mortgaged, and whether I already own it
  trade economics                cash level (buy/sell), and the counterparty's index
  the COLOUR GROUP, resolved     for the deed this action touches: how many of the group
                                 I own, how many one single opponent owns, group size,
                                 whether this action completes MY monopoly, and whether
                                 the deed is DEAD (an opponent holds one, so the group can
                                 never be mine)

`is_dead` and `completes_my_monopoly` are the two the champion's own rules turn on, and
neither existed in v1 in any form.

CONTRACT.  This is a different row layout, so it carries a different
`FEATURE_CONTRACT`.  A model fitted under one and served under the other is a hard error
in `TreeAgent`, not a silent 25%.

v1 lives in ``act_lib.py`` and is imported from there unchanged; v2 is built on top,
so the two layouts stay separable and a model can never be served under the wrong one.
"""

from __future__ import annotations

import numpy as np  # noqa: E402

import engine_shim  # noqa: F401
from engine.actions import OFFSETS  # noqa: E402
from engine.constants import PROPERTIES, PROPERTY_IDS  # noqa: E402
from act_lib import (ACT_DIM as ACT_DIM_V1, OBS_DIM, PROP_SECTIONS, RE_SECTIONS, SEC_NAMES, SEC_STARTS, TRADE_SECTIONS, _STATIC, action_features as action_features_v1)

N_NEW = 15
ACT_DIM = ACT_DIM_V1 + N_NEW          # 36
FEAT_DIM = OBS_DIM + ACT_DIM          # 336
FEATURE_CONTRACT_V2 = f"obs{OBS_DIM}+act{ACT_DIM}=feat{FEAT_DIM}/v2"

_NP = len(PROPERTY_IDS)
_COLORS = sorted({PROPERTIES[p]["color"] for p in PROPERTY_IDS})
_COLOR_IDX = {c: i for i, c in enumerate(_COLORS)}
#: how many deeds are in each colour group -- the denominator for "do I own the group"
_GROUP_SIZE = {c: sum(1 for p in PROPERTY_IDS if PROPERTIES[p]["color"] == c) for c in _COLORS}
_GROUP_MEMBERS = {c: [p for p in PROPERTY_IDS if PROPERTIES[p]["color"] == c] for c in _COLORS}

_EXCH_LO = int(OFFSETS["exch_trade"])
_BUY_LO = int(OFFSETS["buy_trade"])
_SELL_LO = int(OFFSETS["sell_trade"])
_AUCTION_LO = int(OFFSETS["auction"])
_NUM_TRADE_CASH = (int(OFFSETS["sell_trade"]) - int(OFFSETS["buy_trade"])) // (3 * _NP)


def _decode_trade(a: int) -> tuple[int, int, int, int]:
    """(counterparty, main deed pos, requested deed pos or -1, cash level or -1).

    Mirrors `engine/actions.py:110-127` exactly; that is the only decode of record.
    """
    if _BUY_LO <= a < _AUCTION_LO and not (_EXCH_LO <= a < _AUCTION_LO):
        base = _BUY_LO if a < _SELL_LO else _SELL_LO
        local = a - base
        player = local // (_NP * _NUM_TRADE_CASH)
        rem = local % (_NP * _NUM_TRADE_CASH)
        return player, PROPERTY_IDS[rem // _NUM_TRADE_CASH], -1, rem % _NUM_TRADE_CASH
    if _EXCH_LO <= a < _AUCTION_LO:
        local = a - _EXCH_LO
        player = local // (_NP * (_NP - 1))
        rem = local % (_NP * (_NP - 1))
        offer = rem // (_NP - 1)
        raw = rem % (_NP - 1)
        req = raw if raw < offer else raw + 1
        return player, PROPERTY_IDS[offer], PROPERTY_IDS[req], -1
    return -1, -1, -1, -1


def _group_view(env, pid: int, pos: int) -> tuple[float, float, float, float]:
    """(my share of the group, best opponent's share, group size, is_dead).

    `is_dead` = an opponent holds at least one deed of this colour, so the group can
    never become my monopoly.  This is the exact predicate the champion's war-chest rule
    fires on, and v1 had no way to express it.
    """
    if pos < 0:
        return -1.0, -1.0, -1.0, -1.0
    prop = env.properties.get(pos)
    if prop is None:
        return -1.0, -1.0, -1.0, -1.0
    colour = PROPERTIES[pos]["color"]
    members = _GROUP_MEMBERS[colour]
    size = len(members)
    mine = 0
    others: dict[int, int] = {}
    for m in members:
        mp = env.properties.get(m)
        if mp is None or mp.owner is None:
            continue
        if mp.owner == pid:
            mine += 1
        else:
            others[mp.owner] = others.get(mp.owner, 0) + 1
    best_opp = max(others.values()) if others else 0
    dead = 1.0 if others else 0.0
    return mine / size, best_opp / size, size / 3.0, dead


def action_features_v2(env, pid: int, acts: np.ndarray) -> np.ndarray:
    """(len(acts), 36) float32.  v1's 21 columns unchanged, then 15 new ones."""
    acts = np.asarray(acts, dtype=np.int64)
    n = acts.size
    out = np.zeros((n, ACT_DIM), dtype=np.float32)
    out[:, :ACT_DIM_V1] = action_features_v1(env, pid, acts)

    sec_idx = np.searchsorted(SEC_STARTS, acts, side="right") - 1
    for i in range(n):
        a = int(acts[i])
        name = SEC_NAMES[int(sec_idx[i])]
        b = ACT_DIM_V1

        # ---- the other side of a trade, and its economics ----------------
        cp, main_pos, req_pos, cash = _decode_trade(a)
        out[i, b + 0] = (cp + 1) / 3.0 if cp >= 0 else -1.0
        out[i, b + 1] = cash / max(1, _NUM_TRADE_CASH - 1) if cash >= 0 else -1.0
        if req_pos >= 0:
            rp = env.properties.get(req_pos)
            out[i, b + 2] = req_pos / 39.0
            out[i, b + 3] = _COLOR_IDX[PROPERTIES[req_pos]["color"]] / max(1, len(_COLORS) - 1)
            if rp is not None:
                out[i, b + 4] = rp.price / 400.0
                out[i, b + 5] = rp.houses / 5.0
                out[i, b + 6] = 1.0 if rp.mortgaged else 0.0
                out[i, b + 7] = 1.0 if rp.owner == pid else 0.0
            else:
                out[i, b + 4:b + 8] = -1.0
        else:
            out[i, b + 2:b + 8] = -1.0

        # ---- the colour group this action touches ------------------------
        # For an exchange the deed that matters is the one being REQUESTED (what I gain);
        # for everything else it is the deed the action names.
        if main_pos < 0:
            main_pos = int(round(out[i, 13] * 39.0)) if out[i, 13] >= 0 else -1
        focus = req_pos if req_pos >= 0 else main_pos
        mine, opp, size, dead = _group_view(env, pid, focus)
        out[i, b + 8] = mine
        out[i, b + 9] = opp
        out[i, b + 10] = size
        out[i, b + 11] = dead
        # completes MY monopoly: I would hold every deed of the group after this
        completes = 0.0
        if focus >= 0 and mine >= 0 and size > 0:
            need = round(size * 3.0)
            held = round(mine * need)
            fp = env.properties.get(focus)
            gain = 1.0 if (fp is not None and fp.owner != pid and
                           name in ("buy_trade", "exch_trade")) else 0.0
            completes = 1.0 if (held + gain >= need and dead in (0.0, -1.0)) else 0.0
        out[i, b + 12] = completes
        # hands the counterparty THEIR monopoly: the deed I give away completes a group
        # they are one short of.  The champion vetoes exactly this.
        gives = 0.0
        if name in ("sell_trade", "exch_trade") and main_pos >= 0 and cp >= 0:
            colour = PROPERTIES[main_pos]["color"]
            members = _GROUP_MEMBERS[colour]
            owners = [env.properties[m].owner for m in members if env.properties.get(m)]
            others = [o for o in owners if o is not None and o != pid]
            if others and len(set(others)) == 1 and len(others) == len(members) - 1:
                gives = 1.0
        out[i, b + 13] = gives
        out[i, b + 14] = 1.0 if name in ("buy_trade", "sell_trade", "exch_trade") else 0.0
    return out

