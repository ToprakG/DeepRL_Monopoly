"""Experiment 9 — which trade does the teacher PROPOSE?

Why this experiment exists
--------------------------
It was not in the briefed list. Experiment 6 mapped the accept/decline surface
— the *reply* side — and Phase 2's first agreement measurement then showed the
gap: of 474 disagreements over 2,005 held-out decisions, **307 were trade
proposals the clone cannot make at all** (`exch_trade` 277, `buy_trade` 7,
`sell_trade` 6, plus proactive `mortgage` 10). The teacher spends ~15% of its
decisions proposing trades. No amount of tuning the reply rules reaches 90%
agreement while that family is missing, so it has to be probed.

Method
------
Seat 0 holds two of the three oranges and assorted spares; a rival holds the
completing deed. The teacher is asked what it does, and the proposal is
decoded into (family, target player, offered deed, requested deed, cash level).
Swept over: which spare deeds we hold, our cash, which rival holds the
completing piece, and whether a completing piece exists at all (the control —
with no near-monopoly there should be nothing worth proposing).

Output: probes/p09_trade_proposal.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from competition_agent.proc import managed_pool  # noqa: E402
from competition_agent.probe_harness import (  # noqa: E402
    PROPERTIES, PROPERTY_IDS, ProbeWriter, ask_value, blank_board, describe,
    give, legal, set_pre_roll,
)
from monopoly_game_engine.actions import OFFSETS  # noqa: E402
from monopoly_game_engine.constants import TRADE_CASH_LEVELS  # noqa: E402

SEED = 20250811
SPARES = [[], [5], [5, 25], [5, 25, 37], [37]]
CASHES = [300, 600, 1200, 2500]
COMPLETER_OWNER = [1, 2, 3, None]      # None = deed still with the bank


def decode(action: int):
    """(family, target, offered_sq, requested_sq, cash_level) or family only."""
    n = len(PROPERTY_IDS)
    if OFFSETS["buy_trade"] <= action < OFFSETS["sell_trade"]:
        loc = action - OFFSETS["buy_trade"]
        p = loc // (n * len(TRADE_CASH_LEVELS))
        rem = loc % (n * len(TRADE_CASH_LEVELS))
        return ("buy_trade", p, None, PROPERTY_IDS[rem // len(TRADE_CASH_LEVELS)],
                TRADE_CASH_LEVELS[rem % len(TRADE_CASH_LEVELS)])
    if OFFSETS["sell_trade"] <= action < OFFSETS["exch_trade"]:
        loc = action - OFFSETS["sell_trade"]
        p = loc // (n * len(TRADE_CASH_LEVELS))
        rem = loc % (n * len(TRADE_CASH_LEVELS))
        return ("sell_trade", p, PROPERTY_IDS[rem // len(TRADE_CASH_LEVELS)],
                None, TRADE_CASH_LEVELS[rem % len(TRADE_CASH_LEVELS)])
    if OFFSETS["exch_trade"] <= action < OFFSETS["auction"]:
        loc = action - OFFSETS["exch_trade"]
        p = loc // (n * (n - 1))
        rem = loc % (n * (n - 1))
        off = rem // (n - 1)
        req_raw = rem % (n - 1)
        req = req_raw if req_raw < off else req_raw + 1
        return ("exch_trade", p, PROPERTY_IDS[off], PROPERTY_IDS[req], None)
    return (describe(action), None, None, None, None)


def _job(item):
    spares, cash, owner = item
    _, env = blank_board(seed=SEED)
    give(env, 0, [16, 18] + spares)
    if owner is not None:
        give(env, owner, [19])
        give(env, owner, [21, 23])          # give the rival something too
    give(env, 2 if owner != 2 else 3, [31, 32])
    set_pre_roll(env, 0, cash=cash)

    a = ask_value(env, 0)
    fam, target, off, req, lvl = decode(a)
    return {
        "spares": "|".join(map(str, spares)) or "none",
        "cash": cash,
        "completer_owner": "bank" if owner is None else owner,
        "n_legal": len(legal(env, 0)),
        "action": describe(a),
        "family": fam,
        "target_player": "" if target is None else target,
        "offered_sq": "" if off is None else off,
        "requested_sq": "" if req is None else req,
        "cash_level": "" if lvl is None else lvl,
        "requests_completer": (req == 19),
        "seed": SEED,
    }


def main() -> int:
    items = [(s, c, o) for s in SPARES for c in CASHES
             for o in COMPLETER_OWNER]
    with managed_pool(10) as pool:
        rows = pool.map(_job, items)

    with ProbeWriter("p09_trade_proposal", list(rows[0].keys())) as out:
        for r in rows:
            out.write(**r)

    print(f"{'spares':<12}{'cash':>6}{'completer':>11}  {'action':<34}"
          f"{'wants 19':>9}")
    print("-" * 76)
    for r in rows:
        print(f"{r['spares']:<12}{r['cash']:>6}{str(r['completer_owner']):>11}"
              f"  {r['action'][:34]:<34}{str(r['requests_completer']):>9}")

    fams = {}
    for r in rows:
        fams[r["family"]] = fams.get(r["family"], 0) + 1
    print("\nfamily counts:", fams)
    print(f"wrote {out.path} ({out.rows} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
