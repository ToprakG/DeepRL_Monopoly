"""Isolate the two-sided feasibility filter — filter only, no weight tuning.

D2.4 diagnosed the trade-ranking failure as "every feature is one-sided": the
teacher requires recipient gain >= 0 and both safety gates, so the pick may be
the best *feasible* candidate rather than the best one for us. That diagnosis
has to be tested on its own before anything is fitted on top of it.

So this script changes exactly one thing. Weights stay at the current
`deed_value` baseline (price + rent + mono, all 1.0). No search, no tuning.
The only variable is which candidates are allowed into the argmax:

    none            every legal exchange                    (D2.4 baseline)
    recipient       recipient gain >= 0
    recipient_pos   recipient gain > 0
    safety          both parties' safety gates pass
    recipient+safety    both of the above

If a filter genuinely lifts top-1, the diagnosis holds and weight fitting is
worth doing afterwards. If none of them move it, the diagnosis is wrong as
well, and the valuation needs rethinking from scratch rather than patching.

Reported on the same 60/40 board split as `fit_trade.py`, fixed by the same
seed, so the held-out set is the one already committed to.

Usage: python3 competition_agent/fit_trade_ablation.py
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from competition_agent.proc import managed_pool  # noqa: E402
from competition_agent.probe_harness import ask_value, legal  # noqa: E402
from competition_agent.probes.p09b_trade_ranking import (  # noqa: E402
    SEED, N_BOARDS, _decode, build_board,
)
from competition_agent.spec_model import deed_value, gates_ok  # noqa: E402
from monopoly_game_engine.actions import OFFSETS  # noqa: E402
from monopoly_game_engine.constants import PROPERTY_IDS  # noqa: E402

CACHE = Path(__file__).resolve().parent / "probes" / "ablation_features.json"


def _extract(k: int):
    """Per-candidate: our gain, the recipient's gain, and both safety flags."""
    rng = random.Random(SEED + k)
    env, rival, dev = build_board(rng)
    n = len(PROPERTY_IDS)
    others = [i for i in range(len(env.players)) if i != 0]
    cands = [a for a in legal(env, 0)
             if OFFSETS["exch_trade"] <= a < OFFSETS["auction"]]
    if len(cands) < 2:
        return None

    chosen = int(ask_value(env, 0))
    vcache = {}

    def v(player, sq):
        key = (player, sq)
        if key not in vcache:
            vcache[key] = deed_value(env, player, sq)
        return vcache[key]

    # A pure deed swap moves no cash, so the gates are evaluated at zero spend;
    # what changes for each side is which assets back them.
    our_safe = gates_ok(env, 0, 0)

    rows = []
    for a in cands:
        dec = _decode(a, n, others)
        if dec is None:
            continue
        tgt, off, req = dec
        rows.append({
            "action": a,
            "our_gain": v(0, req) - v(0, off),
            "their_gain": v(tgt, off) - v(tgt, req),
            "our_safe": bool(our_safe),
            "their_safe": bool(gates_ok(env, tgt, 0)),
        })
    if not rows:
        return None
    return {"board": k, "chosen": chosen,
            "proposed": chosen in cands, "cands": rows}


FILTERS = {
    "none":             lambda c: True,
    "recipient>=0":     lambda c: c["their_gain"] >= 0,
    "recipient>0":      lambda c: c["their_gain"] > 0,
    "safety":           lambda c: c["our_safe"] and c["their_safe"],
    "recipient+safety": lambda c: (c["their_gain"] >= 0
                                   and c["our_safe"] and c["their_safe"]),
}


def top1(boards, keep):
    """Top-1 with baseline weights, restricted to candidates passing `keep`."""
    hit = usable = 0
    for b in boards:
        pool = [c for c in b["cands"] if keep(c)]
        if not pool:
            continue                 # filter removed everything: no prediction
        usable += 1
        best = max(pool, key=lambda c: (c["our_gain"], -c["action"]))
        hit += best["action"] == b["chosen"]
    return hit, usable


def survives(boards, keep):
    """Does the teacher's own pick survive the filter? The filter's own test."""
    kept = tot = 0
    for b in boards:
        pick = next((c for c in b["cands"] if c["action"] == b["chosen"]), None)
        if pick is None:
            continue
        tot += 1
        kept += bool(keep(pick))
    return kept, tot


def main() -> int:
    if CACHE.exists():
        data = json.loads(CACHE.read_text())
    else:
        with managed_pool(10) as pool:
            data = [d for d in pool.map(_extract, range(N_BOARDS)) if d]
        CACHE.write_text(json.dumps(data))

    rng = random.Random(20250811)          # same split as fit_trade.py
    ids = sorted(d["board"] for d in data)
    rng.shuffle(ids)
    cut = int(0.6 * len(ids))
    train_ids = set(ids[:cut])
    train = [d for d in data if d["board"] in train_ids and d["proposed"]]
    held = [d for d in data if d["board"] not in train_ids and d["proposed"]]

    print(f"boards: {len(data)}   proposals: train {len(train)}, "
          f"held-out {len(held)}")
    print("weights fixed at the deed_value baseline; only the filter varies\n")

    print(f"{'filter':<20}{'train top-1':>14}{'held-out top-1':>17}"
          f"{'teacher pick kept':>20}")
    print("-" * 71)
    for name, keep in FILTERS.items():
        th, tn = top1(train, keep)
        hh, hn = top1(held, keep)
        kk, kt = survives(train + held, keep)
        tr = f"{th}/{tn} = {100*th/max(tn,1):.1f}%"
        hd = f"{hh}/{hn} = {100*hh/max(hn,1):.1f}%"
        sv = f"{kk}/{kt} = {100*kk/max(kt,1):.1f}%"
        print(f"{name:<20}{tr:>14}{hd:>17}{sv:>20}")

    print("\nReading: 'teacher pick kept' is the filter's own sanity test — a "
          "filter\nthat discards the teacher's actual choice is refuted "
          "outright, whatever\nit does to top-1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
