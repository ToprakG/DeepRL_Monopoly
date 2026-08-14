"""Does joint state-value ranking beat a difference of marginals?

One variable only: the scoring function. Weights are not touched, the split is
the same 60/40 by board as `fit_trade.py`, and nothing is fitted. See
DECISIONS D2.5 for why separability is the suspect.

    marginal difference   deed_value(req) - deed_value(offer)   (the old model)
    joint state delta     state_value(after swap) - state_value(before)
"""

from __future__ import annotations

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
    SEED, _decode, build_board,
)
from competition_agent.spec_model import deed_value, swap_delta  # noqa: E402
from monopoly_game_engine.actions import OFFSETS  # noqa: E402
from monopoly_game_engine.constants import PROPERTY_IDS  # noqa: E402

N_BOARDS = 200


def job(k):
    rng = random.Random(SEED + k)
    env, rival, dev = build_board(rng)
    n = len(PROPERTY_IDS)
    others = [i for i in range(len(env.players)) if i != 0]
    cands = [a for a in legal(env, 0)
             if OFFSETS["exch_trade"] <= a < OFFSETS["auction"]]
    if len(cands) < 2:
        return None
    chosen = int(ask_value(env, 0))
    if chosen not in cands:
        return None                    # proposal boards only
    vc = {}

    def v(p, sq):
        if (p, sq) not in vc:
            vc[(p, sq)] = deed_value(env, p, sq)
        return vc[(p, sq)]

    rows = []
    for a in cands:
        d = _decode(a, n, others)
        if d is None:
            continue
        _, off, req = d
        rows.append({"a": a,
                     "marg": v(0, req) - v(0, off),
                     "joint": swap_delta(env, 0, off, req)})
    return {"board": k, "chosen": chosen, "c": rows} if rows else None


def top1(bs, key):
    hit = 0
    for b in bs:
        best = max(b["c"], key=lambda c: (c[key], -c["a"]))
        hit += best["a"] == b["chosen"]
    return hit, len(bs)


def main() -> int:
    with managed_pool(10) as pool:
        data = [d for d in pool.map(job, range(N_BOARDS)) if d]

    rng = random.Random(20250811)
    ids = sorted(d["board"] for d in data)
    rng.shuffle(ids)
    cut = int(0.6 * len(ids))
    tr = set(ids[:cut])
    train = [d for d in data if d["board"] in tr]
    held = [d for d in data if d["board"] not in tr]

    print(f"proposal boards: train {len(train)}  held-out {len(held)}")
    print(f"{'scoring':<30}{'train top-1':>16}{'held-out top-1':>18}")
    print("-" * 64)
    for key, label in (("marg", "marginal difference (old)"),
                       ("joint", "joint state delta (new)")):
        a, b = top1(train, key)
        c, d = top1(held, key)
        print(f"{label:<30}"
              f"{f'{a}/{b} = {100*a/max(b,1):.1f}%':>16}"
              f"{f'{c}/{d} = {100*c/max(d,1):.1f}%':>18}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
