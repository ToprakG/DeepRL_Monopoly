"""Component ablation on the full pool, plus the threshold-vs-ranking test.

Two questions, deliberately answered in one place because the second one
decides whether the first one's finding is allowed into SPEC.md.

1. **Which component drives the trade ranking?** Score candidates by each
   component of `deed_value` alone — price, rent, monopoly — and compare
   against the combined model. Run on the **full 400-board pool** (7,675
   candidates, 118 proposals), board-level 60/40 split, with a Wilson interval
   on every arm. The earlier 33/22-board read was too small to call anything
   decisive, and saying so for one column while quoting 4/33 for another was
   inconsistent.

2. **Why doesn't a dominant monopoly term break buy and auction?** If
   `max_group_rent / 2**missing` really swamps price and rent, it ought to
   damage every decision that uses `deed_value` — yet buy agrees 96–98% and
   auction 90.5%. Two candidate explanations, and they are tested rather than
   assumed:

   (a) *Buy never touches it.* Established by inspection: `_buy` calls only
       `gates_ok(env, pid, price)`. No `deed_value`, no ceiling. Buy's
       agreement is therefore evidence about the safety gates and nothing
       else, and cannot be cited either for or against the valuation.

   (b) *Auction is a threshold, not a ranking.* `_auction` compares one
       quantity against a cash gate: bid iff `high_bid + inc <= ceiling`. A
       term that dominates the ceiling shifts it far above the standing bid,
       where the comparison is insensitive to its exact size. Ranking is
       different in kind — a dominant additive term collapses the ordering
       because every candidate touching a group outranks every candidate that
       does not.

   The test perturbs the *monopoly term specifically* (not the overall scale,
   which cannot change a ranking at all) and measures both effects on the same
   arms. If auction agreement barely moves while trade top-1 moves, the
   threshold-vs-ranking distinction is real. If auction agreement collapses
   too, then the valuation is simply wrong everywhere and the distinction is a
   story rather than a finding.

Usage: python3 competition_agent/ablation_reconcile.py
"""

from __future__ import annotations

import json
import math
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
from competition_agent.probe_harness import (  # noqa: E402
    PROPERTY_IDS, ask_value, blank_board, legal,
)
from competition_agent.probes.p09b_trade_ranking import (  # noqa: E402
    SEED, build_board,
)
from competition_agent.spec_model import (  # noqa: E402
    SHORT_TURNS, gates_ok, marginal_monopoly_value, multi_turn_landings,
    rent_for,
)
from monopoly_game_engine.actions import (  # noqa: E402
    AUCTION_BID_INCREMENTS, AuctionAction, OFFSETS,
)
from monopoly_game_engine.constants import PROPERTIES  # noqa: E402

CACHE = Path(__file__).resolve().parent / "probes" / "fit_trade_features.json"
AUCTION_PASS = int(AuctionAction.PASS)
N_AUCTION = 500

# Each arm weights the three components of deed_value differently.
ARMS = {
    "combined (current)": {"price": 1.0, "rent": 1.0, "mono": 1.0},
    "price only":         {"price": 1.0, "rent": 0.0, "mono": 0.0},
    "rent only":          {"price": 0.0, "rent": 1.0, "mono": 0.0},
    "monopoly only":      {"price": 0.0, "rent": 0.0, "mono": 1.0},
    "no monopoly":        {"price": 1.0, "rent": 1.0, "mono": 0.0},
    "monopoly x0.1":      {"price": 1.0, "rent": 1.0, "mono": 0.1},
}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - m), min(1.0, c + m)


# --------------------------------------------------------------------------
# part 1 — ranking ablation on the full pool
# --------------------------------------------------------------------------
def ranking_ablation():
    data = json.loads(CACHE.read_text())
    rng = random.Random(20250811)                    # same split rule as before
    ids = sorted(d["board"] for d in data)
    rng.shuffle(ids)
    cut = int(0.6 * len(ids))
    tr = set(ids[:cut])
    train = [d for d in data if d["board"] in tr and d["proposed"]]
    held = [d for d in data if d["board"] not in tr and d["proposed"]]

    print(f"pool: {len(data)} boards, "
          f"{sum(len(b['cands']) for b in data)} candidates, "
          f"{sum(1 for b in data if b['proposed'])} proposals")
    print(f"split: train {len(train)} proposals / held-out {len(held)} "
          f"proposals\n")

    def top1(boards, w):
        hit = 0
        for b in boards:
            best = max(b["cands"],
                       key=lambda c: (sum(w.get(f, 0.0) * c["d"][f]
                                          for f in w), -c["action"]))
            hit += best["action"] == b["chosen"]
        return hit, len(boards)

    # a random-pick reference, so "better than chance" is checkable
    mean_c = (sum(len(b["cands"]) for b in held) / len(held)) if held else 0
    print(f"{'arm':<22}{'train top-1':>22}{'held-out top-1':>26}")
    print("-" * 72)
    rows = []
    for name, w in ARMS.items():
        a, b = top1(train, w)
        c, d = top1(held, w)
        _, lo1, hi1 = wilson(a, b)
        _, lo2, hi2 = wilson(c, d)
        rows.append((name, c, d, lo2, hi2))
        print(f"{name:<22}"
              f"{f'{a}/{b} {100*a/max(b,1):5.1f}% [{100*lo1:4.1f},{100*hi1:4.1f}]':>22}"
              f"{f'{c}/{d} {100*c/max(d,1):5.1f}% [{100*lo2:4.1f},{100*hi2:4.1f}]':>26}")
    print("-" * 72)
    print(f"random-pick reference: ~{100/max(mean_c,1):.1f}% "
          f"(mean {mean_c:.1f} candidates per board)")
    return rows


# --------------------------------------------------------------------------
# part 2 — does the same perturbation damage auction (a threshold decision)?
# --------------------------------------------------------------------------
def _ceiling(env, pid, sq, w):
    price = PROPERTIES[sq]["price"]
    prop = env.properties[sq]
    saved = prop.owner
    prop.owner = pid
    try:
        rent = 0.0
        for opp in env.players:
            if opp.player_id == pid or opp.bankrupt:
                continue
            for land, p in multi_turn_landings(opp.position, SHORT_TURNS):
                if land == sq:
                    rent += p * rent_for(env, sq)
    finally:
        prop.owner = saved
    mono = marginal_monopoly_value(env, pid, sq)
    return max(0.0, w["price"] * price + w["rent"] * rent + w["mono"] * mono)


def _auction_job(k):
    rng = random.Random(SEED + 50000 + k)
    env, rival, dev = build_board(rng)
    free = [s for s in PROPERTY_IDS if env.properties[s].owner is None]
    if not free:
        return None
    sq = rng.choice(free)
    env.phase = "auction"
    env.auction_property_id = sq
    env.auction_high_bid = rng.choice([0, 30, 80, 150, 260, 420, 700])
    env.auction_high_bidder = rival
    env.auction_bidders = [0, 1, 2, 3]
    env.auction_current_pid = 0
    env.players[0].cash = rng.choice([120, 300, 700, 1400, 2600])

    lg = legal(env, 0)
    if len(lg) < 2:
        return None
    t = int(ask_value(env, 0))

    out = {}
    for name, w in ARMS.items():
        ceil = _ceiling(env, 0, sq, w)
        high = env.auction_high_bid
        best = None
        for i, inc in enumerate(AUCTION_BID_INCREMENTS):
            a = AUCTION_PASS + 1 + i
            if a not in lg:
                continue
            total = high + inc
            if total > ceil:
                continue
            if not gates_ok(env, 0, total):
                continue
            best = a
        out[name] = (best if best is not None else AUCTION_PASS) == t
    return out


def auction_reconcile():
    with managed_pool(10) as pool:
        rows = [r for r in pool.map(_auction_job, range(N_AUCTION)) if r]
    print(f"\n\nauction states measured: {len(rows)}")
    print(f"{'arm':<22}{'auction agreement':>30}")
    print("-" * 52)
    res = {}
    for name in ARMS:
        k = sum(1 for r in rows if r[name])
        p, lo, hi = wilson(k, len(rows))
        res[name] = p
        print(f"{name:<22}"
              f"{f'{k}/{len(rows)} {100*p:5.1f}% [{100*lo:4.1f},{100*hi:4.1f}]':>30}")
    return res


def main() -> int:
    print("=" * 72)
    print("PART 1 — trade ranking, full pool, Wilson 95% intervals")
    print("=" * 72)
    rank = ranking_ablation()

    print("\n" + "=" * 72)
    print("PART 2 — same perturbations on auction (a THRESHOLD decision)")
    print("=" * 72)
    auc = auction_reconcile()

    print("\n" + "=" * 72)
    print("RECONCILIATION")
    print("=" * 72)
    base_r = dict((n, c / max(d, 1)) for n, c, d, _, _ in rank)
    print(f"{'arm':<22}{'trade top-1':>14}{'auction':>12}"
          f"{'Δtrade':>10}{'Δauction':>11}")
    print("-" * 69)
    b_r = base_r["combined (current)"]
    b_a = auc["combined (current)"]
    for name in ARMS:
        print(f"{name:<22}{100*base_r[name]:>13.1f}%{100*auc[name]:>11.1f}%"
              f"{100*(base_r[name]-b_r):>+9.1f}{100*(auc[name]-b_a):>+10.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
