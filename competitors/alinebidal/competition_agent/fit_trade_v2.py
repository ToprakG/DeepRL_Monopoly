"""Fit the trade ranking on harvested real-play decisions.

Data: `probes/trade_harvest.jsonl` — 2,508 proposals from 60 teacher-driven
games, split 60/40 **by game seed** (decisions inside a game share a board).
1,520 train / 988 held-out, against 118 total in the synthetic pool.

Features are the raw quantities `analyze_trades.py` showed carry signal, all
differenced requested-minus-offered where that makes sense. The monopoly term
is deliberately absent: D2.6 showed it determines the ordering by itself and
gets it wrong, and D2.5 showed nothing built on it moves the argmax.

Weights are searched on train only; held-out is computed once per reported
configuration. Whatever wins here is then measured on held-out *play*
agreement before it ships (D2.7).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SRC = Path(__file__).resolve().parent / "probes" / "trade_harvest.jsonl"
OUT = Path(__file__).resolve().parent / "probes" / "trade_weights.json"

FEATURES = ("d_rent", "d_price", "completes", "d_ours", "off_mort", "d_houses")


def feats(c):
    r, o = c["req"], c["off"]
    return {
        "d_rent": r["rent_if_ours"] - o["rent_if_ours"],
        "d_price": (r["price"] - o["price"]) / 100.0,
        "completes": 1.0 if r["ours_in_group"] == r["group_size"] - 1 else 0.0,
        "d_ours": float(r["ours_in_group"] - o["ours_in_group"]),
        "off_mort": 1.0 if o["mortgaged"] else 0.0,
        "d_houses": float(r["houses"] - o["houses"]),
    }


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - m), min(1.0, c + m)


def top1(rs, w):
    hit = 0
    for r in rs:
        best = max(r["F"], key=lambda t: (sum(w[k] * t[1][k] for k in w),
                                          -t[0]))
        hit += best[0] == r["chosen"]
    return hit, len(rs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=6000)
    args = ap.parse_args()

    recs = [json.loads(l) for l in SRC.open()]
    prop = [r for r in recs if r["proposed"]]
    for r in prop:
        r["F"] = [(c["a"], feats(c)) for c in r["cands"]]

    rng = random.Random(20250811)
    seeds = sorted({r["seed"] for r in recs})
    rng.shuffle(seeds)
    tr = set(seeds[:int(0.6 * len(seeds))])
    train = [r for r in prop if r["seed"] in tr]
    held = [r for r in prop if r["seed"] not in tr]
    mean_c = sum(r["n_cands"] for r in prop) / len(prop)

    print(f"train {len(train)}  held-out {len(held)}  "
          f"(random top-1 ~= {100/mean_c:.2f}%)")

    base = {"d_rent": 1.0, "d_price": 0.0, "completes": 0.0,
            "d_ours": 0.0, "off_mort": 0.0, "d_houses": 0.0}
    a, b = top1(train, base)
    c, d = top1(held, base)
    print(f"\nbest single feature (rent only): train {100*a/b:.2f}%  "
          f"held-out {100*c/d:.2f}%")

    best_w, best_s = dict(base), a / b
    srng = random.Random(11)
    for i in range(args.iters):
        if i < args.iters // 2:
            w = {k: srng.uniform(-2, 3) for k in FEATURES}
        else:
            w = {k: best_w[k] + srng.gauss(0, 0.3) for k in FEATURES}
        s = top1(train, w)[0] / len(train)
        if s > best_s:
            best_w, best_s = w, s

    hc, hd = top1(held, best_w)
    p, lo, hi = wilson(hc, hd)
    print(f"\nfitted on train:")
    print("  " + "  ".join(f"{k}={best_w[k]:+.3f}" for k in FEATURES))
    print(f"  train    top-1 {100*best_s:.2f}%")
    print(f"  held-out top-1 {100*p:.2f}%  95% CI [{100*lo:.2f}, {100*hi:.2f}]")
    print(f"  vs random      {100/mean_c:.2f}%")
    print(f"  vs rent-only   {100*c/d:.2f}%")

    OUT.write_text(json.dumps({"weights": best_w, "train_top1": best_s,
                               "held_top1": p, "ci": [lo, hi]}, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
