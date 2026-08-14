"""What does the teacher's trade choice actually correlate with?

D2.5 listed a third possible outcome for the component ablation: that no
component carries the signal and the teacher is not ranking trades by a
state-value difference at all. That outcome landed, so this measures the
decision instead of proposing another valuation.

Method: for every harvested proposal, score the candidate set by a single
candidate feature at a time and record where the teacher's actual choice
lands. A feature that ranks the choice first far more often than chance is
carrying signal; one that does not, is not. No fitting, no weights — each
scoring is one raw quantity, so the result says which quantity the decision
tracks rather than how to combine several.

Split by game seed (decisions within a game share a board and are correlated).
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SRC = Path(__file__).resolve().parent / "probes" / "trade_harvest.jsonl"


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - m), min(1.0, c + m)


# Each scorer takes one candidate {off, req, tgt} and returns a number.
def completes_group(c):
    r = c["req"]
    return 1.0 if r["ours_in_group"] == r["group_size"] - 1 else 0.0


SCORERS = {
    "price(req) - price(off)":      lambda c: c["req"]["price"] - c["off"]["price"],
    "rent(req) - rent(off)":        lambda c: c["req"]["rent_if_ours"] - c["off"]["rent_if_ours"],
    "req completes our group":      completes_group,
    "ours_in_group(req)":           lambda c: c["req"]["ours_in_group"],
    "ours_in_group(req) - (off)":   lambda c: c["req"]["ours_in_group"] - c["off"]["ours_in_group"],
    "theirs_in_group(off)":         lambda c: c["off"]["theirs_in_group"],
    "-theirs_in_group(off)":        lambda c: -c["off"]["theirs_in_group"],
    "base_rent(req) - (off)":       lambda c: c["req"]["base_rent"] - c["off"]["base_rent"],
    "req not mortgaged":            lambda c: 0.0 if c["req"]["mortgaged"] else 1.0,
    "off mortgaged (dump it)":      lambda c: 1.0 if c["off"]["mortgaged"] else 0.0,
    "houses(req) - houses(off)":    lambda c: c["req"]["houses"] - c["off"]["houses"],
}


def main() -> int:
    recs = [json.loads(l) for l in SRC.open()]
    prop = [r for r in recs if r["proposed"]]

    rng = random.Random(20250811)
    seeds = sorted({r["seed"] for r in recs})
    rng.shuffle(seeds)
    cut = int(0.6 * len(seeds))
    tr_seeds = set(seeds[:cut])
    train = [r for r in prop if r["seed"] in tr_seeds]
    held = [r for r in prop if r["seed"] not in tr_seeds]

    mean_c = sum(r["n_cands"] for r in prop) / len(prop)
    print(f"proposals: {len(prop)}  (train {len(train)} / held-out {len(held)})")
    print(f"mean candidates per decision: {mean_c:.1f}  "
          f"=> random top-1 ~= {100/mean_c:.2f}%\n")

    def top1(rs, fn):
        hit = 0
        for r in rs:
            best = max(r["cands"], key=lambda c: (fn(c), -c["a"]))
            hit += best["a"] == r["chosen"]
        return hit, len(rs)

    print(f"{'single-feature scorer':<30}{'train top-1':>20}{'held-out top-1':>24}")
    print("-" * 74)
    results = []
    for name, fn in SCORERS.items():
        a, b = top1(train, fn)
        c, d = top1(held, fn)
        p, lo, hi = wilson(c, d)
        results.append((c / max(d, 1), name, lo, hi))
        print(f"{name:<30}{f'{100*a/max(b,1):6.2f}%':>20}"
              f"{f'{100*p:6.2f}% [{100*lo:.2f},{100*hi:.2f}]':>24}")
    print("-" * 74)
    results.sort(reverse=True)
    print(f"best single feature: {results[0][1]}  "
          f"({100*results[0][0]:.2f}%, CI [{100*results[0][2]:.2f}, "
          f"{100*results[0][3]:.2f}])")
    print(f"random reference   : {100/mean_c:.2f}%")

    # what the chosen deed looks like, versus the pool it was chosen from
    print("\n=== chosen candidate vs the candidate pool ===")
    def avg(rs, sel, key):
        v = [sel(c)[key] for r in rs for c in r["cands"]]
        return sum(v) / len(v) if v else 0.0

    def avg_chosen(rs, side, key):
        v = []
        for r in rs:
            c = next((c for c in r["cands"] if c["a"] == r["chosen"]), None)
            if c:
                v.append(c[side][key])
        return sum(v) / len(v) if v else 0.0

    print(f"{'quantity':<28}{'chosen':>10}{'pool avg':>11}")
    for side, key in (("req", "price"), ("req", "rent_if_ours"),
                      ("req", "ours_in_group"), ("req", "base_rent"),
                      ("off", "price"), ("off", "rent_if_ours"),
                      ("off", "ours_in_group"), ("off", "theirs_in_group")):
        ch = avg_chosen(prop, side, key)
        po = avg(prop, lambda c, s=side: c[s], key)
        print(f"{side+'.'+key:<28}{ch:>10.2f}{po:>11.2f}")

    # does it ever request a deed it cannot use?
    comp = sum(1 for r in prop
               for c in r["cands"] if c["a"] == r["chosen"]
               and c["req"]["ours_in_group"] == c["req"]["group_size"] - 1)
    print(f"\nchoices where the requested deed completes our group: "
          f"{comp}/{len(prop)} = {100*comp/len(prop):.1f}%")
    tgt = Counter(c["tgt"] for r in prop for c in r["cands"]
                  if c["a"] == r["chosen"])
    print(f"target player distribution: {dict(tgt)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
