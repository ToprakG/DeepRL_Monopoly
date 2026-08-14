"""Pair two single-arm field runs and test the difference.

`survival_ablation.py` compares arms inside one process, which cannot work when
the two arms differ by an environment variable read at import time — a refitted
`TRADE_WEIGHTS` is fixed the moment `spec_policy` loads. So each configuration
is run as its own process over the same seed set, and this pairs the two
outputs afterwards.

The pairing is the point. Board luck dominates a 2,000-game field sample, and
it cancels exactly when both arms play the same seed from the same seat. The
noise floor for this harness was measured directly: two identical
configurations over 2,000 seeds flipped **0** games, so any discordance here is
the configuration and not the harness.

    python3 competition_agent/pair_runs.py probes/a.json probes/b.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - m), min(1.0, c + m)


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def load(path, arm):
    rows = json.loads(Path(path).read_text())
    return {r["seed"]: r for r in rows if r["arm"] == arm}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("variant")
    ap.add_argument("--arm", default="none")
    ap.add_argument("--labels", default="baseline,variant")
    ap.add_argument("--metric", default="leader_win",
                    choices=["leader_win", "decisive_win"])
    args = ap.parse_args()
    la, lb = args.labels.split(",")

    A = load(args.baseline, args.arm)
    B = load(args.variant, args.arm)
    seeds = sorted(set(A) & set(B))
    if not seeds:
        print("no shared seeds")
        return 1

    ka = sum(A[s][args.metric] for s in seeds)
    kb = sum(B[s][args.metric] for s in seeds)
    pa, loa, hia = wilson(ka, len(seeds))
    pb, lob, hib = wilson(kb, len(seeds))
    bka = sum(A[s]["bankrupt"] for s in seeds)
    bkb = sum(B[s]["bankrupt"] for s in seeds)

    print(f"paired on {len(seeds)} shared seeds   metric {args.metric}")
    print(f"  {la:<14}{ka}/{len(seeds)}  {100*pa:5.2f}%  "
          f"[{100*loa:.2f}, {100*hia:.2f}]   bankrupt {100*bka/len(seeds):.1f}%")
    print(f"  {lb:<14}{kb}/{len(seeds)}  {100*pb:5.2f}%  "
          f"[{100*lob:.2f}, {100*hib:.2f}]   bankrupt {100*bkb/len(seeds):.1f}%")

    b = sum(1 for s in seeds if A[s][args.metric] and not B[s][args.metric])
    c = sum(1 for s in seeds if B[s][args.metric] and not A[s][args.metric])
    z = (c - b) / math.sqrt(b + c) if b + c else 0.0
    pv = 2 * (1 - _phi(abs(z)))
    print(f"\n  delta {100*(pb-pa):+.2f}pp")
    print(f"  McNemar: {la}-only {b}, {lb}-only {c}, discordant {b+c}, "
          f"z {z:+.2f}, p {pv:.4f}   "
          f"{'SIGNIFICANT' if pv < 0.05 else 'not significant'}")
    # A configuration change that flips almost nothing is worth knowing about
    # even when the win rate moves: it bounds how much the change can matter.
    print(f"  games whose outcome changed at all: {b+c}/{len(seeds)} "
          f"({100*(b+c)/len(seeds):.2f}%)   harness noise floor 0/2000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
