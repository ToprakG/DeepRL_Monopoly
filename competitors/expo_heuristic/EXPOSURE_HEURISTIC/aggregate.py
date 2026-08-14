"""Pool per-seed arena JSON blocks into one seat-balanced summary."""

import json
import math
import sys
from pathlib import Path


def wilson(wins, games):
    if games <= 0:
        return (0.0, 1.0)
    z = 1.959963984540054
    rate = wins / games
    denom = 1 + z * z / games
    center = (rate + z * z / (2 * games)) / denom
    radius = z * math.sqrt(
        rate * (1 - rate) / games + z * z / (4 * games * games)
    ) / denom
    return (max(0.0, center - radius), min(1.0, center + radius))


def main(paths):
    totals, games, rounds, elapsed = {}, 0, [], 0.0
    for path in paths:
        try:
            data = json.loads(Path(path).read_text())
        except Exception:
            continue
        games += data["scored_games"]
        rounds.append(data["mean_rounds"] * data["scored_games"])
        elapsed = max(elapsed, data["elapsed_s"])
        for policy, stats in data["policies"].items():
            slot = totals.setdefault(policy, {"a": 0, "w": 0, "s": 0, "nw": 0.0})
            slot["a"] += stats["appearances"]
            slot["w"] += stats["wins"]
            slot["s"] += round(stats["survival_rate"] * stats["appearances"])
            slot["nw"] += stats["mean_net_worth"] * stats["appearances"]

    print(f"blocks={len(paths)}  scored_games={games}  "
          f"mean_rounds={sum(rounds)/max(1,games):.1f}")
    print(f"{'policy':16s} {'n':>4s} {'wins':>5s} {'win%':>7s} "
          f"{'wilson95':>16s} {'surv%':>7s} {'mean_nw':>9s}")
    for policy, s in sorted(totals.items(), key=lambda x: -x[1]["w"] / max(1, x[1]["a"])):
        lo, hi = wilson(s["w"], s["a"])
        print(f"{policy:16s} {s['a']:4d} {s['w']:5d} "
              f"{100*s['w']/max(1,s['a']):6.1f}% "
              f"[{100*lo:5.1f},{100*hi:5.1f}] "
              f"{100*s['s']/max(1,s['a']):6.1f}% "
              f"{s['nw']/max(1,s['a']):9.0f}")


if __name__ == "__main__":
    main(sys.argv[1:])
