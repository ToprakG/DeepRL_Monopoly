"""Re-derive the propose/don't-propose threshold for a re-fitted scorer.

`TRADE_GATE = 3.92` is a score floor, and a score floor is only meaningful on
the scale of the weights it was fitted with. A refit changes that scale, so
shipping new weights with the old gate would silently change how often the
agent proposes at all — a second, uncontrolled change riding along with the
one being tested.

The gate is chosen the same way the ranking is: on the teacher's own behaviour.
For every state with at least two legal exchanges the corpus records whether
the teacher proposed, so the threshold is picked to maximise agreement with
that binary decision on train, and reported once on held-out. The propose rate
it produces is printed alongside, because a gate that matches the teacher's
40.4% rate but disagrees state-by-state is not the same thing as one that
matches the decisions.

    python3 competition_agent/calibrate_gate.py --weights A --out probes/w_A.json
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import random
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from competition_agent.fit_trade_v3 import BASE, EXTRA, features  # noqa: E402

PROBES = Path(__file__).resolve().parent / "probes"
NAMES = list(BASE) + list(EXTRA)


def load_maxscores(paths, w):
    """-> (seed, best score over the state's candidates, teacher proposed?)."""
    out = []
    for p in paths:
        fh = (io.TextIOWrapper(gzip.open(p, "rb")) if p.suffix == ".gz"
              else p.open())
        with fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                cands = r.get("cands") or []
                if len(cands) < 2:
                    continue
                X = np.asarray([features(c) for c in cands], np.float32)
                out.append((r["seed"], float((X @ w).max()),
                            bool(r["proposed"])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", choices=["A", "B", "L", "shipped"],
                    default="A")
    ap.add_argument("--src", type=str, default=None)
    ap.add_argument("--fit", type=str,
                    default=str(PROBES / "trade_weights_v3.json"))
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--mode", choices=["rate", "accuracy"], default="rate",
                    help="rate: match the teacher's propose FREQUENCY, which "
                         "is the convention TRADE_GATE=3.92 follows, so an A/B "
                         "changes only the ranking. accuracy: maximise "
                         "state-by-state agreement, which here halves the "
                         "propose rate and would confound the comparison.")
    args = ap.parse_args()

    blob = json.loads(Path(args.fit).read_text())
    if args.weights == "shipped":
        from competition_agent.spec_policy import TRADE_W
        wd = dict(TRADE_W)
    else:
        wd = blob[f"weights_{args.weights}"]
    w = np.asarray([wd.get(n, 0.0) for n in NAMES], np.float32)

    paths = (sorted(Path().glob(args.src)) if args.src
             else sorted((PROBES / "trade_shards").glob("*.jsonl.gz")))
    rows = load_maxscores(paths, w)

    seeds = sorted({r[0] for r in rows})
    random.Random(20250811).shuffle(seeds)
    tr = set(seeds[: int(0.7 * len(seeds))])
    train = [r for r in rows if r[0] in tr]
    held = [r for r in rows if r[0] not in tr]

    print(f"states with >=2 exchanges: {len(rows)}  "
          f"({len(train)} train / {len(held)} held-out)")
    print(f"teacher proposed in {100 * sum(r[2] for r in rows) / len(rows):.1f}%")

    st = np.asarray([r[1] for r in train])
    yt = np.asarray([r[2] for r in train])
    grid = np.quantile(st, np.linspace(0.0, 0.995, 400))
    acc = [(float(((st >= g) == yt).mean()), float(g)) for g in grid]
    best_acc, acc_gate = max(acc)
    # Rate matching: the threshold above which exactly the teacher's share of
    # states falls. Fitted on train only, like everything else.
    rate_gate = float(np.quantile(st, 1.0 - yt.mean()))

    sh = np.asarray([r[1] for r in held])
    yh = np.asarray([r[2] for r in held])
    always = float(max(yh.mean(), 1 - yh.mean()))
    for label, g in (("accuracy-max", acc_gate), ("rate-matched", rate_gate)):
        print(f"\n  {label} gate {g:.4f}")
        print(f"    held agreement  {100 * float(((sh >= g) == yh).mean()):.2f}%"
              f"   (always-guess {100 * always:.2f}%)")
        print(f"    held propose rate {100 * float((sh >= g).mean()):.1f}%  "
              f"vs teacher {100 * yh.mean():.1f}%")

    gate = rate_gate if args.mode == "rate" else acc_gate
    ha = float(((sh >= gate) == yh).mean())
    rate = float((sh >= gate).mean())
    print(f"\nchosen ({args.mode}) gate {gate:.4f}")


    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "weights": {n: float(v) for n, v in zip(NAMES, w)},
        "gate": gate,
        "source_fit": args.weights, "mode": args.mode,
        "held_gate_agreement": ha, "held_propose_rate": rate,
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
