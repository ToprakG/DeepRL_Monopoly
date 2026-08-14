"""Calibrate Candidate D's propose/don't-propose threshold on its own scale.

`calibrate_gate.py` does this for the linear scorer; the ranker needs the same
treatment for the same reason. Its scores are unnormalised logits with no
relation to the linear scorer's units, so reusing either 3.92 or the refit's
13.16 would change how often the agent proposes at all — a second change riding
along with the ranking one.

Convention matched to the shipped gate: `TRADE_GATE = 3.92` turns out to be the
*accuracy-maximising* threshold for the shipped weights on this corpus (the
grid search recovers 3.8942), not the rate-matching one. So accuracy-max is
what an A/B has to reproduce, and the resulting propose rate is printed so any
drift is visible rather than assumed away.

Threshold picked on the 700 training games, reported once on the 300 held out.
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
import torch  # noqa: E402

from competition_agent.train_rank import RankHead, cand_features  # noqa: E402

PROBES = Path(__file__).resolve().parent / "probes"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str,
                    default=str(PROBES.parent / "rank_head_1000.pt"))
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--mode", choices=["accuracy", "rate"], default="accuracy")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = RankHead(ck.get("hidden", 256), ck.get("dropout", 0.2))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    torch.set_num_threads(1)

    rows = []
    paths = sorted((PROBES / "trade_shards").glob("*.jsonl.gz"))
    with torch.no_grad():
        for p in paths:
            with io.TextIOWrapper(gzip.open(p, "rb")) as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    cands = r.get("cands") or []
                    if len(cands) < 2 or "obs" not in r:
                        continue
                    obs = np.asarray(r["obs"], np.float32)
                    cf = np.asarray([cand_features(c) for c in cands],
                                    np.float32)
                    x = np.concatenate(
                        [np.repeat(obs[None, :], len(cf), 0), cf], 1)
                    s = model(torch.from_numpy(x)).numpy()
                    rows.append((r["seed"], float(s.max()),
                                 bool(r["proposed"])))

    seeds = sorted({r[0] for r in rows})
    random.Random(20250811).shuffle(seeds)
    tr = set(seeds[: int(0.7 * len(seeds))])
    st = np.asarray([r[1] for r in rows if r[0] in tr])
    yt = np.asarray([r[2] for r in rows if r[0] in tr])
    sh = np.asarray([r[1] for r in rows if r[0] not in tr])
    yh = np.asarray([r[2] for r in rows if r[0] not in tr])

    print(f"states with >=2 exchanges: {len(rows)}  "
          f"({len(st)} train / {len(sh)} held-out)")
    print(f"teacher proposed in {100 * yh.mean():.1f}% of held-out states")

    grid = np.quantile(st, np.linspace(0.0, 0.995, 400))
    acc_gate = float(max(((((st >= g) == yt).mean()), float(g))
                         for g in grid)[1])
    rate_gate = float(np.quantile(st, 1.0 - yt.mean()))
    always = float(max(yh.mean(), 1 - yh.mean()))
    for label, g in (("accuracy-max", acc_gate), ("rate-matched", rate_gate)):
        print(f"\n  {label} gate {g:.4f}")
        print(f"    held agreement    {100 * float(((sh >= g) == yh).mean()):.2f}%"
              f"   (always-guess {100 * always:.2f}%)")
        print(f"    held propose rate {100 * float((sh >= g).mean()):.1f}%")

    gate = acc_gate if args.mode == "accuracy" else rate_gate
    out = Path(args.out)
    out.write_text(json.dumps({
        "ckpt": str(Path(args.ckpt).resolve()), "gate": gate,
        "mode": args.mode,
        "held_agreement": float(((sh >= gate) == yh).mean()),
        "held_propose_rate": float((sh >= gate).mean()),
        "held_top1": ck.get("held_top1"),
    }, indent=2))
    print(f"\nchosen ({args.mode}) gate {gate:.4f}\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
