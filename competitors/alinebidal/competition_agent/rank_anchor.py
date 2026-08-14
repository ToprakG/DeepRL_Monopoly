"""Put the fitted ranker and Candidate D on the same corpus and the same split.

Why this exists
---------------
D6.4 concluded that Candidate D failed, on 25.62% held-out top-1 against a
"hand-fitted anchor" of 29.86%. Those two numbers were never measured on the
same data: 29.86% comes from D2.9's fit on the ORIGINAL harvest (2,508
proposals), while 25.62% comes from the 120-game re-harvest Candidate D was
trained on (4,916 proposals). Comparing across corpora is the same error that
produced the oracle-ceiling mistake in Phase 4, caught there and repeated here.

This replays `spec_policy.TRADE_W` over Candidate D's corpus under Candidate
D's own seed split, then scores the trained checkpoint on the identical states,
and compares them pairwise — McNemar rather than two independent intervals,
because both models answer the same 1,327 questions and the pairing removes the
between-state variance that dominates an unpaired test.

Note on "train" vs "held-out" for the fitted ranker: both are out-of-sample for
it, since its weights were fitted on a corpus that no longer exists. The gap
between the two subsets is therefore a property of the seeds, not of fitting,
and it is reported so that Candidate D's number is read against the same
subset rather than against the corpus average.
"""

from __future__ import annotations

import argparse
import json
import math
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

from competition_agent.spec_policy import TRADE_W  # noqa: E402
from competition_agent.train_rank import RankHead, cand_features  # noqa: E402

SRC = Path(__file__).resolve().parent / "probes" / "trade_harvest.jsonl"
CKPT = Path(__file__).resolve().parent / "rank_head.pt"


def fitted_score(c) -> float:
    r, o = c["req"], c["off"]
    return (TRADE_W["d_rent"] * (r["rent_if_ours"] - o["rent_if_ours"])
            + TRADE_W["d_price"] * ((r["price"] - o["price"]) / 100.0)
            + TRADE_W["completes"] * (1.0 if r["ours_in_group"]
                                      == r["group_size"] - 1 else 0.0)
            + TRADE_W["d_ours"] * (r["ours_in_group"] - o["ours_in_group"])
            + TRADE_W["off_mort"] * (1.0 if o["mortgaged"] else 0.0)
            + TRADE_W["d_houses"] * (r["houses"] - o["houses"]))


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default=str(SRC))
    ap.add_argument("--ckpt", type=str, default=str(CKPT))
    args = ap.parse_args()

    # train_rank.load()'s filter, reproduced so the state sets are identical.
    states = []
    for line in Path(args.src).open():
        if not line.strip():
            continue
        r = json.loads(line)
        if "obs" not in r or not r["proposed"]:
            continue
        tgt = -1
        for i, c in enumerate(r["cands"]):
            if c["a"] == r["chosen"]:
                tgt = i
        if tgt < 0 or len(r["cands"]) < 2:
            continue
        states.append((r["seed"], r["obs"], r["cands"], tgt))

    # train_rank.main()'s split, reproduced with the same seed.
    seeds = sorted({s[0] for s in states})
    random.Random(20250811).shuffle(seeds)
    tr = set(seeds[: int(0.7 * len(seeds))])
    held = [s for s in states if s[0] not in tr]
    train = [s for s in states if s[0] in tr]
    mean_c = sum(len(s[2]) for s in states) / len(states)

    print(f"corpus {Path(args.src).name}: {len(states)} proposals, "
          f"{len(seeds)} seeds, mean {mean_c:.1f} candidates "
          f"-> random {100 / mean_c:.2f}%")
    print(f"split by game seed: {len(train)} train / {len(held)} held-out")

    fit_hits = {}
    for split, ss in (("train", train), ("held-out", held)):
        k = sum(1 for _, _, cands, tgt in ss
                if max(range(len(cands)), key=lambda i: fitted_score(cands[i]))
                == tgt)
        p, lo, hi = wilson(k, len(ss))
        fit_hits[split] = k
        print(f"  fitted ranker   {split:<9} {k}/{len(ss)}  {100 * p:5.2f}%  "
              f"[{100 * lo:.2f}, {100 * hi:.2f}]")

    ck = Path(args.ckpt)
    if not ck.exists():
        print(f"\n{ck.name} missing — cannot score Candidate D")
        return 1
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    model = RankHead(blob.get("hidden", 256), blob.get("dropout", 0.2))
    model.load_state_dict(blob["state_dict"])
    model.eval()

    fit_ok, net_ok = [], []
    with torch.no_grad():
        for _, obs, cands, tgt in held:
            o = np.asarray(obs, np.float32)
            cf = np.asarray([cand_features(c) for c in cands], np.float32)
            x = np.concatenate([np.repeat(o[None, :], len(cands), 0), cf], 1)
            net_ok.append(int(model(torch.from_numpy(x)).argmax().item())
                          == tgt)
            fit_ok.append(max(range(len(cands)),
                              key=lambda i: fitted_score(cands[i])) == tgt)

    kn = sum(net_ok)
    pn, lon, hin = wilson(kn, len(net_ok))
    print(f"  Candidate D     held-out  {kn}/{len(net_ok)}  {100 * pn:5.2f}%  "
          f"[{100 * lon:.2f}, {100 * hin:.2f}]"
          f"   (checkpoint records {100 * blob.get('held_top1', 0):.2f}%)")

    b = sum(1 for f, n in zip(fit_ok, net_ok) if f and not n)
    c = sum(1 for f, n in zip(fit_ok, net_ok) if n and not f)
    z = (c - b) / math.sqrt(b + c) if b + c else 0.0
    pv = 2 * (1 - _phi(abs(z)))
    print(f"\npaired on the same {len(held)} states:")
    print(f"  fitted-only right {b}   Candidate-D-only right {c}   "
          f"both/neither {len(held) - b - c}")
    print(f"  McNemar z = {z:+.2f}   p = {pv:.4f}   "
          f"{'SIGNIFICANT' if pv < 0.05 else 'not significant'}")
    print(f"  difference (D - fitted) = "
          f"{100 * (kn - fit_hits['held-out']) / len(held):+.2f}pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
