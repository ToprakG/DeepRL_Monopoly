"""Candidate D — learn the teacher's revealed preference over trades.

The signal
----------
Every proposal the teacher made states that the chosen exchange outranked every
other legal exchange in that state. That is a ranking constraint, and it has no
credit-assignment problem: the label *is* the preference, not an outcome
1,000 decisions away. D6.3 established that outcome-based learning cannot reach
early states at all, so this uses a different signal rather than a better
optimiser over the same one.

It also targets where strength actually lives — the pinned-oracle ablation
measured trade families carrying ~19 of ~23 available win-rate points, with
every other family recovering +0.0pp.

Loss: listwise softmax, not pairwise
------------------------------------
Softmax cross-entropy over each state's candidate set, chosen over
Bradley-Terry pairs. Reasons:

1. It matches how the policy is used. At play time the ranker sees one state's
   whole candidate set and takes an argmax over it. Listwise trains exactly
   that operation; pairwise trains a different one and hopes it transfers.
2. Candidate sets here average ~65 and reach several hundred. Pairwise
   expansion is quadratic in that and would drown the positive in negatives
   from the same state, while the softmax normalises per state for free.
3. The masked-softmax head from Phase 3 already worked mechanically at this
   scale; only its inputs were wrong (four hand-picked features). Keeping the
   loss and changing the representation isolates the variable being tested.

Representation
--------------
The full 300-dim observation **concatenated with** the candidate's own
features. D2.5 and the Phase 3 head both capped out on hand-picked features
alone; the observation carries board context those features cannot express,
and the candidate features carry deed-specific structure the observation
encodes only diffusely.

Split by game seed. Decisions inside one game share a board and would leak.
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
import torch.nn as nn  # noqa: E402

SRC = Path(__file__).resolve().parent / "probes" / "trade_harvest.jsonl"
CKPT = Path(__file__).resolve().parent / "rank_head.pt"
OBS_DIM = 300
CAND_DIM = 14


def cand_features(c) -> list:
    """Candidate-side features. Deliberately raw — no valuation baked in."""
    r, o = c["req"], c["off"]
    return [
        r["price"] / 400.0, o["price"] / 400.0,
        r["rent_if_ours"] / 100.0, o["rent_if_ours"] / 100.0,
        r["ours_in_group"] / 3.0, o["ours_in_group"] / 3.0,
        r["theirs_in_group"] / 3.0, o["theirs_in_group"] / 3.0,
        1.0 if r["ours_in_group"] == r["group_size"] - 1 else 0.0,
        1.0 if o["mortgaged"] else 0.0,
        1.0 if r["mortgaged"] else 0.0,
        r["houses"] / 5.0, o["houses"] / 5.0,
        r["base_rent"] / 50.0,
    ]


class RankHead(nn.Module):
    """Scores one (state, candidate) pair. Applied to every candidate, then
    softmaxed within the state."""

    def __init__(self, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM + CAND_DIM, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def sources(pattern=None):
    """Prefer the sharded corpus when it exists; fall back to the single file.

    Part C replaced the single 120-game file with one gzipped shard per game so
    a 1,000-game collection could stream and resume. The feature set below is
    unchanged, so a run over the shards differs from the original only in how
    much data it sees — which is the variable Part C tests.
    """
    if pattern:
        return sorted(Path().glob(pattern)) or [Path(pattern)]
    shards = sorted((SRC.parent / "trade_shards").glob("*.jsonl.gz"))
    return shards if shards else [SRC]


def load(pattern=None):
    """States where the teacher proposed -> (obs, candidate matrix, target)."""
    states = []
    for p in sources(pattern):
        fh = (io.TextIOWrapper(gzip.open(p, "rb")) if p.suffix == ".gz"
              else p.open())
        with fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if "obs" not in r or not r["proposed"]:
                    continue
                cands, tgt = [], -1
                for i, c in enumerate(r["cands"]):
                    cands.append(cand_features(c))
                    if c["a"] == r["chosen"]:
                        tgt = i
                if tgt < 0 or len(cands) < 2:
                    continue
                states.append((r["seed"], np.asarray(r["obs"], np.float32),
                               np.asarray(cands, np.float32), tgt))
    return states


def top1(model, states, bs_states=64):
    model.eval()
    hit = 0
    with torch.no_grad():
        for seed, obs, cands, tgt in states:
            x = np.concatenate(
                [np.repeat(obs[None, :], len(cands), 0), cands], axis=1)
            s = model(torch.from_numpy(x))
            hit += int(s.argmax().item() == tgt)
    return hit, len(states)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default=None,
                    help="harvest glob; defaults to the sharded "
                         "corpus, then the single legacy file")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--ckpt", type=str, default=str(CKPT),
                    help="where to write; the default would "
                         "overwrite the 120-game checkpoint that "
                         "D7.2 is measured against")
    args = ap.parse_args()

    torch.manual_seed(20250811)
    states = load(args.src)
    seeds = sorted({s[0] for s in states})
    random.Random(20250811).shuffle(seeds)
    tr = set(seeds[: int(0.7 * len(seeds))])
    train = [s for s in states if s[0] in tr]
    held = [s for s in states if s[0] not in tr]
    mean_c = sum(len(s[2]) for s in states) / max(len(states), 1)

    print(f"proposals {len(states)}  train {len(train)}  held-out {len(held)}  "
          f"(split by game seed)")
    print(f"mean candidates/state {mean_c:.1f}  -> random top-1 "
          f"{100/mean_c:.2f}%")
    if not train or not held:
        print("not enough data")
        return 1

    model = RankHead(args.hidden, args.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = 0.0

    for ep in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train)
        for seed, obs, cands, tgt in train:
            x = np.concatenate(
                [np.repeat(obs[None, :], len(cands), 0), cands], axis=1)
            opt.zero_grad()
            scores = model(torch.from_numpy(x)).unsqueeze(0)
            loss = nn.functional.cross_entropy(
                scores, torch.tensor([tgt]))
            loss.backward()
            opt.step()
        h, n = top1(model, held)
        acc = h / max(n, 1)
        if acc > best:
            best = acc
            torch.save({"state_dict": model.state_dict(),
                        "hidden": args.hidden, "dropout": args.dropout,
                        "held_top1": acc, "n_held": n,
                        "n_train": len(train), "n_proposals": len(states),
                        "corpus": args.src or "trade_shards"},
                       Path(args.ckpt))
        if ep % 3 == 0 or ep == 1:
            th, tn = top1(model, train[:400])
            print(f"  ep {ep:>2}  train top-1 {100*th/tn:5.2f}%  "
                  f"held-out top-1 {100*acc:5.2f}%")

    print(f"\nbest held-out top-1: {100*best:.2f}%")
    print(f"  anchors: hand-fitted 29.86%, Phase 3 head 38.51%, "
          f"random {100/mean_c:.2f}%")
    print("  top-1 is DIAGNOSTIC ONLY. Beating 38.51% is permission to run "
          "Part C, not success.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
