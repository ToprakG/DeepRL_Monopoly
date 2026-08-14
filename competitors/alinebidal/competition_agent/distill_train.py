"""Phase 3 — train the trade head on collected DAgger data.

Model
-----
A masked-softmax scorer: the 300-dim observation goes through an MLP that emits
one logit per action in the trade range, illegal actions are masked to -inf,
and the loss is cross-entropy against the teacher's chosen action. This is a
ranking objective over exactly the candidate set the policy will face, which is
what the hand-fitted ranker was doing badly (29.86% top-1, capturing 37.3% of
the oracle ceiling — D2.15).

The action space is restricted to the trade families the measurements justify
owning (buy_trade / sell_trade / exch_trade / ACCEPT / DECLINE); everything
else stays with the rules at 90-99% agreement and +0.0pp measured cost.

Evaluation is top-1 on a held-out split **by game seed**, never by decision:
decisions inside a game share a board and would leak.
"""

from __future__ import annotations

import argparse
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

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from monopoly_game_engine.actions import OFFSETS  # noqa: E402

DAGGER = Path(__file__).resolve().parent / "probes" / "dagger"
CKPT = Path(__file__).resolve().parent / "trade_head.pt"

TRADE_LO, TRADE_HI = OFFSETS["buy_trade"], OFFSETS["auction"]
ACCEPT, DECLINE = 7, 8
# Compact index space: the trade block, plus accept/decline appended.
N_TRADE = TRADE_HI - TRADE_LO
N_OUT = N_TRADE + 2
OBS_DIM = 300


def to_idx(a: int) -> int:
    if a == ACCEPT:
        return N_TRADE
    if a == DECLINE:
        return N_TRADE + 1
    return a - TRADE_LO


def from_idx(i: int) -> int:
    if i == N_TRADE:
        return ACCEPT
    if i == N_TRADE + 1:
        return DECLINE
    return i + TRADE_LO


def relevant(a: int) -> bool:
    return TRADE_LO <= a < TRADE_HI or a in (ACCEPT, DECLINE)


class TradeHead(nn.Module):
    def __init__(self, hidden: int = 512, dropout: float = 0.0):
        super().__init__()
        # Capacity and dropout are configurable because the first run
        # overfitted hard: train 97.76% against held-out 38.51%, with held-out
        # peaking at epoch 5 and declining after (D3.5).
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, N_OUT),
        )

    def forward(self, obs):
        return self.net(obs)


def load(paths):
    rows = []
    for p in paths:
        for line in Path(p).open():
            if not line.strip():
                continue
            r = json.loads(line)
            legal = [a for a in r["legal"] if relevant(a)]
            # Only supervise states where the label is inside our scope and a
            # real choice exists.
            if len(legal) >= 2 and relevant(r["label"]):
                rows.append((r["seed"], r["obs"], legal, r["label"]))
    return rows


def batches(rows, bs, shuffle=True):
    idx = list(range(len(rows)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [rows[j] for j in idx[i:i + bs]]
        obs = torch.tensor([c[1] for c in chunk], dtype=torch.float32)
        mask = torch.full((len(chunk), N_OUT), float("-inf"))
        for k, c in enumerate(chunk):
            for a in c[2]:
                mask[k, to_idx(a)] = 0.0
        tgt = torch.tensor([to_idx(c[3]) for c in chunk], dtype=torch.long)
        yield obs, mask, tgt


def top1(model, rows, bs=256):
    model.eval()
    hit = n = 0
    with torch.no_grad():
        for obs, mask, tgt in batches(rows, bs, shuffle=False):
            pred = (model(obs) + mask).argmax(dim=1)
            hit += (pred == tgt).sum().item()
            n += len(tgt)
    return hit, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default=str(CKPT))
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--wd", type=float, default=0.0)
    args = ap.parse_args()

    torch.manual_seed(20250811)
    random.seed(20250811)

    rows = load(args.data)
    seeds = sorted({r[0] for r in rows})
    random.Random(20250811).shuffle(seeds)
    tr = set(seeds[:int(0.7 * len(seeds))])
    train = [r for r in rows if r[0] in tr]
    held = [r for r in rows if r[0] not in tr]
    print(f"states {len(rows)}  train {len(train)}  held-out {len(held)}  "
          f"(split by game seed)")
    if not train or not held:
        print("not enough data")
        return 1

    model = TradeHead(hidden=args.hidden, dropout=args.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.wd)
    print(f"hidden={args.hidden} dropout={args.dropout} wd={args.wd}")
    lossf = nn.CrossEntropyLoss()

    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for obs, mask, tgt in batches(train, args.bs):
            opt.zero_grad()
            loss = lossf(model(obs) + mask, tgt)
            loss.backward()
            opt.step()
            tot += loss.item()
        h, n = top1(model, held)
        acc = h / max(n, 1)
        if acc > best:
            best = acc
            torch.save({"state_dict": model.state_dict(),
                        "held_top1": acc, "n_held": n,
                        "hidden": args.hidden,
                        "dropout": args.dropout}, args.out)
        if ep % 5 == 0 or ep == 1:
            th, tn = top1(model, train)
            print(f"  ep {ep:>3}  loss {tot/max(len(train)//args.bs,1):6.3f}  "
                  f"train top-1 {100*th/tn:5.2f}%  "
                  f"held-out top-1 {100*acc:5.2f}%")

    print(f"\nbest held-out top-1: {100*best:.2f}%")
    print(f"hand-fitted ranker reference: 29.86%  (D2.9)")
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
