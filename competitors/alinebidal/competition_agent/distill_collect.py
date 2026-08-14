"""Phase 3 — DAgger data collection for the trade families.

Scope, and why it is narrow
---------------------------
The brief describes a hybrid where "rules decide when confident, the network
breaks ties / covers uncovered states". The measurements say exactly where
that boundary sits, so the network is aimed there rather than everywhere:

    family           agreement   win-rate cost (D2.13)
    ROLL_DICE           99.5%        +0.0pp
    BUY_PROPERTY        98.4%        +0.0pp
    turn flow           96.9%        +0.0pp
    auction             90.5%        +0.0pp
    liquidation         23.8%        +0.0pp
    trade reply         78.1%        +6.7pp
    trade proposal       8.0%       +13.3pp

Every family except the two trade ones recovers **zero** win rate when pinned
to the teacher's ground truth, so a network covering them can only add risk and
latency. Liquidation is the clearest case: 23.8% agreement, the worst in the
project, and worth nothing. The learned component therefore replaces the rule
pipeline only for `exch_trade` / `buy_trade` / `sell_trade` proposals and
`ACCEPT_TRADE` / `DECLINE_TRADE` replies. Everything else keeps the rules,
which are already at 90-99%.

Why a network rather than more hand-fitting
-------------------------------------------
D2.15 measured the hand-fitted ranker capturing 37.3% of the oracle ceiling
(+7.9pp of +21.3pp). D2.5, D2.6 and D2.12 each independently established that
the limit is the *feature set*, not the weights. A model over the engine's
300-dim observation is not confined to the four hand-chosen features that cap
the current ranker at 29.86% top-1.

DAgger, not plain distillation
------------------------------
D2.11 flagged covariate shift: the existing fit was harvested from
teacher-vs-teacher play, but the states that decide a match are the ones the
*clone* reaches — and the clone reaches worse positions (it bankrupts in
86-93% of head-to-heads against the teacher's 60%). So each iteration collects
states from the CURRENT policy driving, and asks the teacher only for the
label. Iteration 0 is the teacher driving, which reproduces the existing
harvest and gives a comparison point.

Output: one .jsonl per iteration under probes/dagger/, holding the 300-dim
observation, the legal action list, and the teacher's chosen action.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ASU_FROZEN_TEACHER import ASUValueV1  # noqa: E402
from ASU_FROZEN_TEACHER.evaluate import _ScriptedAdapter, _new_seeded_game  # noqa: E402
from monopoly_game_engine.actions import OFFSETS  # noqa: E402
from monopoly_game_engine.agents_fixed import FP_AGENT_CLASSES  # noqa: E402

from competition_agent.proc import managed_pool  # noqa: E402
from competition_agent.spec_policy import SpecPolicy  # noqa: E402

OUTDIR = Path(__file__).resolve().parent / "probes" / "dagger"

# The two families the network is responsible for, by action-id range.
TRADE_LO, TRADE_HI = OFFSETS["buy_trade"], OFFSETS["auction"]
ACCEPT, DECLINE = 7, 8


def is_trade_decision(legal) -> bool:
    """True if this state offers a trade choice the network would own."""
    has_offer = any(TRADE_LO <= a < TRADE_HI for a in legal)
    has_reply = ACCEPT in legal and DECLINE in legal
    return has_offer or has_reply


def _play(job):
    seed, driver, max_steps = job
    game = _new_seeded_game(seed)
    env = game.env
    teacher = ASUValueV1(0)
    student = SpecPolicy(0) if driver == "student" else None
    opp = {i: _ScriptedAdapter(FP_AGENT_CLASSES[i - 1](i), i) for i in (1, 2, 3)}

    rows, steps = [], 0
    while steps < max_steps and not env.done:
        actor = env.whose_turn()
        if actor == 0:
            legal = [int(a) for a in env.get_allowed_actions(0)]
            label = int(teacher.choose_action(env))       # always the label
            if len(legal) > 1 and is_trade_decision(legal):
                rows.append({
                    "seed": seed,
                    "obs": [round(float(x), 5)
                            for x in env._get_state(0).tolist()],
                    "legal": legal,
                    "label": label,
                })
            if driver == "student":
                try:
                    act = int(student.choose_action(env))
                    if act not in set(legal):
                        act = label
                except Exception:                        # noqa: BLE001
                    act = label
            else:
                act = label
        else:
            act = int(opp[actor].choose_action(env))
        game.step(act)
        steps += 1
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", choices=["teacher", "student"],
                    default="student",
                    help="who acts. 'student' is the DAgger setting: states "
                         "come from the current policy, labels from the "
                         "teacher (D2.11).")
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--seed-base", type=int, default=940000)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"iter{args.iteration}_{args.driver}.jsonl"

    # Stream per game and resume by seed. This is the THIRD time this harness
    # pattern was needed (bench.py D0.8, pinned_ablation, now here) and the
    # second time it was learned by losing work: 2h11 of collection was
    # discarded because pool.map holds everything in memory until the end.
    # See DECISIONS D3.4.
    done_seeds = set()
    if out.exists() and not args.no_resume:
        for line in out.open():
            if line.strip():
                done_seeds.add(json.loads(line)["seed"])
        if done_seeds:
            print(f"resuming: {len(done_seeds)} seed(s) already collected",
                  flush=True)

    jobs = [(args.seed_base + args.iteration * 10000 + k, args.driver,
             args.max_steps) for k in range(args.seeds)
            if args.seed_base + args.iteration * 10000 + k not in done_seeds]
    print(f"{len(jobs)} game(s) to play, {len(done_seeds)} reused", flush=True)

    rows = []
    if out.exists() and not args.no_resume:
        rows = [json.loads(l) for l in out.open() if l.strip()]

    import time
    t0 = time.time()
    with out.open("a") as sink, managed_pool(args.workers) as pool:
        for i, batch in enumerate(pool.imap_unordered(_play, jobs), 1):
            for r in batch:
                sink.write(json.dumps(r) + "\n")
                rows.append(r)
            sink.flush()
            if i % 10 == 0 or i == len(jobs):
                el = time.time() - t0
                rate = i / max(el / 60, 1e-9)
                eta = (len(jobs) - i) / max(rate, 1e-9)
                print(f"  {i}/{len(jobs)} games  {len(rows)} states  "
                      f"{rate:.1f} g/min  ETA {eta:.0f} min", flush=True)

    n_prop = sum(1 for r in rows
                 if any(TRADE_LO <= a < TRADE_HI for a in r["legal"]))
    n_reply = sum(1 for r in rows if ACCEPT in r["legal"])
    took = sum(1 for r in rows if TRADE_LO <= r["label"] < TRADE_HI)
    print(f"iteration {args.iteration}  driver={args.driver}  "
          f"seeds {args.seed_base + args.iteration*10000}..")
    print(f"  trade decision states : {len(rows)}")
    print(f"    offer-capable       : {n_prop}")
    print(f"    reply-capable       : {n_reply}")
    print(f"    teacher proposed    : {took} ({100*took/max(len(rows),1):.1f}%)")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
