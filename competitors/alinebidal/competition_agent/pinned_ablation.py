"""Cost-weighted defect analysis: which family actually loses the games?

Agreement percentages rank defects by how often the clone is *wrong*. That is
the wrong ranking for a competition. A family with 23.8% agreement that fires
21 times a game costs less than one with 78% agreement that fires 800 times,
and neither number tells you which loses games — only a win rate does.

Method: pin one family at a time to the teacher's ground truth. The agent plays
`spec_policy` everywhere except decisions belonging to the pinned family, where
it takes the teacher's actual action. Everything else stays as-is. The win-rate
recovery from pinning family F is then a direct measure of what F's defects
cost, in the only currency that matters.

A decision's family is defined by the teacher's chosen action, the same
convention `agreement.py` uses, so the ablation lines up with the agreement
table it is meant to reweight.

This is a diagnostic, not a shippable policy — the oracle consults the teacher.

    none      baseline: pure spec_policy                    (~20%)
    all       upper bound: effectively the teacher          (~50%)
    <family>  spec everywhere except that family

Every arm is seat-rotated (spec on 0,2 then on 1,3) so seat effects cancel.
"""

from __future__ import annotations

import argparse
import json
import math
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
from ASU_FROZEN_TEACHER.evaluate import _new_seeded_game  # noqa: E402
from monopoly_game_engine.actions import OFFSETS, action_to_description  # noqa: E402

from competition_agent.proc import ensure_hash_seed, managed_pool  # noqa: E402
from competition_agent.spec_policy import SpecPolicy  # noqa: E402

FAMILY_SETS = {
    "none": set(),
    "trade_reply": {"ACCEPT_TRADE", "DECLINE_TRADE"},
    "trade_proposal": {"buy_trade", "sell_trade", "exch_trade"},
    "liquidation": {"mortgage", "sell_house", "sell_hotel", "sell_prop",
                    "DECLARE_BANKRUPT"},
    "unmortgage": {"unmortgage"},
    "development": {"improve_house", "improve_hotel"},
    "auction": {"auction"},
    "turn_flow": {"END_TURN", "ROLL_DICE", "DO_NOTHING"},
    # Both trade families together. Individually trade_proposal measured +10.0
    # and trade_reply -7.5, and a negative from pinning to ground truth is not
    # credible, so the two are likely interacting: proposals the clone makes
    # are answered by replies the clone also gets wrong, and fixing one side
    # while the other stays broken may be worse than fixing neither. This arm
    # separates "trade_proposal is the cost" from "trade as a whole is".
    "trade_both": {"ACCEPT_TRADE", "DECLINE_TRADE",
                   "buy_trade", "sell_trade", "exch_trade"},
    "all": {"*"},
}


def family(action: int) -> str:
    best = "binary"
    for name, start in sorted(OFFSETS.items(), key=lambda kv: kv[1]):
        if action >= start:
            best = name
    return action_to_description(action) if best == "binary" else best


class PinnedPolicy:
    """spec_policy, except decisions of `pinned` families take the teacher's."""

    def __init__(self, player_id: int, pinned: set):
        self.pid = player_id
        self.pinned = pinned
        self.spec = SpecPolicy(player_id)
        self.teacher = ASUValueV1(player_id)
        self.fired = 0
        self.total = 0

    def choose_action(self, env) -> int:
        self.total += 1

        # With nothing pinned this policy IS spec_policy, so the teacher call
        # is pure waste - and it doubled the cost of the baseline arms, which
        # are exactly the arms that need the most games. Deferred to the
        # illegal-action fallback path, which is rare. Behaviour is unchanged.
        if not self.pinned:
            try:
                a = int(self.spec.choose_action(env))
            except Exception:                              # noqa: BLE001
                return int(self.teacher.choose_action(env))
            if a in {int(x) for x in env.get_allowed_actions(self.pid)}:
                return a
            return int(self.teacher.choose_action(env))

        t = int(self.teacher.choose_action(env))
        if "*" in self.pinned or family(t) in self.pinned:
            self.fired += 1
            return t
        try:
            a = int(self.spec.choose_action(env))
        except Exception:                                  # noqa: BLE001
            return t
        legal = {int(x) for x in env.get_allowed_actions(self.pid)}
        return a if a in legal else t


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - m), min(1.0, c + m)


def _game(args):
    seed, arm, spec_seats, max_steps = args
    pinned = FAMILY_SETS[arm]
    game = _new_seeded_game(seed)
    env = game.env
    agents = {}
    for s in range(4):
        agents[s] = (PinnedPolicy(s, pinned) if s in spec_seats
                     else ASUValueV1(s))
    steps = 0
    while not env.done and steps < max_steps:
        actor = env.whose_turn()
        game.step(int(agents[actor].choose_action(env)))
        steps += 1
    active = [p.player_id for p in env.players if not p.bankrupt]
    winner = active[0] if len(active) == 1 else env.winner()
    fired = sum(agents[s].fired for s in spec_seats)
    total = sum(agents[s].total for s in spec_seats)
    return {"arm": arm, "seed": seed, "seats": list(spec_seats),
            "spec_win": winner in spec_seats,
            "fired": fired, "total": total}


def main() -> int:
    # The strong field contains `fixed-d`, whose colour-target set
    # iterates in hash order. Pin it before any game is played.
    ensure_hash_seed()
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20,
                    help="games per seat arrangement (x2 arrangements)")
    ap.add_argument("--arms", type=str, default=",".join(FAMILY_SETS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--seed-base", type=int, default=920000)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--tag", type=str, default="",
                    help="suffix for the output/partial filenames. Needed when "
                         "two runs share an arm name but differ in policy "
                         "config (e.g. the same 'none' arm with proposals on "
                         "vs off), which would otherwise collide in the "
                         "resume file keyed on (arm, seed, seats).")
    args = ap.parse_args()

    jobs = []
    for arm in args.arms.split(","):
        for k in range(args.games):
            jobs.append((args.seed_base + k, arm, (0, 2), args.max_steps))
            jobs.append((args.seed_base + k, arm, (1, 3), args.max_steps))

    # Stream results as they land and skip any already recorded. Same fix as
    # DECISIONS D0.8 applied to bench.py: a plain pool.map leaves a long run
    # opaque while it works and unrecoverable if it is interrupted. This run
    # took 1h23m+ with no visibility, which is exactly the failure that fix
    # was written for; it should have been carried over at the time.
    stem = "pinned_ablation" + (f"_{args.tag}" if args.tag else "")
    out = Path(__file__).resolve().parent / "probes" / f"{stem}.json"
    partial = out.with_suffix(".partial.jsonl")
    done = {}
    if partial.exists() and not args.no_resume:
        for line in partial.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["arm"], r["seed"], tuple(r.get("seats", ())))] = r
        if done:
            print(f"resuming: {len(done)} game(s) already recorded")

    todo = [j for j in jobs
            if (j[1], j[0], tuple(j[2])) not in done]
    rows = list(done.values())
    print(f"{len(todo)} game(s) to play, {len(rows)} reused")

    with partial.open("a") as sink, managed_pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_game, todo), 1):
            rows.append(r)
            sink.write(json.dumps(r) + "\n")
            sink.flush()
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} games", flush=True)

    out.write_text(json.dumps(rows, indent=1))

    base = None
    print(f"\n{'pinned family':<18}{'spec win rate':>26}"
          f"{'Δ vs none':>11}{'oracle fired':>15}")
    print("-" * 71)
    res = []
    for arm in args.arms.split(","):
        sub = [r for r in rows if r["arm"] == arm]
        k = sum(1 for r in sub if r["spec_win"])
        p, lo, hi = wilson(k, len(sub))
        fired = sum(r["fired"] for r in sub)
        tot = sum(r["total"] for r in sub)
        if arm == "none":
            base = p
        res.append((p - (base or 0), arm, p, lo, hi, fired, tot))
        print(f"{arm:<18}"
              f"{f'{k}/{len(sub)}  {100*p:5.1f}% [{100*lo:4.1f},{100*hi:4.1f}]':>26}"
              f"{f'{100*(p-(base or 0)):+.1f}':>11}"
              f"{f'{fired}/{tot} = {100*fired/max(tot,1):.1f}%':>15}")

    print("\n=== cost ranking (win-rate recovery per family) ===")
    for d, arm, p, lo, hi, fired, tot in sorted(res, reverse=True):
        if arm in ("none", "all"):
            continue
        share = 100 * fired / max(tot, 1)
        per = (100 * d / share) if share else 0.0
        print(f"{arm:<18} recovers {100*d:+5.1f}pp  "
              f"on {share:5.2f}% of decisions  "
              f"=> {per:+6.2f}pp per 1% of decisions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
