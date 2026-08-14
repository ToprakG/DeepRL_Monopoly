"""Audit-trail integrity: differential test of the modified teacher.

Why this exists
---------------
`ASU_FROZEN_TEACHER/core.py` carries 258 added / 32 removed uncommitted lines.
The earlier evidence for "behaviour preserving" was a digest match on three
ordinary-play seeds, which is too weak: function-name analysis shows the
change adds `_fast_copy_env / _fast_copy_player / _fast_copy_property /
_fast_copy_trade_offer`, i.e. a hand-rolled replacement for deepcopy. A fast
copy that drops a rarely-populated field (auction bidders, pending trades,
debt bookkeeping) would leave ordinary play identical while silently changing
auction, jail, liquidation, bankruptcy or trade behaviour — exactly the
branches the probe corpus depends on.

Method
------
Rather than compare against a stale stored digest, run **both versions of the
policy** and compare their selected actions directly. The pre-modification
`core.py` is checked out of git into a shadow package; this script is invoked
once per version as a subprocess with that shadow package first on `sys.path`,
emits the selected action id for every decision point, and a driver diffs the
two streams.

This certifies the property that actually matters — same state, same action —
and it reads no source: each version is executed, never inspected.

Only selected action ids are recorded, so the opacity discipline of
DECISIONS.md D0.3 holds here too.

Usage
-----
    python3 competition_agent/certify_teacher.py --emit new  --out new.json
    python3 competition_agent/certify_teacher.py --emit old  --out old.json
    python3 competition_agent/certify_teacher.py --compare old.json new.json
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

REPO = Path(__file__).resolve().parents[1]

GAME_SEEDS = list(range(1, 21))          # 20 ordinary-play seeds
ROLLOUT_SEEDS = [1, 2, 3, 4, 5]          # rollout is ~460x costlier
ROLLOUT_DECISIONS = 20
MAX_STEPS = 1200


# --------------------------------------------------------------------------
# synthetic scenarios — one builder per required branch
# --------------------------------------------------------------------------
def _reset(env, cash=1500):
    for prop in env.properties.values():
        prop.owner = None
        prop.mortgaged = False
        prop.houses = 0
        prop.is_monopoly = False
    for p in env.players:
        p.cash = cash
        p.position = 0
        p.in_jail = False
        p.jail_turns = 0
        p.gooj_card = False
        p.bankrupt = False
        p.properties = []
    env.phase = "pre_roll"
    env.has_rolled = False
    env.current_turn_idx = 0
    env.turn_order = list(range(len(env.players)))
    env.debt_player = None
    env.debt_amount = 0
    env.debt_creditor = None
    env.consecutive_doubles = 0
    env.extra_roll_pending = False
    env.last_dice = (0, 0)
    env.out_of_turn_pids = []
    env.pending_trades = {}
    env.auction_property_id = None
    env.auction_high_bid = 0
    env.auction_high_bidder = None
    env.auction_bidders = []
    env.auction_current_pid = None
    env.houses_available = 32
    env.hotels_available = 12
    env.round = 1
    env.done = False
    env._update_monopolies()
    return env


def _give(env, pid, squares, houses=0, mortgaged=False):
    for sq in squares:
        prop = env.properties[sq]
        if prop.owner is not None:
            env.players[prop.owner].properties.remove(prop)
        prop.owner = pid
        prop.mortgaged = mortgaged
        env.players[pid].properties.append(prop)
    env._update_monopolies()
    for sq in squares:
        prop = env.properties[sq]
        if houses and prop.is_real_estate:
            prop.houses = houses


def scenario_auction_multi(env):
    """(a) auction for Boardwalk with all four seats still bidding."""
    _reset(env)
    _give(env, 1, [37])                       # rival holds Park Place
    env.phase = "auction"
    env.auction_property_id = 39
    env.auction_high_bid = 120
    env.auction_high_bidder = 2
    env.auction_bidders = [0, 1, 2, 3]
    env.auction_current_pid = 0
    return 0


def scenario_jail_pay(env):
    """(b1) in jail, cash available, no card -> pay-bail branch."""
    _reset(env)
    env.current_turn_idx = 0
    p = env.players[0]
    p.in_jail, p.jail_turns, p.gooj_card, p.position, p.cash = True, 1, False, 10, 900
    _give(env, 1, [31, 32, 34], houses=3)     # developed opposition
    env.phase = "pre_roll"
    return 0


def scenario_jail_card(env):
    """(b2) identical, but holding a get-out-of-jail-free card."""
    scenario_jail_pay(env)
    env.players[0].gooj_card = True
    return 0


def scenario_jail_late(env):
    """(b3) third jail turn — the forced-exit boundary."""
    scenario_jail_pay(env)
    env.players[0].jail_turns = 3
    env.players[0].cash = 60
    return 0


def scenario_debt_liquidation(env):
    """(c) rent owed exceeds cash: forced mortgage / house sale."""
    _reset(env)
    env.current_turn_idx = 0
    p = env.players[0]
    p.cash = 40
    p.position = 34
    _give(env, 0, [6, 8, 9], houses=2)        # liquidatable light blues
    _give(env, 0, [5, 15])                    # plus railroads
    _give(env, 1, [31, 32, 34], houses=4)
    env.phase = "post_roll"
    env.has_rolled = True
    env.debt_player = 0
    env.debt_amount = 1100
    env.debt_creditor = 1
    return 0


def scenario_bankruptcy(env):
    """(d) debt with nothing left to liquidate -> bankruptcy branch."""
    _reset(env)
    env.current_turn_idx = 0
    p = env.players[0]
    p.cash = 5
    p.position = 39
    _give(env, 1, [37, 39], houses=4)
    env.phase = "post_roll"
    env.has_rolled = True
    env.debt_player = 0
    env.debt_amount = 1700
    env.debt_creditor = 1
    return 0


def _pending_offer(env, sender, to, offered_sq, requested_sq,
                   cash_offered=0, cash_requested=0):
    """Inject a real incoming TradeOffer so ACCEPT/DECLINE become legal."""
    from monopoly_game_engine.env import TradeOffer

    env.pending_trades[sender] = TradeOffer(
        from_player=sender,
        to_player=to,
        offered_prop=None if offered_sq is None else env.properties[offered_sq],
        requested_prop=(
            None if requested_sq is None else env.properties[requested_sq]
        ),
        cash_offered=cash_offered,
        cash_requested=cash_requested,
    )


def scenario_trade_offers(env):
    """(e) seat 0 facing an incoming offer it should want: the completing
    orange deed for a spare railroad plus modest cash."""
    _reset(env)
    _give(env, 0, [16, 18])                   # seat 0 holds 2 of 3 oranges
    _give(env, 0, [5])                        # and a spare railroad
    _give(env, 1, [19])                       # rival holds the completing deed
    _give(env, 1, [21, 23])
    env.phase = "out_of_turn"
    env.current_turn_idx = 1
    env.out_of_turn_pids = [0, 2, 3]
    _pending_offer(env, sender=1, to=0, offered_sq=19, requested_sq=5,
                   cash_offered=0, cash_requested=50)
    return 0


def scenario_trade_offer_bad(env):
    """(e2) the same shape of offer, priced so it should be refused:
    a low-value deed demanded for the seat's own completing piece."""
    scenario_trade_offers(env)
    _pending_offer(env, sender=1, to=0, offered_sq=1, requested_sq=16,
                   cash_offered=0, cash_requested=400)
    return 0


def scenario_build_boundary(env):
    """(f) monopoly + cash at the 4-house/hotel boundary, bank constrained."""
    _reset(env)
    env.current_turn_idx = 0
    env.players[0].cash = 2000
    _give(env, 0, [16, 18, 19], houses=4)     # orange monopoly, 4 houses each
    env.houses_available = 2                  # bank nearly out of houses
    env.hotels_available = 1                  # exactly one hotel left
    env.phase = "pre_roll"
    return 0


def scenario_build_plain(env):
    """(f2) same monopoly, unconstrained bank — the control for (f)."""
    scenario_build_boundary(env)
    env.houses_available = 32
    env.hotels_available = 12
    return 0


SCENARIOS = [
    ("auction_multi_bidder", scenario_auction_multi),
    ("jail_pay_bail", scenario_jail_pay),
    ("jail_gooj_card", scenario_jail_card),
    ("jail_third_turn", scenario_jail_late),
    ("debt_liquidation", scenario_debt_liquidation),
    ("bankruptcy", scenario_bankruptcy),
    ("trade_offer_favourable", scenario_trade_offers),
    ("trade_offer_unfavourable", scenario_trade_offer_bad),
    ("build_house_hotel_boundary", scenario_build_boundary),
    ("build_unconstrained_control", scenario_build_plain),
]


# --------------------------------------------------------------------------
# emit
# --------------------------------------------------------------------------
def _load():
    """Import the teacher version selected by the ASU_SHADOW env var.

    Done inside workers as well as the parent so that a spawned process picks
    up the same version. The shadow directory is prepended to sys.path, so a
    checked-out copy of the previous core.py shadows the working tree's.
    """
    shadow = os.environ.get("ASU_SHADOW") or None
    if shadow and shadow not in sys.path:
        sys.path.insert(0, shadow)
    if str(REPO) not in sys.path:
        sys.path.append(str(REPO))
    import ASU_FROZEN_TEACHER as A
    from ASU_FROZEN_TEACHER import ASURolloutV1, ASUValueV1
    from ASU_FROZEN_TEACHER.evaluate import _ScriptedAdapter, _new_seeded_game
    from monopoly_game_engine.agents_fixed import FP_AGENT_CLASSES
    return (A, ASUValueV1, ASURolloutV1, _ScriptedAdapter, _new_seeded_game,
            FP_AGENT_CLASSES)


def _play(seed: int, variant: str, limit: int | None):
    """Record seat-0 action ids for one seeded game under one variant."""
    (_, Value, Rollout, Adapter, new_game, FIXED) = _load()
    game = new_game(seed)
    env = game.env
    seat0 = (Value if variant == "value" else Rollout)(0)
    opp = {i: Adapter(FIXED[i - 1](i), i) for i in (1, 2, 3)}
    acts, steps = [], 0
    while steps < MAX_STEPS and not env.done:
        if limit is not None and len(acts) >= limit:
            break
        actor = env.whose_turn()
        if actor == 0:
            a = int(seat0.choose_action(env))
            acts.append(a)
        else:
            a = int(opp[actor].choose_action(env))
        game.step(a)
        steps += 1
    return seed, variant, acts, steps


def _play_job(job):
    return _play(*job)


def _scenario_job(name):
    (_, Value, Rollout, _A, new_game, _F) = _load()
    build = dict(SCENARIOS)[name]
    rec = {}
    for variant, cls in (("value", Value), ("rollout", Rollout)):
        game = new_game(99)
        pid = build(game.env)
        try:
            legal = [int(x) for x in game.env.get_allowed_actions(pid)]
            rec[variant] = int(cls(pid).choose_action(game.env))
            rec[variant + "_n_legal"] = len(legal)
        except Exception as exc:                           # noqa: BLE001
            rec[variant] = f"ERROR:{type(exc).__name__}:{exc}"
    return name, rec


def emit(shadow: str | None, workers: int) -> dict:
    """Run a specific teacher version and record its selected actions."""
    import multiprocessing as mp

    from competition_agent.proc import managed_pool

    if shadow:
        os.environ["ASU_SHADOW"] = shadow
    A = _load()[0]
    origin = Path(A.__file__).resolve().parent
    out: dict = {"origin": str(origin), "games": {}, "scenarios": {}}

    jobs = [(s, "value", None) for s in GAME_SEEDS]
    jobs += [(s, "rollout", ROLLOUT_DECISIONS) for s in ROLLOUT_SEEDS]

    with managed_pool(workers) as pool:
        for seed, variant, acts, steps in pool.imap_unordered(_play_job, jobs):
            rec = out["games"].setdefault(str(seed), {})
            rec[f"{variant}_actions"] = acts
            if variant == "value":
                rec["steps"] = steps
            print(f"  seed {seed:>3} {variant:<8} {len(acts):>4} decisions",
                  flush=True)
        for name, rec in pool.imap_unordered(
            _scenario_job, [n for n, _ in SCENARIOS]
        ):
            out["scenarios"][name] = rec
            print(f"  scenario {name:<30} "
                  f"value={rec.get('value')} rollout={rec.get('rollout')}",
                  flush=True)
    return out


def compare(old_path: Path, new_path: Path) -> int:
    old = json.loads(old_path.read_text())
    new = json.loads(new_path.read_text())

    mismatches, checked = [], 0

    for seed, o in old["games"].items():
        n = new["games"].get(seed, {})
        for key in ("value_actions", "rollout_actions"):
            ov, nv = o.get(key), n.get(key)
            if ov is None and nv is None:
                continue
            checked += len(ov or [])
            if ov != nv:
                first = next(
                    (i for i, (a, b) in enumerate(zip(ov or [], nv or []))
                     if a != b),
                    min(len(ov or []), len(nv or [])),
                )
                mismatches.append(
                    f"seed {seed} {key}: diverges at decision {first} "
                    f"(len {len(ov or [])} vs {len(nv or [])})"
                )

    for name, o in old["scenarios"].items():
        n = new["scenarios"].get(name, {})
        for variant in ("value", "rollout"):
            checked += 1
            if o.get(variant) != n.get(variant):
                mismatches.append(
                    f"scenario {name} [{variant}]: "
                    f"{o.get(variant)!r} -> {n.get(variant)!r}"
                )

    print(f"old package: {old['origin']}")
    print(f"new package: {new['origin']}")
    print(f"decision points compared: {checked}")
    print(f"ordinary-play seeds: {len(old['games'])}   "
          f"synthetic scenarios: {len(old['scenarios'])}")
    if mismatches:
        print(f"\nFAIL — {len(mismatches)} mismatch(es):")
        for m in mismatches[:40]:
            print(f"  - {m}")
        return 1
    print("\nPASS — every compared decision is identical across versions.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", choices=["old", "new"])
    ap.add_argument("--shadow", type=str, default=None)
    ap.add_argument("--out", type=str)
    ap.add_argument("--compare", nargs=2, metavar=("OLD", "NEW"))
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    if args.compare:
        return compare(Path(args.compare[0]), Path(args.compare[1]))

    payload = emit(args.shadow, args.workers)
    Path(args.out).write_text(json.dumps(payload, indent=1) + "\n")
    n = sum(len(g.get("value_actions", [])) + len(g.get("rollout_actions", []))
            for g in payload["games"].values())
    print(f"emitted {args.out}: {n} action ids, "
          f"{len(payload['scenarios'])} scenarios")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
