"""Evaluation set for the families ordinary play never reaches.

Why
---
The full held-out breakdown showed jail and liquidation are effectively
untested: 2 jail decisions across 16 games, and **zero** `sell_house`,
`sell_hotel`, `sell_prop` or `DECLARE_BANKRUPT` — the teacher simply never
went into debt. SPEC F1–F5 (liquidation order) and G1–G5 (jail) therefore have
strong probe evidence but no agreement validation at all, and their headline
rates were noise on tiny samples.

This builds the missing population directly: diverse boards that *start* in
debt or in jail, so the clone and the teacher are both forced through those
branches. Agreement is then measured the same way as ordinary play.

Board diversity is a requirement here for the same reason as 9b (see
DECISIONS D2.2): a single debt configuration would tell us about that
configuration and nothing else. Deed allocation, development, positions, cash,
debt size, creditor, jail turn and card holding are all randomised, and the
diversity achieved is reported rather than assumed.

Output: probes/p10_debt_jail_eval.csv
"""

from __future__ import annotations

import collections
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from competition_agent.proc import managed_pool  # noqa: E402
from competition_agent.probe_harness import (  # noqa: E402
    COLOR_GROUPS, PROPERTY_IDS, ProbeWriter, ask_value, blank_board, describe,
    legal, set_pre_roll,
)
from competition_agent.spec_policy import SpecPolicy  # noqa: E402
from monopoly_game_engine.actions import OFFSETS, action_to_description  # noqa: E402

SEED = 880001
N_DEBT = 250
N_JAIL = 250
MAX_CHAIN = 14          # follow a debt through its whole liquidation sequence


def family(action: int) -> str:
    best = "binary"
    for name, start in sorted(OFFSETS.items(), key=lambda kv: kv[1]):
        if action >= start:
            best = name
    return action_to_description(action) if best == "binary" else best


def _random_board(rng):
    _, env = blank_board(seed=SEED)
    deeds = list(PROPERTY_IDS)
    rng.shuffle(deeds)

    n_own = rng.randint(3, 8)
    ours = deeds[:n_own]
    for sq in ours:
        env.properties[sq].owner = 0
        env.players[0].properties.append(env.properties[sq])

    idx = n_own
    for pid in (1, 2, 3):
        take = deeds[idx:idx + rng.randint(2, 6)]
        idx += len(take)
        for sq in take:
            env.properties[sq].owner = pid
            env.players[pid].properties.append(env.properties[sq])
    env._update_monopolies()

    for squares in COLOR_GROUPS.values():
        if all(env.properties[s].is_monopoly for s in squares) and rng.random() < 0.6:
            h = rng.randint(1, 5)
            for s in squares:
                if env.properties[s].is_real_estate:
                    env.properties[s].houses = h

    for sq in ours:
        if rng.random() < 0.2 and env.properties[sq].houses == 0:
            env.properties[sq].mortgaged = True

    for p in env.players:
        p.position = rng.randrange(40)
        p.cash = rng.choice([0, 50, 150, 400, 900, 1800])
    return env


def _job(item):
    kind, k = item
    rng = random.Random(SEED + k * 7 + (0 if kind == "debt" else 3))
    env = _random_board(rng)
    teacher = __import__(
        "ASU_FROZEN_TEACHER", fromlist=["ASUValueV1"]).ASUValueV1(0)
    clone = SpecPolicy(0)
    rows = []

    if kind == "debt":
        creditor = rng.choice([1, 2, 3])
        env.current_turn_idx = 0
        env.phase = "post_roll"
        env.has_rolled = True
        env.players[0].cash = rng.choice([0, 20, 80])
        env.debt_player = 0
        env.debt_creditor = creditor
        env.debt_amount = rng.choice([120, 300, 650, 1100, 1800, 2600])

        # follow the whole liquidation chain, not just its first step
        from ASU_FROZEN_TEACHER.evaluate import _new_seeded_game  # noqa: F401
        for _ in range(MAX_CHAIN):
            if env.done or env.debt_player != 0:
                break
            lg = legal(env, 0)
            if len(lg) < 2:
                break
            t = int(teacher.choose_action(env))
            try:
                c = int(clone.choose_action(env))
            except Exception as exc:                       # noqa: BLE001
                rows.append((kind, "ERROR", f"{type(exc).__name__}", "", False))
                break
            rows.append((kind, family(t), action_to_description(t),
                         action_to_description(c), t == c))
            # advance along the TEACHER's choice: off-policy, so clone errors
            # do not compound and each step is judged on the same state
            try:
                env.step(t)
            except Exception:                              # noqa: BLE001
                break
    else:
        p = env.players[0]
        p.in_jail = True
        p.position = 10
        p.jail_turns = rng.randint(0, 3)
        p.gooj_card = rng.random() < 0.5
        p.cash = rng.choice([0, 40, 60, 150, 300, 800, 2000])
        env.current_turn_idx = 0
        for phase in ("pre_roll", "post_roll"):
            env.phase = phase
            env.has_rolled = False
            lg = legal(env, 0)
            if len(lg) < 2:
                continue
            t = int(teacher.choose_action(env))
            try:
                c = int(clone.choose_action(env))
            except Exception as exc:                       # noqa: BLE001
                rows.append((kind, "ERROR", f"{type(exc).__name__}", "", False))
                continue
            rows.append((f"jail_{phase}", family(t),
                         action_to_description(t),
                         action_to_description(c), t == c))
    return rows


def main() -> int:
    jobs = ([("debt", k) for k in range(N_DEBT)]
            + [("jail", k) for k in range(N_JAIL)])
    with managed_pool(10) as pool:
        batches = pool.map(_job, jobs)
    rows = [r for b in batches for r in b]

    with ProbeWriter("p10_debt_jail_eval",
                     ["kind", "family", "teacher", "clone", "agree"]) as out:
        for r in rows:
            out.write(kind=r[0], family=r[1], teacher=r[2], clone=r[3],
                      agree=r[4])

    by = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        by[r[1]][0] += int(r[4])
        by[r[1]][1] += 1
    bykind = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        bykind[r[0]][0] += int(r[4])
        bykind[r[0]][1] += 1

    print(f"decisions collected: {len(rows)}\n")
    print(f"{'family':<20}{'agree':>8}{'n':>8}{'rate':>9}")
    print("-" * 45)
    for f, (a, n) in sorted(by.items(), key=lambda kv: -kv[1][1]):
        print(f"{f:<20}{a:>8}{n:>8}{100*a/n:>8.1f}%")
    print("-" * 45)
    print(f"\n{'scenario':<20}{'agree':>8}{'n':>8}{'rate':>9}")
    for f, (a, n) in sorted(bykind.items()):
        print(f"{f:<20}{a:>8}{n:>8}{100*a/n:>8.1f}%")
    tot_a = sum(int(r[4]) for r in rows)
    print(f"\nTOTAL {tot_a}/{len(rows)} = {100*tot_a/max(len(rows),1):.1f}%")
    print(f"wrote {out.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
