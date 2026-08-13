"""Replay jail field games: first set, first three houses, skipped builds."""

from __future__ import annotations

import json
import multiprocessing as mp
from collections import Counter
from pathlib import Path

from ASU_FROZEN_TEACHER.core import preserve_global_rng
from ASU_FROZEN_TEACHER.evaluate import _new_seeded_game
from monopoly_game_engine.actions import OFFSETS, ActionType
from monopoly_game_engine.constants import COLOR_GROUPS
from oracle.agent import OracleConfig
from oracle.eval_h2h import JAIL_ID, _H2HFactory, rotate_lineup
from oracle.plus_steals import full_sets

LINEUP = ("oracle-jail-v1", "slayer-v1", "underdog-v1", "inncenta-heuristic")
REAL = tuple(c for c in COLOR_GROUPS if c not in ("railroad", "utility"))
TRADE_LO = OFFSETS["buy_trade"]
AUCTION_LO = OFFSETS["auction"]
HOUSE_LO = OFFSETS["improve_house"]
SELL_H = OFFSETS["sell_house"]


def _sets(env, pid: int) -> list[str]:
    return [
        color
        for color in REAL
        if all(env.properties[sq].owner == pid for sq in COLOR_GROUPS[color])
    ]


def _houses(env, pid: int) -> int:
    return sum(max(0, min(4, int(getattr(p, "houses", 0)))) for p in env.players[pid].properties)


def _three(env, pid: int) -> bool:
    return any(int(getattr(p, "houses", 0) or 0) >= 3 for p in env.players[pid].properties)


def replay(index: int) -> dict:
    specs = rotate_lineup(LINEUP, index)
    policies = [s.policy_id for s in specs]
    us = policies.index(JAIL_ID)
    seed = 9_710_000 + index
    factory = _H2HFactory(OracleConfig(), seed=seed)
    with preserve_global_rng():
        game = _new_seeded_game(seed)
        agents = [factory.build(spec, seat) for seat, spec in enumerate(specs)]
        first_set: dict[str, dict] = {}
        first_three: dict[str, dict] = {}
        trades: Counter[str] = Counter()
        skipped_build = 0
        prev_sets = [0] * 4
        decisions = 0
        while not game.env.done and decisions < 50_000:
            actor = game.env.whose_turn()
            env = game.env
            legal = list(env.get_allowed_actions(actor))
            action = agents[actor].choose_action(env)
            if actor == us:
                if action == int(ActionType.ACCEPT_TRADE):
                    trades["accept"] += 1
                elif action == int(ActionType.DECLINE_TRADE):
                    trades["decline"] += 1
                elif TRADE_LO <= action < AUCTION_LO:
                    trades["offer"] += 1
                if _sets(env, us) and any(HOUSE_LO <= a < SELL_H for a in legal):
                    if not (HOUSE_LO <= action < SELL_H):
                        skipped_build += 1
            game.step(action)
            decisions += 1
            env = game.env
            for i, pol in enumerate(policies):
                nset = full_sets(env, i)
                if nset > prev_sets[i] and pol not in first_set:
                    first_set[pol] = {
                        "round": int(env.round),
                        "colors": _sets(env, i),
                        "houses": _houses(env, i),
                        "cash": float(env.players[i].cash),
                    }
                if _three(env, i) and pol not in first_three:
                    first_three[pol] = {
                        "round": int(env.round),
                        "houses": _houses(env, i),
                        "sets": _sets(env, i),
                        "cash": float(env.players[i].cash),
                    }
                prev_sets[i] = nset
        winner = env.winner() if env.done else None
        return {
            "index": index,
            "winner": None if winner is None else policies[winner],
            "rounds": int(env.round),
            "first_set": first_set,
            "first_three": first_three,
            "us_had_set": JAIL_ID in first_set,
            "us_had_three": JAIL_ID in first_three,
            "us_set_first": bool(first_set)
            and min(first_set, key=lambda p: first_set[p]["round"]) == JAIL_ID,
            "us_three_first": bool(first_three)
            and min(first_three, key=lambda p: first_three[p]["round"]) == JAIL_ID,
            "skipped_build_with_set": skipped_build,
            "trades": dict(trades),
        }


def main() -> None:
    with mp.get_context("spawn").Pool(6) as pool:
        rows = pool.map(replay, range(24))
    out = Path(__file__).resolve().parent / "jail_seed971_probe.json"
    out.write_text(json.dumps(rows, indent=2) + "\n")
    wins = Counter(r["winner"] for r in rows)
    print("replay winners", dict(wins))
    print("us had set", sum(r["us_had_set"] for r in rows), "/24")
    print("us set first", sum(r["us_set_first"] for r in rows), "/24")
    print("us had three", sum(r["us_had_three"] for r in rows), "/24")
    print("us three first", sum(r["us_three_first"] for r in rows), "/24")
    first_who = Counter()
    three_who = Counter()
    for r in rows:
        first_who[
            min(r["first_set"], key=lambda p: r["first_set"][p]["round"])
            if r["first_set"]
            else "(none)"
        ] += 1
        three_who[
            min(r["first_three"], key=lambda p: r["first_three"][p]["round"])
            if r["first_three"]
            else "(none)"
        ] += 1
    print("first monopoly", dict(first_who))
    print("first three houses", dict(three_who))
    print(
        "us set but never 3 houses",
        [r["index"] for r in rows if r["us_had_set"] and not r["us_had_three"]],
    )
    print(
        "mean skipped_build_with_set",
        sum(r["skipped_build_with_set"] for r in rows) / 24,
    )
    print(
        "trade offer/accept/decline",
        sum(r["trades"].get("offer", 0) for r in rows),
        sum(r["trades"].get("accept", 0) for r in rows),
        sum(r["trades"].get("decline", 0) for r in rows),
    )
    print("\n=== losses ===")
    for r in rows:
        if r["winner"] == JAIL_ID:
            continue
        who = (
            min(r["first_set"], key=lambda p: r["first_set"][p]["round"])
            if r["first_set"]
            else "(none)"
        )
        t3 = (
            min(r["first_three"], key=lambda p: r["first_three"][p]["round"])
            if r["first_three"]
            else "(none)"
        )
        fs = r["first_set"].get(who, {})
        print(
            f"g{r['index']:02d} win={r['winner']:<22} r={r['rounds']:3d} "
            f"first_set={who} {fs.get('colors')} r{fs.get('round')} "
            f"first3={t3} us_set={r['us_had_set']} us3={r['us_had_three']} "
            f"skip={r['skipped_build_with_set']} "
            f"off={r['trades'].get('offer', 0)} acc={r['trades'].get('accept', 0)} "
            f"dec={r['trades'].get('decline', 0)}"
        )
    print("\n=== wins ===")
    for r in rows:
        if r["winner"] != JAIL_ID:
            continue
        print(
            f"g{r['index']:02d} r={r['rounds']} first_set={r['first_set']} "
            f"first3={list(r['first_three'])} trades={r['trades']}"
        )


if __name__ == "__main__":
    main()
