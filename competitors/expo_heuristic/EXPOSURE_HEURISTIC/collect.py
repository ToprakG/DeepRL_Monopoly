"""
Parallel self-play data collection for EXPO.

Two jobs, both fanned out across every available core:

``arena``   -- play games and record only the outcome. Used to measure a
               weight vector's strength.
``dataset`` -- play games and record (300-float observation, EXPO action)
               pairs. This is the distillation corpus for the DDQN / PPO /
               MonopolyZero students described in REPO_STUDY_NOTES.md.

EXPO decides in ~0.07 ms, so a single core plays a full game in about a
second. That is what makes bulk collection practical here; the ASU teacher
costs ~894 ms per decision and cannot fill a dataset at this scale.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import EXPOSURE_HEURISTIC  # noqa: F401  (bootstraps the engine import path)
from monopoly_game_engine.actions import ACTION_SPACE_SIZE, ActionType
from monopoly_game_engine.agents_fixed import FP_AGENT_CLASSES
from monopoly_game_engine.constants import NUM_PLAYERS
from monopoly_game_engine.env import MonopolyEnv

from EXPOSURE_HEURISTIC.agent import ExpoHeuristicAgent, configure

FIXED_IDS = {f"fixed-{c}": cls for c, cls in zip("abcdef", FP_AGENT_CLASSES)}


def _make(policy_id: str, seat: int):
    if policy_id == "expo":
        return ExpoHeuristicAgent(seat), True
    if policy_id in FIXED_IDS:
        return FIXED_IDS[policy_id](seat), False
    raise ValueError(f"collect.py supports expo/fixed-* only, got {policy_id!r}")


def play(policy_ids, seed, max_rounds=200, max_decisions=20000, record=False):
    random.seed(seed)
    env = MonopolyEnv(agent_ids=[0], max_rounds=max_rounds)
    agents, strict = [], []
    for seat, pid in enumerate(policy_ids):
        agent, is_strict = _make(pid, seat)
        agents.append(agent)
        strict.append(is_strict)

    observations, actions, actors, masks = [], [], [], []
    decisions = 0
    while not env.done and decisions < max_decisions:
        actor = env.whose_turn()
        allowed = env.get_allowed_actions(actor)
        if record and policy_ids[actor] == "expo" and len(allowed) > 1:
            observations.append(env._get_state(actor).astype(np.float32))
            actors.append(actor)
            mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
            mask[allowed] = True
            masks.append(np.packbits(mask))
        action = agents[actor].choose_action(env)
        if action not in allowed:
            if strict[actor]:
                raise RuntimeError(f"illegal action from {policy_ids[actor]}")
            action = (
                int(ActionType.END_TURN)
                if int(ActionType.END_TURN) in allowed
                else allowed[0]
            )
        if record and policy_ids[actor] == "expo" and len(allowed) > 1:
            actions.append(action)
        env.step(action)
        decisions += 1

    truncated = decisions >= max_decisions and not env.done
    result = {
        "policies": list(policy_ids),
        "seed": seed,
        "rounds": env.round,
        "decisions": decisions,
        "truncated": truncated,
        "winner": None if truncated else env.winner(),
        "net_worth": [p.net_worth() for p in env.players],
        "bankrupt": [p.bankrupt for p in env.players],
    }
    if record:
        result["obs"] = np.asarray(observations, dtype=np.float32)
        result["act"] = np.asarray(actions, dtype=np.int32)
        result["actor"] = np.asarray(actors, dtype=np.int8)
        result["mask"] = (
            np.asarray(masks, dtype=np.uint8)
            if masks
            else np.zeros((0, (ACTION_SPACE_SIZE + 7) // 8), dtype=np.uint8)
        )
    return result


def _worker(job):
    seed, focus, opponents, weights, max_rounds, record = job
    if weights:
        configure(**weights)
    out = []
    for focus_seat in range(NUM_PLAYERS):
        seats = [None] * NUM_PLAYERS
        seats[focus_seat] = focus
        others = [s for s in range(NUM_PLAYERS) if s != focus_seat]
        for seat, opponent in zip(others, opponents):
            seats[seat] = opponent
        record_out = play(seats, seed, max_rounds=max_rounds, record=record)
        record_out["focus_seat"] = focus_seat
        out.append(record_out)
    return out


def collect(seeds, focus="expo", opponents=("fixed-b", "fixed-d", "fixed-e"),
            weights=None, workers=None, max_rounds=200, record=False):
    """Run one seat-balanced block per seed, fanned across ``workers``."""
    workers = workers or os.cpu_count() or 4
    jobs = [
        (seed, focus, tuple(opponents), weights, max_rounds, record)
        for seed in seeds
    ]
    started = time.time()
    with Pool(processes=workers) as pool:
        blocks = pool.map(_worker, jobs, chunksize=1)
    games = [g for block in blocks for g in block]
    return games, time.time() - started


def summarise(games, focus="expo"):
    scored = [g for g in games if not g["truncated"]]
    appearances = [
        (g, seat)
        for g in scored
        for seat, pid in enumerate(g["policies"])
        if pid == focus
    ]
    wins = sum(1 for g, seat in appearances if g["winner"] == seat)
    return {
        "games": len(games),
        "scored": len(scored),
        "focus": focus,
        "wins": wins,
        "win_rate": wins / len(appearances) if appearances else 0.0,
        "mean_rounds": sum(g["rounds"] for g in scored) / max(1, len(scored)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="EXPO parallel collector")
    parser.add_argument("--mode", choices=("arena", "dataset"), default="arena")
    parser.add_argument("--games", type=int, default=400,
                        help="seat-balanced blocks = games // 4")
    parser.add_argument("--opponents", nargs=3,
                        default=["fixed-b", "fixed-d", "fixed-e"])
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--max-rounds", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("artifacts/expo_data"))
    args = parser.parse_args(argv)

    seeds = list(range(max(1, args.games // NUM_PLAYERS)))
    games, elapsed = collect(
        seeds, opponents=args.opponents, workers=args.workers,
        max_rounds=args.max_rounds, record=(args.mode == "dataset"),
    )
    stats = summarise(games)
    stats["elapsed_s"] = round(elapsed, 1)
    stats["games_per_second"] = round(len(games) / max(1e-9, elapsed), 2)
    stats["workers"] = args.workers
    print(json.dumps(stats, indent=2))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(stats, indent=2))

    if args.mode == "dataset":
        from EXPOSURE_HEURISTIC.distill import build_dataset
        report = build_dataset(games, args.out)
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
