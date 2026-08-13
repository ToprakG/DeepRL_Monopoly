"""Time choose_action on shared frozen snapshots. Prints p50/mean/p95 in ms."""

from __future__ import annotations

import copy
import os
import random
import statistics
import time

os.environ.setdefault("PYTHONHASHSEED", "0")

from competitors.factory import COMPETITOR_IDS, build_competitor
from monopoly_game_engine.agents_fixed import FPAgentB
from monopoly_game_engine.env import MonopolyEnv
from oracle.agent import OracleAgent, OracleConfig
from oracle.plus_agent import OraclePlusAgent
from oracle_v2.agent import OracleV2Agent, default_v2_config
from ASU_FROZEN_TEACHER import ASUValueV1

N_SNAPS = 36
N_EXPENSIVE = 10
N_SEARCH = 3


def collect_snapshots(n: int) -> list:
    snaps = []
    seed = 0
    while len(snaps) < n:
        random.seed(10_000 + seed)
        env = MonopolyEnv(agent_ids=[0], max_rounds=80)
        bots = [FPAgentB(i) for i in range(4)]
        steps = 0
        while not env.done and steps < 400 and len(snaps) < n:
            pid = env.whose_turn()
            legal = env.get_allowed_actions(pid)
            if len(legal) > 1 and steps % 7 == 0:
                snaps.append(copy.deepcopy(env))
            action = bots[pid].choose_action(env)
            if action not in legal:
                action = legal[0]
            env.step(action)
            steps += 1
        seed += 1
    return snaps


def build(name: str, seat: int):
    if name in COMPETITOR_IDS:
        return build_competitor(name, seat)
    if name == "asu-value-v1":
        return ASUValueV1(seat)
    if name == "fixed-b":
        return FPAgentB(seat)
    if name == "oracle-plus-v1":
        return OraclePlusAgent(seat, OracleConfig(), seed=seat)
    if name == "oracle-fast-v1@32":
        cfg = default_v2_config(simulations=32, rollout_horizon=16, rollouts_per_leaf=1)
        return OracleV2Agent(seat, cfg, seed=seat)
    if name == "oracle-rollout-v1@32":
        return OracleAgent(
            seat,
            OracleConfig(simulations=32, rollout_horizon=16, rollouts_per_leaf=1),
            seed=seat,
        )
    raise KeyError(name)


def time_agent(name: str, snaps: list[object]) -> dict:
    times = []
    for i, env in enumerate(snaps):
        seat = int(env.whose_turn())
        agent = build(name, seat)
        if i == 0:
            agent.choose_action(env)
        started = time.perf_counter()
        agent.choose_action(env)
        times.append(time.perf_counter() - started)
    ms = [t * 1000.0 for t in times]
    ms_sorted = sorted(ms)
    p50 = ms_sorted[len(ms_sorted) // 2]
    p95 = ms_sorted[min(len(ms_sorted) - 1, int(0.95 * (len(ms_sorted) - 1)))]
    return {
        "n": len(ms),
        "p50_ms": p50,
        "mean_ms": statistics.fmean(ms),
        "p95_ms": p95,
        "min_ms": ms_sorted[0],
        "max_ms": ms_sorted[-1],
    }


def fmt(ms: float) -> str:
    if ms >= 100:
        return f"{ms:8.0f} ms"
    if ms >= 1:
        return f"{ms:8.2f} ms"
    if ms >= 0.01:
        return f"{ms:8.3f} ms"
    return f"{ms * 1000:8.1f} µs"


def main() -> None:
    print(f"collecting {N_SNAPS} branched snapshots...", flush=True)
    snaps = collect_snapshots(N_SNAPS)
    expensive = snaps[:N_EXPENSIVE]
    search = snaps[:N_SEARCH]
    order = [
        ("fixed-b", snaps),
        ("boom-hybrid", snaps),
        ("slayer-v1", snaps),
        ("alinebidal-final", snaps),
        ("expo-heuristic-v1", snaps),
        ("underdog-v1", snaps),
        ("oracle-plus-v1", snaps),
        ("inncenta-heuristic", expensive),
        ("asu-value-v1", expensive),
        ("code-exposure", expensive),
        ("oracle-fast-v1@32", search),
        ("oracle-rollout-v1@32", search),
    ]
    rows = []
    for name, sample in order:
        print(f"timing {name} n={len(sample)}...", flush=True)
        row = time_agent(name, sample)
        row["name"] = name
        rows.append(row)
        print(
            f"  p50={fmt(row['p50_ms'])}  mean={fmt(row['mean_ms'])}  "
            f"p95={fmt(row['p95_ms'])}",
            flush=True,
        )
    print()
    print(f"{'agent':<24} {'n':>3} {'p50':>12} {'mean':>12} {'p95':>12} {'max':>12}")
    rows.sort(key=lambda r: r["p50_ms"])
    for row in rows:
        print(
            f"{row['name']:<24} {row['n']:>3} "
            f"{fmt(row['p50_ms']):>12} {fmt(row['mean_ms']):>12} "
            f"{fmt(row['p95_ms']):>12} {fmt(row['max_ms']):>12}"
        )


if __name__ == "__main__":
    main()
