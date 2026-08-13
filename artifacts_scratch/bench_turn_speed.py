"""Time a full seat-turn: PRE_ROLL of P until PRE_ROLL of the next player."""

from __future__ import annotations

import os
import random
import statistics
import time

os.environ.setdefault("PYTHONHASHSEED", "0")

from competitors.factory import COMPETITOR_IDS, build_competitor
from monopoly_game_engine.agents_fixed import FPAgentB
from monopoly_game_engine.env import PHASE_PRE_ROLL, MonopolyEnv
from oracle.agent import OracleAgent, OracleConfig
from oracle.plus_agent import OraclePlusAgent
from oracle_v2.agent import OracleV2Agent, default_v2_config
from ASU_FROZEN_TEACHER import ASUValueV1

FAST_TURNS = 48
SLOW_TURNS = 16
SEARCH_TURNS = 8


def make_agent(name: str, seat: int):
    if name in COMPETITOR_IDS:
        return build_competitor(name, seat)
    if name == "asu-value-v1":
        return ASUValueV1(seat)
    if name == "fixed-b":
        return FPAgentB(seat)
    if name == "oracle-plus-v1":
        return OraclePlusAgent(seat, OracleConfig(), seed=seat)
    if name == "oracle-fast-v1@32":
        return OracleV2Agent(
            seat,
            default_v2_config(simulations=32, rollout_horizon=16, rollouts_per_leaf=1),
            seed=seat,
        )
    if name == "oracle-rollout-v1@32":
        return OracleAgent(
            seat,
            OracleConfig(simulations=32, rollout_horizon=16, rollouts_per_leaf=1),
            seed=seat,
        )
    raise KeyError(name)


def play_turns(names: tuple[str, ...], n_turns: int, seed: int) -> list[dict]:
    random.seed(seed)
    env = MonopolyEnv(agent_ids=[0], max_rounds=120)
    agents = [make_agent(names[i], i) for i in range(4)]
    rows = []
    started = None
    active0 = None
    n_dec = 0
    n_active = 0
    safety = 0
    while len(rows) < n_turns and not env.done and safety < 80_000:
        safety += 1
        if env.phase == PHASE_PRE_ROLL and started is None:
            started = time.perf_counter()
            active0 = env.active_player_id()
            n_dec = 0
            n_active = 0
        pid = env.whose_turn()
        legal = env.get_allowed_actions(pid)
        t0 = time.perf_counter()
        action = agents[pid].choose_action(env)
        dt = time.perf_counter() - t0
        if action not in legal:
            action = legal[0]
        n_dec += 1
        if pid == active0:
            n_active += 1
        env.step(action)
        if (
            started is not None
            and env.phase == PHASE_PRE_ROLL
            and env.active_player_id() != active0
        ):
            rows.append(
                {
                    "turn_s": time.perf_counter() - started,
                    "decisions": n_dec,
                    "active_decisions": n_active,
                    "last_dt_s": dt,
                }
            )
            started = time.perf_counter()
            active0 = env.active_player_id()
            n_dec = 0
            n_active = 0
    return rows


def summarize(ms: list[float], counts: list[int]) -> str:
    ms_sorted = sorted(ms)
    p50 = ms_sorted[len(ms_sorted) // 2]
    p95 = ms_sorted[min(len(ms_sorted) - 1, int(0.95 * (len(ms_sorted) - 1)))]
    dec50 = sorted(counts)[len(counts) // 2]
    return p50, p95, statistics.fmean(ms), dec50, statistics.fmean(counts)


def fmt_s(seconds: float) -> str:
    ms = seconds * 1000.0
    if ms >= 1000:
        return f"{seconds:7.2f} s"
    if ms >= 1:
        return f"{ms:7.1f} ms"
    return f"{ms * 1000:7.0f} µs"


def main() -> None:
    jobs = [
        ("fixed-b ×4", ("fixed-b",) * 4, FAST_TURNS),
        ("boom-hybrid ×4", ("boom-hybrid",) * 4, FAST_TURNS),
        ("expo-heuristic-v1 ×4", ("expo-heuristic-v1",) * 4, FAST_TURNS),
        ("underdog-v1 ×4", ("underdog-v1",) * 4, FAST_TURNS),
        ("slayer-v1 ×4", ("slayer-v1",) * 4, FAST_TURNS),
        ("alinebidal-final ×4", ("alinebidal-final",) * 4, FAST_TURNS),
        ("oracle-plus-v1 ×4", ("oracle-plus-v1",) * 4, FAST_TURNS),
        ("inncenta-heuristic +3×fixed-b", ("inncenta-heuristic", "fixed-b", "fixed-b", "fixed-b"), SLOW_TURNS),
        ("asu-value-v1 +3×fixed-b", ("asu-value-v1", "fixed-b", "fixed-b", "fixed-b"), SLOW_TURNS),
        ("code-exposure +3×fixed-b", ("code-exposure", "fixed-b", "fixed-b", "fixed-b"), SLOW_TURNS),
        ("oracle-fast-v1@32 +3×fixed-b", ("oracle-fast-v1@32", "fixed-b", "fixed-b", "fixed-b"), SEARCH_TURNS),
        ("oracle-rollout-v1@32 +3×fixed-b", ("oracle-rollout-v1@32", "fixed-b", "fixed-b", "fixed-b"), SEARCH_TURNS),
    ]
    print(f"{'table':<36} {'n':>3} {'p50 turn':>10} {'p95':>10} {'mean':>10} {'dec/turn':>9}")
    for label, names, n in jobs:
        print(f"running {label}...", flush=True)
        rows = play_turns(names, n, seed=42)
        if not rows:
            print(f"{label:<36} no turns")
            continue
        p50, p95, mean, dec50, decmean = summarize(
            [r["turn_s"] for r in rows],
            [r["decisions"] for r in rows],
        )
        print(
            f"{label:<36} {len(rows):>3} {fmt_s(p50):>10} {fmt_s(p95):>10} "
            f"{fmt_s(mean):>10} {decmean:6.1f} (p50 {dec50})",
            flush=True,
        )


if __name__ == "__main__":
    main()
