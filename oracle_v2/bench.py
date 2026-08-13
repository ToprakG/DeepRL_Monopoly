"""Compare v1 vs v2 action agreement and latency on the same roots."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from monopoly_bench.engine import SharedGame
from oracle.agent import OracleAgent, OracleConfig
from oracle_v2.agent import OracleV2Agent


def collect_roots(games: int, seed: int, max_rounds: int = 30) -> list:
    roots = []
    for game_id in range(games):
        game = SharedGame.new(seed + game_id, max_rounds=max_rounds)
        steps = 0
        while not game.env.done and steps < 80 and len(roots) < games * 4:
            actor = game.env.whose_turn()
            legal = game.env.get_allowed_actions(actor)
            if len(legal) > 1:
                roots.append((game.clone(), actor))
            game.step(legal[0] if len(legal) == 1 else legal[0])
            steps += 1
    return roots


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sims", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = OracleConfig(
        simulations=args.sims,
        rollout_horizon=16,
        rollouts_per_leaf=1,
        max_depth=64,
        max_width=16,
        deadline_s=None,
        early_stop_visit_lead=None,
    )
    roots = collect_roots(args.games, args.seed)
    agree = 0
    v1_t = v2_t = 0.0
    rows = []
    for index, (game, actor) in enumerate(roots):
        v1 = OracleAgent(actor, cfg, seed=index)
        v2 = OracleV2Agent(actor, cfg, seed=index)
        t0 = time.perf_counter()
        a1 = v1.search_action(game.env)
        v1_t += time.perf_counter() - t0
        t0 = time.perf_counter()
        a2 = v2.search_action(game.env)
        v2_t += time.perf_counter() - t0
        match = int(a1.chosen_action == a2.chosen_action)
        agree += match
        rows.append(
            {
                "v1_action": int(a1.chosen_action),
                "v2_action": int(a2.chosen_action),
                "v1_s": a1.latency_s,
                "v2_s": a2.latency_s,
                "v2_tt_hits": v2.search.tt_hits,
                "v2_leaf_evals": v2.search.leaf_evals,
                "match": match,
            }
        )
        print(
            f"{index + 1}/{len(roots)} match={match} v1={a1.latency_s:.2f}s "
            f"v2={a2.latency_s:.2f}s leaves={v2.search.leaf_evals} tt={v2.search.tt_hits}",
            flush=True,
        )
    n = max(len(roots), 1)
    report = {
        "n": len(roots),
        "sims": args.sims,
        "agreement": agree / n,
        "v1_mean_s": v1_t / n,
        "v2_mean_s": v2_t / n,
        "speedup": (v1_t / v2_t) if v2_t else None,
        "rows": rows,
    }
    text = json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2)
    print(text, flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
