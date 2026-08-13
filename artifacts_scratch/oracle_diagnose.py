"""Three oracle diagnostics: greedy-only H2H, visit discrimination, leaf sanity."""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
from collections import Counter
from pathlib import Path

import numpy as np

from ASU_FROZEN_TEACHER.core import ASUValueV1, preserve_global_rng
from ASU_FROZEN_TEACHER.evaluate import DEFAULT_MAX_DECISIONS, _run_game
from monopoly_bench.config import SearchConfig
from monopoly_bench.engine import SharedGame, clone_env
from monopoly_bench.search import MaxNPUCT
from monopoly_game_engine.actions import OFFSETS, ActionType, AuctionAction, action_to_description
from monopoly_game_engine.agents_fixed import FPAgentA, FPAgentB
from monopoly_game_engine.env import PHASE_OUT_OF_TURN, TradeOffer

from oracle.agent import UniformPriorModel
from oracle.eval_h2h import ASU_VALUE_ID, FIXED_A_ID, FIXED_B_ID, rotate_lineup, summarize_policies
from oracle.rollout_leaf import net_worth_margin_vector, rollout_leaf_value
from oracle.rollout_policy import greedy_rollout_action

OUT = Path("artifacts_scratch/oracle_diagnose.json")
GREEDY_ID = "greedy-rollout"
WORKERS = max(1, min(8, os.cpu_count() or 1))


class GreedySeat:
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.counts = Counter()

    def choose_action(self, env) -> int:
        action = greedy_rollout_action(env, self.player_id)
        if action == int(ActionType.BUY_PROPERTY):
            self.counts["buy"] += 1
        elif action == int(ActionType.ACCEPT_TRADE):
            self.counts["accept_trade"] += 1
        elif action == int(ActionType.DECLINE_TRADE):
            self.counts["decline_trade"] += 1
        elif OFFSETS["improve_house"] <= action < OFFSETS["sell_house"]:
            self.counts["build"] += 1
        elif action == int(ActionType.ROLL_DICE):
            self.counts["roll"] += 1
        elif action == int(ActionType.END_TURN):
            self.counts["end_turn"] += 1
        elif action == int(AuctionAction.PASS):
            self.counts["auction_pass"] += 1
        elif action in {int(a) for a in AuctionAction}:
            self.counts["auction_bid"] += 1
        else:
            self.counts["other"] += 1
        return action


class Scripted:
    def __init__(self, agent, player_id: int):
        self.agent = agent
        self.player_id = player_id
        self.fallbacks = 0

    def choose_action(self, env) -> int:
        allowed = env.get_allowed_actions(self.player_id)
        action = self.agent.choose_action(env)
        if action in allowed:
            return action
        self.fallbacks += 1
        if int(ActionType.END_TURN) in allowed:
            return int(ActionType.END_TURN)
        return allowed[0]


class Factory:
    def __init__(self):
        self.greedy_agents: list[GreedySeat] = []

    def build(self, spec, player_id: int):
        if spec.policy_id == GREEDY_ID:
            agent = GreedySeat(player_id)
            self.greedy_agents.append(agent)
            return agent
        if spec.policy_id == ASU_VALUE_ID:
            return ASUValueV1(player_id)
        if spec.policy_id == FIXED_A_ID:
            return Scripted(FPAgentA(player_id), player_id)
        if spec.policy_id == FIXED_B_ID:
            return Scripted(FPAgentB(player_id), player_id)
        raise ValueError(spec.policy_id)


def _greedy_game_job(payload: dict) -> tuple[int, dict, dict]:
    factory = Factory()
    specs = tuple(type("S", (), {"policy_id": p})() for p in payload["policies"])
    with preserve_global_rng():
        result = _run_game(
            specs,
            focus_seat=payload["focus_seat"],
            seed=payload["seed"],
            max_decisions=payload["max_decisions"],
            factory=factory,
        )
    counts = Counter()
    for agent in factory.greedy_agents:
        counts.update(agent.counts)
    return int(payload["index"]), result, dict(counts)


def run_diag1(games: int = 20) -> dict:
    print("=" * 60)
    print(f"DIAG 1: greedy rollout_policy alone vs ASU ({games} games, workers={WORKERS})")
    print("=" * 60)
    lineup_base = (GREEDY_ID, ASU_VALUE_ID, FIXED_A_ID, FIXED_B_ID)
    payloads = []
    for index in range(games):
        specs = rotate_lineup(lineup_base, index)
        focus = next(i for i, s in enumerate(specs) if s.policy_id == GREEDY_ID)
        payloads.append(
            {
                "index": index,
                "policies": [s.policy_id for s in specs],
                "focus_seat": focus,
                "seed": index,
                "max_decisions": DEFAULT_MAX_DECISIONS,
            }
        )

    results: list[dict | None] = [None] * games
    action_totals: Counter = Counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=WORKERS) as pool:
        for index, result, counts in pool.imap_unordered(_greedy_game_job, payloads):
            results[index] = result
            action_totals.update(counts)
            print(
                f"  game {index + 1}/{games} winner={result['winner_policy']} "
                f"rounds={result['rounds']}",
                flush=True,
            )

    games_out = [r for r in results if r is not None]
    summaries = summarize_policies(games_out)
    margins = []
    for game in games_out:
        pols = game["policies"]
        g = pols.index(GREEDY_ID)
        a = pols.index(ASU_VALUE_ID)
        margins.append(float(game["final_net_worth"][g]) - float(game["final_net_worth"][a]))
    mean = sum(margins) / len(margins)
    se = (sum((m - mean) ** 2 for m in margins) / (len(margins) - 1)) ** 0.5 / len(margins) ** 0.5
    richer = sum(m > 0 for m in margins)
    diag1 = {
        "win_rates": summaries,
        "margin_mean": mean,
        "margin_se": se,
        "greedy_richer": richer,
        "n": len(margins),
        "action_counts": dict(action_totals),
    }
    print("\nWIN RATES:")
    for key, value in summaries.items():
        print(
            f"  {key}: {value['wins']}/{value['games']} ({value['win_rate']:.3f}) "
            f"wilson={[round(x, 3) for x in value['wilson_95']]}"
        )
    print(f"MARGIN greedy-ASU: mean={mean:.1f} ± SE {se:.1f} richer={richer}/{len(margins)}")
    print(f"ACTION COUNTS: {dict(action_totals)}")
    return diag1


def print_visits(label: str, game: SharedGame, sims: int = 200) -> dict:
    env = game.env
    actor = env.whose_turn()
    legal = env.get_allowed_actions(actor)
    search = MaxNPUCT(
        UniformPriorModel(),
        SearchConfig(simulations=sims, max_depth=32, max_width=64),
        leaf_fn=lambda e: rollout_leaf_value(e, num_rollouts=1, horizon=16),
    )
    result = search.choose_action(env, actor, decision_seed=12345)
    visits = sorted(result.visits.items(), key=lambda kv: (-kv[1], kv[0]))
    total = sum(n for _, n in visits) or 1
    print(
        f"\n[{label}] actor={actor} legal={len(legal)} "
        f"sims={result.simulations} chosen={result.chosen_action} "
        f"latency={result.latency_s:.1f}s"
    )
    print(f"  root_value={list(np.round(result.root_value, 4))}")
    rows = []
    for action, n in visits[:12]:
        name = action_to_description(action)
        frac = n / total
        print(f"  action={action:>5} visits={n:>4} ({frac:5.1%})  {name}")
        rows.append({"action": action, "visits": n, "frac": frac, "name": name})
    probs = np.array([n / total for _, n in visits], dtype=np.float64)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    max_ent = math.log(len(probs)) if len(probs) else 1.0
    ratio = entropy / max_ent if max_ent > 0 else 1.0
    top1 = visits[0][1] / total if visits else 0.0
    print(f"  top1_frac={top1:.3f} entropy_ratio={ratio:.3f} (1.0=uniform)")
    return {
        "label": label,
        "legal": len(legal),
        "entropy_ratio": ratio,
        "top1_frac": top1,
        "top": rows,
        "chosen": result.chosen_action,
        "root_value": list(map(float, result.root_value)),
    }


def run_diag2() -> list[dict]:
    print("\n" + "=" * 60)
    print("DIAG 2: search visit discrimination at N=200")
    print("=" * 60)
    diag2: list[dict] = []

    # Prefer midgame multi-action roots; fall back to seeded roots.
    g = SharedGame.new(42, max_rounds=200)
    for step in range(400):
        if g.env.done:
            break
        actor = g.env.whose_turn()
        action = ASUValueV1(actor).choose_action(g.env)
        legal = g.env.get_allowed_actions(actor)
        if action not in legal:
            action = legal[0]
        g.step(action)
        legal_now = g.env.get_allowed_actions(g.env.whose_turn())
        if g.env.round >= 15 and 3 <= len(legal_now) <= 40:
            diag2.append(print_visits(f"midgame round={g.env.round} step={step}", g, sims=200))
            if len(diag2) >= 3:
                return diag2

    for seed in (7, 11, 19, 23, 29, 31):
        g2 = SharedGame.new(seed, max_rounds=200)
        for _ in range(120):
            if g2.env.done:
                break
            actor = g2.env.whose_turn()
            legal = g2.env.get_allowed_actions(actor)
            if 3 <= len(legal) <= 40:
                diag2.append(print_visits(f"seed={seed} round={g2.env.round}", g2, sims=200))
                break
            action = greedy_rollout_action(g2.env, actor)
            if action not in legal:
                action = legal[0]
            g2.step(action)
        if len(diag2) >= 3:
            break
    return diag2


def setup_buy_state() -> SharedGame:
    game = SharedGame.new(100, max_rounds=200)
    env = game.env
    env.turn_order = [0, 1, 2, 3]
    env.current_turn_idx = 0
    env.players[0].cash = 1500
    env.players[0].position = 1
    env.phase = "post_roll"
    env.has_rolled = True
    env.debt_player = None
    prop = env.properties[1]
    if prop.owner is not None:
        env.players[prop.owner].properties = [
            p for p in env.players[prop.owner].properties if p.square_id != 1
        ]
        prop.owner = None
    assert int(ActionType.BUY_PROPERTY) in env.get_allowed_actions(0)
    return game


def setup_bad_trade_state() -> SharedGame:
    game = SharedGame.new(101, max_rounds=200)
    env = game.env
    env.turn_order = [1, 0, 2, 3]
    env.current_turn_idx = 0
    for sq in (1, 3):
        prop = env.properties[sq]
        if prop.owner is not None:
            env.players[prop.owner].properties = [
                p for p in env.players[prop.owner].properties if p.square_id != sq
            ]
        prop.owner = 0
        prop.mortgaged = False
        if prop not in env.players[0].properties:
            env.players[0].properties.append(prop)
    env.players[0].cash = 800
    env.players[1].cash = 2000
    env._update_monopolies()
    env.pending_trades = {
        1: TradeOffer(
            from_player=1,
            to_player=0,
            offered_prop=None,
            requested_prop=env.properties[1],
            cash_offered=1,
            cash_requested=0,
        )
    }
    env.phase = PHASE_OUT_OF_TURN
    env.out_of_turn_pids = [0]
    legal = env.get_allowed_actions(0)
    assert int(ActionType.ACCEPT_TRADE) in legal and int(ActionType.DECLINE_TRADE) in legal
    return game


def setup_build_state():
    game = SharedGame.new(102, max_rounds=200)
    env = game.env
    env.turn_order = [0, 1, 2, 3]
    env.current_turn_idx = 0
    for sq in (6, 8, 9):
        prop = env.properties[sq]
        if prop.owner is not None:
            env.players[prop.owner].properties = [
                p for p in env.players[prop.owner].properties if p.square_id != sq
            ]
        prop.owner = 0
        prop.houses = 0
        prop.mortgaged = False
        if prop not in env.players[0].properties:
            env.players[0].properties.append(prop)
    env.players[0].cash = 2000
    env._update_monopolies()
    env.phase = "pre_roll"
    env.has_rolled = False
    legal = env.get_allowed_actions(0)
    builds = [a for a in legal if OFFSETS["improve_house"] <= a < OFFSETS["sell_house"]]
    return game, builds


def run_diag3() -> list[dict]:
    print("\n" + "=" * 60)
    print("DIAG 3: leaf values on hand-picked states")
    print("=" * 60)
    diag3 = []

    print("\n[3a] Good buy available (Mediterranean Avenue)")
    gb = setup_buy_state()
    base = rollout_leaf_value(gb.env, num_rollouts=8, horizon=24, seed=1)
    buy_env = clone_env(gb.env)
    buy_env.step(int(ActionType.BUY_PROPERTY))
    after_buy = rollout_leaf_value(buy_env, num_rollouts=8, horizon=24, seed=1)
    skip_env = clone_env(gb.env)
    legal = skip_env.get_allowed_actions(0)
    skip = (
        int(ActionType.END_TURN)
        if int(ActionType.END_TURN) in legal
        else next(a for a in legal if a != int(ActionType.BUY_PROPERTY))
    )
    skip_env.step(skip)
    after_skip = rollout_leaf_value(skip_env, num_rollouts=8, horizon=24, seed=1)
    print(
        f"  base V0={base[0]:.4f}  after_BUY V0={after_buy[0]:.4f}  "
        f"after_SKIP V0={after_skip[0]:.4f}"
    )
    print(
        f"  prefers BUY? {after_buy[0] > after_skip[0]}  "
        f"(delta={after_buy[0] - after_skip[0]:+.4f})"
    )
    print(
        f"  vectors base={np.round(base, 4).tolist()} "
        f"buy={np.round(after_buy, 4).tolist()} skip={np.round(after_skip, 4).tolist()}"
    )
    diag3.append(
        {
            "case": "good_buy",
            "base": base.tolist(),
            "after_buy": after_buy.tolist(),
            "after_skip": after_skip.tolist(),
            "prefers_buy": bool(after_buy[0] > after_skip[0]),
        }
    )

    print("\n[3b] Bad trade offered (give monopoly deed for $1)")
    bt = setup_bad_trade_state()
    after_accept_env = clone_env(bt.env)
    after_accept_env.step(int(ActionType.ACCEPT_TRADE))
    after_accept = rollout_leaf_value(after_accept_env, num_rollouts=8, horizon=24, seed=2)
    after_decline_env = clone_env(bt.env)
    after_decline_env.step(int(ActionType.DECLINE_TRADE))
    after_decline = rollout_leaf_value(after_decline_env, num_rollouts=8, horizon=24, seed=2)
    print(f"  ACCEPT V0={after_accept[0]:.4f}  DECLINE V0={after_decline[0]:.4f}")
    print(
        f"  prefers DECLINE? {after_decline[0] > after_accept[0]}  "
        f"(delta={after_decline[0] - after_accept[0]:+.4f})"
    )
    raw_a = [float(p.net_worth()) for p in after_accept_env.players]
    raw_d = [float(p.net_worth()) for p in after_decline_env.players]
    print(f"  immediate NW after ACCEPT={raw_a} DECLINE={raw_d}")
    greedy_choice = greedy_rollout_action(bt.env, 0)
    print(
        f"  greedy_rollout_action chooses "
        f"{'ACCEPT' if greedy_choice == int(ActionType.ACCEPT_TRADE) else 'DECLINE/OTHER'} "
        f"(action={greedy_choice})"
    )
    diag3.append(
        {
            "case": "bad_trade",
            "after_accept": after_accept.tolist(),
            "after_decline": after_decline.tolist(),
            "prefers_decline": bool(after_decline[0] > after_accept[0]),
            "nw_accept": raw_a,
            "nw_decline": raw_d,
            "greedy_action": greedy_choice,
            "greedy_accepts_bad_trade": greedy_choice == int(ActionType.ACCEPT_TRADE),
        }
    )

    print("\n[3c] Build available on light-blue monopoly")
    bs, builds = setup_build_state()
    print(f"  legal builds={builds[:5]} (n={len(builds)})")
    base_b = rollout_leaf_value(bs.env, num_rollouts=8, horizon=24, seed=3)
    if builds:
        build_env = clone_env(bs.env)
        build_env.step(builds[0])
        after_build = rollout_leaf_value(build_env, num_rollouts=8, horizon=24, seed=3)
    else:
        after_build = base_b
    print(
        f"  base V0={base_b[0]:.4f}  after_BUILD V0={after_build[0]:.4f}  "
        f"prefers_build_vs_base? {after_build[0] > base_b[0]}"
    )
    greedy_b = greedy_rollout_action(bs.env, 0)
    print(
        f"  greedy chooses action={greedy_b} "
        f"is_build={OFFSETS['improve_house'] <= greedy_b < OFFSETS['sell_house']}"
    )
    diag3.append(
        {
            "case": "build",
            "n_build_actions": len(builds),
            "base": base_b.tolist(),
            "after_build": after_build.tolist(),
            "prefers_build_vs_base": bool(after_build[0] > base_b[0]),
            "greedy_action": greedy_b,
            "greedy_builds": bool(OFFSETS["improve_house"] <= greedy_b < OFFSETS["sell_house"]),
        }
    )

    print("\n[3d] Static margin sign: rich seat should dominate softmax")
    rich = SharedGame.new(200, max_rounds=200).env
    rich.players[0].cash = 5000
    rich.players[1].cash = 100
    rich.players[2].cash = 100
    rich.players[3].cash = 100
    value = net_worth_margin_vector(rich)
    print(
        f"  cash={[p.cash for p in rich.players]} V={np.round(value, 4).tolist()} "
        f"V0_largest? {bool(value[0] == value.max())}"
    )
    diag3.append(
        {
            "case": "margin_sign",
            "value": value.tolist(),
            "v0_largest": bool(value[0] == value.max()),
        }
    )
    return diag3


def main() -> None:
    # Fast leaf sanity first, then visits, then parallel greedy H2H.
    diag3 = run_diag3()
    OUT.write_text(json.dumps({"diag3_leaf": diag3}, indent=2) + "\n", encoding="utf-8")

    diag2 = run_diag2()
    OUT.write_text(
        json.dumps({"diag2_visits": diag2, "diag3_leaf": diag3}, indent=2) + "\n",
        encoding="utf-8",
    )

    diag1 = run_diag1(20)
    report = {"diag1_greedy_h2h": diag1, "diag2_visits": diag2, "diag3_leaf": diag3}
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
