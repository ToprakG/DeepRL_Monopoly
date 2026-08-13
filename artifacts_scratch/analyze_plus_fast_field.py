"""Replay plus_fast_field games and log why plus died."""

from __future__ import annotations

import json
import multiprocessing as mp
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from ASU_FROZEN_TEACHER.core import preserve_global_rng
from ASU_FROZEN_TEACHER.evaluate import _new_seeded_game
from monopoly_game_engine.actions import ActionType, AuctionAction
from monopoly_game_engine.constants import COLOR_GROUPS, PROPERTIES, TRADE_CASH_LEVELS
from monopoly_game_engine.env import PHASE_AUCTION
from oracle.agent import OracleConfig
from oracle.eval_h2h import ORACLE_PLUS_ID, _H2HFactory, _Spec, rotate_lineup
from oracle.plus_agent import OraclePlusAgent, resolve_plus_config
from oracle.plus_steals import (
    HOUSE_LO,
    HOTEL_LO,
    SELL_HOUSE_LO,
    build_first_action,
    cash_floor,
    full_sets,
    max_live_rent,
    next_roll_threat,
    unowned_count,
)

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "artifacts_scratch" / "plus_fast_field_ckpt"
OUT = ROOT / "artifacts_scratch" / "plus_fast_field_fail.json"
LINEUP = ("oracle-plus-v1", "underdog-v1", "slayer-v1", "alinebidal-final")
SEED0 = 9_500_000
REAL = tuple(c for c in COLOR_GROUPS if c not in ("railroad", "utility"))


def _sets(env, pid: int) -> list[str]:
    out = []
    for color in REAL:
        group = COLOR_GROUPS[color]
        if all(env.properties[sq].owner == pid for sq in group):
            out.append(color)
    return out


def _hotels(env, pid: int) -> int:
    return sum(1 for p in env.players[pid].properties if getattr(p, "houses", 0) >= 5)


def _houses(env, pid: int) -> int:
    return sum(max(0, min(4, int(getattr(p, "houses", 0)))) for p in env.players[pid].properties)


def _board_snapshot(env, policies: list[str]) -> dict:
    return {
        "round": int(env.round),
        "unowned": unowned_count(env),
        "players": [
            {
                "id": pol,
                "cash": float(env.players[i].cash),
                "nw": float(env.players[i].net_worth()),
                "bankrupt": bool(env.players[i].bankrupt),
                "sets": _sets(env, i),
                "houses": _houses(env, i),
                "hotels": _hotels(env, i),
                "deeds": len(env.players[i].properties),
                "in_jail": bool(getattr(env.players[i], "in_jail", False)),
            }
            for i, pol in enumerate(policies)
        ],
    }


class Probe(OraclePlusAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats = Counter()
        self.events: list[dict] = []

    def choose_action(self, env):
        legal = list(env.get_allowed_actions(self.player_id))
        pid = self.player_id
        cash = float(env.players[pid].cash)
        floor = cash_floor(env, pid)
        live = max_live_rent(env, pid)
        nxt = next_roll_threat(env, pid)

        house_legal = [a for a in legal if HOUSE_LO <= a < SELL_HOUSE_LO]
        if house_legal and build_first_action(env, pid, legal) is None:
            self.stats["build_skipped_floor"] += 1
            if cash < floor:
                self.stats["build_skipped_cash_below_floor"] += 1

        if int(ActionType.BUY_PROPERTY) in legal:
            square = int(env.players[pid].position)
            prop = env.properties.get(square)
            if prop is not None and prop.owner is None:
                price = float(prop.price)
                if cash - price < floor:
                    self.stats["buy_skipped_floor"] += 1

        if int(ActionType.PAY_BAIL) in legal:
            self.stats["jail_menu"] += 1

        if getattr(env, "phase", None) == PHASE_AUCTION:
            self.stats["auction_menu"] += 1
            square = getattr(env, "auction_property_id", None)
            if square is not None:
                worth = self._property_worth(env, int(square))
                high = float(getattr(env, "auction_high_bid", 0) or 0)
                if min(worth, cash - floor) - high <= 0:
                    self.stats["auction_headroom_zero"] += 1

        # one-away completing trade blocked by floor
        me = env.players[pid]
        for color, group in COLOR_GROUPS.items():
            if color in ("railroad", "utility"):
                continue
            owned = [sq for sq in group if env.properties[sq].owner == pid]
            if len(owned) + 1 != len(group):
                continue
            need = [
                sq
                for sq in group
                if env.properties[sq].owner not in (pid, None)
                and not env.players[env.properties[sq].owner].bankrupt
                and env.properties[sq].houses == 0
            ]
            if not need:
                continue
            sq = need[0]
            cheap = float(PROPERTIES[sq]["price"]) * TRADE_CASH_LEVELS[0]
            mid = float(PROPERTIES[sq]["price"]) * TRADE_CASH_LEVELS[1]
            if me.cash - cheap < floor:
                self.stats["complete_trade_blocked_floor"] += 1
            elif me.cash - mid < floor:
                self.stats["complete_trade_only_cheap"] += 1

        action = super().choose_action(env)
        if action == int(ActionType.PAY_BAIL):
            self.stats["paid_bail"] += 1
        elif int(ActionType.PAY_BAIL) in legal:
            self.stats["jail_sit"] += 1
        if HOUSE_LO <= action < SELL_HOUSE_LO:
            self.stats["built"] += 1
        if action == int(ActionType.BUY_PROPERTY):
            self.stats["bought"] += 1
        if action == int(ActionType.ACCEPT_TRADE):
            self.stats["accepted_trade"] += 1
        if action == int(ActionType.DECLINE_TRADE):
            self.stats["declined_trade"] += 1
        if getattr(env, "phase", None) == PHASE_AUCTION and action == int(AuctionAction.PASS):
            self.stats["auction_pass"] += 1
        self.stats["cash_vs_floor_sum"] += cash - floor
        self.stats["floor_sum"] += floor
        self.stats["live_sum"] += live
        self.stats["next_sum"] += nxt
        self.stats["decisions"] += 1
        return action


def _replay(index: int) -> dict:
    specs = rotate_lineup(LINEUP, index)
    policies = [s.policy_id for s in specs]
    us = policies.index(ORACLE_PLUS_ID)
    config = resolve_plus_config(
        OracleConfig(
            one_ply=False,
            denial=True,
            completing_trade=True,
            auction=True,
            inncenta_trade=True,
        )
    )
    factory = _H2HFactory(config, seed=SEED0 + index)
    with preserve_global_rng():
        game = _new_seeded_game(SEED0 + index)
        agents = []
        probe = None
        for seat, spec in enumerate(specs):
            agent = factory.build(spec, seat)
            if spec.policy_id == ORACLE_PLUS_ID:
                probe = Probe(seat, config, seed=SEED0 + index + seat)
                agents.append(probe)
            else:
                agents.append(agent)
        first_set: dict[str, int] = {}
        first_hotel: dict[str, int] = {}
        first_opp_set_snap = None
        first_opp_hotel_snap = None
        plus_set_snap = None
        bankrupt_snap = None
        prev_sets = [0] * 4
        prev_hotels = [0] * 4
        prev_broke = [False] * 4
        decisions = 0
        while not game.env.done and decisions < 50_000:
            actor = game.env.whose_turn()
            action = agents[actor].choose_action(game.env)
            game.step(action)
            decisions += 1
            env = game.env
            for i, pol in enumerate(policies):
                nset = full_sets(env, i)
                nh = _hotels(env, i)
                if nset > prev_sets[i] and pol not in first_set:
                    first_set[pol] = int(env.round)
                    snap = _board_snapshot(env, policies)
                    if i == us:
                        plus_set_snap = snap
                    elif first_opp_set_snap is None:
                        first_opp_set_snap = snap
                if nh > prev_hotels[i] and pol not in first_hotel:
                    first_hotel[pol] = int(env.round)
                    if i != us and first_opp_hotel_snap is None:
                        first_opp_hotel_snap = _board_snapshot(env, policies)
                if env.players[i].bankrupt and not prev_broke[i] and i == us:
                    bankrupt_snap = _board_snapshot(env, policies)
                prev_sets[i] = nset
                prev_hotels[i] = nh
                prev_broke[i] = bool(env.players[i].bankrupt)
        winner = env.winner() if env.done else None
        stats = dict(probe.stats) if probe is not None else {}
        n = max(1, int(stats.get("decisions", 1)))
        stats["mean_floor"] = stats.get("floor_sum", 0) / n
        stats["mean_live"] = stats.get("live_sum", 0) / n
        stats["mean_next"] = stats.get("next_sum", 0) / n
        stats["mean_cash_minus_floor"] = stats.get("cash_vs_floor_sum", 0) / n
        for k in ("floor_sum", "live_sum", "next_sum", "cash_vs_floor_sum"):
            stats.pop(k, None)
        return {
            "index": index,
            "policies": policies,
            "us": us,
            "winner": None if winner is None else policies[winner],
            "rounds": int(env.round),
            "nw": [float(p.net_worth()) for p in env.players],
            "first_set": first_set,
            "first_hotel": first_hotel,
            "plus_had_set": ORACLE_PLUS_ID in first_set,
            "plus_had_hotel": ORACLE_PLUS_ID in first_hotel,
            "plus_set_first": bool(first_set)
            and min(first_set, key=first_set.get) == ORACLE_PLUS_ID,
            "opp_set_before_plus": (
                any(p != ORACLE_PLUS_ID for p in first_set)
                and (
                    ORACLE_PLUS_ID not in first_set
                    or min(v for p, v in first_set.items() if p != ORACLE_PLUS_ID)
                    < first_set[ORACLE_PLUS_ID]
                )
            ),
            "stats": stats,
            "end": _board_snapshot(env, policies),
            "first_opp_set": first_opp_set_snap,
            "first_opp_hotel": first_opp_hotel_snap,
            "plus_set": plus_set_snap,
            "plus_broke": bankrupt_snap,
        }


def main() -> int:
    with mp.get_context("spawn").Pool(6) as pool:
        rows = pool.map(_replay, range(24))
    OUT.write_text(json.dumps(rows, indent=2) + "\n")

    wins = Counter(r["winner"] for r in rows)
    print("replay winners", dict(wins))
    wipe = [r for r in rows if r["nw"][r["us"]] == 0]
    cap = [r for r in rows if r["rounds"] >= 200 and r["nw"][r["us"]] > 0]
    print(f"wipe {len(wipe)} cap-alive {len(cap)} plus-wins {wins.get(ORACLE_PLUS_ID, 0)}")

    def _agg(label, games):
        if not games:
            print(f"\n=== {label} empty ===")
            return
        print(f"\n=== {label} n={len(games)} ===")
        print("  plus had set", sum(g["plus_had_set"] for g in games), "/", len(games))
        print("  plus had hotel", sum(g["plus_had_hotel"] for g in games), "/", len(games))
        print("  plus set first", sum(g["plus_set_first"] for g in games), "/", len(games))
        print("  opp set before plus", sum(g["opp_set_before_plus"] for g in games), "/", len(games))
        first_who = Counter()
        for g in games:
            if g["first_set"]:
                first_who[min(g["first_set"], key=g["first_set"].get)] += 1
            else:
                first_who["(none)"] += 1
        print("  first monopoly", dict(first_who))
        hotel_who = Counter()
        for g in games:
            if g["first_hotel"]:
                hotel_who[min(g["first_hotel"], key=g["first_hotel"].get)] += 1
            else:
                hotel_who["(none)"] += 1
        print("  first hotel", dict(hotel_who))
        sums = Counter()
        for g in games:
            sums.update(g["stats"])
        n = len(games)
        keys = [
            "jail_sit",
            "paid_bail",
            "jail_menu",
            "build_skipped_floor",
            "build_skipped_cash_below_floor",
            "built",
            "buy_skipped_floor",
            "bought",
            "complete_trade_blocked_floor",
            "complete_trade_only_cheap",
            "accepted_trade",
            "declined_trade",
            "auction_headroom_zero",
            "auction_menu",
            "auction_pass",
        ]
        for k in keys:
            print(f"  {k:32s} {sums[k]/n:8.1f}/game  total={sums[k]}")
        print(f"  mean_floor {sum(g['stats']['mean_floor'] for g in games)/n:.0f}")
        print(f"  mean_live  {sum(g['stats']['mean_live'] for g in games)/n:.0f}")
        print(f"  mean_next  {sum(g['stats']['mean_next'] for g in games)/n:.0f}")
        print(
            "  mean cash-floor",
            f"{sum(g['stats']['mean_cash_minus_floor'] for g in games)/n:.0f}",
        )
        if label.startswith("wipe"):
            print("  bankrupt snaps:")
            for g in games:
                b = g["plus_broke"] or g["end"]
                us = b["players"][g["us"]]
                opps = [p for i, p in enumerate(b["players"]) if i != g["us"]]
                hotels = [(p["id"], p["hotels"], p["sets"]) for p in opps if p["hotels"] or p["sets"]]
                print(
                    f"    g{g['index']:02d} r={b['round']} us_cash={us['cash']:.0f} "
                    f"us_sets={us['sets']} us_h={us['houses']}/{us['hotels']} "
                    f"floor_live_board hotels/sets={hotels} win={g['winner']}"
                )
        if label.startswith("cap"):
            print("  end board:")
            for g in games:
                e = g["end"]
                bits = " ".join(
                    f"{p['id'][:6]} nw={p['nw']:.0f} set={p['sets']} h={p['houses']}/{p['hotels']}"
                    for p in e["players"]
                )
                print(f"    g{g['index']:02d} win={g['winner']:<18} {bits}")

    _agg("ALL", rows)
    _agg("wipeouts", wipe)
    _agg("cap-alive (incl our 3 wins)", cap)
    _agg("our wins", [r for r in rows if r["winner"] == ORACLE_PLUS_ID])
    _agg("cap losses", [r for r in cap if r["winner"] != ORACLE_PLUS_ID])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
