"""Colour-agnostic book loop. Optional deepcopy 1-ply sits off by default.

The body (``oracle.plus_loop``) scores each spend as one engine-book step
(2.5× deed, 5× if the set completes) and takes it when the delta is positive.
No colour is the plan. DealBuilder is not the fallback. deepcopy 1-ply is
still a toggle, not the body.

Flags (defaults for ``oracle-plus-v1``):
- ``one_ply`` — deepcopy legal shortlist (off: closed-form book loop)
- ``solvency`` — prefer solvent 1-ply successors (default off)
- ``denial`` — add a slice of the best opponent's book gain
- ``completing_trade`` — swap when both gain book and we gain more
- ``auction`` — 0.62 of book value, smallest legal raise
- ``cash_gate`` / ``build_first`` / ``race_buy`` / ``lethal_jail`` — on
- ``phase_switch`` — blend toward engine net worth as the cap approaches
- ``inncenta_trade`` — unused by the book loop (list-price completing cash-buy)
- ``leaf`` — asu (default) / networth / asu_plus / clone / rollout
"""

from __future__ import annotations

import random
from dataclasses import replace

from monopoly_game_engine.actions import OFFSETS, ActionType
from monopoly_game_engine.constants import COLOR_GROUPS, PROPERTIES
from monopoly_game_engine.env import PHASE_AUCTION, MonopolyEnv

from oracle.agent import OracleConfig
from oracle.plus_loop import (
    debt_action,
    idle_action,
    needs_plan_cash,
    plan_auction_action,
    plan_build_action,
    plan_buy_action,
    plan_incoming_action,
    plan_trade_action,
)
from oracle.plus_steals import (
    AUCTION_KINDS,
    FIRST_RACE_COLOURS,
    asu_delta_auction,
    cap_weight,
    dead_mortgage_action,
    full_sets,
    has_trade_action,
    is_dominated_cash_action,
    lethal_jail_action,
    scrap_buy_action,
    spend_floor,
    thaw_unmortgage_action,
    unowned_count,
    would_complete,
)

ORACLE_PLUS_ID = "oracle-plus-v1"
TRADE_LO = OFFSETS["buy_trade"]
AUCTION_LO = OFFSETS["auction"]
MAX_TRADE_CANDIDATES = 12
MAX_CANDIDATES = 60


def _flag(value: bool | None, default: bool = True) -> bool:
    return default if value is None else bool(value)


def resolve_plus_config(config: OracleConfig) -> OracleConfig:
    """Fill unspecified plus toggles. Default leaf is ASU (Inncenta's scorer)."""

    leaf = config.leaf if config.leaf and config.leaf != "rollout" else "asu"
    kind = (config.auction_kind or "inncenta").strip().lower()
    if kind not in AUCTION_KINDS:
        kind = "inncenta"
    return replace(
        config,
        leaf=leaf,
        one_ply=_flag(config.one_ply, False),
        solvency=_flag(config.solvency, False),
        denial=_flag(config.denial, True),
        completing_trade=_flag(config.completing_trade, True),
        auction=_flag(config.auction, True),
        inncenta_trade=_flag(config.inncenta_trade, True),
        networth_mix=float(config.networth_mix or 0.0),
        auction_kind=kind,
        family_body="rules",
        one_ply_trades=_flag(config.one_ply_trades, False),
        phase_switch=_flag(config.phase_switch, True),
        cash_gate=_flag(config.cash_gate, True),
        build_first=_flag(config.build_first, True),
        race_buy=_flag(config.race_buy, True),
        lethal_jail=_flag(config.lethal_jail, True),
    )


def denial_value(env: MonopolyEnv, pid: int, square: int) -> float:
    """What holding ``square`` denies opponents (Alinebidal B5)."""

    data = PROPERTIES.get(square)
    if data is None:
        return 0.0
    group = COLOR_GROUPS.get(data["color"]) or ()
    size = len(group)
    if size < 2:
        return 0.0
    rents = data.get("rent") or [data.get("price", 0)]
    hotel = float(rents[-1])
    total = 0.0
    for opp in env.players:
        if opp.player_id == pid or opp.bankrupt:
            continue
        owned = sum(1 for sq in group if env.properties[sq].owner == opp.player_id)
        if owned == 0:
            continue
        missing_after = size - (owned + 1)
        if missing_after < 0:
            continue
        closeness = (owned + 1) / size
        total += (hotel / (2 ** missing_after)) * closeness
    return total


def _acquired_square(env: MonopolyEnv, pid: int, action: int) -> int | None:
    if action == int(ActionType.BUY_PROPERTY):
        return int(env.players[pid].position)
    if AUCTION_LO <= action < AUCTION_LO + 8:
        square = getattr(env, "auction_property_id", None)
        return None if square is None else int(square)
    return None


def _own_score(env: MonopolyEnv, pid: int, config: OracleConfig) -> float:
    player = env.players[pid]
    if player.bankrupt:
        return -1.0e9
    kind = config.leaf
    if kind == "asu":
        from ASU_FROZEN_TEACHER import evaluate_value

        score = float(evaluate_value(env, pid).total)
    elif kind == "asu_plus":
        from asu_plus import evaluate_value_plus

        score = float(evaluate_value_plus(env, pid).total)
    elif kind in ("clone", "rollout"):
        from oracle.leaves import build_leaf_fn

        score = float(build_leaf_fn(config)(env)[pid])
    else:
        score = float(player.net_worth())
    mix = min(1.0, max(0.0, float(config.networth_mix or 0.0)))
    if config.phase_switch:
        mix = max(mix, cap_weight(env))
        if unowned_count(env) == 0:
            mix = max(mix, 0.5)
    if mix <= 0.0:
        return score
    return (1.0 - mix) * score + mix * float(player.net_worth())


class OraclePlusAgent:
    """H2H ``choose_action(env)`` policy with steal-toggles."""

    policy_id = ORACLE_PLUS_ID

    def __init__(self, player_id: int, config: OracleConfig | None = None, *, seed: int = 0):
        self.player_id = player_id
        self.config = resolve_plus_config(config or OracleConfig())
        self.rng = random.Random(0xC0FFEE + int(seed) + player_id)

    def _shortlist(self, legal: list[int]) -> list[int]:
        filtered = [a for a in legal if not is_dominated_cash_action(a, legal)]
        pool = filtered or legal
        if len(pool) <= MAX_CANDIDATES:
            return pool
        trades, others = [], []
        for action in pool:
            (trades if TRADE_LO <= action < AUCTION_LO else others).append(action)
        picked = others[: MAX_CANDIDATES - MAX_TRADE_CANDIDATES]
        if trades:
            k = min(MAX_TRADE_CANDIDATES, len(trades), MAX_CANDIDATES - len(picked))
            picked.extend(self.rng.sample(trades, k))
        return picked or pool[:MAX_CANDIDATES]

    def _property_worth(self, env: MonopolyEnv, square: int) -> float:
        prop = env.properties.get(square)
        if prop is None:
            return 0.0
        price = float(prop.price)
        worth = 2.5 * price
        group = COLOR_GROUPS.get(prop.color) or ()
        mine = sum(1 for sq in group if env.properties[sq].owner == self.player_id)
        if would_complete(env, self.player_id, int(square)) or mine == len(group) - 1:
            worth = 5.0 * price
        elif prop.color in FIRST_RACE_COLOURS:
            worth = 5.0 * price
        if self.config.denial:
            worth += float(self.config.denial_weight) * denial_value(
                env, self.player_id, int(square)
            )
        return worth

    def _auction_action(self, env: MonopolyEnv, legal: list[int]) -> int | None:
        return plan_auction_action(
            env, self.player_id, legal, deny=self.config.denial
        )

    def _survives(self, future: MonopolyEnv) -> bool:
        if not self.config.solvency:
            return True
        from competitors.inncenta.evaluator import is_solvent

        try:
            return bool(is_solvent(future, self.player_id))
        except Exception:
            return True

    def _score(self, env: MonopolyEnv, action: int, future: MonopolyEnv) -> float:
        value = _own_score(future, self.player_id, self.config)
        ours_before = full_sets(env, self.player_id)
        ours_after = full_sets(future, self.player_id)
        if ours_after < ours_before:
            value -= 1.0e7
        for opp in env.players:
            if opp.player_id == self.player_id or opp.bankrupt:
                continue
            if full_sets(future, opp.player_id) > full_sets(env, opp.player_id):
                value -= 1.0e7
        if self.config.cash_gate:
            cash = float(future.players[self.player_id].cash)
            if cash < spend_floor(future, self.player_id):
                value -= 1.0e6
        if self.config.denial:
            square = _acquired_square(env, self.player_id, action)
            if square is not None:
                value += float(self.config.denial_weight) * denial_value(
                    env, self.player_id, square
                )
        return value

    def _one_ply(self, env: MonopolyEnv, legal: list[int]) -> int:
        best_action, best_value = legal[0], float("-inf")
        safe_action, safe_value = None, float("-inf")
        state = random.getstate()
        try:
            from oracle_v2.clone import fast_clone_env

            for action in self._shortlist(legal):
                try:
                    future = fast_clone_env(env)
                    future.step(action)
                except Exception:
                    continue
                value = self._score(env, action, future)
                if value > best_value:
                    best_action, best_value = action, value
                if self._survives(future) and value > safe_value:
                    safe_action, safe_value = action, value
        finally:
            random.setstate(state)
        if safe_action is not None:
            return int(safe_action)
        return int(best_action)

    def choose_action(self, env: MonopolyEnv) -> int:
        pid = self.player_id
        legal = list(env.get_allowed_actions(pid))
        if not legal:
            return int(ActionType.END_TURN)
        if len(legal) == 1:
            return int(legal[0])
        deny = bool(self.config.denial)
        incoming = plan_incoming_action(env, pid, legal)
        if incoming is not None:
            return incoming

        if getattr(env, "phase", None) == PHASE_AUCTION:
            if self.config.auction:
                bid = (
                    asu_delta_auction(env, pid, legal)
                    if self.config.auction_kind == "asu_delta"
                    else plan_auction_action(env, pid, legal, deny=deny)
                )
                if bid is not None:
                    return bid
            if self.config.one_ply:
                action = self._one_ply(env, legal)
                return action if action in legal else int(legal[0])
            return idle_action(legal)

        if env.debt_player == pid:
            return debt_action(env, pid, legal)

        if self.config.lethal_jail:
            jail = lethal_jail_action(env, pid, legal)
            if jail is not None:
                return jail
        thawed = thaw_unmortgage_action(env, pid, legal)
        if thawed is not None:
            return thawed
        if self.config.build_first:
            built = plan_build_action(env, pid, legal)
            if built is not None:
                return built
        force = needs_plan_cash(env, pid, legal, deny=deny)
        parked = dead_mortgage_action(env, pid, legal, force=force)
        if parked is not None:
            return parked
        if self.config.race_buy:
            buy = plan_buy_action(env, pid, legal, deny=deny)
            if buy is not None:
                return buy
        scrap = scrap_buy_action(env, pid, legal)
        if scrap is not None:
            return scrap
        if self.config.completing_trade:
            trade = plan_trade_action(
                env, pid, legal, pay_up=self.config.inncenta_trade
            )
            if trade is not None:
                return trade
        if self.config.one_ply and (
            not self.config.one_ply_trades or has_trade_action(legal)
        ):
            action = self._one_ply(env, legal)
            return action if action in legal else int(legal[0])
        return idle_action(legal)


__all__ = [
    "ORACLE_PLUS_ID",
    "OraclePlusAgent",
    "denial_value",
    "resolve_plus_config",
]
