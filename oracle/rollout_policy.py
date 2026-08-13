"""DealMaker + Builder hybrid used as the oracle's cheap rollout policy.

DealMaker contributes cheap-buffer buying and monopoly-completing trade offers.
Builder contributes aggressive development (and mortgage-to-fund) once a set
is complete, plus accepting only monopoly-completing incoming trades.
"""

from __future__ import annotations

from typing import List, Optional

from monopoly_game_engine.actions import OFFSETS, ActionType
from monopoly_game_engine.agents_fixed import (
    FixedPolicyAgent,
    TheDealMaker,
    _buy_trade_action,
    _exchange_action,
    _sell_trade_action,
)
from monopoly_game_engine.constants import COLOR_GROUPS, PROPERTY_IDS, REAL_ESTATE_IDS
from monopoly_game_engine.env import MonopolyEnv

# DealMaker buy buffer; Builder-style development floor.
BUY_BUFFER = 100
BUILD_CASH_FLOOR = 50
MORTGAGE_CASH_TRIGGER = 300
# Relaxed buffer for the 1st/2nd square of a color group no opponent has
# touched yet -- early-game land grab, targets opponents that don't
# contest early group pieces.
EARLY_GROUP_BUFFER = 20


class DealBuilderRollout(FixedPolicyAgent):
    """Competent non-search policy: acquire like DealMaker, develop like Builder."""

    def _should_accept_trade(self, offer, env) -> bool:
        # Builder instinct, any colour: only complete a monopoly.
        if offer.offered_prop is None:
            return False
        pid = self.player_id
        color = offer.offered_prop.color
        if color in ("railroad", "utility"):
            return False
        group = COLOR_GROUPS.get(color, [])
        if not group:
            return False
        would_own = sum(1 for sq in group if env.properties[sq].owner == pid) + 1
        return would_own == len(group)

    def _handle_jail(self, allowed: List[int], player) -> Optional[int]:
        if int(ActionType.USE_GOOJ_CARD) in allowed:
            return int(ActionType.USE_GOOJ_CARD)
        return None

    def _should_buy(self, player, prop, env) -> bool:
        # DealMaker: buy anything affordable with a small buffer, relaxed
        # further for the 1st/2nd piece of a group nobody else has touched.
        buffer = BUY_BUFFER
        color = prop.color
        if color not in ("railroad", "utility"):
            group = COLOR_GROUPS.get(color, [])
            contested = any(
                env.properties[sq].owner not in (None, self.player_id)
                for sq in group
                if sq != prop.square_id
            )
            mine_already = sum(
                1
                for sq in group
                if sq != prop.square_id and env.properties[sq].owner == self.player_id
            )
            if not contested and mine_already <= 1:
                buffer = EARLY_GROUP_BUFFER
        return player.can_afford(prop.price + buffer)

    def _best_build_action(self, allowed: List[int], env: MonopolyEnv) -> Optional[int]:
        # Builder: develop any completed monopoly ASAP; mortgage junk to fund.
        player = env.players[self.player_id]
        for index, sq in enumerate(REAL_ESTATE_IDS):
            prop = env.properties[sq]
            if prop.owner != self.player_id or not prop.is_monopoly:
                continue
            hp = prop.data["house_price"]
            if player.can_afford(hp + BUILD_CASH_FLOOR):
                for key in ("improve_hotel", "improve_house"):
                    action = OFFSETS[key] + index
                    if action in allowed:
                        return action
            else:
                funded = self._mortgage_for_build(allowed, env)
                if funded is not None:
                    return funded
        return None

    def _mortgage_for_build(self, allowed: List[int], env: MonopolyEnv) -> Optional[int]:
        for sq in PROPERTY_IDS:
            prop = env.properties.get(sq)
            if (
                prop is None
                or prop.owner != self.player_id
                or prop.is_monopoly
                or prop.houses > 0
                or prop.mortgaged
            ):
                continue
            action = OFFSETS["mortgage"] + PROPERTY_IDS.index(sq)
            if action in allowed:
                return action
        return None

    def _make_trade_offer(self, allowed: List[int], env: MonopolyEnv) -> Optional[int]:
        pid = self.player_id

        # Sell-to-completer: some opponents accept any trade that completes
        # their color group regardless of price. If we own exactly one
        # square in a group and a single non-bankrupt opponent owns every
        # other square, offer it at the richest legal price tier.
        for color, group in COLOR_GROUPS.items():
            if color in ("railroad", "utility"):
                continue
            owned_by_me = [sq for sq in group if env.properties[sq].owner == pid]
            if len(owned_by_me) != 1:
                continue
            sq = owned_by_me[0]
            owners = {env.properties[other].owner for other in group if other != sq}
            if len(owners) != 1:
                continue
            (target,) = owners
            if target is None or target == pid or env.players[target].bankrupt:
                continue
            for price_idx in (2, 1, 0):
                action = _sell_trade_action(pid, target, sq, price_idx, env, allowed)
                if action is not None:
                    return action

        # DealMaker monopoly tempo: bargain buy-offers, then colour exchanges.
        # Skip premium sell-spam — it is not monopoly-completing and dilutes search.
        for color, group in COLOR_GROUPS.items():
            if color in ("railroad", "utility"):
                continue
            owned = [sq for sq in group if env.properties[sq].owner == pid]
            need = [
                sq
                for sq in group
                if env.properties[sq].owner not in (pid, None)
                and not env.players[env.properties[sq].owner].bankrupt
            ]
            if len(owned) + 1 == len(group) and need:
                sq = need[0]
                target = env.properties[sq].owner
                action = _buy_trade_action(pid, target, sq, 0, env, allowed)
                if action is not None:
                    return action

        for color, group in COLOR_GROUPS.items():
            if color in ("railroad", "utility"):
                continue
            owned_here = [sq for sq in group if env.properties[sq].owner == pid]
            if not owned_here:
                continue
            need = [
                sq
                for sq in group
                if env.properties[sq].owner not in (pid, None)
                and not env.players[env.properties[sq].owner].bankrupt
            ]
            if not need:
                continue
            for req_sq in need:
                target = env.properties[req_sq].owner
                for offer_sq in owned_here:
                    if env.properties[offer_sq].is_monopoly:
                        continue
                    action = _exchange_action(pid, target, offer_sq, req_sq, env, allowed)
                    if action is not None:
                        return action
        return None

    def _maybe_mortgage(self, allowed: List[int], env: MonopolyEnv) -> Optional[int]:
        player = env.players[self.player_id]
        if player.cash >= MORTGAGE_CASH_TRIGGER:
            return None
        return TheDealMaker._maybe_mortgage(self, allowed, env)


_AGENTS: dict[int, DealBuilderRollout] = {}


def greedy_rollout_action(env: MonopolyEnv, player_id: int) -> int:
    """Legal DealMaker+Builder action for ``player_id`` in ``env``."""

    agent = _AGENTS.get(player_id)
    if agent is None:
        agent = DealBuilderRollout(player_id)
        _AGENTS[player_id] = agent
    action = agent.choose_action(env)
    allowed = env.get_allowed_actions(player_id)
    if action in allowed:
        return int(action)
    if int(ActionType.END_TURN) in allowed:
        return int(ActionType.END_TURN)
    return int(allowed[0])


__all__ = [
    "BUY_BUFFER",
    "BUILD_CASH_FLOOR",
    "DealBuilderRollout",
    "greedy_rollout_action",
]
