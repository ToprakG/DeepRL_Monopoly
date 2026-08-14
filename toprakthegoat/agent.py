"""First-monopoly / four-house lock policy (``toprakthegoat-v1``).

Jail-v1 body plus contest. Build to 4, never hotel. Brown locks 8 of 32 houses.
"""

from __future__ import annotations

from monopoly_game_engine.actions import ActionType
from monopoly_game_engine.env import PHASE_AUCTION, MonopolyEnv

from oracle.plus_loop import idle_action
from oracle.plus_steals import lethal_jail_action, thaw_unmortgage_action

from .loop import (
    active_colour,
    auction_action,
    build_action,
    buy_action,
    debt_action,
    fund_three_action,
    incoming_action,
    trade_action,
)

GOAT_ID = "toprakthegoat-v1"


class GoatAgent:
    policy_id = GOAT_ID

    def __init__(self, player_id: int, config=None, *, seed: int = 0):
        self.player_id = player_id
        self.config = config
        self.seed = seed

    def choose_action(self, env: MonopolyEnv) -> int:
        pid = self.player_id
        legal = list(env.get_allowed_actions(pid))
        if not legal:
            return int(ActionType.END_TURN)
        if len(legal) == 1:
            return int(legal[0])
        plan = active_colour(env, pid)

        incoming = incoming_action(env, pid, legal, plan)
        if incoming is not None:
            return incoming

        if getattr(env, "phase", None) == PHASE_AUCTION:
            bid = auction_action(env, pid, legal, plan)
            return bid if bid is not None else idle_action(legal)

        if env.debt_player == pid:
            return debt_action(env, pid, legal, plan)

        jail = lethal_jail_action(env, pid, legal)
        if jail is not None:
            return jail
        thawed = thaw_unmortgage_action(env, pid, legal)
        if thawed is not None:
            return thawed
        built = build_action(env, pid, legal, plan)
        if built is not None:
            return built
        funded = fund_three_action(env, pid, legal, plan)
        if funded is not None:
            return funded
        buy = buy_action(env, pid, legal, plan)
        if buy is not None:
            return buy
        trade = trade_action(env, pid, legal, plan)
        if trade is not None:
            return trade
        return idle_action(legal)


__all__ = ["GOAT_ID", "GoatAgent"]
