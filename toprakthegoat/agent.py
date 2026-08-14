"""First-monopoly / four-house lock policy (``toprakthegoat-v1``).

Jail-v1 body plus contest. Build to 4, never hotel. Brown locks 8 of 32 houses.
Max-N PUCT only on buy / auction-open / build; everything else stays the heuristic.
"""

from __future__ import annotations

from monopoly_game_engine.actions import ActionType
from monopoly_game_engine.env import PHASE_AUCTION, MonopolyEnv

from .helpers import idle_action, lethal_jail_action, thaw_unmortgage_action
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
SEARCH_KINDS = frozenset({"buy", "build", "auction"})


def heuristic_action(env: MonopolyEnv, pid: int, legal: list[int]) -> int:
    """Deterministic goat body. Search must call this, never ``choose_action``."""

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


class GoatAgent:
    policy_id = GOAT_ID

    def __init__(
        self,
        player_id: int,
        config=None,
        *,
        seed: int = 0,
        search: bool = False,
    ):
        self.player_id = player_id
        self.config = config
        self.seed = seed
        self.search_enabled = bool(search)
        self._seed = int(seed)
        self._decisions = 0
        self._puct = None
        self._build_menu_labeled: set[tuple[int, int]] = set()
        self.last_used_search = False

    def _engine(self):
        if self._puct is None:
            from .puct import GoatMaxNPUCT

            self._puct = GoatMaxNPUCT()
        return self._puct

    def _decision_seed(self) -> int:
        seed = self._seed + self._decisions * 1_000_003 + self.player_id * 17
        self._decisions += 1
        return seed

    def _should_search(self, env: MonopolyEnv, legal: list[int]) -> bool:
        if not self.search_enabled or len(legal) <= 1:
            return False
        if env.round != getattr(self, "_last_round", None):
            self._build_menu_labeled.clear()
            self._last_round = env.round
        from oracle.hybrid_config import checkpoint_kind

        kind = checkpoint_kind(env, legal)
        if kind not in SEARCH_KINDS:
            return False
        if kind == "build":
            key = (int(env.round), self.player_id)
            if key in self._build_menu_labeled:
                return False
            self._build_menu_labeled.add(key)
        return True

    def choose_action(self, env: MonopolyEnv) -> int:
        pid = self.player_id
        legal = list(env.get_allowed_actions(pid))
        if not legal:
            return int(ActionType.END_TURN)
        if len(legal) == 1:
            self._decisions += 1
            self.last_used_search = False
            return int(legal[0])

        if self._should_search(env, legal):
            result = self._engine().choose_action(env, pid, self._decision_seed())
            action = int(result.chosen_action)
            if action in legal:
                self.last_used_search = True
                return action
            self.last_used_search = False
            return heuristic_action(env, pid, legal)

        self._decisions += 1
        self.last_used_search = False
        return heuristic_action(env, pid, legal)


__all__ = ["GOAT_ID", "GoatAgent", "heuristic_action"]
