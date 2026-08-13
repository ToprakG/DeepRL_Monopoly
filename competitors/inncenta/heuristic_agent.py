
from __future__ import annotations

import copy
import random

from . import evaluator as EV


def _own_value(env, pid):
    return EV.evaluate(env, pid)


def _asu_value(env, pid):
    from ASU_FROZEN_TEACHER import evaluate_value
    return evaluate_value(env, pid).total


_VALUE_FN = _own_value


def use_value_fn(name: str) -> None:
    global _VALUE_FN
    _VALUE_FN = {"own": _own_value, "asu": _asu_value}[name]
from monopoly_game_engine.actions import (
    AUCTION_ACTION_TO_INCREMENT, ActionType, OFFSETS,
)
from monopoly_game_engine.constants import (
    PROPERTY_IDS, REAL_ESTATE_IDS, TRADE_CASH_LEVELS,
)

N_PROP = len(PROPERTY_IDS)
N_RE = len(REAL_ESTATE_IDS)

TRADE_LO = OFFSETS["buy_trade"]
AUCTION_LO = OFFSETS["auction"]
MAX_TRADE_CANDIDATES = 12
MAX_CANDIDATES = 60


def _family(action: int) -> str:
    for name, start in sorted(OFFSETS.items(), key=lambda kv: kv[1], reverse=True):
        if action >= start:
            return name
    return "binary"


class HeuristicAgent:

    name = "heuristic"

    def __init__(self, player_id: int, rng: random.Random | None = None):
        self.player_id = player_id
        self.rng = rng or random.Random(0xC0FFEE + player_id)
        self.simulations = 0


    def _shortlist(self, env, legal: list[int]) -> list[int]:
        if len(legal) <= MAX_CANDIDATES:
            return legal
        trades, others = [], []
        for action in legal:
            (trades if TRADE_LO <= action < AUCTION_LO else others).append(action)
        picked = others[:MAX_CANDIDATES - MAX_TRADE_CANDIDATES]
        if trades:
            k = min(MAX_TRADE_CANDIDATES, len(trades), MAX_CANDIDATES - len(picked))
            picked.extend(self.rng.sample(trades, k))
        return picked or legal[:MAX_CANDIDATES]


    def _simulate(self, env, action: int, want_state: bool = False):
        state = random.getstate()
        try:
            future = copy.deepcopy(env)
            future.step(action)
            self.simulations += 1
            value = _VALUE_FN(future, self.player_id)
            return (value, future) if want_state else value
        except Exception:
            return None
        finally:
            random.setstate(state)


    def _property_worth(self, env, square: int) -> float:
        prop = env.properties.get(square)
        if prop is None:
            return 0.0
        rivals = sum(1 for i, p in enumerate(env.players)
                     if i != self.player_id and not p.bankrupt)
        worth = EV.W_INCOME * EV.B.expected_rent(square, 0, False) * max(rivals, 1)


        colour = getattr(prop, "color", None)
        group = EV.B.SETS.get(colour)
        if group:
            mine = sum(1 for s in group
                       if env.properties[s].owner == self.player_id)
            if mine == len(group) - 1:
                worth += EV.W_POTENT * sum(
                    EV.B.expected_rent(s, EV.BUILD_TARGET, True) for s in group
                ) * max(rivals, 1)
        return worth

    def _auction_action(self, env, legal: list[int]) -> int | None:
        bids = [a for a in legal if AUCTION_LO < a < AUCTION_LO + 5]
        if not bids:
            return None
        square = getattr(env, "auction_property_id", None)
        if square is None:
            return None
        worth = self._property_worth(env, square)
        high = float(getattr(env, "auction_high_bid", 0) or 0)
        cash = env.players[self.player_id].cash


        frac = getattr(EV, "AUCTION_RESERVE_FRAC", None)
        reserve = EV.CASH_FLOOR if frac is None else max(50.0, frac * cash)
        headroom = min(worth, cash - reserve) - high
        if headroom <= 0:
            return None

        for action, step in sorted(AUCTION_ACTION_TO_INCREMENT.items(),
                                   key=lambda kv: -kv[1]):
            if int(action) in bids and step <= headroom:
                return int(action)
        return None

    def _completing_trade(self, env, legal: list[int]) -> int | None:
        me = env.players[self.player_id]
        others = [i for i in range(len(env.players)) if i != self.player_id]
        best = None
        for colour, group in EV.B.SETS.items():
            owned = [s for s in group if env.properties[s].owner == self.player_id]
            if len(owned) != len(group) - 1:
                continue
            missing = next(s for s in group if s not in owned)
            prop = env.properties[missing]
            if prop.owner is None or prop.owner == self.player_id:
                continue
            if prop.houses > 0:
                continue
            try:
                player_idx = others.index(prop.owner)
                prop_idx = PROPERTY_IDS.index(missing)
            except ValueError:
                continue

            for price_idx in (2, 1, 0):
                cost = int(prop.price * TRADE_CASH_LEVELS[price_idx])
                if me.cash - cost < EV.CASH_FLOOR:
                    continue
                action = (OFFSETS["buy_trade"]
                          + player_idx * (len(PROPERTY_IDS) * len(TRADE_CASH_LEVELS))
                          + prop_idx * len(TRADE_CASH_LEVELS)
                          + price_idx)
                if action in legal:
                    quality = EV.B.set_quality().get(colour, 0.0)
                    if best is None or quality > best[0]:
                        best = (quality, action)
                    break
        return best[1] if best else None

    def _survives(self, future) -> bool:

        mult = getattr(EV, "HARD_GATE", None)
        if mult is not None:
            try:
                return (EV.liquid_assets(future, self.player_id)
                        >= mult * EV.exposure_per_round(future, self.player_id))
            except Exception:
                return True


        if not getattr(EV, "SOLVENCY_GATE", True):
            return True
        try:
            return EV.is_solvent(future, self.player_id)
        except Exception:
            return True


    def choose_action(self, env) -> int:
        legal = list(env.get_allowed_actions(self.player_id))
        if not legal:
            return int(ActionType.END_TURN)
        if len(legal) == 1:
            return legal[0]

        bid = self._auction_action(env, legal)
        if bid is not None:
            return bid

        trade = self._completing_trade(env, legal)
        if trade is not None:
            return trade

        best_action, best_value = legal[0], float("-inf")
        safe_action, safe_value = None, float("-inf")
        for action in self._shortlist(env, legal):
            out = self._simulate(env, action, want_state=True)
            if out is None:
                continue
            value, future = out
            if value > best_value:
                best_action, best_value = action, value
            if future is not None and self._survives(future) and value > safe_value:
                safe_action, safe_value = action, value

        if safe_action is not None:
            best_action = safe_action


        return best_action if best_action in legal else legal[0]
