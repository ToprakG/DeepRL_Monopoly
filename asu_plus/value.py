"""ASU+ value: ASU base total plus cheap cash / endgame / block / liquidity terms."""

from __future__ import annotations

from dataclasses import dataclass

from ASU_FROZEN_TEACHER.core import (  # noqa: E402
    _hypothetical_group_rent,
    evaluate_value,
    safety_breakdown,
)
from ASU_FROZEN_TEACHER.types import ValueBreakdown  # noqa: E402
from monopoly_game_engine.constants import COLOR_GROUPS, MAX_HOUSES, NUM_PLAYERS  # noqa: E402


@dataclass(frozen=True, slots=True)
class ASUPlusWeights:
    """Tunable additives folded into ASU's ``ValueBreakdown.total``."""

    w_cash: float = 1.0
    w_endgame: float = 1.0
    endgame_start_round: int = 120
    w_block: float = 0.25
    w_liq: float = 0.5

    def ablate(self, term: str) -> "ASUPlusWeights":
        """Return weights with one new term zeroed (``cash`` / ``endgame`` / ``block`` / ``liq``)."""

        if term == "cash":
            return ASUPlusWeights(
                w_cash=0.0,
                w_endgame=self.w_endgame,
                endgame_start_round=self.endgame_start_round,
                w_block=self.w_block,
                w_liq=self.w_liq,
            )
        if term == "endgame":
            return ASUPlusWeights(
                w_cash=self.w_cash,
                w_endgame=0.0,
                endgame_start_round=self.endgame_start_round,
                w_block=self.w_block,
                w_liq=self.w_liq,
            )
        if term == "block":
            return ASUPlusWeights(
                w_cash=self.w_cash,
                w_endgame=self.w_endgame,
                endgame_start_round=self.endgame_start_round,
                w_block=0.0,
                w_liq=self.w_liq,
            )
        if term == "liq":
            return ASUPlusWeights(
                w_cash=self.w_cash,
                w_endgame=self.w_endgame,
                endgame_start_round=self.endgame_start_round,
                w_block=self.w_block,
                w_liq=0.0,
            )
        raise ValueError(f"Unknown ablation term {term!r}")


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _developed_group_rent(color: str, squares: tuple[int, ...]) -> float:
    """Hotel-level (or full railroad/utility) monopoly rent proxy for a color group."""

    enabled = tuple(True for _ in squares)
    if color in ("railroad", "utility"):
        levels = tuple(0 for _ in squares)
    else:
        # Developed = hotels on every deed (ASU uses level 5 for hotel rent index).
        levels = tuple(MAX_HOUSES + 1 for _ in squares)
    return float(_hypothetical_group_rent(color, squares, levels, enabled))


def blocking_bonus(env, player_id: int) -> float:
    """Bonus for holding the last deed that prevents an opponent monopoly."""

    bonus = 0.0
    for color, group in COLOR_GROUPS.items():
        squares = tuple(group)
        size = len(squares)
        if size < 2:
            continue
        my_count = 0
        opp_counts: dict[int, int] = {}
        for square in squares:
            owner = env.properties[square].owner
            if owner is None:
                continue
            if owner == player_id:
                my_count += 1
            else:
                opp_counts[owner] = opp_counts.get(owner, 0) + 1
        if my_count != 1:
            continue
        for opponent, count in opp_counts.items():
            if count != size - 1:
                continue
            if env.players[opponent].bankrupt:
                continue
            bonus += _developed_group_rent(color, squares)
    return bonus


def cash_term(env, player_id: int, weights: ASUPlusWeights) -> float:
    return weights.w_cash * float(env.players[player_id].cash)


def endgame_margin_term(env, player_id: int, weights: ASUPlusWeights) -> float:
    own = float(env.players[player_id].net_worth())
    best_opp = max(
        (
            float(player.net_worth())
            for player in env.players
            if player.player_id != player_id and not player.bankrupt
        ),
        default=0.0,
    )
    lead = own - best_opp
    start = weights.endgame_start_round
    span = max(1, 200 - start)
    ramp = _clamp01((float(env.round) - start) / span)
    return weights.w_endgame * ramp * lead


def liquidity_term(env, player_id: int, weights: ASUPlusWeights) -> float:
    safety = safety_breakdown(env, player_id)
    return weights.w_liq * min(0.0, float(safety.cash_floor_margin))


def evaluate_value_plus(
    env,
    player_id: int,
    weights: ASUPlusWeights | None = None,
) -> ValueBreakdown:
    """ASU base breakdown with ASU+ additives folded into ``total``."""

    if weights is None:
        weights = ASUPlusWeights()
    if not 0 <= player_id < NUM_PLAYERS:
        raise ValueError(f"player_id must be in [0, {NUM_PLAYERS - 1}]")

    base = evaluate_value(env, player_id)
    player = env.players[player_id]
    if player.bankrupt or env.done:
        return base

    cash = cash_term(env, player_id, weights)
    endgame = endgame_margin_term(env, player_id, weights)
    block = weights.w_block * blocking_bonus(env, player_id)
    liq = liquidity_term(env, player_id, weights)
    return ValueBreakdown(
        base.m_assets,
        base.r_short,
        base.r_long,
        base.m_monopoly,
        base.terminal_utility,
        base.total + cash + endgame + block + liq,
    )


__all__ = [
    "ASUPlusWeights",
    "blocking_bonus",
    "cash_term",
    "endgame_margin_term",
    "evaluate_value_plus",
    "liquidity_term",
]
