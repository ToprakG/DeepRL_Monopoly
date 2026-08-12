"""Hybrid distillation config: cheap play, oracle labels at event checkpoints.

Intended production labeling:
- DealMaker+Builder hybrid drives every move (all hybrid seats).
- 128-sim Max-N search runs only at sparse event checkpoints to record a
  teacher label (play action stays hybrid).
- Lineups mix pure hybrid self-play with a pool of frozen opponents.

Checkpoint policy (kept sparse on purpose):
- buy once when BUY_PROPERTY is legal
- incoming trade accept/decline once
- build once per pre-roll menu (not every successive house)
- auction once at opening bid (high_bid==0), not every bid tick
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from monopoly_game_engine.actions import OFFSETS, ActionType
from monopoly_game_engine.constants import COLOR_GROUPS
from monopoly_game_engine.env import PHASE_AUCTION, PHASE_PRE_ROLL, MonopolyEnv

from .agent import OracleConfig

# Locked defaults for the distillation phase (smallest comfortable H2H budget).
HYBRID_SIMS = 128
HYBRID_HORIZON = 16
HYBRID_ROLLOUTS = 1
HYBRID_MAX_WIDTH = 16
HYBRID_PRIOR_PEAK = 0.55


@dataclass(frozen=True, slots=True)
class HybridLabelConfig:
    """Calibrated teacher-label settings for hybrid checkpoint distillation."""

    simulations: int = HYBRID_SIMS
    rollout_horizon: int = HYBRID_HORIZON
    rollouts_per_leaf: int = HYBRID_ROLLOUTS
    max_depth: int = 64
    max_width: int = HYBRID_MAX_WIDTH
    c_puct: float = 1.5
    margin_temperature: float = 2000.0
    prior_peak: float = HYBRID_PRIOR_PEAK
    # Play always uses hybrid; search is label-only at checkpoints.
    play_with_hybrid: bool = True
    label_checkpoints_only: bool = True
    # Alternate self-play (4× hybrid play) with pool opponents.
    # Labels come from one focus seat only (even in self) so cost ≈ H2H oracle seat.
    mix_self_play: bool = True
    mix_pool: bool = True
    label_one_seat_only: bool = True

    def oracle_config(self) -> OracleConfig:
        return OracleConfig(
            simulations=self.simulations,
            c_puct=self.c_puct,
            max_depth=self.max_depth,
            max_width=self.max_width,
            rollout_horizon=self.rollout_horizon,
            rollouts_per_leaf=self.rollouts_per_leaf,
            margin_temperature=self.margin_temperature,
            prior_peak=self.prior_peak,
        )

    def as_dict(self) -> dict:
        return asdict(self)


def is_buy_checkpoint(env: MonopolyEnv, legal_set: set[int]) -> bool:
    return int(ActionType.BUY_PROPERTY) in legal_set


def is_trade_checkpoint(env: MonopolyEnv, legal_set: set[int]) -> bool:
    """Only trades that complete *our* monopoly or strip one of ours."""

    if (
        int(ActionType.ACCEPT_TRADE) not in legal_set
        and int(ActionType.DECLINE_TRADE) not in legal_set
    ):
        return False
    actor = env.whose_turn()
    offer = env._incoming_trade(actor)
    if offer is None:
        return False
    if offer.offered_prop is not None:
        color = offer.offered_prop.color
        if color not in ("railroad", "utility"):
            group = COLOR_GROUPS.get(color, [])
            owned = sum(1 for sq in group if env.properties[sq].owner == actor)
            if group and owned + 1 == len(group):
                return True
    if offer.requested_prop is not None and offer.requested_prop.is_monopoly:
        return True
    return False


def is_build_opportunity(legal_set: set[int]) -> bool:
    improve_lo = OFFSETS["improve_house"]
    improve_hi = OFFSETS["sell_house"]
    return any(improve_lo <= action < improve_hi for action in legal_set)


def is_auction_open_checkpoint(env: MonopolyEnv) -> bool:
    """One label per auction — only the opening call, not every bid tick."""

    if env.phase != PHASE_AUCTION:
        return False
    return int(getattr(env, "auction_high_bid", 0) or 0) == 0


def checkpoint_kind(env: MonopolyEnv, legal: Iterable[int]) -> str | None:
    """Return 'buy'|'trade'|'build'|'auction' or None."""

    legal_set = {int(action) for action in legal}
    if is_auction_open_checkpoint(env):
        return "auction"
    if env.phase == PHASE_AUCTION:
        return None
    if is_buy_checkpoint(env, legal_set):
        return "buy"
    if is_trade_checkpoint(env, legal_set):
        return "trade"
    if env.phase == PHASE_PRE_ROLL and is_build_opportunity(legal_set):
        return "build"
    return None


def is_event_checkpoint(
    env: MonopolyEnv,
    legal: Iterable[int],
    *,
    already_labeled_build_menu: bool = False,
    already_labeled_trade_round: bool = False,
) -> bool:
    """Sparse buy / build / property-trade / auction-open decisions.

    Caps: one build label per pre-roll menu, one trade label per round,
    one auction label at opening bid only. This avoids the 100+ label/game
    blow-up from bid ticks and trade-offer spam.
    """

    kind = checkpoint_kind(env, legal)
    if kind is None:
        return False
    if kind == "build" and already_labeled_build_menu:
        return False
    if kind == "trade" and already_labeled_trade_round:
        return False
    return True


def lineup_kind_for_game(game_index: int, config: HybridLabelConfig) -> str:
    """Alternate self / pool when both are enabled."""

    if config.mix_self_play and config.mix_pool:
        return "self" if game_index % 2 == 0 else "pool"
    if config.mix_pool and not config.mix_self_play:
        return "pool"
    return "self"


__all__ = [
    "HYBRID_SIMS",
    "HybridLabelConfig",
    "is_event_checkpoint",
    "lineup_kind_for_game",
]
