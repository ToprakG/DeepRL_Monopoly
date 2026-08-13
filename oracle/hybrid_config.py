"""Hybrid distillation config: cheap play, oracle labels at richer checkpoints.

Intended production labeling:
- DealMaker+Builder hybrid drives every move (all hybrid seats).
- 128-sim Max-N search records soft visit + value labels at:
  - event checkpoints (buy / build / incoming trade / auction open)
  - a small random subsample of other multi-legal decisions (routine)
- Soft visit distributions are always written (never one-hot stubs).
- Lineups mix pure hybrid self-play with a pool of frozen opponents.

Checkpoint policy:
- buy once when BUY_PROPERTY is legal
- every incoming accept/decline once per seat/round (includes declines)
- build once per pre-roll menu (not every successive house)
- auction once at opening bid (high_bid==0), not every bid tick
- routine decisions at ``routine_label_prob`` (END_TURN menus, offers, etc.)
- ``--broad-value`` raises routine_label_prob to 0.25 for denser value labels
- Distill can blend Max-N backups with the game-winner one-hot (``--blend-outcomes``)
"""

from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

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
# ~1/12 multi-legal non-event decisions → enough END_TURN/structure without
# blowing up to 100+ labels/game.
DEFAULT_ROUTINE_LABEL_PROB = 0.08
# `--broad-value` labeling: denser routine subsample for value coverage.
BROAD_ROUTINE_LABEL_PROB = 0.25
# Distill value mix: 0 = Max-N backups only, 1 = game-winner one-hot only.
DEFAULT_VALUE_OUTCOME_MIX = 0.0
BROAD_VALUE_OUTCOME_MIX = 0.5


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
    # Play always uses hybrid; search is label-only at selected decisions.
    play_with_hybrid: bool = True
    label_checkpoints_only: bool = True
    # Fraction of non-event multi-legal decisions to label with Max-N.
    routine_label_prob: float = DEFAULT_ROUTINE_LABEL_PROB
    # Alternate self-play (4× hybrid play) with pool opponents.
    # Labels come from one focus seat only (even in self) so cost ≈ H2H oracle seat.
    mix_self_play: bool = True
    mix_pool: bool = True
    label_one_seat_only: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.routine_label_prob) <= 1.0:
            raise ValueError("routine_label_prob must be in [0, 1]")

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
            deadline_s=None,
            early_stop_visit_lead=None,
        )

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def broad_value(cls, **overrides: Any) -> "HybridLabelConfig":
        """Same hybrid recipe with denser routine labels for value coverage."""

        payload = dict(overrides)
        payload.setdefault("routine_label_prob", BROAD_ROUTINE_LABEL_PROB)
        return cls(**payload)


def is_buy_checkpoint(env: MonopolyEnv, legal_set: set[int]) -> bool:
    return int(ActionType.BUY_PROPERTY) in legal_set


def is_monopoly_relevant_trade(env: MonopolyEnv) -> bool:
    """True when the offer completes our color set or strips a monopoly we hold."""

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


def is_trade_checkpoint(env: MonopolyEnv, legal_set: set[int]) -> bool:
    """Any incoming accept/decline decision (junk offers included → teaches DECLINE)."""

    if (
        int(ActionType.ACCEPT_TRADE) not in legal_set
        and int(ActionType.DECLINE_TRADE) not in legal_set
    ):
        return False
    return env._incoming_trade(env.whose_turn()) is not None


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
    """Sparse buy / build / incoming-trade / auction-open decisions.

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


def should_label_routine(
    *,
    seed: int,
    step: int,
    actor: int,
    prob: float,
) -> bool:
    """Deterministic Bernoulli subsample for non-event multi-legal decisions."""

    if prob <= 0.0:
        return False
    if prob >= 1.0:
        return True
    # Cheap stable hash → [0, 1).
    mixed = (int(seed) * 1_000_003 + int(step) * 97 + int(actor) * 17) & 0xFFFFFFFF
    unit = (mixed % 10_000) / 10_000.0
    return unit < float(prob)


def hybrid_label_config_from_args(args: Namespace) -> HybridLabelConfig:
    """Build hybrid config from label_gen CLI. Explicit routine prob wins over ``--broad-value``."""

    overrides: dict[str, Any] = {}
    routine = getattr(args, "routine_label_prob", None)
    if routine is not None:
        overrides["routine_label_prob"] = float(routine)
    elif getattr(args, "broad_value", False):
        overrides["routine_label_prob"] = BROAD_ROUTINE_LABEL_PROB
    if getattr(args, "calibrate", False):
        return HybridLabelConfig(**overrides)
    return HybridLabelConfig(
        simulations=int(args.sims),
        rollout_horizon=int(args.horizon),
        rollouts_per_leaf=int(args.rollouts),
        margin_temperature=float(args.margin_temperature),
        **overrides,
    )


def resolve_value_outcome_mix(
    *,
    mix: float | None = None,
    blend_outcomes: bool | None = None,
    broad_value: bool = False,
) -> float:
    """``--value-outcome-mix`` wins, then ``--blend-outcomes`` / ``--no-blend-outcomes``, then ``--broad-value``."""

    if mix is not None:
        mix_f = float(mix)
        if not 0.0 <= mix_f <= 1.0:
            raise ValueError("value_outcome_mix must be in [0, 1]")
        return mix_f
    if blend_outcomes is True:
        return BROAD_VALUE_OUTCOME_MIX
    if blend_outcomes is False:
        return DEFAULT_VALUE_OUTCOME_MIX
    if broad_value:
        return BROAD_VALUE_OUTCOME_MIX
    return DEFAULT_VALUE_OUTCOME_MIX


def blend_value_vectors(
    backups: np.ndarray,
    outcomes: np.ndarray,
    mix: float,
) -> np.ndarray:
    """Convex mix of search backups and winner one-hots. Truncated (all-zero) rows keep backups."""

    mix_f = float(mix)
    if not 0.0 <= mix_f <= 1.0:
        raise ValueError("value_outcome_mix must be in [0, 1]")
    backups_f = np.asarray(backups, dtype=np.float64)
    outcomes_f = np.asarray(outcomes, dtype=np.float64)
    if backups_f.shape != outcomes_f.shape:
        raise ValueError(f"value/outcome shape mismatch: {backups_f.shape} vs {outcomes_f.shape}")
    if mix_f <= 0.0:
        return backups_f.astype(np.float32, copy=True)
    valid = outcomes_f.sum(axis=-1, keepdims=True) > 0.5
    if mix_f >= 1.0:
        blended = np.where(valid, outcomes_f, backups_f)
        return blended.astype(np.float32)
    mixed = (1.0 - mix_f) * backups_f + mix_f * outcomes_f
    blended = np.where(valid, mixed, backups_f)
    return blended.astype(np.float32)


def lineup_kind_for_game(game_index: int, config: HybridLabelConfig) -> str:
    """Alternate self / pool when both are enabled."""

    if config.mix_self_play and config.mix_pool:
        return "self" if game_index % 2 == 0 else "pool"
    if config.mix_pool and not config.mix_self_play:
        return "pool"
    return "self"


__all__ = [
    "BROAD_ROUTINE_LABEL_PROB",
    "BROAD_VALUE_OUTCOME_MIX",
    "DEFAULT_ROUTINE_LABEL_PROB",
    "DEFAULT_VALUE_OUTCOME_MIX",
    "HYBRID_SIMS",
    "HybridLabelConfig",
    "blend_value_vectors",
    "checkpoint_kind",
    "hybrid_label_config_from_args",
    "is_event_checkpoint",
    "is_monopoly_relevant_trade",
    "is_trade_checkpoint",
    "lineup_kind_for_game",
    "resolve_value_outcome_mix",
    "should_label_routine",
]
