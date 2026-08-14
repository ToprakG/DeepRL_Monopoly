"""Kuzey's Monopoly heuristic — the r5c champion.

    from heuristic import Champion
    agent = Champion()
    action = agent.choose_action(game, player_id, decision_seed)

That three-line contract is the whole interface. `game` may be the raw `MonopolyEnv` or
any wrapper with an `.env` attribute; the agent returns a legal action id.

WHAT YOU GET
    Champion      the measured champion. Base heuristic + two changes.  <- use this
    ChampionPlus  Champion + one extra rule that is free but only pays against
                  opponents that accept cash-for-deed trades. Ships enabled.
    Spine         the raw base heuristic, before any of tonight's changes.
    SpineH100     the previous champion, kept so improvements stay comparable.

See CLAUDE.md in the folder above for what each change does and what it measured.
"""

from __future__ import annotations

from . import _bind  # noqa: F401  -- MUST be first: binds engine/ as monopoly_game_engine
from ._bind import ENGINE_DIR, ROOT, engine_is_ours, unwrap
from .champion import SpineH100
from .champion import StChest as Champion
from .champion import StChestScrap as ChampionPlus
from .champion import StChestScrapScore as ChampionScore
from .spine import Spine

__all__ = [
    "Champion",
    "ChampionPlus",
    "ChampionScore",
    "Spine",
    "SpineH100",
    "ROOT",
    "ENGINE_DIR",
    "engine_is_ours",
    "unwrap",
]

__version__ = "r5c"
