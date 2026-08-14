"""Build H2H-compatible wrappers for the public competitor policies."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

INNCENTA_ID = "inncenta-heuristic"
BOOM_ID = "boom-hybrid"
ALINEBIDAL_ID = "alinebidal-final"
SLAYER_ID = "slayer-v1"
CODE_EXPOSURE_ID = "code-exposure"
EXPO_HEURISTIC_ID = "expo-heuristic-v1"
UNDERDOG_ID = "underdog-v1"

# Fetched 2026-08-14. SHAs in competitors/PINNED.json.
COMPETITOR_IDS = (
    INNCENTA_ID,
    BOOM_ID,
    ALINEBIDAL_ID,
    SLAYER_ID,
    CODE_EXPOSURE_ID,
    EXPO_HEURISTIC_ID,
    UNDERDOG_ID,
)
# Default oracle+3 field. Other competitors stay available via --lineup.
FIELD_COMPETITOR_IDS = (INNCENTA_ID, ALINEBIDAL_ID, SLAYER_ID)
ALINEBIDAL_ROOT = Path(__file__).resolve().parent / "alinebidal"


def _ensure_alinebidal_path() -> None:
    root = str(ALINEBIDAL_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def build_competitor(policy_id: str, player_id: int):
    if policy_id == INNCENTA_ID:
        from competitors.inncenta import InncentaAgent

        return InncentaAgent(player_id)
    if policy_id == BOOM_ID:
        from competitors.boom import BoomHybridAgent

        return BoomHybridAgent(player_id)
    if policy_id == ALINEBIDAL_ID:
        _ensure_alinebidal_path()
        from competition_agent.final_agent import FinalAgent

        return FinalAgent(player_id, rng_seed=player_id)
    if policy_id == SLAYER_ID:
        from competitors.asu_slayer import SlayerV1

        return SlayerV1(player_id)
    if policy_id == CODE_EXPOSURE_ID:
        from competitors.code_exposure import CodeExposureAgent

        return CodeExposureAgent(player_id)
    if policy_id == EXPO_HEURISTIC_ID:
        from competitors.expo_heuristic import ExpoHeuristicAgent

        return ExpoHeuristicAgent(player_id)
    if policy_id == UNDERDOG_ID:
        from competitors.underdog_agent import UnderdogAgent

        return UnderdogAgent(player_id)
    raise ValueError(f"Unknown competitor policy {policy_id!r}")


__all__ = [
    "ALINEBIDAL_ID",
    "BOOM_ID",
    "CODE_EXPOSURE_ID",
    "COMPETITOR_IDS",
    "EXPO_HEURISTIC_ID",
    "FIELD_COMPETITOR_IDS",
    "INNCENTA_ID",
    "SLAYER_ID",
    "UNDERDOG_ID",
    "build_competitor",
]
