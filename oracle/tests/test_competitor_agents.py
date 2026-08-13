"""Competitor agents return a legal action on a fresh env."""

from __future__ import annotations

from competitors.factory import (
    BOOM_ID,
    CODE_EXPOSURE_ID,
    COMPETITOR_IDS,
    EXPO_HEURISTIC_ID,
    FIELD_COMPETITOR_IDS,
    SLAYER_ID,
    UNDERDOG_ID,
    build_competitor,
)
from monopoly_game_engine.env import MonopolyEnv
from oracle.eval_h2h import (
    COMPETITOR_LINEUP,
    ORACLE_V2_ID,
    _parse_lineup,
    _parse_lineups,
)
from ASU_FROZEN_TEACHER.evaluate import AgentSpec, _run_game


def test_competitor_ids_fill_the_field_lineup():
    assert COMPETITOR_LINEUP[0] == ORACLE_V2_ID
    assert COMPETITOR_LINEUP[1:] == FIELD_COMPETITOR_IDS
    assert SLAYER_ID in COMPETITOR_LINEUP
    assert BOOM_ID in COMPETITOR_IDS
    assert BOOM_ID not in COMPETITOR_LINEUP
    assert CODE_EXPOSURE_ID in COMPETITOR_IDS
    assert EXPO_HEURISTIC_ID in COMPETITOR_IDS
    assert UNDERDOG_ID in COMPETITOR_IDS
    assert set(FIELD_COMPETITOR_IDS).issubset(set(COMPETITOR_IDS))


def test_four_team_lineup_does_not_require_oracle():
    field = _parse_lineup(
        ",".join(
            (
                "inncenta-heuristic",
                "alinebidal-final",
                "slayer-v1",
                "expo-heuristic-v1",
            )
        )
    )
    assert len(field) == 4
    assert "oracle-fast-v1" not in field
    fields = _parse_lineups(
        [
            "inncenta-heuristic,alinebidal-final,slayer-v1,expo-heuristic-v1;"
            "inncenta-heuristic,alinebidal-final,slayer-v1,code-exposure"
        ]
    )
    assert len(fields) == 2
    assert fields[1][3] == "code-exposure"


def test_each_competitor_returns_a_legal_opening_action():
    env = MonopolyEnv(agent_ids=[0], max_rounds=5)
    for policy_id in COMPETITOR_IDS:
        agent = build_competitor(policy_id, 0)
        legal = set(env.get_allowed_actions(0))
        action = agent.choose_action(env)
        assert action in legal, f"{policy_id} chose {action} not in {legal}"


class _EndTurnFactory:
    def build(self, spec, player_id):
        class _Agent:
            fallbacks = 0

            def choose_action(self, env):
                from monopoly_game_engine.actions import ActionType

                allowed = env.get_allowed_actions(player_id)
                if int(ActionType.END_TURN) in allowed:
                    return int(ActionType.END_TURN)
                return allowed[0]

        return _Agent()


def test_game_timeout_truncates_immediately():
    specs = tuple(AgentSpec("fixed-a", "fixed-a") for _ in range(4))
    result = _run_game(specs, 0, 0, 20_000, _EndTurnFactory(), game_timeout_s=0.0)
    assert result["truncated"] is True
    assert result["timed_out"] is True
    assert result["winner"] is None
    assert result["decisions"] == 0
