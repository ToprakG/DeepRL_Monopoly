"""Field best-response pool, mixed tables, and promotion bar."""

from __future__ import annotations

from monopoly_bench.adapters import CompetitorAdapter
from monopoly_bench.engine import SharedGame
from oracle.field_br import (
    FOUR_REAL_OPPONENTS,
    LEARNER_ID,
    TOURNAMENT_OPPONENTS,
    mixed_table_jobs,
    promotion_beats_incumbent,
    summarize_field_records,
)


def test_competitor_adapter_legal_for_four_real_opponents():
    game = SharedGame.new(3, max_rounds=1)
    actor = game.env.whose_turn()
    legal = game.env.get_allowed_actions(actor)
    for policy_id in FOUR_REAL_OPPONENTS:
        decision = CompetitorAdapter(policy_id).choose_action(game, actor, 7)
        assert decision.action in legal, policy_id
        assert not decision.fallback, policy_id


def test_mixed_tables_are_one_learner_plus_three_pool():
    jobs = mixed_table_jobs(generation=1, games=32, seed_base=0, snapshots=["snap.pt"])
    assert len(jobs) == 32
    categories = {job["category"] for job in jobs}
    assert {"tournament", "four_real", "mixed"} <= categories
    tournament = [job for job in jobs if job["category"] == "tournament"]
    assert tournament
    for job in jobs:
        names = job["policies"]
        assert names.count(LEARNER_ID) == 1
        assert job["policies"][job["learner_seat"]] == LEARNER_ID
        assert len(names) == 4
        assert all(names)
    for job in tournament:
        opp = {name for name in job["policies"] if name != LEARNER_ID}
        assert opp == set(TOURNAMENT_OPPONENTS)


def test_summarize_field_records_per_opponent_and_promotion_bar():
    records = []
    opponents = list(TOURNAMENT_OPPONENTS)
    for index in range(8):
        learner_seat = index % 4
        policies = [""] * 4
        policies[learner_seat] = LEARNER_ID
        opp_iter = iter(opponents)
        for seat in range(4):
            if not policies[seat]:
                policies[seat] = next(opp_iter)
        winner = learner_seat if index < 2 else (learner_seat + 1) % 4
        worth = [100.0, 100.0, 100.0, 100.0]
        worth[winner] = 500.0
        records.append(
            {
                "game_id": index,
                "seed": index,
                "learner_seat": learner_seat,
                "policies": policies,
                "winner": winner,
                "completed": True,
                "crashes": 0,
                "final_net_worth": worth,
            }
        )
    summary = summarize_field_records(records)
    assert summary["learner_wins"] == 2
    assert summary["completed"] == 8
    assert abs(summary["learner_win_rate"] - 0.25) < 1e-9
    assert set(summary["opponents"]) == set(TOURNAMENT_OPPONENTS)
    assert promotion_beats_incumbent(0.18, 0.25)
    assert not promotion_beats_incumbent(0.25, 0.25)
    assert not promotion_beats_incumbent(0.25, 0.20)
