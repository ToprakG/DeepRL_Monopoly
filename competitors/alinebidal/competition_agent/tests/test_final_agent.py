"""The frozen deliverable must load, be configured as shipped, and never raise.

An exception in a tournament is a forfeit; a suboptimal move is not. These
tests guard the difference.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ASU_FROZEN_TEACHER import ASUValueV1
from ASU_FROZEN_TEACHER.evaluate import _new_seeded_game
from competition_agent.final_agent import FinalAgent


def test_shipped_configuration():
    from competition_agent.beyond.denial import ENABLED as denial
    from competition_agent.beyond.endgame import ENABLED as endgame
    assert denial is True, "denial must ship enabled (+5.3pp, D5.1)"
    assert endgame is False, "endgame must ship disabled (conflicts, D5.2)"


def test_always_returns_a_legal_action():
    game = _new_seeded_game(970001)
    env = game.env
    agents = {0: FinalAgent(0), 2: FinalAgent(2),
              1: ASUValueV1(1), 3: ASUValueV1(3)}
    steps = 0
    while not env.done and steps < 400:
        actor = env.whose_turn()
        legal = {int(a) for a in env.get_allowed_actions(actor)}
        action = int(agents[actor].choose_action(env))
        assert action in legal, f"illegal action {action} for seat {actor}"
        game.step(action)
        steps += 1
    assert steps > 0


def test_survives_a_broken_inner_policy():
    """A failure inside the policy must degrade, not propagate."""
    game = _new_seeded_game(970002)
    env = game.env
    agent = FinalAgent(env.whose_turn())

    class Boom:
        def choose_action(self, _env):
            raise RuntimeError("inner failure")

    agent.policy = Boom()
    legal = {int(a) for a in env.get_allowed_actions(agent.player_id)}
    assert int(agent.choose_action(env)) in legal
