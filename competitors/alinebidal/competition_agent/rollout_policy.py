"""Phase 4 probe — truncated rollout over OUR leaf evaluation.

The question this answers
------------------------
Experiment 8 measured that the *teacher's* lookahead changes the *teacher's*
decisions on 35% of boundary states. That says nothing about us: our policy
agrees with the teacher 77.5% overall and 8% on trade proposals, and our leaf
evaluation is the component measured wrong three separate times (D2.5, D2.6,
D2.12).

If the leaf evaluation is the weak component, rollout over it does not repair
the weakness — it **amplifies** it, because every playout is scored by the same
faulty valuation. That is a testable claim, and this is the test.

Design
------
Shortlist the top-K candidates by the fast policy, run M seeded playouts of P
plies each with all seats driven by `spec_policy`, score leaves with
`spec_model.state_value`, and take the best mean. Common random numbers across
candidates: the same seed list is reused for every candidate so the comparison
is paired and the variance from dice cancels.

The live env is never mutated — every playout runs on a `deepcopy`, per the
project's standing constraint.

K/M/P are deliberately small. This is a diagnostic on whether rollout helps at
all, not a tuned competition configuration; if the sign is negative there is no
budget worth tuning.
"""

from __future__ import annotations

import copy
import os
import random
from typing import List, Optional

from competition_agent.spec_model import state_value
from competition_agent.spec_policy import SpecPolicy


class RolloutPolicy:
    """`spec_policy` with a truncated rollout over its own evaluation."""

    policy_id = "rollout_spec_v1"

    def __init__(self, player_id: int, rng_seed: int = 0,
                 k: Optional[int] = None, m: Optional[int] = None,
                 p: Optional[int] = None):
        self.player_id = player_id
        self.spec = SpecPolicy(player_id, rng_seed)
        self.K = int(os.environ.get("ROLLOUT_K", k or 4))
        self.M = int(os.environ.get("ROLLOUT_M", m or 3))
        self.P = int(os.environ.get("ROLLOUT_P", p or 6))  # OUR decisions, not plies
        self.rolled = 0
        self.changed = 0
        self.decisions = 0

    # ------------------------------------------------------------------
    def _shortlist(self, env, legal: List[int]) -> List[int]:
        """Spec's pick, plus the best alternatives by one-ply state value.

        The first version padded with legal actions sampled by stride, which
        injected candidates the rule pipeline would never consider — including
        selling one's own deeds — and that is half of why the first run
        produced 0/200 (D4.3).

        Ranking by the immediate post-action state value is honest about what
        is available: the valuation is the component under suspicion, so the
        test becomes "does deepening THIS valuation help over the rules", which
        is the actual Phase 4 question.
        """
        pick = self.spec.choose_action(env)
        scored = []
        for a in legal:
            if a == pick:
                continue
            try:
                sim = copy.deepcopy(env)
                sim.step(a)
                scored.append((state_value(sim, self.player_id), a))
            except Exception:                              # noqa: BLE001
                continue
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [pick] + [a for _, a in scored[: self.K - 1]]

    def _playout(self, env, action: int, seed: int) -> float:
        """Apply `action`, then play until OUR seat has acted `P` more times.

        Horizon alignment is the fix for D4.3. Scoring after a fixed number of
        *plies* compared leaves at different points in the turn cycle: END_TURN
        passes the turn so its plies are opponents acting, while sell_prop
        retains it so its plies are ours. That rewarded any action keeping the
        turn — including liquidating our own assets — and produced 0/200 with
        100% bankruptcy.

        Counting OUR OWN decisions instead puts every candidate's leaf at the
        same point in the cycle, so the comparison is like-for-like.
        """
        sim = copy.deepcopy(env)
        rng = random.Random(seed)
        try:
            sim.step(action)
        except Exception:                                  # noqa: BLE001
            return float("-inf")
        agents = {s: SpecPolicy(s) for s in range(len(sim.players))}
        our_turns, plies = 0, 0
        while our_turns < self.P and plies < self.P * 12:
            if getattr(sim, "done", False):
                break
            plies += 1
            actor = sim.whose_turn()
            try:
                a = int(agents[actor].choose_action(sim))
                legal = [int(x) for x in sim.get_allowed_actions(actor)]
                if a not in legal:
                    a = rng.choice(legal)
                sim.step(a)
                if actor == self.player_id:
                    our_turns += 1
            except Exception:                              # noqa: BLE001
                break
        return state_value(sim, self.player_id)

    # ------------------------------------------------------------------
    def choose_action(self, env) -> int:
        self.decisions += 1
        legal = [int(a) for a in env.get_allowed_actions(self.player_id)]
        if len(legal) == 1:
            return legal[0]

        fast = self.spec.choose_action(env)
        if len(legal) < 2:
            return fast

        cands = self._shortlist(env, legal)
        if len(cands) < 2:
            return fast

        self.rolled += 1
        # Common random numbers: the same seeds for every candidate, so the
        # dice are identical across the comparison and only the action differs.
        seeds = [1000 + i for i in range(self.M)]
        best, best_score = fast, float("-inf")
        for a in cands:
            vals = [self._playout(env, a, s) for s in seeds]
            vals = [v for v in vals if v != float("-inf")]
            if not vals:
                continue
            score = sum(vals) / len(vals)
            if score > best_score:
                best, best_score = a, score

        if best != fast:
            self.changed += 1
        return best if best in set(legal) else fast


__all__ = ["RolloutPolicy"]
