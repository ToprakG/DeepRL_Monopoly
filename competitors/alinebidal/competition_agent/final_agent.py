"""Tournament entry point — the frozen deliverable.

    from competition_agent.final_agent import FinalAgent
    agent = FinalAgent(player_id)
    action = agent.choose_action(env)

Configuration, and why it is this one
-------------------------------------
`spec_policy` with the denial valuation term enabled and the endgame term
disabled. Every component here was selected by a head-to-head win rate against
`ASUValueV1`, seat-rotated on a common seed base — not by agreement, not by a
projection, and not by an architectural argument.

    configuration                     win rate vs teacher      n
    floor (no trade proposals)        18.3% [14.4, 23.1]      300
    spec_policy (baseline)            25.7% [21.1, 30.9]      300
    + learned trade head (Phase 3)    24.0% [18.6, 30.4]      200
    + rollout layer (Phase 4)         14.5% [10.3, 20.0]      200
    + denial + endgame                22.0% [18.2, 26.3]      400
    **+ denial only  <- SHIPPED**     31.0% [25.0, 37.7]      200

What was tried and rejected, each on its own measurement:

- **Learned trade head** (Phase 3). Reached 38.51% held-out top-1 against the
  hand-fitted ranker's 29.86% — a real agreement gain that converted to −1.7pp
  of win rate. Regularisation across a 4x capacity range moved held-out top-1
  by 0.5pp, so ~38.5% is a ceiling for that model class, not an artefact.
- **Rollout layer** (Phase 4). −11.2pp at p=0.003, worse than making no trade
  proposals at all. Lookahead over a leaf evaluation that is itself wrong
  amplifies the error: the same function scores the leaves *and* ranks the
  shortlist, so depth multiplies it.
- **Endgame switch** (Phase 5 module 3). Conflicts with denial — it feeds the
  safety gate, making every purchase harder, while denial's mechanism is
  buying deeds to block opponents. The survival hypothesis stands (we bankrupt
  at ~88% against the teacher's ~60%); the threshold-widening implementation
  does not. See DECISIONS D5.2.

Known limits, stated plainly
----------------------------
- Win rate is **~31% against a 50% parity**, so the teacher still wins clearly.
  The measured ceiling for perfect trade decisions alone is 40.0%, and for
  perfect trade proposals *and* replies 45.8% — parity is not reachable by
  trade work alone.
- The leaf valuation was long recorded here as the root cause of the remaining
  gap, by four independent diagnoses (D2.5, D2.6, D2.12, D4.4). **Corrected in
  D7.8:** `spec_model.state_value` is never called by this agent — its only
  callers are `swap_delta` (used by a test) and `rollout_policy` (not shipped).
  The one valuation this agent uses is `deed_value`, and only inside
  `_trade_reply`. Any work aimed at "the leaf valuation" has to target that
  call site or it changes nothing measurable.
- Denial's +5.3pp is the project's only positive result; its confirmation
  status is recorded in DECISIONS D5.3.

The teacher is never consulted at runtime. This agent is derived entirely from
the behavioural specification in SPEC.md, which was built from probe outputs
alone — `core.py` and `evaluate.py` were never read.
"""

from __future__ import annotations

import os
from pathlib import Path

# Frozen configuration. Set before importing anything that reads these, since
# the beyond/ modules latch their flags at import time.
os.environ.setdefault("BEYOND_DENIAL", "1")     # +5.3pp, DECISIONS D5.1/D5.3
os.environ.setdefault("BEYOND_ENDGAME", "0")    # conflicts with denial, D5.2

# Candidate D — the learned trade-proposal ranker. Measured against every
# field on paired seeds (D7.6): ASU 2v2 +6.17pp (p=0.0065), weak field
# +7.90pp (p<0.0001), strong field +0.45pp which does not survive Bonferroni.
# The alternative of simply proposing less often was tested and is 7.23pp
# WORSE than doing nothing (D7.7), so the network is carrying the gain, not
# the threshold.
#
# `spec_policy` resolves this path relative to the package and falls back to
# the linear scorer if the checkpoint or torch is unavailable, so a broken
# install costs the trade branch rather than the agent.
os.environ.setdefault(
    "TRADE_RANKER",
    str(Path(__file__).resolve().parent / "probes" / "rank_gate_1000.json"))

from competition_agent.spec_policy import SpecPolicy  # noqa: E402


class FinalAgent:
    """The competition entry. Wraps `spec_policy` in the shipped configuration.

    Never raises on a decision: any internal failure falls back to a legal
    action rather than propagating, because an exception in a tournament is a
    forfeit while a suboptimal move is merely a suboptimal move.
    """

    policy_id = "final_agent_v1"
    config = {"BEYOND_DENIAL": "1", "BEYOND_ENDGAME": "0",
              "TRADE_RANKER": "probes/rank_gate_1000.json"}

    def __init__(self, player_id: int, rng_seed: int = 0):
        self.player_id = player_id
        self.policy = SpecPolicy(player_id, rng_seed)

    def choose_action(self, env) -> int:
        legal = [int(a) for a in env.get_allowed_actions(self.player_id)]
        if not legal:
            return 0
        if len(legal) == 1:
            return legal[0]
        try:
            action = int(self.policy.choose_action(env))
        except Exception:                                  # noqa: BLE001
            return legal[0]
        return action if action in set(legal) else legal[0]


__all__ = ["FinalAgent"]
