"""Phase 3 — the hybrid: rules everywhere, learned head on trade only.

Boundary set by measurement, not by architecture preference. Every family
outside trade recovers +0.0pp when pinned to the teacher's ground truth
(D2.13), so the rules keep them at their measured 90–99% agreement. The two
trade families carry ~19 of the ~23 available win-rate points, and the
hand-fitted ranker captures only 33.8% of that (D2.17), so the head owns them.

Fallbacks are explicit: if the checkpoint is missing, if the head's pick is
somehow illegal, or if anything raises, the rule pipeline answers. The hybrid
is never worse than `spec_policy` by construction — a property worth having
given how often in this project a plausible improvement measured negative.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import torch

from competition_agent.distill_train import (
    N_OUT, TradeHead, from_idx, relevant, to_idx,
)
from competition_agent.spec_policy import SpecPolicy

CKPT = Path(__file__).resolve().parent / "trade_head.pt"


class HybridPolicy:
    """`spec_policy` with the learned head substituted on trade decisions."""

    policy_id = "hybrid_v1"

    def __init__(self, player_id: int, rng_seed: int = 0,
                 ckpt: Optional[str] = None, enabled: Optional[bool] = None):
        self.player_id = player_id
        self.spec = SpecPolicy(player_id, rng_seed)
        # Feature flag, so the head can be switched off for an A/B on
        # identical code rather than across two commits (the lesson of D2.7).
        if enabled is None:
            enabled = os.environ.get("HYBRID_HEAD", "1") != "0"
        self.enabled = enabled
        self.model = None
        self.head_used = 0
        self.decisions = 0

        path = Path(ckpt) if ckpt else CKPT
        if self.enabled and path.exists():
            try:
                blob = torch.load(path, map_location="cpu")
                m = TradeHead()
                m.load_state_dict(blob["state_dict"])
                m.eval()
                self.model = m
            except Exception:                              # noqa: BLE001
                self.model = None                          # fall back to rules

    def choose_action(self, env) -> int:
        self.decisions += 1
        legal = [int(a) for a in env.get_allowed_actions(self.player_id)]
        if len(legal) == 1:
            return legal[0]

        if self.model is not None:
            scope = [a for a in legal if relevant(a)]
            # Only take over when the trade choice is real: at least two
            # in-scope options. A single legal trade action is not a decision.
            if len(scope) >= 2:
                try:
                    obs = torch.tensor(
                        env._get_state(self.player_id), dtype=torch.float32
                    ).unsqueeze(0)
                    mask = torch.full((1, N_OUT), float("-inf"))
                    for a in scope:
                        mask[0, to_idx(a)] = 0.0
                    with torch.no_grad():
                        pick = from_idx(
                            int((self.model(obs) + mask).argmax(dim=1).item())
                        )
                    if pick in set(legal):
                        self.head_used += 1
                        return pick
                except Exception:                          # noqa: BLE001
                    pass                                   # rules answer

        return self.spec.choose_action(env)


__all__ = ["HybridPolicy"]
