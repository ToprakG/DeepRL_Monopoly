"""Phase 5 module 3 — endgame objective switch, as a valuation term.

The measured basis
------------------
Net-worth maximisation is not win-probability maximisation, and this clone has
a specific, measured symptom of the difference: across head-to-heads it goes
bankrupt in **80–93%** of games while the teacher does so in **57–70%**. It is
not losing on assets, it is dying.

The safety gates (SPEC D1–D5) are calibrated to reproduce the *teacher's*
caution, which is itself tuned for net worth. Late in a game that is the wrong
target: with the bank drained of houses and monopolies developed, one landing
can end you, and a cushion sized for the average case is not sized for the
case that matters.

What this term does
-------------------
`stage(env)` reads how far the game has progressed from board state rather
than round count — bank house/hotel depletion, monopolies formed, and how
developed they are. It runs 0 (opening) to 1 (fully developed endgame).

`cushion_multiplier` then scales the safety requirement with stage, so the
agent grows more conservative exactly as the board grows lethal. This is the
"switch from net worth to survival" the brief describes, expressed as a change
in the gate the valuation already uses rather than as a separate objective —
consistent with D4.4's finding that layering on top of the valuation is what
failed.

Flags: BEYOND_ENDGAME=1 to enable, BEYOND_ENDGAME_W to scale the effect.
"""

from __future__ import annotations

import os

from monopoly_game_engine.constants import COLOR_GROUPS

ENABLED = os.environ.get("BEYOND_ENDGAME", "0") != "0"
WEIGHT = float(os.environ.get("BEYOND_ENDGAME_W", "1.0"))


def stage(env) -> float:
    """How far into the game, from board state. 0 = opening, 1 = endgame.

    Round count is deliberately not used: two games at the same round can be
    at completely different stages depending on how the deeds fell.
    """
    houses = getattr(env, "houses_available", 32)
    hotels = getattr(env, "hotels_available", 12)
    # Bank depletion: the single best signal that development is advanced.
    depletion = 1.0 - (houses / 32.0) * 0.5 - (hotels / 12.0) * 0.5

    monos = 0
    developed = 0.0
    for squares in COLOR_GROUPS.values():
        owners = {env.properties[s].owner for s in squares}
        if len(owners) == 1 and None not in owners:
            monos += 1
            h = sum(env.properties[s].houses for s in squares)
            developed += min(h / (5.0 * len(squares)), 1.0)
    mono_frac = min(monos / 4.0, 1.0)          # 4+ monopolies = fully on
    dev_frac = min(developed / 4.0, 1.0)

    return max(0.0, min(1.0, 0.4 * depletion + 0.3 * mono_frac + 0.3 * dev_frac))


def cushion_multiplier(env) -> float:
    """Scale on the safety cushion. 1.0 early, up to ~2.5x in a lethal endgame.

    Applied to the gate-1 floor, which SPEC A2/D1 measured at $200. That floor
    reproduces the teacher, and the teacher bankrupts far less often than we
    do — but it is a constant, and the danger it guards against is not.
    """
    if not ENABLED:
        return 1.0
    return 1.0 + WEIGHT * 1.5 * stage(env)


__all__ = ["stage", "cushion_multiplier", "ENABLED"]
