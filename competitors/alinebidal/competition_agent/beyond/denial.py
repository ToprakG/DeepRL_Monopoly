"""Phase 5 module 2 — denial value, as a valuation term.

Why a term and not a module
---------------------------
D4.4 identified the leaf valuation as the root cause for the fourth
independent time. Layering "modules" on top of a function measured wrong four
times repeats the Phase 4 mistake, so denial is implemented as an additional
term inside the valuation and measured the only way that has proved reliable
in this project: win rate, behind a flag, A/B on identical code.

What it prices
--------------
SPEC B3/B5 measured a concrete weakness in the teacher we cloned: it pays the
**same** for the first deed of a colour group as for the second (own1/own0 =
1.00 to the dollar in 18/18 cases) and escalates only on the completing deed.
It therefore never bids or trades defensively for a group it has no presence
in — so an opponent can take the first two deeds of a group unopposed.

The clone inherits that blind spot. This term prices what a deed *denies*:
holding a deed an opponent needs to complete a group is worth the monopoly
that opponent does not get.

    denial(sq) = sum over opponents of
                   (their group rent potential / 2**their_missing_after)
                   weighted by how close they are to completing it

Only groups where an opponent already has presence are counted, mirroring the
B5 clause the probes established — a group nobody is near is not being denied.

Flag: BEYOND_DENIAL=1 (default off, so it is opt-in and A/B-able).
"""

from __future__ import annotations

import os

from monopoly_game_engine.constants import COLOR_GROUPS, PROPERTIES

from competition_agent.spec_model import max_group_rent

ENABLED = os.environ.get("BEYOND_DENIAL", "0") != "0"
WEIGHT = float(os.environ.get("BEYOND_DENIAL_W", "1.0"))


def denial_value(env, pid: int, sq: int) -> float:
    """What holding `sq` denies opponents, in the valuation's own units.

    Zero when no opponent has presence in the group, per SPEC B5: a group
    nobody is near is not being denied anything.
    """
    if not ENABLED:
        return 0.0

    group = COLOR_GROUPS[PROPERTIES[sq]["color"]]
    size = len(group)
    total = 0.0

    for opp in env.players:
        if opp.player_id == pid or opp.bankrupt:
            continue
        owned = sum(1 for s in group
                    if env.properties[s].owner == opp.player_id)
        if owned == 0:
            continue                      # B5 — no presence, nothing denied

        # If they took this deed they would hold owned+1 and miss the rest.
        missing_after = size - (owned + 1)
        if missing_after < 0:
            continue
        gain_to_them = max_group_rent(env, sq) / (2 ** missing_after)

        # Weight by how much of the group is already theirs: denying the
        # completing piece of a near-monopoly is worth far more than denying
        # a group they have barely started.
        closeness = (owned + 1) / size
        total += gain_to_them * closeness

    return WEIGHT * total


__all__ = ["denial_value", "ENABLED"]
