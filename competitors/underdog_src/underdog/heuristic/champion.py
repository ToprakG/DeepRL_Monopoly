"""r5b STACKING brief -- put round 1's survivors in one agent and measure the INTERACTION.

Nothing here is a new idea.  Every rule below was written, screened and reported by a
round-1 agent; this module's entire job is to answer a question none of them could answer
alone: **do they add up?**  The deliverable is a table of increments over `build_relief0`,
not a champion.

THE FOUR PIECES, as round 1 measured them against `build_relief0` (paired, N=120):

    piece              what it does                                strongish   mixed   duel
    f1_chest_relief    mortgage deeds that can NEVER be a monopoly    +4.2      +7.5    +4.2
                       and hold the cash to contest auctions, while
                       there is still unowned property on the board
    end_scrapbuy       near the cap, buy deeds for cash at LIST and    0.0      +8.3     --
                       drop the peer veto (a deed scores 2.5x list
                       at the cap, cash scores 1.0x)
    cu_thaw            redeem a mortgaged deed inside a group we      +0.8       0.0     0.0
                       own outright, so the build ladder unfreezes
    cu_reach           size the cash reserve against the worst rent   +0.8       --      --
                       REACHABLE IN ONE ROLL, not the worst on the
                       whole board

`cu_thaw` and `cu_reach` are each worth ONE game in 120.  The brief's own power note says
N=120 cannot certify +3 pp, let alone +0.8, so they are carried as passengers inside the
bigger stacks and are never screened alone -- round 1 already did that and the answer was
"one game".  The two pieces that can actually move a number are the war chest and the
near-cap buy, and the interesting fact about them is below.

==============================================================================
THE COLLISION, AND HOW IT WAS RESOLVED
==============================================================================
`combine.py --inspect f1_chest_relief end_scrapbuy cu_thaw cu_reach` reports three
conflicting names.  Two are spurious and one is real:

  * `BUILD_RESERVE_RELIEF` and `_build` -- SPURIOUS.  All four variants are
    `build_relief0`, and each module reproduced the same `_build` locally rather than
    import another agent's file.  `combine.py` cannot see that they are the same code, so
    it flags them.  This module keeps exactly ONE copy (`_Relief0Build`) and every variant
    inherits it, which makes the conflict disappear rather than resolving it.

  * `_unmortgage` -- REAL, and it is a genuine strategic disagreement:
        `_WarChestMixin` keeps a mortgaged deed mortgaged ON PURPOSE (otherwise the parent
            redeems next turn what the war chest mortgaged last turn, and the pair
            oscillates instead of holding a war chest);
        `_ThawMixin` wants mortgaged deeds redeemed SOONER than the parent's gate allows.

    RESOLUTION -- and this is the finding, not a compromise: **the two rules address
    DISJOINT SETS OF DEEDS, so there is no disagreement to split.**  `_ThawMixin` fires
    only when `ctx.is_development_group(square)` is true, which `_Context` defines
    (mono/spine.py:993) as owning EVERY deed of that colour.  `_WarChestMixin._dead_deeds`
    collects only deeds whose colour group contains a deed owned by SOMEONE ELSE.  A deed
    cannot satisfy both tests, and both tests skip railroads and utilities.  So the merge
    is a chain, not a choice:

        _ThawMixin  ->  _WarChestMixin  ->  Spine._unmortgage
        "is a group of ours frozen by a mortgage?  thaw the cheapest one."
              else "hide the dead deeds from the parent while the board is still open."
                    else "the parent's ordinary gated redemption."

    Both branches keep their full round-1 behaviour and neither is silently dropped.  With
    `cu_reach` in the stack a fourth link goes on the FRONT, `_ReachMixin`, which only
    rescopes `ctx.reserve` for the duration of the call, so it lowers the parent's gate at
    the end of the chain without touching either rule above it.

THE OTHER STRUCTURAL FACT, which is a prediction this module then tests: **the war chest
and the near-cap buy CANNOT OVERLAP IN TIME.**  `_chest_target` returns 0.0 the moment
`unowned_deeds` hits 0, and `_WarChestMixin._unmortgage` gates its veto on
`ctx.unowned_deeds > 0`; the round-banded census says the board is bought out by round 25
and has 0 unowned deeds from round 50 on.  `_endgame_weight` is 0.0 until round 100 of 200.
So the war chest is a rounds-0-to-25 rule and the near-cap buy is a rounds-100-to-200 rule,
with a 75-round gap between them.  If two rules never see the same board state, their
effects should be additive -- and that is exactly what this module screens for.

==============================================================================
RESULTS -- filled in from the measurement JSONs this module produced.
==============================================================================
See the STACK RESULTS block at the bottom of this file.

PROVENANCE.  `_Relief0Build` and `_EndgameMixin`/`_ScrapMixin`/`_ScrapBuyMixin` are copied
from `plans/r5b/variants/endgame.py`; `_WarChestMixin` from `plans/r5b/variants/free1.py`;
`_ThawMixin` and `_ReachMixin` from `plans/r5b/variants/cashuse.py`.  Copied rather than
imported for the reason both of those modules state in their own docstrings: nine agents are
editing this directory at once, and a stack that imports four other agents' files is a stack
whose measurements can be invalidated by an edit somebody else makes tomorrow.
"""

from __future__ import annotations

import os
import sys


from typing import Iterable, Optional  # noqa: E402

from engine.actions import OFFSETS  # noqa: E402
from engine.constants import PROPERTY_IDS, TRADE_CASH_LEVELS  # noqa: E402
from .spine import (  # noqa: E402
    GROUP_ORDER,
    GROUP_POWER,
    GROUP_RENT_PER_ROUND,
    _COLOR_OF,
    _GROUP_SQUARES,
    _HOUSE_PRICE,
    _MORTGAGE_OF,
    _PRICE_OF,
    _PROP_INDEX,
    _REAL_GROUPS,
)
from .spine import Spine


class SpineH100(Spine):
    """The r5 champion: Spine with the one constant that mattered set correctly."""
    name = "spine_h100"
    VALUE_HORIZON = 100.0

_N_PROPS = len(PROPERTY_IDS)
_N_CASH_LEVELS = len(TRADE_CASH_LEVELS)
_EXCH_STRIDE = _N_PROPS * (_N_PROPS - 1)
_TRADE_STRIDE = _N_PROPS * _N_CASH_LEVELS

# engine/state.py:43 -- Property.calculate_net_worth, the only thing that decides a capped
# game.  A deed scores 2.5x list, or 5.0x inside a monopoly; cash scores 1.0x.
SCORE_MULT = 2.5
SCORE_MULT_SET = 5.0

# 2d6 cannot roll a 1 and cannot roll past 12: the whole set of squares that can charge us
# rent before we get another decision.
_ONE_ROLL = tuple(range(2, 13))


# ==========================================================================
# THE ONE SHARED PARENT.  `build_relief0`, kept in a single copy so that the
# `_build` / `BUILD_RESERVE_RELIEF` "conflicts" combine.py reports simply do not
# exist inside this module.
# (verbatim from plans/r5b/variants/endgame.py `_Relief0Build`)
# ==========================================================================
class _Relief0Build(SpineH100):
    """BUILD_RESERVE_RELIEF = 0.0 -- the only survivor of 44 screens, re-confirmed on a
    second independent seed (strongish +5.0 z=+2.15, mixed +6.7 z=+2.35).  The climb to
    BUILD_TARGET may spend every dollar of cash; only escalation past BUILD_TARGET keeps a
    floor.  `_build_floor` and `_escalate_buffer` are split out of the ladder so a variant
    can override one line instead of the whole method."""

    BUILD_RESERVE_RELIEF = 0.0

    def _build_floor(self, ctx, step_level: int) -> float:
        if step_level <= self.BUILD_TARGET:
            return ctx.reserve * self.BUILD_RESERVE_RELIEF
        return ctx.reserve

    def _escalate_buffer(self, ctx, squares, house_price: float) -> float:
        return ctx.reserve + self.ESCALATE_LUMPS * len(squares) * house_price

    def _build(self, ctx, aset: set) -> Optional[int]:
        env, pid = ctx.env, ctx.pid
        cash = env.players[pid].cash
        best = None

        for group in GROUP_ORDER:
            squares = _GROUP_SQUARES[group]
            props = [env.properties[sq] for sq in squares]
            if any(p.owner != pid or p.mortgaged for p in props):
                continue
            if not props[0].is_monopoly:
                continue
            levels = [p.houses for p in props]
            floor_level = min(levels)
            if floor_level >= 5:
                continue

            target = self.BUILD_TARGET
            if floor_level >= target:
                target = self.BUILD_ESCALATE
                if target <= floor_level:
                    continue
            step_level = floor_level + 1
            house_price = _HOUSE_PRICE[squares[0]]
            k = sum(1 for lv in levels if lv == floor_level)
            lump = k * house_price

            if cash - lump < self._build_floor(ctx, step_level):
                continue
            if step_level > self.BUILD_TARGET:
                if cash - lump < self._escalate_buffer(ctx, squares, house_price):
                    continue
            if step_level == 5 and not ctx.hotels_ok(env):
                continue

            plan_cost = sum(
                (min(target, 5) - lv) * house_price for lv in levels if lv < target
            )
            if plan_cost <= 0:
                continue
            gain = (
                GROUP_RENT_PER_ROUND[group][min(target, 5)]
                - ctx.current_group_rent(group)
            ) * ctx.opp_scale
            score = gain / plan_cost

            action = self._pick_house_action(ctx, group, step_level, floor_level, aset)
            if action is None:
                continue
            key = (score, GROUP_POWER[group])
            if best is None or key > best[0]:
                best = (key, action)

        return None if best is None else best[1]


class StBase(_Relief0Build):
    """`build_relief0` under an `st_` name.  Exists so that a stack can be paired against
    its own parent inside this module if a future round wants it; NOT screened here,
    because `build_relief0` is already measured at N=120 on the same seed and panels and
    every number below is paired against that file."""

    name = "st_base"


# ==========================================================================
# PIECE 1 of 4 -- the near-cap net-worth rules
# (verbatim from plans/r5b/variants/endgame.py: `_EndgameMixin`, `EndScrap`, `EndScrapBuy`,
#  restated as plain mixins so they can be composed with the war chest)
# ==========================================================================
class _EndgameMixin:
    """Where are we on the clock, and what will a deed SCORE when it stops?"""

    ENDGAME_SPAN = 100.0

    def _endgame_weight(self, env) -> float:
        cap = float(getattr(env, "max_rounds", 200) or 200)
        left = cap - float(env.round)
        if left >= self.ENDGAME_SPAN:
            return 0.0
        if left <= 0.0:
            return 1.0
        return 1.0 - left / self.ENDGAME_SPAN

    def _score_value(self, ctx, square: int, player: int) -> float:
        price = float(_PRICE_OF[square])
        return price * (SCORE_MULT_SET if ctx.completes_for(square, player) else SCORE_MULT)


class _ScrapMixin:
    """`end_scrap` -- never hand a deed away for less than it will score at the cap.

    ROUND 1 MEASURED THIS HALF AS A DEAD BRANCH: `end_scrap` is outcome-identical to
    `build_relief0` on BOTH panels, 0 of 120 games differing each time.  It is carried
    only because `_ScrapBuyMixin` is written as an override of it and splitting them
    would be a transcription of somebody else's code into a different shape, which is how
    stacks acquire bugs.  It contributes nothing and costs nothing.
    """

    def _give_value(self, ctx, square: int) -> float:
        rent_value = ctx.deed_value(square, ctx.pid)
        w = self._endgame_weight(ctx.env)
        if w <= 0.0:
            return rent_value
        return max(rent_value, w * self._score_value(ctx, square, ctx.pid))

    def _exchange_candidates(self, ctx, target: int, aset: set) -> Iterable[tuple]:
        env, pid = ctx.env, ctx.pid
        t_idx = ctx.other_index[target]
        base = OFFSETS["exch_trade"] + t_idx * _EXCH_STRIDE
        mine = [
            sq for sq in PROPERTY_IDS
            if env.properties[sq].owner == pid and env.properties[sq].houses == 0
        ]
        theirs = [
            sq for sq in PROPERTY_IDS
            if env.properties[sq].owner == target and env.properties[sq].houses == 0
        ]
        for want in theirs:
            my_gain = ctx.deed_value(want, pid)
            their_loss = ctx.deed_value(want, target)
            for give in mine:
                if ctx.completes_for(give, target):
                    continue
                if env.properties[give].is_monopoly:
                    continue
                offer_idx = _PROP_INDEX[give]
                req_idx = _PROP_INDEX[want]
                req_raw = req_idx if req_idx < offer_idx else req_idx - 1
                action = base + offer_idx * (_N_PROPS - 1) + req_raw
                if action not in aset:
                    continue
                edge = my_gain - self._give_value(ctx, give)
                if edge <= self.TRADE_OFFER_MARGIN:
                    continue
                peer = ctx.deed_value(give, target) - their_loss
                if peer <= self.TRADE_PEER_MARGIN:
                    continue
                yield (edge + 0.25 * peer, action)

    def _cash_candidates(self, ctx, target: int, aset: set) -> Iterable[tuple]:
        env, pid = ctx.env, ctx.pid
        t_idx = ctx.other_index[target]
        buy_base = OFFSETS["buy_trade"] + t_idx * _TRADE_STRIDE
        sell_base = OFFSETS["sell_trade"] + t_idx * _TRADE_STRIDE

        for square in PROPERTY_IDS:
            prop = env.properties[square]
            if prop.houses:
                continue
            idx = _PROP_INDEX[square]

            if prop.owner == target:
                value = self._buy_value(ctx, square, target)
                their_value = ctx.deed_value(square, target)
                for level, multiplier in enumerate(TRADE_CASH_LEVELS):
                    action = buy_base + idx * _N_CASH_LEVELS + level
                    if action not in aset:
                        continue
                    cost = int(_PRICE_OF[square] * multiplier)
                    edge = value - cost
                    if edge <= self.TRADE_OFFER_MARGIN:
                        continue
                    if cost - their_value <= self.TRADE_PEER_MARGIN:
                        continue
                    yield (edge, action)

            elif prop.owner == pid:
                if prop.is_monopoly:
                    continue
                if ctx.completes_for(square, target):
                    continue
                value = self._give_value(ctx, square)
                their_value = ctx.deed_value(square, target)
                for level, multiplier in enumerate(TRADE_CASH_LEVELS):
                    if multiplier < self.JUNK_SELL_MULT:
                        continue
                    action = sell_base + idx * _N_CASH_LEVELS + level
                    if action not in aset:
                        continue
                    proceeds = int(_PRICE_OF[square] * multiplier)
                    edge = proceeds - value
                    if edge <= self.TRADE_OFFER_MARGIN:
                        continue
                    if their_value - proceeds <= self.TRADE_PEER_MARGIN:
                        continue
                    yield (edge, action)

    def _buy_value(self, ctx, square: int, target: int) -> float:
        return ctx.deed_value(square, ctx.pid)


class _ScrapBuyMixin(_ScrapMixin):
    """`end_scrapbuy` -- the half that carried all of round 1's +8.3 pp on PANEL-MIXED.

    Near the cap a deed we can buy is worth `price * 2.5` and the cash is worth `price`,
    so: price the deed on its SCORE, offer exactly LIST (a lowball is the one price a
    rational counterparty must refuse), and drop the peer veto (the offer slot it is
    protecting has zero alternative use -- at round >= 150 the agent makes 0 purchases,
    wins 0 auctions, builds 0 houses and takes 0 rescue actions).

    ITS DOWNSIDE IS BOUNDED AT ZERO AND THAT IS WHY IT IS IN EVERY STACK HERE: on
    PANEL-STRONGISH all three opponents refuse cash-for-deed offers by construction, and
    round 1 measured the variant as IDENTICAL there -- 0 of 120 games differ.  It buys its
    gain from `gambler` specifically, whose accept test is `offer.net_worth() >= -50` over
    LIST PRICES, which a purchase at list satisfies exactly.
    """

    ENDGAME_BUY_MULT = 1.0

    def _buy_value(self, ctx, square: int, target: int) -> float:
        rent_value = ctx.deed_value(square, ctx.pid)
        w = self._endgame_weight(ctx.env)
        if w <= 0.0:
            return rent_value
        return max(rent_value, w * self._score_value(ctx, square, ctx.pid))

    def _cash_candidates(self, ctx, target: int, aset: set) -> Iterable[tuple]:
        w = self._endgame_weight(ctx.env)
        if w <= 0.0:
            yield from super()._cash_candidates(ctx, target, aset)
            return

        env, pid = ctx.env, ctx.pid
        t_idx = ctx.other_index[target]
        buy_base = OFFSETS["buy_trade"] + t_idx * _TRADE_STRIDE

        # The SELL side keeps end_scrap's rule verbatim; only the BUY side is rewritten.
        for cand in super()._cash_candidates(ctx, target, aset):
            action = cand[1]
            if not (buy_base <= action < buy_base + _TRADE_STRIDE):
                yield cand

        for square in PROPERTY_IDS:
            prop = env.properties[square]
            if prop.owner != target or prop.houses:
                continue
            idx = _PROP_INDEX[square]
            for level, multiplier in enumerate(TRADE_CASH_LEVELS):
                if multiplier != self.ENDGAME_BUY_MULT:
                    continue
                action = buy_base + idx * _N_CASH_LEVELS + level
                if action not in aset:
                    continue
                cost = int(_PRICE_OF[square] * multiplier)
                if env.players[pid].cash - cost < ctx.reserve * self.BUY_RESERVE_RELIEF:
                    continue
                edge = self._buy_value(ctx, square, target) - cost
                if edge <= self.TRADE_OFFER_MARGIN:
                    continue
                # NO PEER VETO.
                yield (edge, action)


# ==========================================================================
# PIECE 2 of 4 -- the war chest
# (verbatim from plans/r5b/variants/free1.py `_WarChestMixin`)
# ==========================================================================
class _WarChestMixin:
    """Turn deeds that can never become a monopoly into bidding cash, while there is still
    something on the board to bid for.  Round 1's best all-round piece: +4.2 strongish,
    +7.5 mixed, +4.2 duel over `build_relief0`, positive on all three panels."""

    CHEST_CLEARING_MULT = 1.0

    def _dead_deeds(self, ctx) -> list:
        env, pid = ctx.env, ctx.pid
        out = []
        for square in PROPERTY_IDS:
            prop = env.properties[square]
            if prop.owner != pid or prop.houses:
                continue
            colour = _COLOR_OF[square]
            if colour not in _REAL_GROUPS:
                continue          # a railroad is never dead; each one raises the set
            for sq in _GROUP_SQUARES[colour]:
                other = env.properties[sq].owner
                if other is not None and other != pid:
                    out.append(square)
                    break
        return out

    def _chest_target(self, ctx) -> float:
        env = ctx.env
        biggest = 0
        for square in PROPERTY_IDS:
            if env.properties[square].owner is None:
                price = _PRICE_OF[square]
                if price > biggest:
                    biggest = price
        if biggest <= 0:
            return 0.0
        return float(self.AUCTION_RESERVE + self.CHEST_CLEARING_MULT * biggest)

    def _war_chest(self, ctx, aset: set) -> Optional[int]:
        env, pid = ctx.env, ctx.pid
        if env.players[pid].cash >= self._chest_target(ctx):
            return None
        best = None
        for square in self._dead_deeds(ctx):
            if env.properties[square].mortgaged:
                continue
            action = OFFSETS["mortgage"] + _PROP_INDEX[square]
            if action not in aset:
                continue
            raised = _MORTGAGE_OF[square]
            key = (ctx.square_income(square) / max(raised, 1), -raised)
            if best is None or key < best[0]:
                best = (key, action)
        return None if best is None else best[1]

    def _mortgage_to_build(self, ctx, aset: set) -> Optional[int]:
        action = super()._mortgage_to_build(ctx, aset)
        if action is not None:
            return action
        return self._war_chest(ctx, aset)

    def _unmortgage(self, ctx, aset: set) -> Optional[int]:
        if ctx.unowned_deeds > 0:
            dead = self._dead_deeds(ctx)
            if dead:
                frozen = {OFFSETS["unmortgage"] + _PROP_INDEX[s] for s in dead}
                return super()._unmortgage(ctx, aset - frozen)
        return super()._unmortgage(ctx, aset)


# ==========================================================================
# PIECE 3 of 4 -- thaw a frozen build ladder
# (verbatim from plans/r5b/variants/cashuse.py `_ThawMixin`)
# ==========================================================================
class _ThawMixin:
    """Redeem a mortgaged deed inside a colour group we own OUTRIGHT, using any cash we
    actually have, instead of only cash above `reserve + UNMORTGAGE_BUFFER`.  One mortgage
    freezes the whole group's build ladder, so this redemption is not competing with the
    ladder for cash -- it IS the ladder.

    It ends with `super()._unmortgage(...)`, which is what makes it composable with the war
    chest: see the collision note at the top of this file.  The two rules select from
    disjoint deed sets (`is_development_group` requires we own the whole colour;
    `_dead_deeds` requires an opponent to own part of it), so chaining them loses nothing.
    """

    def _unmortgage(self, ctx, aset: set) -> Optional[int]:
        env, pid = ctx.env, ctx.pid
        cash = env.players[pid].cash
        best = None
        for square in PROPERTY_IDS:
            prop = env.properties[square]
            if prop.owner != pid or not prop.mortgaged:
                continue
            if not ctx.is_development_group(square):
                continue
            action = OFFSETS["unmortgage"] + _PROP_INDEX[square]
            if action not in aset:
                continue
            cost = int(_MORTGAGE_OF[square] * 1.1)
            if cash - cost < 0:
                continue
            if best is None or cost < best[0]:
                best = (cost, action)
        if best is not None:
            return best[1]
        return super()._unmortgage(ctx, aset)


# ==========================================================================
# PIECE 4 of 4 -- insure against the rent you can actually be charged
# (verbatim from plans/r5b/variants/cashuse.py `_ReachMixin`)
# ==========================================================================
class _ReachMixin:
    """`reserve = min(CAP, BASE + RENT_K * worst rent WITHIN ONE ROLL)`.  Same arithmetic,
    same constants; only WHICH rent it is computed against changes.

    Placed FIRST in every stack that uses it, so the rescoped reserve is in force for the
    whole `_unmortgage` chain below it.  It cannot fight the war chest or the thaw rule,
    because neither of those reads `ctx.reserve` -- it only relaxes the PARENT's gate at
    the end of the chain.
    """

    def _reach_reserve(self, ctx) -> float:
        env, pid = ctx.env, ctx.pid
        pos = env.players[pid].position
        worst = 0.0
        for step in _ONE_ROLL:
            prop = env.properties.get((pos + step) % 40)
            if prop is None or prop.owner is None or prop.owner == pid or prop.mortgaged:
                continue
            owner = env.players[prop.owner]
            rent = prop.get_rent(7, owner.railroads_owned(), owner.utilities_owned())
            if rent > worst:
                worst = rent
        return float(
            min(self.RESERVE_CAP, self.RESERVE_BASE + self.RESERVE_RENT_K * worst)
        )

    def _scoped(self, ctx, fn, *args):
        saved = ctx.reserve
        ctx.reserve = self._reach_reserve(ctx)
        try:
            return fn(*args)
        finally:
            ctx.reserve = saved

    def _should_buy(self, ctx, square):
        return self._scoped(ctx, super()._should_buy, ctx, square)

    def _build(self, ctx, aset):
        return self._scoped(ctx, super()._build, ctx, aset)

    def _build_shortfall(self, ctx):
        return self._scoped(ctx, super()._build_shortfall, ctx)

    def _unmortgage(self, ctx, aset):
        return self._scoped(ctx, super()._unmortgage, ctx, aset)

    def _jail(self, ctx, aset):
        return self._scoped(ctx, super()._jail, ctx, aset)

    def _accept_offer(self, ctx):
        return self._scoped(ctx, super()._accept_offer, ctx)


# ==========================================================================
# THE STACKS
# ==========================================================================
class StChest(_WarChestMixin, _Relief0Build):
    """THE CHAMPION.  `build_relief0` + the war chest, and nothing else.

    Identical in behaviour to the variant measured as `f1_chest_relief`: the same
    `_WarChestMixin` over the same `build_relief0` base.  Kept as its own class here so
    the shipped package exposes the thing the tournament actually crowned, rather than
    only the stack built on top of it.

    Measured, paired, N=360, base seed 20261211 (disjoint from the seed it was selected
    on), against the previous champion `spine_h100`:
        PANEL-STRONGISH  75.0%  vs 60.0%   +15.0 pp  z +6.64  BETTER
        PANEL-MIXED      68.3%  vs 52.2%   +16.1 pp  z +7.44  BETTER
        PANEL-STRONG     63.6%  vs 58.9%    +4.7 pp  z +2.75  BETTER
        PANEL-DUEL       45.0%  vs 33.3%   +11.7 pp  z +5.05  BETTER
    Better on every panel, no WORSE cell, 360/360 games, 0 clamps, 0 illegal actions,
    0 crashes.
    """

    name = "champion"


class StChestScrap(_WarChestMixin, _ScrapBuyMixin, _EndgameMixin, _Relief0Build):
    """PAIR: the war chest + the near-cap buy.  Round 1's two biggest increments.

    NO METHOD IS CLAIMED TWICE.  The war chest owns `_mortgage_to_build` / `_unmortgage`,
    the scrap-buy owns `_exchange_candidates` / `_cash_candidates` / the value hooks, and
    `_build` comes from the single shared `_Relief0Build`.

    And they cannot even see each other: `_chest_target` returns 0.0 once `unowned_deeds`
    is 0 (round ~25), `_endgame_weight` is 0.0 until round 100.  This is the stack whose
    increment SHOULD be the sum of the parts, and the panel where both parts are non-zero
    is PANEL-MIXED (+7.5 and +8.3 respectively).
    """

    name = "st_chest_scrap"


class StChestThaw(_ThawMixin, _WarChestMixin, _Relief0Build):
    """PAIR: the war chest + the thaw, i.e. the stack that contains THE ONE REAL COLLISION.

    Both pieces override `_unmortgage`; the chain is thaw -> war chest -> parent, and the
    top of this file argues why that loses nothing from either rule.  The near-cap buy is
    deliberately absent so this screen measures the collision and nothing else.
    """

    name = "st_chest_thaw"


class StCore(_ThawMixin, _WarChestMixin, _ScrapBuyMixin, _EndgameMixin, _Relief0Build):
    """TRIPLE: war chest + thaw + near-cap buy.  Everything that was non-negative on
    PANEL-MIXED in round 1, with the collision resolved."""

    name = "st_core"


class StAll(_ReachMixin, _ThawMixin, _WarChestMixin, _ScrapBuyMixin, _EndgameMixin,
            _Relief0Build):
    """THE FULL STACK: reach + thaw + war chest + near-cap buy, all on `build_relief0`.

    `cu_stack` (thaw + reach) was +1.7 on strongish and -1.7 on mixed, so the reach rule is
    the one piece here with a measured SIGN FLIP between panels.  That is why `st_core` and
    `st_all` are both screened on both panels: the difference between them is exactly the
    reach rule's contribution inside a stack, which is the interaction round 1 could not
    see.
    """

    name = "st_all"


class _ScoreBuyMixin:
    """Price an unowned deed at what it SCORES, not only at the rent it earns.

    Measured against the six competition agents (240 instrumented games,
    `docs/GAUNTLET.md`): of every chance to buy an unowned deed we took half
    and declined half, and *every* decline was for low value -- none for the
    cash floor. The median deed was valued at 0.91x list, so `_should_buy`'s
    final `value >= price` test rejected it.

    That test measures the wrong quantity. `Ctx.deed_value` is rent income over
    a horizon, but `Property.calculate_net_worth` -- which decides every capped
    game and underwrites every liquidation -- prices an unmortgaged deed at
    2.5x list, 5.0x inside a completed group, against cash at 1.0x. Buying at
    list converts $P of score into $2.5P whatever the rent does, so declining
    is a guaranteed loss of 1.5x the price in the only currency the engine
    ranks players by. `_score_value` already states that rule; the shipped
    agent consulted it only through `_endgame_weight`, which is zero until
    round 100, by which time the board is bought.

    The reserve floor is deliberately kept: cash is what stops you going
    bankrupt, and we go bankrupt in 68% of games.

    Paired, identical tables and seeds, against expo/aline/slayer:
        120 games  20.8% -> 25.0%  (+4.2 pp)
        240 games  25.4% -> 27.5%  (+2.1 pp)
    Never measured worse; not individually significant. Two variants on the
    same idea were measured and rejected: bidding auctions off the score value
    (-4.2 pp) and thinning the reserve floor as well (+0.0 pp).
    """

    def _should_buy(self, ctx, square: int) -> bool:
        env, pid = ctx.env, ctx.pid
        prop = env.properties.get(square)
        if prop is None or prop.owner is not None:
            return False
        price = _PRICE_OF[square]
        cash = env.players[pid].cash
        value = max(ctx.deed_value(square, pid),
                    self._score_value(ctx, square, pid))
        critical = value >= 1.5 * price
        floor = ctx.reserve * (self.BUY_RESERVE_RELIEF if critical else 1.0)
        if cash - price < floor:
            return False
        return value >= price


class StChestScrapScore(_ScoreBuyMixin, StChestScrap):
    """THE SUBMITTED AGENT: ChampionPlus plus score-priced buying."""

    name = "st_chest_scrap_score"


VARIANTS = {
    "champion": StChest,
    "st_chest_scrap_score": StChestScrapScore,
    "st_base": StBase,
    "st_chest_scrap": StChestScrap,
    "st_chest_thaw": StChestThaw,
    "st_core": StCore,
    "st_all": StAll,
}


# ==========================================================================
# STACK RESULTS
# 8 screens, N=120, base_seed 20260811, PYTHONHASHSEED=0, workers=1.
# ALL HEALTH-CLEAN: 0 clamps, 0 illegal actions, 0 crashes, 120/120 completed,
# p50 decision latency 57-77 us against the champion's ~76 us budget.
# ==========================================================================
#
# THE TABLE THE BRIEF ASKED FOR -- increments over `build_relief0`, paired on identical
# seeds via pairdiff.py.  Rows above the line are round 1's singles, re-read from their
# JSONs; rows below are this module's stacks.
#
#                        PANEL-STRONGISH                PANEL-MIXED
#                        rate    d_pp     z             rate    d_pp     z
#   build_relief0        64.2%     --     --            58.3%     --     --
#   f1_chest_relief      68.3%   +4.2  +1.15            65.8%   +7.5  +2.09  BETTER
#   end_scrapbuy         64.2%   +0.0  +0.00 IDENT      66.7%   +8.3  +3.29  BETTER
#   cu_thaw              65.0%   +0.8  +1.00            58.3%   +0.0  +0.00  IDENT
#   cu_reach             65.0%   +0.8  +1.00              --      --     --
#   ------------------------------------------------------------------------
#   st_chest_scrap       68.3%   +4.2  +1.15            72.5%  +14.2  +3.56  BETTER  <<<
#   st_chest_thaw        68.3%   +4.2  +1.15            65.8%   +7.5  +2.09  BETTER
#   st_core              68.3%   +4.2  +1.15            71.7%  +13.3  +3.26  BETTER
#   st_all               68.3%   +4.2  +1.15            71.7%  +13.3  +3.26  BETTER
#
# 1. THE TWO BIG PIECES ADD UP, AND THAT IS THE RESULT.
#    On PANEL-MIXED the war chest is +7.5 and the near-cap buy is +8.3; their sum is
#    +15.8 and the measured stack is +14.2.  The interaction is -1.6 pp -- two games out
#    of 120, far inside the ~2.6 pp paired standard error.  `st_chest_scrap` is 72.5% on
#    PANEL-MIXED against the champion's 50.8% and `build_relief0`'s 58.3%.
#    The mechanism was PREDICTED BEFORE THE SCREEN and then instrumented: the two rules
#    cannot see the same board.  Over 48 probe games the war chest fired in rounds 3-36
#    and NEVER LATER; the near-cap buy fired in rounds 101-199 and NEVER EARLIER.
#
# 2. THE COLLISION WAS NOT A COLLISION, AND THE THAW RULE IS A PASSENGER.
#    `st_chest_thaw` -- the stack built specifically to resolve the `_unmortgage` clash --
#    is BIT-IDENTICAL to `f1_chest_relief` on both panels: 0 of 120 games differ on
#    strongish, 0 of 120 on mixed.  The thaw branch still fires 2.3/game (strongish) and
#    4.8/game (mixed) inside the stack; it changes actions and changes no outcomes.
#    cu_thaw's own +0.8 pp -- its one flipped game -- is a game the war chest already
#    flips.  Two rules that both reach into the mortgage ledger were NOT additive; the
#    smaller one was absorbed whole.
#
# 3. CARRYING THE PASSENGERS COSTS A GAME.
#    st_core (chest+scrapbuy+thaw) and st_all (+reach) are both +13.3 on mixed against
#    st_chest_scrap's +14.2 -- one game worse, and identical to it on strongish.  Paired
#    directly: thaw GIVEN chest+scrapbuy is -0.8 pp on mixed (0 gained, 1 lost) and
#    IDENTICAL on strongish; reach GIVEN the other three is IDENTICAL on strongish and
#    1-gained-1-lost on mixed.  Nothing here is significant, but nothing is positive
#    either, and the anti-goal says a rule a network must imitate has to earn its place.
#    SHIP THE PAIR, NOT THE FULL STACK.
#
# 4. FIRING FREQUENCY IS NOT VALUE, MEASURED CLEANLY.
#    The near-cap buy fires 39.3 times per game on PANEL-STRONGISH and flips ZERO games;
#    it fires 16.4 times per game on PANEL-MIXED and is worth +6.7 pp there (paired
#    `st_chest_scrap` vs `f1_chest_relief`, z=+2.92).  The rule that fires 2.4x MORE OFTEN
#    is the one worth nothing.  Its downside stays bounded at zero exactly as round 1
#    claimed: on strongish `st_chest_scrap`'s win vector is identical to the war chest's.
#
# 5. WHAT WAS NOT DONE.  All eight screens are on base_seed 20260811, the seed round 1
#    selected on.  The +14.2 is NOT replicated on an independent seed and the screen
#    budget was spent.  The cheapest possible confirmation is ONE screen -- the paired bar
#    already exists on disk:
#        screen.py --variant st_chest_scrap --panel mixed --games 120 --base-seed 20260814
#        pairdiff.py --a st_chest_scrap --b build_relief0 --panel mixed --meas <replication>
#    Corroboration that does exist, from files this module did not produce: both
#    components replicate on other seeds (chest +6.7/+4.2/+6.7/+4.2 pp on seeds 20260814
#    and 20260911 across three panels; scrapbuy +15.8 pp vs spine_h100 on seed 20260812,
#    matching its 20260811 value to the game), and another agent's PANEL-STRONG screen of
#    `st_chest_scrap` reads 74.2%, +9.2 pp over build_relief0 at z=+3.16.
