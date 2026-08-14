"""Greedy 1-ply agent over our own evaluator. This is the TEACHER, not the ship.

Purpose: measure whether evaluator.evaluate() is any good, before investing in
search, weight tuning, or distillation. If a 1-ply agent using it cannot beat
the fixed personalities, nothing downstream will work either.

Two engine facts shape the implementation:

* env.step() consumes the GLOBAL random module (env.py:102 random.shuffle,
  env.py:615 random.randint). Simulating a candidate action therefore advances
  the real game's dice unless the RNG state is saved and restored. _simulate()
  does that; forgetting it silently corrupts every game.

* The legal action set can contain thousands of trade offers (measured: 5997
  offer candidates at a single decision). Evaluating all of them by simulation
  is hopeless, so candidates are shortlisted by family first.
"""

from __future__ import annotations

import copy
import random
import os
import sys
from pathlib import Path

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

def _find_repo() -> Path:
    """Locate DeepRL_Monopoly without hardcoding anyone's home directory.

    Order: DEEPRL_MONOPOLY_ROOT env var, already-importable, then walk up from
    this file looking for a directory containing monopoly_game_engine.
    """
    env = os.environ.get("DEEPRL_MONOPOLY_ROOT")
    if env and (Path(env) / "monopoly_game_engine").is_dir():
        return Path(env)
    try:
        import monopoly_game_engine as _m
        return Path(_m.__file__).resolve().parent.parent
    except Exception:
        pass
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "monopoly_game_engine").is_dir():
            return parent
        for sib in parent.iterdir() if parent.is_dir() else []:
            if sib.is_dir() and (sib / "monopoly_game_engine").is_dir():
                return sib
    raise RuntimeError(
        "Could not locate DeepRL_Monopoly. Set DEEPRL_MONOPOLY_ROOT, or place "
        "this directory next to the DeepRL_Monopoly checkout.")


ROOT = _find_repo()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluator as EV  # noqa: E402


def _own_value(env, pid):
    return EV.evaluate(env, pid)


def _asu_value(env, pid):   # not reachable: "asu" is rejected by use_value_fn
    """ASU's position evaluation, used inside OUR agent structure.

    Our agent loses to ASU 91-9, and the difference is not the structure --
    our explicit auction bidding (+21pp) and monopoly-completing trades (+20pp)
    are things ASU's 1-ply search is blind to. The difference is the VALUATION.
    ASU is also weak exactly where our explicit logic is strong: it scores 35.0%
    on `hard` against our 48.8%. So our shortlisting and special-case logic on
    top of ASU's evaluator is a genuinely different agent, not a copy, and has a
    real chance of beating both parents.
    """
    from ASU_FROZEN_TEACHER import evaluate_value
    return evaluate_value(env, pid).total


def _nw_value(env, pid):
    """Our own net-worth-native evaluator. Imports nothing from ASU."""
    import nw_eval
    return nw_eval.evaluate(env, pid)


_VALUE_FN = _own_value


def use_value_fn(name: str) -> None:
    """'own' | 'asu' | 'nw_v1'..'nw_v3' (variant selects the term set)."""
    global _VALUE_FN
    if name.startswith("nw"):
        import nw_eval
        parts = name.split("_")
        if len(parts) > 1:
            nw_eval.VARIANT = parts[1].upper()
        # nw_v4_lc0  -> LIQUIDITY_CREDIT 0.0 ; nw_v4_lc50 -> 0.50
        for tok in parts[2:]:
            if tok.startswith("lc"):
                nw_eval.LIQUIDITY_CREDIT = int(tok[2:]) / 100.0
        _VALUE_FN = _nw_value
        return
    if name == "asu":
        raise ValueError("the ASU evaluator may not be used in a submission")
    _VALUE_FN = {"own": _own_value}[name]
from monopoly_game_engine.actions import (  # noqa: E402
    AUCTION_ACTION_TO_INCREMENT, ActionType, OFFSETS,
)
from monopoly_game_engine.constants import (  # noqa: E402
    PROPERTY_IDS, REAL_ESTATE_IDS, TRADE_CASH_LEVELS,
)

N_PROP = len(PROPERTY_IDS)
N_RE = len(REAL_ESTATE_IDS)

TRADE_LO = OFFSETS["buy_trade"]
AUCTION_LO = OFFSETS["auction"]
MAX_TRADE_CANDIDATES = 12      # sampled, not exhaustive
MAX_CANDIDATES = 40            # hard cap on simulations per decision
# 60->40 measured at 3.9% decision divergence; 30 jumps to 13.4%.
USE_FASTENV = True
CASH_GATE = False   # UNTESTED -- off until measured
CASH_GATE_FLOOR = 50.0
CASH_GATE_MULT = 1.0
# MEASURED against the strongest rival: at round 30 we lead on deeds (7.5 vs
# 6.8), monopolies (3.4 vs 2.6) and net worth (7422 vs 5978) -- and by round 40
# our houses have gone BACKWARDS (4.8 -> 3.0) and the lead is gone. We run the
# whole mid-game on 217-296 cash against their 434-642, get hit by rent we
# cannot cover, and liquidate at 2-10 net worth destroyed per dollar raised.
# We were bankrupt in 80% of those games.
#
# The evaluator's W_RISK term cannot fix this: measured, a 10x change in it
# alters ZERO decisions. Survival is a constraint, not a number to add -- so
# actions that leave cash below the reserve are filtered out, and only taken
# when nothing else is available.
PROGRESSIVE_GROUP = False  # measured neutral: trades r2 for r3, same mean
GROUP_PROGRESS_FRAC = 0.20
OPEN_GROUP_FRAC = 0.0
# These multiply the FULLY DEVELOPED rent stream of a whole colour group, which
# is large relative to a bare deed. An earlier calibration (0.45 / 0.35) raised
# the value of the first deed of an empty group by ~1160%, pricing a
# low-probability option at close to its exercised value -- a reliable way to
# overbid into bankruptcy. Credit now scales with presence and starts at zero:
# owning none of a group earns nothing, and the discount is quadratic because
# completion becomes likelier only as deeds accumulate.


def _group_is_open(env, group) -> bool:
    """True when no single opponent already holds a majority of this group."""
    owners = {}
    for sq in group:
        o = env.properties[sq].owner
        if o is not None:
            owners[o] = owners.get(o, 0) + 1
    return not owners or max(owners.values()) < len(group) - 1

# Per-decision cost is candidates x (clone + step + evaluate), so the clone
# implementation and MAX_CANDIDATES both scale it linearly. This matters
# competitively: the slowest team in any game over 10 minutes is disqualified.


def _clone(env):
    from pickle import dumps, loads
    return loads(dumps(env, -1))


def _family(action: int) -> str:
    for name, start in sorted(OFFSETS.items(), key=lambda kv: kv[1], reverse=True):
        if action >= start:
            return name
    return "binary"


class HeuristicAgent:
    """choose_action(env) -> int, the interface monopoly_game_engine expects."""

    name = "heuristic"

    def __init__(self, player_id: int, rng: random.Random | None = None):
        self.player_id = player_id
        self.rng = rng or random.Random(0xC0FFEE + player_id)
        self.simulations = 0

    # -- candidate selection ------------------------------------------------
    def _shortlist(self, env, legal: list[int]) -> list[int]:
        """Trim a possibly-huge legal set to something we can simulate."""
        if len(legal) <= MAX_CANDIDATES:
            return legal
        trades, others = [], []
        for action in legal:
            (trades if TRADE_LO <= action < AUCTION_LO else others).append(action)
        picked = others[:MAX_CANDIDATES - MAX_TRADE_CANDIDATES]
        if trades:
            k = min(MAX_TRADE_CANDIDATES, len(trades), MAX_CANDIDATES - len(picked))
            picked.extend(self.rng.sample(trades, k))
        return picked or legal[:MAX_CANDIDATES]

    # -- one-step lookahead -------------------------------------------------
    def _simulate(self, env, action: int, want_state: bool = False):
        """Value of the position after `action`, with global RNG left untouched."""
        state = random.getstate()
        try:
            # pickle round-trip is ~2.6x faster than deepcopy and produces an
            # identical object (verified over 132 positions, 0 mismatches).
            # Cloning happens once per candidate, so this is the dominant cost.
            future = _clone(env) if USE_FASTENV else copy.deepcopy(env)
            future.step(action)
            self.simulations += 1
            value = _VALUE_FN(future, self.player_id)
            return (value, future) if want_state else value
        except Exception:
            return None
        finally:
            random.setstate(state)

    # -- multi-step actions that 1-ply cannot see ---------------------------
    def _property_worth(self, env, square: int) -> float:
        """What winning this deed is worth to us, in cash terms.

        1-ply lookahead values an auction bid at ~0 because the deed is not
        transferred until the auction resolves several steps later. Same for
        trade offers. Those actions therefore need an explicit valuation or the
        agent passes on every one -- measured: 5 auctions entered, 5 passed,
        zero monopolies formed, zero houses built, 4.2% win rate.
        """
        prop = env.properties.get(square)
        if prop is None:
            return 0.0
        rivals = sum(1 for i, p in enumerate(env.players)
                     if i != self.player_id and not p.bankrupt)
        worth = EV.W_INCOME * EV.B.expected_rent(square, 0, False) * max(rivals, 1)

        colour = getattr(prop, "color", None)
        group = EV.B.SETS.get(colour)
        if group:
            n = len(group)
            mine = sum(1 for s in group
                       if env.properties[s].owner == self.player_id)
            developed = EV.B.expected_rent
            full = sum(developed(s, EV.BUILD_TARGET, True) for s in group)

            if mine == n - 1:
                # this deed completes the set: the whole developed stream unlocks
                worth += EV.W_POTENT * full * max(rivals, 1)
            elif PROGRESSIVE_GROUP and mine > 0:
                # PARTIAL group presence is worth paying for too. Valuing only
                # the completing deed means we pay ordinary price for the first
                # and second of a colour group and only contest the third -- by
                # which point an opponent already holds the blocking position.
                # Credit the fraction of the developed stream this deed brings
                # us toward, discounted because completion is not yet certain.
                frac = (mine / (n - 1)) ** 2        # convex: late deeds matter most
                worth += EV.W_POTENT * full * max(rivals, 1) * GROUP_PROGRESS_FRAC * frac
            elif PROGRESSIVE_GROUP and mine == 0 and _group_is_open(env, group):
                # first deed of an unclaimed group: cheap option on the set
                worth += (EV.W_POTENT * full * max(rivals, 1)
                          * GROUP_PROGRESS_FRAC * OPEN_GROUP_FRAC)
        return worth

    def _auction_action(self, env, legal: list[int]) -> int | None:
        """Bid while the price is below what the deed is worth to us."""
        bids = [a for a in legal if AUCTION_LO < a < AUCTION_LO + 5]
        if not bids:
            return None
        square = getattr(env, "auction_property_id", None)
        if square is None:
            return None
        worth = self._property_worth(env, square)
        high = float(getattr(env, "auction_high_bid", 0) or 0)
        cash = env.players[self.player_id].cash

        # An ABSOLUTE reserve becomes unsatisfiable once cash dips: measured vs
        # 3x ASU, cash falls to ~$190 by step 150 while CASH_FLOOR is 279, so
        # `cash - CASH_FLOOR` goes negative and the agent is structurally barred
        # from bidding -- it passed 5 of every 7 auctions and handed ASU the
        # board. A PROPORTIONAL reserve shrinks with the bankroll instead.
        frac = getattr(EV, "AUCTION_RESERVE_FRAC", None)
        reserve = EV.CASH_FLOOR if frac is None else max(50.0, frac * cash)
        headroom = min(worth, cash - reserve) - high
        if headroom <= 0:
            return None                       # pass; caller falls through
        # take the largest increment we can still justify
        for action, step in sorted(AUCTION_ACTION_TO_INCREMENT.items(),
                                   key=lambda kv: -kv[1]):
            if int(action) in bids and step <= headroom:
                return int(action)
        return None

    def _completing_trade(self, env, legal: list[int]) -> int | None:
        """Offer cash for the exact deed that completes one of our colour sets.

        Monopolies are the whole game: with one granted, this agent builds to
        hotels and wins outright; without one it never builds and scores ~25%.
        Landing on the missing deed is luck, so it has to be bought.

        1-ply cannot find this. A trade OFFER changes nothing until the other
        side accepts, so simulating it shows ~0 delta and the offer is never
        made. Worse, the offer sits in a 2,268-wide exch_trade block plus a
        252-wide buy_trade block, so random sampling never lands on the one
        offer that matters. Hence explicit construction.

        Encoding (env.py:840 _make_trade_offer):
            others     = [i for i in range(4) if i != pid]      # ascending
            local      = player_idx*(28*3) + prop_idx*3 + price_idx
            cash asked = int(price * TRADE_CASH_LEVELS[price_idx])
        """
        me = env.players[self.player_id]
        others = [i for i in range(len(env.players)) if i != self.player_id]
        best = None
        for colour, group in EV.B.SETS.items():
            owned = [s for s in group if env.properties[s].owner == self.player_id]
            if len(owned) != len(group) - 1:
                continue
            missing = next(s for s in group if s not in owned)
            prop = env.properties[missing]
            if prop.owner is None or prop.owner == self.player_id:
                continue
            if prop.houses > 0:            # engine refuses developed deeds
                continue
            try:
                player_idx = others.index(prop.owner)
                prop_idx = PROPERTY_IDS.index(missing)
            except ValueError:
                continue
            # most generous price level we can actually pay
            for price_idx in (2, 1, 0):
                cost = int(prop.price * TRADE_CASH_LEVELS[price_idx])
                if me.cash - cost < EV.CASH_FLOOR:
                    continue
                action = (OFFSETS["buy_trade"]
                          + player_idx * (len(PROPERTY_IDS) * len(TRADE_CASH_LEVELS))
                          + prop_idx * len(TRADE_CASH_LEVELS)
                          + price_idx)
                if action in legal:
                    quality = EV.B.set_quality().get(colour, 0.0)
                    if best is None or quality > best[0]:
                        best = (quality, action)
                    break
        return best[1] if best else None

    def _cash_ok(self, future) -> bool:
        """Does this future leave enough CASH to cover the worst next rent?

        Cash only. Counting mortgage capacity here makes us look solvent in two
        thirds of the positions where we are actually short (measured: a
        liquidity-inclusive test fires on 10.2% of positions, cash-only on
        30.2%), and liquidating is what destroys the position in the first place.
        """
        if not CASH_GATE:
            return True
        try:
            import nw_eval
            cash = float(future.players[self.player_id].cash)
            need = CASH_GATE_FLOOR + CASH_GATE_MULT * nw_eval.worst_reachable_rent(
                future, self.player_id)
            return cash >= need
        except Exception:
            return True

    def _survives(self, future) -> bool:
        """Hard safety gate: can we still cover a plausible rent after this?

        ASU filters unsafe actions out of consideration entirely; our W_RISK
        term merely subtracts from a score, so a big enough upside can still
        buy a bankrupting move. This makes the constraint binding instead.
        """
        # Legacy multiplier gate, kept switchable for A/B.
        mult = getattr(EV, "HARD_GATE", None)
        if mult is not None:
            try:
                return (EV.liquid_assets(future, self.player_id)
                        >= mult * EV.exposure_per_round(future, self.player_id))
            except Exception:
                return True
        # Solvency gate. The old test used liquid_assets, which counts mortgage
        # values and therefore almost never looked unsafe -- measured 0/95
        # divergences, i.e. completely inert. This asks the question that
        # actually matters: after this action, can we pay the WORST rent
        # reachable on the next roll?
        if not getattr(EV, "SOLVENCY_GATE", True):
            return True
        try:
            return EV.is_solvent(future, self.player_id)
        except Exception:
            return True

    # -- policy -------------------------------------------------------------
    def choose_action(self, env) -> int:
        legal = list(env.get_allowed_actions(self.player_id))
        if not legal:
            return int(ActionType.END_TURN)
        if len(legal) == 1:
            return legal[0]

        bid = self._auction_action(env, legal)
        if bid is not None:
            return bid

        trade = self._completing_trade(env, legal)
        if trade is not None:
            return trade

        best_action, best_value = legal[0], float("-inf")
        safe_action, safe_value = None, float("-inf")
        for action in self._shortlist(env, legal):
            out = self._simulate(env, action, want_state=True)
            if out is None:
                continue
            value, future = out
            if value > best_value:
                best_action, best_value = action, value
            if (future is not None and value > safe_value
                    and self._cash_ok(future) and self._survives(future)):
                safe_action, safe_value = action, value
        # prefer the best SAFE action; fall back only if nothing is safe
        if safe_action is not None:
            best_action = safe_action

        # never return something the engine would reject
        return best_action if best_action in legal else legal[0]
