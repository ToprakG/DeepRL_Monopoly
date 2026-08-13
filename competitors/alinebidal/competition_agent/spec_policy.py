"""Phase 2 — a policy built from SPEC.md, branch by branch.

Structure
---------
A **priority-ordered rule pipeline**, not a reconstruction of the teacher's
architecture. Rules are consulted in order; the first that returns an action
wins. Each rule is small, names the SPEC ids it implements, and is responsible
for exactly one decision family. The ordering encodes what the probes showed
about which considerations override which.

Every branch cites the SPEC rule that justifies it. Nothing here was derived
from the teacher's source or from `decide()`'s internals.

    forced        - only one legal action
    debt          - forced liquidation ordering            F1-F5
    auction       - ceiling + safety                       B1-B5, D1-D4
    jail          - post-roll exit choice                  G1-G5
    buy           - the buy gate                           A1-A6, D1
    trade_reply   - accept/decline an incoming offer        H1-H4
    build         - development order + safety             E1-E5, D1-D4
    unmortgage    - restore a mortgaged deed               D1-D4
    default       - END_TURN / DO_NOTHING
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from monopoly_game_engine.actions import (
    OFFSETS, ActionType, AuctionAction, action_to_description,
)
from monopoly_game_engine.constants import (
    AUCTION_BID_INCREMENTS, COLOR_GROUPS, JAIL_BAIL, PROPERTIES, PROPERTY_IDS,
    REAL_ESTATE_IDS,
)

from competition_agent.spec_model import (
    MIN_CASH, SHORT_TURNS, auction_ceiling, deed_value, expected_rent_flow,
    gates_ok, liquidatable_worth, marginal_monopoly_value,
    multi_turn_landings, rent_for, worst_reachable_rent,
)

# Fitted on 1,520 harvested real-play proposals, validated on 988 held out by
# game seed: top-1 29.86% [27.09, 32.79] against 1.54% random. See D2.9.
TRADE_W = {
    "d_rent": 0.017832242423844513,
    "d_price": -0.015475111970890465,
    "completes": 3.4696527554465098,
    "d_ours": 0.5053699404996843,
    "off_mort": -0.6594870903815228,
    "d_houses": -3.5409761941310096
}
TRADE_GATE = float(os.environ.get("TRADE_GATE", "3.92"))

# Optional override, for A/B-ing a re-fitted scorer against the shipped one
# without editing the frozen agent. Points at a JSON file holding
# {"weights": {feature: weight, ...}, "gate": <float>}; any feature named in
# `EXT_FEATURES` may appear and anything omitted weighs zero.
#
# When it is unset the ORIGINAL six-term expression runs verbatim. That
# duplication is deliberate: computing the general form always would give the
# same value only up to floating-point summation order, and an exact tie
# resolved differently would move the frozen agent's argmax. The frozen path
# stays byte-for-byte what it was.
# Candidate D: a trained ranker replacing the linear scorer for the proposal
# choice only. Everything else in the policy is untouched, so a win-rate A/B
# isolates the ranking. Set TRADE_RANKER to a JSON holding {"ckpt": path,
# "gate": float}; the gate must be calibrated on the ranker's own score scale
# (calibrate_rank_gate.py), because a threshold is meaningless across scales.
TRADE_RANKER_PATH = os.environ.get("TRADE_RANKER", "")
TRADE_RANKER = None
TRADE_RANKER_GATE = None

TRADE_WEIGHTS_PATH = os.environ.get("TRADE_WEIGHTS", "")
TRADE_W_EXT = None
TRADE_GATE_EXT = None
if TRADE_WEIGHTS_PATH:
    import json as _json
    _blob = _json.loads(Path(TRADE_WEIGHTS_PATH).read_text())
    TRADE_W_EXT = dict(_blob["weights"])
    TRADE_GATE_EXT = float(_blob.get("gate", TRADE_GATE))

# Order and units must match `fit_trade_v3.features()` exactly, or a weight
# vector fitted there means something different here.
EXT_FEATURES = (
    "d_rent", "d_price", "completes", "d_ours", "off_mort", "d_houses",
    "off_price", "off_rent", "off_breaks_ours", "req_theirs",
    "off_group_size", "mutual_swap", "req_price", "req_mort",
    "off_completes_theirs", "req_base_rent",
)


HERE = Path(__file__).resolve().parent
TRADE_RANKER_FAILED = False


def _resolve(p) -> Path:
    """Accept an absolute path, but fall back to this package directory.

    The config files were written with absolute paths from the machine that
    trained the model. A tournament entry that only runs from one checkout is
    not an entry, so a stale absolute path degrades to a lookup by name next
    to this file rather than an error.
    """
    q = Path(p)
    if q.is_absolute() and q.exists():
        return q
    for cand in (HERE / q, HERE / q.name, HERE / "probes" / q.name):
        if cand.exists():
            return cand
    return q


def _load_ranker():
    """Deferred: torch is not imported unless a ranker is actually requested,
    and pool workers each pay the load once rather than inheriting a tensor.

    Returns None on any failure, having set `TRADE_RANKER_FAILED` so the
    attempt is made once. The caller then uses the linear scorer. A missing
    checkpoint or an unavailable torch must cost the trade branch, not the
    whole agent: `FinalAgent` catches exceptions by returning the first legal
    action, which would be catastrophic applied to every decision in a game.
    """
    global TRADE_RANKER, TRADE_RANKER_GATE, TRADE_RANKER_FAILED
    if TRADE_RANKER is not None or TRADE_RANKER_FAILED:
        return TRADE_RANKER
    if not TRADE_RANKER_PATH:
        TRADE_RANKER_FAILED = True
        return None
    try:
        import json as _j

        import torch

        from competition_agent.train_rank import RankHead
        blob = _j.loads(_resolve(TRADE_RANKER_PATH).read_text())
        ck = torch.load(_resolve(blob["ckpt"]), map_location="cpu",
                        weights_only=False)
        m = RankHead(ck.get("hidden", 256), ck.get("dropout", 0.2))
        m.load_state_dict(ck["state_dict"])
        m.eval()
        torch.set_num_threads(1)
        TRADE_RANKER = m
        TRADE_RANKER_GATE = float(blob["gate"])
        return m
    except Exception:                                      # noqa: BLE001
        TRADE_RANKER_FAILED = True
        return None


def rank_features(fr, fo) -> list:
    """`train_rank.cand_features` in the policy's own units.

    Kept byte-identical to the training-time function, including the divisors.
    A mismatch here would not raise; it would just feed the network a different
    input than it was trained on and look like a weak model.
    """
    return [
        fr["price"] / 400.0, fo["price"] / 400.0,
        fr["rent"] / 100.0, fo["rent"] / 100.0,
        fr["ours"] / 3.0, fo["ours"] / 3.0,
        fr["theirs"] / 3.0, fo["theirs"] / 3.0,
        1.0 if fr["ours"] == fr["size"] - 1 else 0.0,
        1.0 if fo["mort"] else 0.0,
        1.0 if fr["mort"] else 0.0,
        fr["houses"] / 5.0, fo["houses"] / 5.0,
        fr["base_rent"] / 50.0,
    ]


def ext_features(fr, fo) -> dict:
    """The extended candidate features, from the same per-deed facts the
    shipped scorer already builds (plus `theirs` and `base_rent`)."""
    completes = 1.0 if fr["ours"] == fr["size"] - 1 else 0.0
    off_completes = 1.0 if fo["theirs"] == fo["size"] - 1 else 0.0
    return {
        "d_rent": fr["rent"] - fo["rent"],
        "d_price": (fr["price"] - fo["price"]) / 100.0,
        "completes": completes,
        "d_ours": float(fr["ours"] - fo["ours"]),
        "off_mort": 1.0 if fo["mort"] else 0.0,
        "d_houses": float(fr["houses"] - fo["houses"]),
        "off_price": fo["price"] / 100.0,
        "off_rent": fo["rent"] / 10.0,
        "off_breaks_ours": 1.0 if fo["ours"] >= 2 else 0.0,
        "req_theirs": float(fr["theirs"]),
        "off_group_size": float(fo["size"]),
        "mutual_swap": completes * off_completes,
        "req_price": fr["price"] / 100.0,
        "req_mort": 1.0 if fr["mort"] else 0.0,
        "off_completes_theirs": off_completes,
        "req_base_rent": fr["base_rent"] / 10.0,
    }

DO_NOTHING = int(ActionType.DO_NOTHING)
END_TURN = int(ActionType.END_TURN)
ROLL_DICE = int(ActionType.ROLL_DICE)
BUY = int(ActionType.BUY_PROPERTY)
USE_GOOJ = int(ActionType.USE_GOOJ_CARD)
PAY_BAIL = int(ActionType.PAY_BAIL)
DECLARE_BANKRUPT = int(ActionType.DECLARE_BANKRUPT)
ACCEPT_TRADE = int(ActionType.ACCEPT_TRADE)
DECLINE_TRADE = int(ActionType.DECLINE_TRADE)
AUCTION_PASS = int(AuctionAction.PASS)


def _sq_of(action: int, family: str, table: List[int]) -> int:
    return table[action - OFFSETS[family]]


class SpecPolicy:
    """A policy whose every branch cites a SPEC.md rule."""

    policy_id = "spec_policy_v1"

    def __init__(self, player_id: int, rng_seed: int = 0):
        self.player_id = player_id

    # ------------------------------------------------------------------
    def choose_action(self, env) -> int:
        pid = self.player_id
        legal = [int(a) for a in env.get_allowed_actions(pid)]
        if len(legal) == 1:
            return legal[0]                       # forced; nothing to decide
        allowed = set(legal)

        for rule in (self._debt, self._auction, self._jail, self._buy,
                     self._trade_reply, self._propose_trade, self._build,
                     self._unmortgage):
            action = rule(env, pid, allowed, legal)
            if action is not None and action in allowed:
                return action

        # SPEC G1 — in pre_roll, END_TURN advances the phase rather than
        # declining anything, so it is the correct default, not a passive one.
        if END_TURN in allowed:
            return END_TURN
        return legal[0]

    # ------------------------------------------------------------------
    # F1-F5 — forced liquidation under debt
    # ------------------------------------------------------------------
    def _debt(self, env, pid, allowed, legal) -> Optional[int]:
        if getattr(env, "debt_player", None) != pid:
            return None

        # F5 — bankruptcy only when nothing remains. The engine already
        # reduces the menu to DECLARE_BANKRUPT in that case.
        if allowed == {DECLARE_BANKRUPT}:
            return DECLARE_BANKRUPT

        # F3 — cheapest asset first, by mortgage value. F2 — mortgages rank
        # ahead of house sales where both are legal, which is partly the
        # engine's doing (a deed carrying houses cannot be mortgaged) and
        # partly preference; ordering by raised-cash reproduces both.
        candidates = []
        for a in legal:
            if OFFSETS["mortgage"] <= a < OFFSETS["unmortgage"]:
                sq = _sq_of(a, "mortgage", PROPERTY_IDS)
                candidates.append((env.properties[sq].mortgage_v, 0, a))
            elif OFFSETS["sell_prop"] <= a < OFFSETS["buy_trade"]:
                sq = _sq_of(a, "sell_prop", PROPERTY_IDS)
                candidates.append((env.properties[sq].mortgage_v, 2, a))
            elif OFFSETS["sell_house"] <= a < OFFSETS["sell_hotel"]:
                sq = _sq_of(a, "sell_house", REAL_ESTATE_IDS)
                # F4 — strip one deed at a time; prefer the deed already
                # being stripped, i.e. the least developed among those legal.
                hp = PROPERTIES[sq]["house_price"] // 2
                candidates.append((hp, 1, a))
            elif OFFSETS["sell_hotel"] <= a < OFFSETS["sell_prop"]:
                sq = _sq_of(a, "sell_hotel", REAL_ESTATE_IDS)
                hp = PROPERTIES[sq]["house_price"] // 2
                candidates.append((hp, 1, a))
        if not candidates:
            return DECLARE_BANKRUPT if DECLARE_BANKRUPT in allowed else None
        candidates.sort(key=lambda t: (t[1], t[0], t[2]))
        return candidates[0][2]

    # ------------------------------------------------------------------
    # B1-B5 + D1-D4 — auctions
    # ------------------------------------------------------------------
    def _auction(self, env, pid, allowed, legal) -> Optional[int]:
        if env.phase != "auction" or env.auction_current_pid != pid:
            return None
        sq = env.auction_property_id
        if sq is None:
            return None

        ceiling = auction_ceiling(env, pid, sq)          # B1
        high = env.auction_high_bid

        # B2 — the value teacher takes the LARGEST legal increment whose total
        # stays within the ceiling and passes safety (82/82 opening bids were
        # the maximum increment).
        best = None
        for i, inc in enumerate(AUCTION_BID_INCREMENTS):
            action = AUCTION_PASS + 1 + i
            if action not in allowed:
                continue
            total = high + inc
            if total > ceiling:
                continue
            if not gates_ok(env, pid, total, liq_delta=0):   # D1-D4
                continue
            best = action                                # keep the largest
        return best if best is not None else AUCTION_PASS

    # ------------------------------------------------------------------
    # G1-G5 — jail
    # ------------------------------------------------------------------
    def _jail(self, env, pid, allowed, legal) -> Optional[int]:
        me = env.players[pid]
        if not me.in_jail:
            return None

        # G1 (revised) — in pre_roll the jail *exit* choice is deferred, but
        # that is all that is deferred. The original reading of p07 was
        # "END_TURN in pre_roll while jailed", which p07's setup supported
        # because nothing else was worth doing there. The debt/jail evaluation
        # set refutes it: over 250 jailed pre_roll states the teacher chose
        # END_TURN in only 106, and spent the rest unmortgaging (63) and
        # proposing trades (95). Returning END_TURN here short-circuited every
        # later rule and cost unmortgage 0/63.
        #
        # Correct behaviour: being in jail suppresses no other rule. Fall
        # through, and let END_TURN be reached as the pipeline default.
        if env.phase == "pre_roll":
            return None
        if env.phase != "post_roll" or env.has_rolled:
            return None

        danger = worst_reachable_rent(env, pid)

        # G2 — a free exit is taken readily (63% overall). G5 — but doubles are
        # free too, so the first jail turn is usually spent rolling.
        if USE_GOOJ in allowed and me.jail_turns >= 1:
            return USE_GOOJ

        # G3 — bail is discretionary spending and obeys the $200 floor.
        # G4 — the more rent is waiting outside, the less willing to leave.
        # G5 — buy out later rather than sooner, and more readily when rich.
        if PAY_BAIL in allowed and me.jail_turns >= 2:
            if gates_ok(env, pid, JAIL_BAIL) and danger < me.cash:
                return PAY_BAIL

        return ROLL_DICE if ROLL_DICE in allowed else None

    # ------------------------------------------------------------------
    # A1-A6 + D1 — the buy decision
    # ------------------------------------------------------------------
    def _buy(self, env, pid, allowed, legal) -> Optional[int]:
        if BUY not in allowed:
            return None
        sq = env.players[pid].position
        prop = env.properties.get(sq)
        if prop is None or prop.owner is not None:
            return None

        # A3 — buy iff (cash - price) + E[next-round net rent] >= 200, with the
        # rent term from A4/A5's complete-turn enumeration over real positions.
        # A2 — the gate is on cash AFTER the purchase.
        if gates_ok(env, pid, prop.price, liq_delta=0):
            return BUY
        return None

    # ------------------------------------------------------------------
    # H1-H4 — replying to an incoming trade
    # ------------------------------------------------------------------
    def _trade_reply(self, env, pid, allowed, legal) -> Optional[int]:
        if ACCEPT_TRADE not in allowed:
            return None
        offer = env._incoming_trade(pid)
        if offer is None:
            return None

        gain = 0.0
        if offer.offered_prop is not None:
            gain += deed_value(env, pid, offer.offered_prop.square_id)
        if offer.requested_prop is not None:
            # H4 — no sweetener buys a deed out of a near-monopoly.
            gain -= deed_value(env, pid, offer.requested_prop.square_id)
        gain += offer.cash_offered - offer.cash_requested

        # H2 — the counterparty must be able to fund it: the teacher refuses
        # offers that are generous but unaffordable to the proposer.
        proposer = env.players[offer.from_player]
        if offer.cash_offered > 0:
            if not gates_ok(env, offer.from_player, offer.cash_offered):
                return DECLINE_TRADE

        # our own safety on any cash we pay
        if offer.cash_requested > 0 and not gates_ok(env, pid,
                                                     offer.cash_requested):
            return DECLINE_TRADE

        return ACCEPT_TRADE if gain > 0 else DECLINE_TRADE

    # ------------------------------------------------------------------
    # I1-I5 — proposing a trade
    # ------------------------------------------------------------------
    def _propose_trade(self, env, pid, allowed, legal) -> Optional[int]:
        """Offer a spare deed for the piece that completes one of our groups.

        I1 — every proposal observed was an `exch_trade` (36/36 in p09);
        `buy_trade` and `sell_trade` were never chosen there.
        I5 — cash is irrelevant to the choice (identical at $300 and $2,500).

        I2 (revised) — the first version of this rule assumed the requested
        deed is always the piece completing one of our groups, which p09's
        narrow setup supported. Held-out play refuted it: over 281 real
        proposals the teacher requested 23, 25, 37, 12, 9, 31, 27, 35 … and
        *offered* valuable deeds (13, 24, 9, 21), not spares. Agreement on the
        requested deed was 27/189. It is running a general search over
        exchange pairs, not a completion heuristic.

        I3 (revised) — implemented as that search: score every legal exchange
        as a counterfactual transfer, keep the pairs that are positive for us
        and non-negative for the counterparty, and take the best. Deed values
        come from the same A4/A6 projection used everywhere else, so a deed is
        worth more when opponents are likelier to land on it.
        """
        n = len(PROPERTY_IDS)
        others = [i for i in range(len(env.players)) if i != pid]

        exch_actions = [a for a in legal
                        if OFFSETS["exch_trade"] <= a < OFFSETS["auction"]]
        if not exch_actions:
            return None

        # I3 (revised again) — scored by the features that were MEASURED to
        # carry signal on 2,508 real proposals (DECISIONS D2.9), not by a
        # difference of deed values. The monopoly term is deliberately absent:
        # D2.6 showed it decides the ordering by itself and decides it wrong.
        facts = {}

        def f(sq):
            if sq not in facts:
                prop = env.properties[sq]
                group = COLOR_GROUPS[PROPERTIES[sq]["color"]]
                saved = prop.owner
                prop.owner = pid
                try:
                    rent = 0.0
                    for opp in env.players:
                        if opp.player_id == pid or opp.bankrupt:
                            continue
                        for land, pr in multi_turn_landings(opp.position,
                                                            SHORT_TURNS):
                            if land == sq:
                                rent += pr * rent_for(env, sq)
                finally:
                    prop.owner = saved
                facts[sq] = {
                    "price": PROPERTIES[sq]["price"],
                    "rent": rent,
                    "ours": sum(1 for t in group
                                if env.properties[t].owner == pid),
                    "size": len(group),
                    "mort": prop.mortgaged,
                    "houses": prop.houses,
                }
                if TRADE_W_EXT is not None or TRADE_RANKER_PATH:
                    # Only the override path needs these, and each costs a
                    # scan of the group, so they are not paid for by the
                    # frozen agent.
                    facts[sq]["theirs"] = sum(
                        1 for t in group
                        if env.properties[t].owner not in (None, pid))
                    facts[sq]["base_rent"] = PROPERTIES[sq]["rent"][0]
            return facts[sq]

        # Candidate D: one batched forward over the whole candidate set,
        # which is the operation the listwise loss trained.
        if TRADE_RANKER_PATH and not TRADE_RANKER_FAILED:
            model = _load_ranker()
        else:
            model = None
        if model is not None:
            import numpy as _np
            import torch as _t
            obs = _np.asarray(env._get_state(pid), dtype=_np.float32)
            rows, acts = [], []
            for a in exch_actions:
                loc = a - OFFSETS["exch_trade"]
                p_idx = loc // (n * (n - 1))
                rem = loc % (n * (n - 1))
                off_idx = rem // (n - 1)
                req_raw = rem % (n - 1)
                req_idx = req_raw if req_raw < off_idx else req_raw + 1
                if p_idx >= len(others):
                    continue
                fo, fr = f(PROPERTY_IDS[off_idx]), f(PROPERTY_IDS[req_idx])
                rows.append(rank_features(fr, fo))
                acts.append(a)
            if not rows:
                return None
            cf = _np.asarray(rows, dtype=_np.float32)
            x = _np.concatenate(
                [_np.repeat(obs[None, :], len(cf), 0), cf], axis=1)
            with _t.no_grad():
                sc = model(_t.from_numpy(x)).numpy()
            k = int(sc.argmax())
            return acts[k] if float(sc[k]) >= TRADE_RANKER_GATE else None

        best_action, best_score = None, None
        for a in exch_actions:
            loc = a - OFFSETS["exch_trade"]
            p_idx = loc // (n * (n - 1))
            rem = loc % (n * (n - 1))
            off_idx = rem // (n - 1)
            req_raw = rem % (n - 1)
            req_idx = req_raw if req_raw < off_idx else req_raw + 1
            if p_idx >= len(others):
                continue
            fo, fr = f(PROPERTY_IDS[off_idx]), f(PROPERTY_IDS[req_idx])
            if TRADE_W_EXT is not None:
                ft = ext_features(fr, fo)
                sc = sum(TRADE_W_EXT.get(k, 0.0) * ft[k] for k in EXT_FEATURES)
                if best_score is None or sc > best_score:
                    best_action, best_score = a, sc
                continue
            sc = (TRADE_W["d_rent"] * (fr["rent"] - fo["rent"])
                  + TRADE_W["d_price"] * ((fr["price"] - fo["price"]) / 100.0)
                  + TRADE_W["completes"] * (1.0 if fr["ours"] == fr["size"] - 1
                                            else 0.0)
                  + TRADE_W["d_ours"] * (fr["ours"] - fo["ours"])
                  + TRADE_W["off_mort"] * (1.0 if fo["mort"] else 0.0)
                  + TRADE_W["d_houses"] * (fr["houses"] - fo["houses"]))
            if best_score is None or sc > best_score:
                best_action, best_score = a, sc

        # I6 — the propose/don't gate. The teacher proposed in 2,508 of 6,032
        # states where an exchange was legal (41.6%), so firing on every
        # positive score is wrong; TRADE_GATE is the score floor.
        gate = TRADE_GATE if TRADE_GATE_EXT is None else TRADE_GATE_EXT
        if best_score is None or best_score < gate:
            return None
        return best_action

    # ------------------------------------------------------------------
    # E1-E5 + D1-D4 — development
    # ------------------------------------------------------------------
    def _build(self, env, pid, allowed, legal) -> Optional[int]:
        cands = []
        for a in legal:
            if OFFSETS["improve_house"] <= a < OFFSETS["improve_hotel"]:
                sq = _sq_of(a, "improve_house", REAL_ESTATE_IDS)
            elif OFFSETS["improve_hotel"] <= a < OFFSETS["sell_house"]:
                sq = _sq_of(a, "improve_hotel", REAL_ESTATE_IDS)
            else:
                continue
            cands.append((sq, a))
        if not cands:
            return None

        # E5 — building is gated by the same cushion as every other spend.
        cost = PROPERTIES[cands[0][0]]["house_price"]
        if not gates_ok(env, pid, cost, liq_delta=0):
            return None

        # E1 — highest base RENT first, not highest price (brown picks Baltic
        # over Mediterranean at equal $60). E2 — ties break on lower square id.
        cands.sort(key=lambda t: (-PROPERTIES[t[0]]["rent"][0], t[0], t[1]))
        return cands[0][1]

    # ------------------------------------------------------------------
    # D1-D4 — restoring a mortgaged deed
    # ------------------------------------------------------------------
    def _unmortgage(self, env, pid, allowed, legal) -> Optional[int]:
        cands = []
        for a in legal:
            if OFFSETS["unmortgage"] <= a < OFFSETS["improve_house"]:
                sq = _sq_of(a, "unmortgage", PROPERTY_IDS)
                cands.append((int(env.properties[sq].mortgage_v * 1.1), sq, a))
        if not cands:
            return None
        cands.sort(key=lambda t: (t[0], t[1]))
        cost, sq, action = cands[0]
        if not gates_ok(env, pid, cost, liq_delta=0):
            return None
        return action


__all__ = ["SpecPolicy"]
