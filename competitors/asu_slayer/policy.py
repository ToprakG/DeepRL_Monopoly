"""``slayer-v1``: a net-worth-exact policy for the ``ppo-plus-v2`` simulator.

The policy scores every decision in the units the engine ranks players by, then
spends cash wherever a dollar buys the most net worth, subject to a solvency
reserve sized by how much rent risk it can carry and still expect to survive
the rest of the game.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from monopoly_game_engine.actions import (
    AUCTION_ACTION_TO_INCREMENT,
    OFFSETS,
    PROPERTY_IDS,
    ActionType,
    AuctionAction,
)
from monopoly_game_engine.constants import (
    JAIL_BAIL,
    NUM_PLAYERS,
    PROPERTIES,
    REAL_ESTATE_IDS,
)
from monopoly_game_engine.env import (
    PHASE_AUCTION,
    PHASE_OUT_OF_TURN,
    PHASE_POST_ROLL,
    PHASE_PRE_ROLL,
)

from .board import (
    exposure,
    group_of,
    income_by_square,
    owned_in_group,
    rent_quantile,
)
from .scoring import (
    acquisition_gain,
    development_outlook,
    disposal_loss,
    improvement_gain,
    liquidation_options,
    mortgage_loss,
)


SLAYER_V1 = "slayer-v1"
_EXCHANGE_STRIDE = len(PROPERTY_IDS) * (len(PROPERTY_IDS) - 1)
_MAX_QUANTILE = 0.999


@dataclass(frozen=True, slots=True)
class SlayerConfig:
    """Tunable weights.

    The reserve is deliberately threat-proportional and holds no credit for
    liquidation. An earlier design added a flat reserve while deeds were still
    unowned and counted mortgage capacity as if it were cash; both were wrong
    in the same direction. The flat term peaked on an empty board, which is
    when buying is cheapest and rent is nearly zero, and the liquidation credit
    created a trap: a strict reserve blocked purchases, owning nothing removed
    the credit, and the gate tightened until the agent never invested at all.
    """

    reserve_floor: float = 50.0
    target_survival: float = 0.70
    expected_game_length: float = 45.0
    min_horizon: float = 4.0
    threat_multiple: float = 1.0
    active_liquidation: bool = False
    build_reserve_fraction: float = 0.25
    auction_value_fraction: float = 0.62
    denial_fraction: float = 0.22
    auction_step_fraction: float = 0.18
    jail_exposure_threshold: float = 95.0
    trade_margin: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.target_survival < 1.0:
            raise ValueError("target_survival must be strictly between 0 and 1")
        if self.min_horizon < 1.0:
            raise ValueError("min_horizon must be at least one turn")
        if self.expected_game_length <= 0.0:
            raise ValueError("expected_game_length must be positive")
        if self.threat_multiple < 0.0:
            raise ValueError("threat_multiple must not be negative")
        if not 0.0 <= self.build_reserve_fraction <= 1.0:
            raise ValueError("build_reserve_fraction must be within [0, 1]")

    def evolve(self, **changes) -> "SlayerConfig":
        return replace(self, **changes)


DEFAULT_CONFIG = SlayerConfig()


class SlayerV1:
    """Deterministic net-worth-maximizing policy."""

    policy_id = SLAYER_V1

    def __init__(self, player_id: int, config: SlayerConfig = DEFAULT_CONFIG):
        if not 0 <= player_id < NUM_PLAYERS:
            raise ValueError(f"player_id must be in [0, {NUM_PLAYERS - 1}]")
        self.player_id = player_id
        self.config = config

    # ── Solvency ──────────────────────────────────────────────────────────

    def _survival_quantile(self, env) -> float:
        """Per-turn rent quantile implied by surviving the rest of the game.

        A fixed per-turn quantile is the wrong shape for this decision, because
        the risk compounds over every turn still to be played. Insuring against
        the 90th percentile each turn sounds safe and is not: measured games run
        about 45 turns, and ``0.90 ** 45`` is under one percent. Choosing the
        survival target first and solving for the per-turn quantile inverts that
        relationship, and it also relaxes correctly near the end of a game,
        where fewer remaining turns mean fewer chances to be ruined.
        """

        config = self.config
        horizon = max(config.min_horizon, config.expected_game_length - env.round)
        quantile = config.target_survival ** (1.0 / horizon)
        # rent_quantile requires a value strictly inside (0, 1); a very long
        # horizon would otherwise round to 1.0.
        return min(quantile, _MAX_QUANTILE)

    def _reserve(self, env) -> float:
        """Cash to keep back against the rent that could arrive next turn.

        On an empty board almost every landing owes nothing, so the reserve is
        the floor at any quantile and the agent still invests freely. It rises
        on its own as opponents develop, and carries no term that depends on
        what we own: crediting our own mortgage capacity here once created a
        trap where owning nothing tightened the gate further.
        """

        config = self.config
        threat = rent_quantile(env, self.player_id, self._survival_quantile(env), 1)
        return config.reserve_floor + config.threat_multiple * threat

    def _affordable(self, env, cost: float, reserve: float) -> bool:
        cash = float(env.players[self.player_id].cash)
        return cost <= cash and (cash - cost) >= reserve

    # ── Valuing an acquisition ────────────────────────────────────────────

    def _acquire_value(self, env, square: int) -> float:
        """What acquiring ``square`` is worth to us, as a decision quantity.

        Two corrections to the raw net-worth delta. Deeds in a colour group an
        opponent has already entered are discounted, because we can never build
        there and unbuilt rent is negligible. Against that, keeping a deed away
        from the opponent who wants it most is worth paying for, whether it is
        currently unowned or already theirs.
        """

        own = acquisition_gain(
            env, self.player_id, square, include_denial=False
        ) * development_outlook(env, self.player_id, square)
        rivals = [
            other.player_id
            for other in env.players
            if other.player_id != self.player_id and not other.bankrupt
        ]
        denial = 0.0
        if rivals:
            denial = max(
                acquisition_gain(env, rival, square, include_denial=False)
                for rival in rivals
            )
        return own + self.config.denial_fraction * denial

    # ── Investment candidates ─────────────────────────────────────────────

    def _investments(self, env, legal: set[int]) -> list[tuple[float, int]]:
        """``(net worth gained per action, action)`` for every way to spend cash.

        Each candidate carries the share of the reserve it must respect. Houses
        are held to a fraction of it, because a house is not consumption: it is
        the only purchase that both raises our income and raises what opponents
        owe us, and measured games are decided by who is the creditor when
        somebody goes bankrupt. Instrumentation found the full reserve refusing
        56% of the builds the policy could legally afford, in games it then
        lost with no houses on the board at all.
        """

        share = self.config.build_reserve_fraction
        # Unmortgaging is the exact inverse of the recovery rule, so the two
        # must never both be relaxed: mortgage, unmortgage, mortgage is an
        # infinite loop. Holding unmortgage to the full reserve keeps the
        # guarantee that no investment can drop cash back under it.
        unmortgage_share = 1.0 if self.config.active_liquidation else share
        candidates: list[tuple[float, float, int, float]] = []
        for index, square in enumerate(PROPERTY_IDS):
            action = OFFSETS["unmortgage"] + index
            if action not in legal:
                continue
            prop = env.properties[square]
            cost = int(prop.mortgage_v * 1.1)
            # Unmortgaging restores exactly what mortgaging destroyed, and it
            # brings an earning deed back, so it is treated as development.
            gain = mortgage_loss(env, square) - cost
            candidates.append((gain, cost, action, unmortgage_share))

        for index, square in enumerate(REAL_ESTATE_IDS):
            cost = float(PROPERTIES[square]["house_price"])
            action = OFFSETS["improve_hotel"] + index
            if action in legal:
                candidates.append(
                    (improvement_gain(env, square, True) - cost, cost, action, share)
                )
            action = OFFSETS["improve_house"] + index
            if action in legal:
                candidates.append(
                    (improvement_gain(env, square, False) - cost, cost, action, share)
                )

        candidates.extend(
            (gain, cost, action, 1.0)
            for gain, cost, action in self._trade_proposals(env, legal)
        )
        reserve = self._reserve(env)
        return sorted(
            (
                (gain, action)
                for gain, cost, action, scale in candidates
                if gain > 0 and self._affordable(env, cost, reserve * scale)
            ),
            key=lambda item: (-item[0], item[1]),
        )

    def _trade_proposals(self, env, legal: set[int]) -> list[tuple[float, float, int]]:
        """Deed-for-deed swaps a rational opponent has a reason to accept.

        Cash-for-deed offers are omitted on purpose. Instrumented games showed
        90 such offers made and none accepted: an opponent holding a deed
        values it above its list price, and no legal cash level reaches that.
        Proposing them burned a fifth of the agent's decisions for nothing.
        A swap is only proposed when both sides gain and we gain more.
        """

        proposals: list[tuple[float, float, int]] = []
        others = [pid for pid in range(NUM_PLAYERS) if pid != self.player_id]
        for action in legal:
            if not OFFSETS["exch_trade"] <= action < OFFSETS["auction"]:
                continue
            local = action - OFFSETS["exch_trade"]
            target = others[local // _EXCHANGE_STRIDE]
            remainder = local % _EXCHANGE_STRIDE
            offer_index = remainder // (len(PROPERTY_IDS) - 1)
            raw = remainder % (len(PROPERTY_IDS) - 1)
            request_index = raw if raw < offer_index else raw + 1
            offered = PROPERTY_IDS[offer_index]
            requested = PROPERTY_IDS[request_index]

            gain = self._acquire_value(env, requested) - disposal_loss(
                env, self.player_id, offered
            )
            counter = acquisition_gain(
                env, target, offered, include_denial=False
            ) - disposal_loss(env, target, requested)
            if counter <= 0 or gain <= counter:
                continue
            proposals.append((gain * self.config.trade_margin, 0.0, action))
        return proposals

    def _raise_cash_action(self, env, legal: set[int]) -> int | None:
        """Mortgage back up to the reserve before a rent bill arrives.

        The reserve was only ever a gate on spending, and that is not enough:
        instrumented games spent about half of their own decisions below it,
        because the reserve climbs as opponents develop while our cash is
        already committed. Refusing to spend cannot recover from that; only
        raising cash can.

        Two restrictions keep this from becoming its own death spiral. Deeds in
        a complete color group are never mortgaged, because mortgaging sets
        their rent to zero and that rent is the income the policy wins with,
        and houses are never sold, which costs 3 to 11 net worth per dollar
        against 2.5 for an ordinary mortgage. Within what is left, the deed
        earning the least is given up first.
        """

        if not self.config.active_liquidation:
            return None
        cash = float(env.players[self.player_id].cash)
        reserve = self._reserve(env)
        if cash >= reserve:
            return None

        earned = income_by_square(env, self.player_id, 1)
        best: tuple[float, float, int] | None = None
        for index, square in enumerate(PROPERTY_IDS):
            action = OFFSETS["mortgage"] + index
            if action not in legal:
                continue
            if owned_in_group(env, self.player_id, square) == len(group_of(square)):
                continue  # never break an earning monopoly
            key = (earned.get(square, 0.0), float(env.properties[square].mortgage_v), action)
            if best is None or key < best:
                best = key
        return None if best is None else best[2]

    # ── Phase handlers ────────────────────────────────────────────────────

    def _jail_action(self, env, legal: set[int]) -> int | None:
        player = env.players[self.player_id]
        if not player.in_jail:
            return None
        expected_out, _worst = exposure(env, self.player_id, 1)
        # Jail is shelter once the board is expensive; leave while it is cheap.
        if expected_out >= self.config.jail_exposure_threshold:
            return None
        if int(ActionType.USE_GOOJ_CARD) in legal:
            return int(ActionType.USE_GOOJ_CARD)
        reserve = self._reserve(env)
        if int(ActionType.PAY_BAIL) in legal and self._affordable(
            env, JAIL_BAIL, reserve
        ):
            return int(ActionType.PAY_BAIL)
        return None

    def _incoming_trade_action(self, env, legal: set[int]) -> int | None:
        """Judge an offer against what it gains its proposer, not just us.

        Every offer on the table was constructed by an opponent to serve that
        opponent, so "my net worth goes up" is too weak a test. Both sides are
        priced on their own holdings only and the offer is declined unless our
        gain is the larger one.
        """

        if int(ActionType.ACCEPT_TRADE) not in legal:
            return None
        entry = env._incoming_trade_entry(self.player_id)
        if entry is None:
            return int(ActionType.DECLINE_TRADE)
        proposer, offer = entry

        mine = float(offer.cash_offered) - float(offer.cash_requested)
        theirs = float(offer.cash_requested) - float(offer.cash_offered)
        if offer.offered_prop is not None:
            square = offer.offered_prop.square_id
            mine += self._acquire_value(env, square)
            theirs -= disposal_loss(env, proposer, square)
        if offer.requested_prop is not None:
            square = offer.requested_prop.square_id
            mine -= disposal_loss(env, self.player_id, square)
            theirs += acquisition_gain(env, proposer, square, include_denial=False)

        reserve = self._reserve(env)
        if (
            mine > 0
            and mine > theirs
            and self._affordable(env, float(offer.cash_requested), reserve)
        ):
            return int(ActionType.ACCEPT_TRADE)
        return int(ActionType.DECLINE_TRADE)

    def _auction_action(self, env, legal: set[int]) -> int:
        config = self.config
        square = env.auction_property_id
        if square is None:
            return int(AuctionAction.PASS)
        ceiling = config.auction_value_fraction * self._acquire_value(env, square)

        reserve = self._reserve(env)
        increments = [
            (AUCTION_ACTION_TO_INCREMENT[AuctionAction(action)], action)
            for action in legal
            if action != int(AuctionAction.PASS)
        ]
        step_target = max(1.0, config.auction_step_fraction * ceiling)
        affordable = [
            (increment, action)
            for increment, action in sorted(increments)
            if env.auction_high_bid + increment <= ceiling
            and self._affordable(env, env.auction_high_bid + increment, reserve)
        ]
        if not affordable:
            return int(AuctionAction.PASS)
        for increment, action in affordable:
            if increment >= step_target:
                return action
        return affordable[-1][1]

    def _debt_action(self, env, legal: set[int]) -> int:
        if int(ActionType.DECLARE_BANKRUPT) in legal and len(legal) == 1:
            return int(ActionType.DECLARE_BANKRUPT)
        options = liquidation_options(env, self.player_id, legal)
        if options:
            return options[0][3]
        return min(legal)

    # ── Entry point ───────────────────────────────────────────────────────

    def choose_action(self, env) -> int:
        legal_list = env.get_allowed_actions(self.player_id)
        if not legal_list:
            raise RuntimeError(f"player {self.player_id} has no legal action")
        if len(legal_list) == 1:
            return legal_list[0]
        legal = set(legal_list)

        if env.phase == PHASE_AUCTION:
            return self._auction_action(env, legal)

        if env.debt_player == self.player_id and env.phase == PHASE_POST_ROLL:
            return self._debt_action(env, legal)

        if env.phase == PHASE_POST_ROLL and not env.has_rolled:
            jail = self._jail_action(env, legal)
            if jail is not None:
                return jail
            return int(ActionType.ROLL_DICE)

        if env.phase == PHASE_POST_ROLL:
            if int(ActionType.BUY_PROPERTY) in legal:
                square = env.players[self.player_id].position
                price = float(PROPERTIES[square]["price"])
                gain = self._acquire_value(env, square) - price
                if gain > 0 and self._affordable(env, price, self._reserve(env)):
                    return int(ActionType.BUY_PROPERTY)
            return int(ActionType.END_TURN)

        if env.phase in (PHASE_PRE_ROLL, PHASE_OUT_OF_TURN):
            response = self._incoming_trade_action(env, legal)
            if response is not None:
                return response
            if env.phase == PHASE_PRE_ROLL:
                jail = self._jail_action(env, legal)
                if jail is not None:
                    return jail
                # Strictly before investing, and never alongside it. Every
                # investment leaves cash at or above the reserve, so raising
                # cash cannot be re-triggered by our own spending and the two
                # rules cannot oscillate against each other.
                recovery = self._raise_cash_action(env, legal)
                if recovery is not None:
                    return recovery
            investments = self._investments(env, legal)
            if investments:
                return investments[0][1]
            return int(ActionType.END_TURN)

        return legal_list[0]


__all__ = ["DEFAULT_CONFIG", "SLAYER_V1", "SlayerConfig", "SlayerV1"]
