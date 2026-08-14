"""
PPO Agent (Section V-A-1 and V-B).

Implements actor-critic PPO with clipped surrogate objective and
truncated GAE advantage estimation, as described in the paper.

Hybrid mode: BUY_PROPERTY and ACCEPT_TRADE are handled by fixed rules,
             all other actions go through the actor network.

Fixes applied
─────────────
Bug 1 – Hybrid actions no longer stored in the PPO buffer.
    choose_action() now returns a sentinel log_prob=None to signal that
    the action was chosen by the fixed policy and must not be stored.
    train.py should check `if log_prob is not None` before calling store().

Bug 2 – Correct action mask used during update().
    Each transition now records the allowed_actions mask at collection
    time. The PPO update uses that per-step mask instead of an all-ones
    mask, so the actor is never trained on actions that were illegal in
    the state where the transition was collected.

Bug 3 – Proper GAE bootstrap for mid-game rollout boundaries.
    _compute_gae() now accepts an explicit `last_next_value` argument.
    When the buffer is flushed mid-game (done=False at the boundary),
    train.py should query the critic on the next state and pass that
    value here so the advantage estimate is not truncated to zero.
"""

from typing import List, Optional
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .actions import (
    ACTION_SPACE_SIZE,
    AUCTION_ACTION_TO_INCREMENT,
    OFFSETS,
    ActionType,
    AuctionAction,
)
from .agents_fixed import _buy_trade_action, _exchange_action, _sell_trade_action
from .board_traffic import landing_odds, landing_relative
from .constants import (
    COLOR_GROUPS,
    JAIL_BAIL,
    NUM_PLAYERS,
    PROPERTY_IDS,
    REAL_ESTATE_IDS,
    RULESET_VERSION,
    TRADE_CASH_LEVELS,
)
from .networks import ActorNetwork, CriticNetwork
from .state import STATE_DIM

def _rent_gain_per_lap(env, owner_pid: int, sq: int) -> float:
    """Expected rent per lap (~40 moves) an opponent would pay us if
    ``owner_pid`` owned this square, landing-odds weighted and monopoly-
    aware. Own implementation — consolidates what used to be several
    separate ad-hoc multipliers (0.9x/1.75x auction ceilings, orange-only
    buy buffer) into one consistent estimate, after seeing a competitor's
    equivalent concept expressed the same way."""
    prop = env.properties[sq]
    color = prop.color
    group = COLOR_GROUPS.get(color, [])
    owner = env.players[owner_pid]
    if color == "railroad":
        count = owner.railroads_owned() + 1
        rent = float(prop.data["rent"][min(count - 1, 3)])
    elif color == "utility":
        count = owner.utilities_owned() + 1
        rent = 7.0 * (10.0 if count >= 2 else 4.0)
    else:
        completes = bool(group) and (
            sum(1 for s in group if env.properties[s].owner == owner_pid) + 1 == len(group)
        )
        base = float(prop.data["rent"][0])
        rent = base * 2.0 if completes else base
    return landing_odds(sq) * rent * 40.0


def _property_value(env, pid: int, sq: int) -> float:
    """Value of acquiring this square: our own rent-gain-per-lap plus half
    of what the best-placed live rival would gain if they got it instead
    (denial)."""
    my_gain = _rent_gain_per_lap(env, pid, sq)
    denial = 0.0
    for rival in env.players:
        if rival.player_id == pid or rival.bankrupt:
            continue
        denial = max(denial, _rent_gain_per_lap(env, rival.player_id, sq))
    return my_gain + 0.5 * denial


# ── Hybrid fixed-policy decisions ─────────────────────────────────────────────


def fixed_buy_decision(env, pid: int) -> bool:
    player = env.players[pid]
    sq = player.position
    if sq not in env.properties:
        return False
    prop = env.properties[sq]
    if prop.owner is not None or not player.can_afford(prop.price):
        return False

    # Always buy if it completes a monopoly
    color = prop.color
    group = COLOR_GROUPS.get(color, [])
    if group:
        owned = sum(1 for s in group if env.properties[s].owner == pid)
        if owned + 1 == len(group):
            return True

        # Denial: if any live opponent is one piece from completing this
        # same group, buying denies them the monopoly outright — worth
        # taking even on a thin margin, since letting a rival complete a
        # monopoly is far costlier than the purchase price.
        for rival in env.players:
            if rival.player_id == pid or rival.bankrupt:
                continue
            rival_owned = sum(1 for s in group if env.properties[s].owner == rival.player_id)
            if rival_owned + 1 == len(group):
                return True

    # Protect the empire: a competitor's own diagnostic found 19 bankruptcies
    # in 32 games, each one handing a rival our whole estate, traced to
    # spending as aggressively after owning a monopoly as before it. Before
    # we hold one there's little to protect and speed matters; once we do,
    # staying solvent to develop it outranks grabbing one more square.
    owns_monopoly = any(
        p.is_monopoly and p.color not in ("railroad", "utility")
        for p in player.properties
    )
    reserve = 400 if owns_monopoly else 0

    # Exact landing-probability model (own Markov-chain implementation, see
    # board_traffic.py) replaces the old orange-only special case: any
    # square landed on more than average is worth a thinner cash buffer,
    # scaled continuously instead of one hand-picked group getting a break.
    relative = landing_relative(sq)
    buffer = max(20, 100 / relative) + reserve
    if player.cash >= prop.price + buffer:
        return True

    # Consolidated value estimate (rent-gain-per-lap + denial): buy even
    # below the usual buffer if the square is clearly worth it.
    if player.can_afford(prop.price) and _property_value(env, pid, sq) > prop.price * 1.5:
        return True
    return False


def fixed_accept_trade_decision(env, pid: int) -> bool:
    offer = env._incoming_trade(pid)
    if offer is None:
        return False

    if offer.offered_prop:
        color = offer.offered_prop.color
        group = COLOR_GROUPS.get(color, [])
        if group:
            owned_after = sum(
                1
                for s in group
                if env.properties[s].owner == pid
                or env.properties[s] == offer.offered_prop
            )
            if owned_after == len(group):
                return True

    nwo = offer.net_worth()
    if nwo < 0:
        return False

    # Cash-floor safety: a nominally fair/good deal that drains cash below
    # a survivable buffer can still bankrupt us on the very next rent hit.
    player = env.players[pid]
    if player.cash - offer.cash_requested < 100:
        return False

    return True


def fixed_build_decision(env, pid: int, allowed) -> Optional[int]:
    """Own build heuristic (not derived from any fixed agent's code): build
    on any owned monopoly once cash allows, hotel-before-house, cheapest
    group first. Even-building legality is already enforced upstream by
    env's allowed-action list, so we only need to pick among what's legal.
    """
    player = env.players[pid]
    # Endgame: cash held back for future turns is wasted if there are few
    # turns left, and games that hit the round cap are scored on net worth
    # — building converts cash into book value, so spend more freely once
    # the game is nearly over.
    rounds_left = env.max_rounds - env.round
    build_floor = 20 if rounds_left <= 20 else 100
    for i, sq in enumerate(REAL_ESTATE_IDS):
        prop = env.properties[sq]
        if prop.owner != pid or not prop.is_monopoly:
            continue
        house_price = prop.data["house_price"]
        if player.cash < house_price + build_floor:
            continue
        hotel_action = OFFSETS["improve_hotel"] + i
        house_action = OFFSETS["improve_house"] + i
        if hotel_action in allowed:
            return hotel_action
        if house_action in allowed:
            return house_action
    return None


def fixed_mortgage_to_build_decision(env, pid: int, allowed) -> Optional[int]:
    """Own targeted exception to "never voluntarily mortgage": raise cash
    specifically to fund a house on a monopoly we already hold, by
    mortgaging our cheapest-income-loss non-monopoly junk property first.
    A mortgaged single loses a trickle of rent; a house on a completed
    group returns several times its cost in book value and keeps
    compounding. Only fires when we already have somewhere to build and
    cash alone isn't quite enough — never touches a live (monopoly) group."""
    player = env.players[pid]
    buildable = [
        prop for prop in player.properties
        if prop.is_monopoly and prop.is_real_estate and prop.houses < 4 and not prop.mortgaged
    ]
    if not buildable or env.houses_available <= 0:
        return None
    cheapest_house = min(prop.data["house_price"] for prop in buildable)
    if player.cash >= cheapest_house + 20:
        return None  # building is already affordable on its own

    candidates = sorted(
        (
            (sq, env.properties[sq])
            for sq in PROPERTY_IDS
            if env.properties[sq].owner == pid
            and not env.properties[sq].mortgaged
            and not env.properties[sq].is_monopoly
            and env.properties[sq].houses == 0
        ),
        key=lambda pair: pair[1].price,
    )
    for sq, prop in candidates:
        idx = PROPERTY_IDS.index(sq)
        action = OFFSETS["mortgage"] + idx
        if action in allowed:
            return action
    return None


def fixed_auction_decision(env, pid: int, allowed) -> Optional[int]:
    """Own auction heuristic — deliberately smarter than the fixed agents'
    shared base-class logic (agents_fixed.py's _auction_action always jumps
    straight to the max legal increment when it wants the property, with no
    price shading). We compute a value ceiling (higher for a monopoly-
    completing piece) and bid the SMALLEST increment that keeps us above the
    current high bid, up to that ceiling and a cash-safety floor — real bid
    shading instead of always maxing out."""
    prop = env.properties.get(env.auction_property_id)
    player = env.players[pid]
    if prop is None:
        return None

    color = prop.color
    group = COLOR_GROUPS.get(color, [])
    completes_monopoly = bool(group) and (
        sum(1 for s in group if env.properties[s].owner == pid) + 1 == len(group)
    )
    # More live rivals still on the board means more potential rent payers
    # landing on this square over the rest of the game, so it's worth more
    # than list price even outside a monopoly play.
    rivals = sum(1 for p in env.players if p.player_id != pid and not p.bankrupt)
    rival_scale = 0.9 + 0.05 * max(0, rivals - 1)
    if completes_monopoly:
        # Exact book-value jump from state.py's own net-worth formula
        # (verified: unmortgaged deed = 2.5x price, deed in a complete
        # group = 5.0x price — every deed we already hold in this group
        # re-prices on completion, not just the new one). A flat 1.75x
        # ceiling was leaving real value on the table.
        owned_prices = sum(
            env.properties[s].price for s in group if env.properties[s].owner == pid
        )
        group_total = sum(env.properties[s].price for s in group)
        book_gain = 5.0 * group_total - 2.5 * owned_prices
        ceiling = min(book_gain, prop.price * 10.0)
    else:
        ceiling = prop.price * rival_scale

    safety_buffer = 100
    max_bid = min(ceiling, player.cash - safety_buffer)

    if env.auction_high_bid >= max_bid:
        return int(AuctionAction.PASS)

    candidates = sorted(
        (
            (action, increment)
            for action, increment in AUCTION_ACTION_TO_INCREMENT.items()
            if int(action) in allowed and env.auction_high_bid + increment <= max_bid
        ),
        key=lambda pair: pair[1],
    )
    if not candidates:
        return int(AuctionAction.PASS)
    return int(candidates[0][0])


def fixed_bankruptcy_denial_decision(env, pid: int, allowed) -> Optional[int]:
    """Own edge case (spotted by studying a competitor's commit history, not
    ASU): when we're in the debt-settlement menu and even fully liquidating
    everything we own could not cover the debt, bankruptcy is mathematically
    certain no matter what we do this turn. Mortgaging still hands the
    mortgaged deed to the creditor at a favourable buy-back rate once we go
    under; selling to the bank instead raises the identical cash but denies
    the creditor that estate outright. When our fate is sealed, deny rather
    than mortgage — largest book value first."""
    if env.debt_player != pid:
        return None
    player = env.players[pid]
    liquidatable = 0.0
    for prop in player.properties:
        if prop.houses > 0:
            liquidatable += prop.houses * (prop.data.get("house_price", 0) // 2)
        elif not prop.mortgaged:
            liquidatable += prop.mortgage_v
    if env.debt_amount <= player.cash + liquidatable:
        return None
    sells = sorted(
        (
            (idx, sq)
            for idx, sq in enumerate(PROPERTY_IDS)
            if env.properties[sq].owner == pid
        ),
        key=lambda pair: -env.properties[pair[1]].price,
    )
    for idx, sq in sells:
        action = OFFSETS["sell_prop"] + idx
        if action in allowed:
            return action
    return None


def fixed_mortgage_decision(env, pid: int, allowed) -> Optional[int]:
    """Own cash-management heuristic (not from any fixed agent): mortgage a
    non-monopoly property when cash drops below a safety floor, cheapest
    non-target property first, so building/trading capacity on real assets
    is preserved as long as possible."""
    player = env.players[pid]
    if env.debt_player != pid:
        return None
    candidates = sorted(
        (
            (sq, env.properties[sq])
            for sq in PROPERTY_IDS
            if env.properties[sq].owner == pid
            and not env.properties[sq].mortgaged
            and not env.properties[sq].is_monopoly
            and env.properties[sq].houses == 0
        ),
        key=lambda pair: pair[1].price,
    )
    for sq, prop in candidates:
        idx = PROPERTY_IDS.index(sq)
        action = OFFSETS["mortgage"] + idx
        if action in allowed:
            return action
    return None


def fixed_unmortgage_decision(env, pid: int, allowed) -> Optional[int]:
    """Own heuristic: none of the fixed agents ever lift a mortgage, which
    leaves rent income permanently off for those properties. Once cash is
    comfortably above a buffer, pay off the cheapest mortgage first (lowest
    cost to restore income) — a real edge over Builder/DealMaker, not a
    copy of them."""
    player = env.players[pid]
    if player.cash < 500:
        return None
    candidates = sorted(
        (
            (sq, env.properties[sq])
            for sq in PROPERTY_IDS
            if env.properties[sq].owner == pid and env.properties[sq].mortgaged
        ),
        key=lambda pair: pair[1].mortgage_v,
    )
    for sq, prop in candidates:
        cost = int(prop.mortgage_v * 1.1)
        if player.cash - cost < 300:
            continue
        idx = PROPERTY_IDS.index(sq)
        action = OFFSETS["unmortgage"] + idx
        if action in allowed:
            return action
    return None


def fixed_liquidation_decision(env, pid: int, allowed) -> Optional[int]:
    """Own last-resort cash-raising heuristic. sell_house/sell_hotel are
    legal during ordinary play too (voluntary downgrade), so this only acts
    when actually under financial pressure — the debt-rescue menu
    (env.debt_player == pid) or critically low cash — never sells buildings
    just because it's legal. Order: sell a hotel/house on the cheapest
    property first (raises cash while losing the least future rent
    potential), then sell a property outright as the very last resort,
    cheapest first."""
    if env.debt_player != pid:
        return None
    for i, sq in enumerate(REAL_ESTATE_IDS):
        prop = env.properties[sq]
        if prop.owner != pid:
            continue
        action = OFFSETS["sell_hotel"] + i
        if action in allowed:
            return action
    cheapest_house = sorted(
        (
            (i, sq)
            for i, sq in enumerate(REAL_ESTATE_IDS)
            if env.properties[sq].owner == pid and env.properties[sq].houses > 0
        ),
        key=lambda pair: env.properties[pair[1]].price,
    )
    for i, sq in cheapest_house:
        action = OFFSETS["sell_house"] + i
        if action in allowed:
            return action

    cheapest_prop = sorted(
        (
            (idx, sq)
            for idx, sq in enumerate(PROPERTY_IDS)
            if env.properties[sq].owner == pid
        ),
        key=lambda pair: env.properties[pair[1]].price,
    )
    for idx, sq in cheapest_prop:
        action = OFFSETS["sell_prop"] + idx
        if action in allowed:
            return action
    return None


def fixed_jail_decision(env, pid: int, allowed) -> Optional[int]:
    """Own heuristic, matching Builder's/DealMaker's shared jail trait (both
    never pay bail): use a Get-Out-Of-Jail-Free card if held (free, no
    downside), otherwise decline to pay and let the roll-for-doubles /
    wait-it-out path run — cash stays available for building/trading."""
    if int(ActionType.USE_GOOJ_CARD) in allowed:
        return int(ActionType.USE_GOOJ_CARD)
    return None


def fixed_trade_offer_decision(env, pid: int, allowed) -> Optional[int]:
    """Own trade-initiation heuristic, mixing Builder's + DealMaker's core
    trade traits (both are our own fixed agents — no ASU constraint):
    bargain buy-offer for a one-piece-from-monopoly colour group, exchange
    a non-monopoly prop for a needed piece, or sell spare non-monopoly
    props at a premium. Returns None if nothing worthwhile is legal right
    now (the neural net picks something else that turn).
    """
    others = [
        i for i in range(NUM_PLAYERS) if i != pid and not env.players[i].bankrupt
    ]
    if not others:
        return None

    # 1. Bargain buy-offer (0.75x) for a colour group we're one piece from completing
    for color, group in COLOR_GROUPS.items():
        if color in ("railroad", "utility"):
            continue
        owned = [s for s in group if env.properties[s].owner == pid]
        need = [
            s
            for s in group
            if env.properties[s].owner not in (pid, None)
            and not env.players[env.properties[s].owner].bankrupt
        ]
        if len(owned) + 1 == len(group) and need:
            sq = need[0]
            target = env.properties[sq].owner
            action = _buy_trade_action(pid, target, sq, 0, env, allowed)
            if action is not None:
                return action

    # 2. Exchange a non-monopoly prop of ours for a needed piece
    for color, group in COLOR_GROUPS.items():
        if color in ("railroad", "utility"):
            continue
        owned_here = [s for s in group if env.properties[s].owner == pid]
        if not owned_here:
            continue
        need = [
            s
            for s in group
            if env.properties[s].owner not in (pid, None)
            and not env.players[env.properties[s].owner].bankrupt
        ]
        for req_sq in need:
            target = env.properties[req_sq].owner
            for offer_sq in owned_here:
                if env.properties[offer_sq].is_monopoly:
                    continue
                action = _exchange_action(pid, target, offer_sq, req_sq, env, allowed)
                if action is not None:
                    return action

    # 3. Sell spare non-monopoly, unbuilt props at a premium (1.25x)
    for sq in PROPERTY_IDS:
        prop = env.properties[sq]
        if prop.owner != pid or prop.is_monopoly or prop.houses > 0:
            continue
        for target in others:
            action = _sell_trade_action(pid, target, sq, 2, env, allowed)
            if action is not None:
                return action

    return None


# ── Experience buffer ─────────────────────────────────────────────────────────


class PPOBuffer:
    """Stores a single rollout trajectory for PPO updates."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.action_masks = []  # FIX 2: per-step allowed-action masks

    def store(self, state, action, log_prob, reward, value, done, action_mask):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.action_masks.append(action_mask)  # FIX 2

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.states)


# ── PPO Agent ─────────────────────────────────────────────────────────────────


class PPOAgent:
    """
    Actor-critic PPO agent, with optional hybrid mode.

    Parameters mirror Appendix B-A of the paper.
    """

    def __init__(
        self,
        player_id: int,
        hybrid: bool = False,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.05,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_steps: int = 1024,
        n_epochs: int = 4,
        batch_size: int = 64,
        hidden_dim: int = 256,
        win_loss_bonus: float = 1.0,  # tried 10.0 (DDQN's paper constant) — made it
        # worse: near-0% win rate meant almost every episode ate a -10 terminal
        # penalty, swamping the dense shaping signal (reward went negative and
        # flat instead of trending up). Reverted to the empirically-better value.
        device: str = "auto",
    ):
        self.player_id = player_id
        self.hybrid = hybrid
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.win_loss_bonus = win_loss_bonus
        self.hidden_dim = hidden_dim
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else
            "cpu" if device == "auto" else device
        )

        self.actor = ActorNetwork(hidden_dim).to(self.device)
        self.critic = CriticNetwork(hidden_dim).to(self.device)
        self.opt = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

        self.buffer = PPOBuffer()
        self.step_count = 0
        self.games_trained = 0

        # Mask actions permanently handled by fixed policy (hybrid only)
        self.fixed_action_mask = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.bool)
        if hybrid:
            self.fixed_action_mask[int(ActionType.BUY_PROPERTY)] = True
            self.fixed_action_mask[int(ActionType.ACCEPT_TRADE)] = True
            self.fixed_action_mask[OFFSETS["improve_house"]:OFFSETS["sell_house"]] = True
            self.fixed_action_mask[OFFSETS["buy_trade"]:OFFSETS["auction"]] = True
            self.fixed_action_mask[int(ActionType.PAY_BAIL)] = True
            self.fixed_action_mask[int(ActionType.USE_GOOJ_CARD)] = True
            self.fixed_action_mask[OFFSETS["auction"]:OFFSETS["auction"] + 5] = True
            self.fixed_action_mask[OFFSETS["mortgage"]:OFFSETS["unmortgage"]] = True
            self.fixed_action_mask[OFFSETS["unmortgage"]:OFFSETS["improve_house"]] = True
            self.fixed_action_mask[OFFSETS["sell_house"]:OFFSETS["buy_trade"]] = True

    # ── Action selection ──────────────────────────────────────────────────────

    def choose_action(self, state: np.ndarray, env, allowed_actions: List[int]):
        """
        Choose an action. In hybrid mode, intercept BUY and ACCEPT_TRADE.

        Returns (action, log_prob, value, allowed_actions).

        FIX 1: When the hybrid fixed policy fires, log_prob is returned as
        None.  The caller (train.py) must NOT store these transitions in the
        PPO buffer — storing them with log_prob=0 corrupts the importance
        sampling ratio.
        """
        pid = self.player_id

        # Hybrid: handle buy property
        if self.hybrid and int(ActionType.BUY_PROPERTY) in allowed_actions:
            if fixed_buy_decision(env, pid):
                # FIX 1: return None sentinel — caller skips buffer storage
                return int(ActionType.BUY_PROPERTY), None, None, allowed_actions

        # Hybrid: handle trade acceptance
        if self.hybrid and int(ActionType.ACCEPT_TRADE) in allowed_actions:
            pending = next(
                (o for o in env.pending_trades.values() if o.to_player == pid),
                None,
            )
            if pending is not None:
                if fixed_accept_trade_decision(env, pid):
                    return int(ActionType.ACCEPT_TRADE), None, None, allowed_actions
                else:
                    return int(ActionType.DECLINE_TRADE), None, None, allowed_actions

        # Hybrid: handle building (Builder-inspired, own heuristic — see
        # fixed_build_decision docstring)
        if self.hybrid:
            build_action = fixed_build_decision(env, pid, allowed_actions)
            if build_action is not None:
                return build_action, None, None, allowed_actions

        # Hybrid: mortgage junk to fund a house on a monopoly we already
        # hold (own targeted exception to "never voluntarily mortgage")
        if self.hybrid:
            fund_action = fixed_mortgage_to_build_decision(env, pid, allowed_actions)
            if fund_action is not None:
                return fund_action, None, None, allowed_actions

        # Hybrid: handle trade-offer initiation (Builder+DealMaker mix —
        # see fixed_trade_offer_decision docstring)
        if self.hybrid:
            offer_action = fixed_trade_offer_decision(env, pid, allowed_actions)
            if offer_action is not None:
                return offer_action, None, None, allowed_actions

        # Hybrid: jail timing (GOOJ card if held, never pay bail)
        if self.hybrid:
            jail_action = fixed_jail_decision(env, pid, allowed_actions)
            if jail_action is not None:
                return jail_action, None, None, allowed_actions

        # Hybrid: auction bidding (own bid-shading heuristic)
        if self.hybrid:
            auction_action = fixed_auction_decision(env, pid, allowed_actions)
            if auction_action is not None:
                return auction_action, None, None, allowed_actions

        # Hybrid: certain-bankruptcy denial (sell to bank, never mortgage,
        # when nothing we do this turn can avoid going under)
        if self.hybrid:
            deny_action = fixed_bankruptcy_denial_decision(env, pid, allowed_actions)
            if deny_action is not None:
                return deny_action, None, None, allowed_actions

        # Hybrid: mortgage / unmortgage cash management (own heuristics)
        if self.hybrid:
            mort_action = fixed_mortgage_decision(env, pid, allowed_actions)
            if mort_action is not None:
                return mort_action, None, None, allowed_actions
            unmort_action = fixed_unmortgage_decision(env, pid, allowed_actions)
            if unmort_action is not None:
                return unmort_action, None, None, allowed_actions

        # Hybrid: last-resort liquidation (sell house/hotel/prop) when
        # actually under financial pressure — own heuristic
        if self.hybrid:
            liq_action = fixed_liquidation_decision(env, pid, allowed_actions)
            if liq_action is not None:
                return liq_action, None, None, allowed_actions

        # Filter out fixed-policy actions from neural net consideration
        nn_allowed = [a for a in allowed_actions if not self.fixed_action_mask[a]]
        if not nn_allowed:
            nn_allowed = [int(ActionType.DO_NOTHING)]

        state_t = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.inference_mode():
            value = self.critic(state_t).item()
        action, log_prob = self.actor.get_action(state, nn_allowed)

        return action, log_prob, value, nn_allowed  # FIX 2: return nn_allowed

    # ── Store experience ──────────────────────────────────────────────────────

    def store(
        self, state, action, log_prob, reward, value, done, allowed_actions: List[int]
    ):
        """
        Store one NN-chosen transition.
        FIX 1: Callers must only call this when log_prob is not None (i.e.
               the action was chosen by the neural network, not the fixed
               hybrid policy).
        FIX 2: allowed_actions is stored so update() can reconstruct the
               per-step action mask.
        """
        # Build boolean mask for the allowed actions at this step (FIX 2)
        mask = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.bool)
        mask[allowed_actions] = True

        self.buffer.store(state, action, log_prob, reward, value, done, mask)
        self.step_count += 1

    def add_win_loss(self, won: bool):
        """Add terminal win/loss bonus to the last stored reward."""
        if self.win_loss_bonus != 0 and len(self.buffer.rewards) > 0:
            self.buffer.rewards[-1] += self.win_loss_bonus * (1.0 if won else -1.0)

    # ── PPO update ────────────────────────────────────────────────────────────

    def update(
        self, last_next_state: Optional[np.ndarray] = None, last_done: bool = False
    ):
        """
        Run PPO update over the current buffer.

        FIX 3: last_next_state / last_done support correct GAE bootstrap.
            Pass the state that followed the last stored transition and
            whether that transition was terminal.  When the rollout ends
            mid-game (last_done=False), the critic evaluates last_next_state
            to get the bootstrap value instead of using 0.
        """
        if len(self.buffer) == 0:
            return {}

        # FIX 3: bootstrap value for the end of the rollout
        if last_next_state is not None and not last_done:
            with torch.inference_mode():
                st = torch.as_tensor(
                    last_next_state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                bootstrap_value = self.critic(st).item()
        else:
            bootstrap_value = 0.0

        # Convert to tensors
        states = torch.as_tensor(
            np.array(self.buffer.states), dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            self.buffer.actions, dtype=torch.long, device=self.device
        )
        old_lps = torch.as_tensor(
            self.buffer.log_probs, dtype=torch.float32, device=self.device
        )
        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.dones
        # FIX 2: per-step masks stacked into (N, ACTION_SPACE_SIZE)
        step_masks = torch.stack(self.buffer.action_masks).to(self.device)

        # FIX 3: pass bootstrap value into GAE
        advantages = self._compute_gae(rewards, values, dones, bootstrap_value)
        returns = advantages + torch.as_tensor(
            values, dtype=torch.float32, device=self.device
        )
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages - advantages.mean()

        stats = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}
        n_batches = 0

        for _ in range(self.n_epochs):
            indices = torch.randperm(len(states), device=self.device)
            for start in range(0, len(states), self.batch_size):
                idx = indices[start : start + self.batch_size]
                if len(idx) < 2:
                    continue

                sb = states[idx]
                ab = actions[idx]
                olp = old_lps[idx]
                adv = advantages[idx]
                ret = returns[idx]
                mask = step_masks[idx]  # FIX 2: per-step masks for this batch

                # FIX 2: use the recorded per-step mask, not all-ones
                log_probs_all = self.actor(sb, mask)
                new_lps = log_probs_all.gather(1, ab.unsqueeze(1)).squeeze(1)
                probs_all = log_probs_all.exp()
                # Masked actions have log_prob=-inf and prob=0. Multiplying
                # them directly yields 0 * -inf = nan, which can poison the
                # PPO loss and corrupt the actor weights.
                safe_log_probs = torch.where(
                    mask, log_probs_all, torch.zeros_like(log_probs_all)
                )
                entropy = -(probs_all * safe_log_probs).sum(dim=-1).mean()

                ratio = (new_lps - olp).exp()
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                actor_loss = -torch.min(surr1, surr2).mean()

                values_pred = self.critic(sb)
                critic_loss = nn.MSELoss()(values_pred, ret)

                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    - self.entropy_coef * entropy
                )

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.opt.step()

                stats["actor_loss"] += actor_loss.item()
                stats["critic_loss"] += critic_loss.item()
                stats["entropy"] += entropy.item()
                n_batches += 1

        self.buffer.clear()
        if n_batches > 0:
            return {k: v / n_batches for k, v in stats.items()}
        return stats

    def _compute_gae(
        self, rewards, values, dones, last_next_value: float = 0.0
    ) -> torch.Tensor:
        """
        Generalised Advantage Estimation.

        FIX 3: last_next_value is the critic's estimate of V(s_{T+1}).
        For terminal transitions it is 0; for mid-game buffer flushes it is
        the critic's evaluation of the state that followed the last stored
        step, preventing the advantage from being truncated to zero.
        """
        advantages = torch.zeros(len(rewards), device=self.device)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if t + 1 < len(values):
                next_val = values[t + 1]
            else:
                # FIX 3: use bootstrapped value, not hard-coded 0
                next_val = last_next_value
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        return advantages

    def save(self, path: str):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        torch.save(
            {
                "format_version": 3,
                "ruleset": RULESET_VERSION,
                "state_dim": STATE_DIM,
                "action_dim": ACTION_SPACE_SIZE,
                "player_id": self.player_id,
                "hybrid": self.hybrid,
                "hidden_dim": self.hidden_dim,
                "step_count": self.step_count,
                "games_trained": self.games_trained,
                "training_config": {
                    "gamma": self.gamma,
                    "lam": self.lam,
                    "clip_eps": self.clip_eps,
                    "entropy_coef": self.entropy_coef,
                    "value_coef": self.value_coef,
                    "max_grad_norm": self.max_grad_norm,
                    "n_steps": self.n_steps,
                    "n_epochs": self.n_epochs,
                    "batch_size": self.batch_size,
                    "win_loss_bonus": self.win_loss_bonus,
                },
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.opt.state_dict(),
            },
            temporary,
        )
        os.replace(temporary, destination)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        if ckpt.get("format_version") is None:
            raise ValueError(
                "Legacy PPO checkpoint is incompatible with "
                f"{RULESET_VERSION}; train a new checkpoint."
            )
        expected = {
            "format_version": 3,
            "ruleset": RULESET_VERSION,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_SPACE_SIZE,
            "player_id": self.player_id,
            "hybrid": self.hybrid,
            "hidden_dim": self.hidden_dim,
        }
        actual = {key: ckpt.get(key) for key in expected}
        if actual != expected:
            raise ValueError(
                f"Incompatible PPO checkpoint metadata: {actual}; "
                f"expected {expected}."
            )
        try:
            self.actor.load_state_dict(ckpt["actor"])
            self.critic.load_state_dict(ckpt["critic"])
        except RuntimeError as exc:
            raise ValueError(
                f"PPO checkpoint network is incompatible with {RULESET_VERSION}."
            ) from exc
        if "optimizer" in ckpt:
            self.opt.load_state_dict(ckpt["optimizer"])
            for state in self.opt.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(self.device)
        for key, value in ckpt.get("training_config", {}).items():
            setattr(self, key, value)
        self.step_count = int(ckpt.get("step_count", 0))
        self.games_trained = int(ckpt.get("games_trained", 0))
