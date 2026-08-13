"""Complete discrete key for transposition / leaf cache (no float hashing)."""

from __future__ import annotations

from monopoly_game_engine.constants import PROPERTY_IDS
from monopoly_game_engine.env import MonopolyEnv


def position_key(env: MonopolyEnv) -> tuple:
    """Hashable snapshot of everything that affects legal actions and transitions."""

    players = tuple(
        (
            int(player.cash),
            int(player.position),
            bool(player.in_jail),
            int(player.jail_turns),
            bool(player.gooj_card),
            bool(player.bankrupt),
        )
        for player in env.players
    )
    properties = tuple(
        (
            env.properties[sq].owner,
            bool(env.properties[sq].mortgaged),
            int(env.properties[sq].houses),
            bool(env.properties[sq].is_monopoly),
        )
        for sq in PROPERTY_IDS
    )
    trades = tuple(
        sorted(
            (
                int(offer.from_player),
                int(offer.to_player),
                None if offer.offered_prop is None else int(offer.offered_prop.square_id),
                None if offer.requested_prop is None else int(offer.requested_prop.square_id),
                int(offer.cash_offered),
                int(offer.cash_requested),
            )
            for offer in env.pending_trades.values()
        )
    )
    return (
        int(env.round),
        str(env.phase),
        int(env.current_turn_idx),
        tuple(int(pid) for pid in env.turn_order),
        bool(env.has_rolled),
        int(env.consecutive_doubles),
        bool(env.extra_roll_pending),
        bool(env.done),
        tuple(int(d) for d in env.last_dice),
        tuple(int(pid) for pid in env.out_of_turn_pids),
        env.auction_property_id,
        tuple(int(pid) for pid in (env.auction_bidders or [])),
        env.auction_current_pid,
        int(env.auction_high_bid or 0),
        env.auction_high_bidder,
        int(env.houses_available),
        int(env.hotels_available),
        env.debt_player,
        env.debt_creditor,
        int(env.debt_amount or 0),
        env.player_needs_funds,
        players,
        properties,
        trades,
    )


def key_seed(key: tuple) -> int:
    return hash(key) & 0x7FFFFFFF
