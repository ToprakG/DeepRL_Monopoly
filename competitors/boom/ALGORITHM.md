# Our agent — algorithm description

## Summary

A hybrid PPO agent (`monopoly_game_engine/agent_ppo.py`) where 9 of the
~12 decision families are hand-designed, deterministic heuristics, and the
neural network controls the rest (roll/end-turn sequencing, forced
bankruptcy). This is not pure black-box RL and not imitation of any other
agent — every heuristic below is our own design, with the reasoning stated.

## Why hybrid, not pure RL

Pure self-play RL against Builder+DealMaker (+ASU) needed far more training
volume than our time budget allowed to show a strong win rate (a few
thousand games only reached 2-5%). Hand-designing the well-understood
decisions (buying, building, trading, jail, auctions, mortgaging,
liquidation) removes most of the exploration burden from the network and
is testable in seconds via `tools/evaluate`-style scripts instead of hours
of training. A 50-game eval with an *untrained* network plus these 9
heuristics alone scored 32% against Builder+DealMaker+Hoarder.

## The 9 heuristics (all in `agent_ppo.py`)

1. **`fixed_buy_decision`** — always buy if it completes a monopoly.
   Otherwise buy if cash covers price + a buffer (thinner, $20, for the
   orange group — statistically the most-landed-on squares on the board
   due to post-jail dice distribution — larger, $100, elsewhere).
2. **`fixed_accept_trade_decision`** — accept if it completes a monopoly.
   Otherwise accept only if the offer's net worth is non-negative *and*
   accepting still leaves at least $100 cash (a nominally fair trade that
   leaves us broke can still cause bankruptcy on the next rent hit).
3. **`fixed_build_decision`** — build on any owned monopoly once cash
   allows (hotel before house, cheapest group first). Even-building
   legality is already enforced by the environment's allowed-action list.
4. **`fixed_trade_offer_decision`** — bargain buy-offer (0.75x) for a
   colour group one piece from completion, then exchange a non-monopoly
   piece for a needed one, then sell spare non-monopoly props at a
   premium (1.25x).
5. **`fixed_jail_decision`** — use a Get-Out-Of-Jail-Free card if held
   (free, no downside); otherwise never pay bail, keeping cash available.
6. **`fixed_auction_decision`** — computes a value ceiling (1.75x price if
   the property completes a monopoly, 0.9x otherwise) and bids the
   *smallest* legal increment that stays ahead, up to that ceiling and a
   cash-safety floor. The shared fixed-agent base class always jumps
   straight to the maximum legal increment with no price shading — this
   is a deliberate improvement over that, not a copy of it.
7. **`fixed_mortgage_decision`** — mortgage the cheapest non-monopoly,
   unbuilt property once cash drops below $200.
8. **`fixed_unmortgage_decision`** — once cash is comfortably above $500
   (leaving a $300 buffer), pay off the cheapest mortgage first to restore
   rent income. None of the fixed baseline agents ever do this.
9. **`fixed_liquidation_decision`** — only fires under real financial
   pressure (the environment's debt-rescue state, or cash under $50): sell
   a hotel/house on the cheapest property first, then sell a property
   outright as the last resort, cheapest first. Guarded so it never sells
   buildings during ordinary play just because the action happens to be
   legal.

## What the network still controls

Roll/end-turn sequencing and forced bankruptcy declaration — the
remaining decisions are either close to forced by the ruleset or low-
stakes enough that a lightly-trained network handles them reasonably.

## Opponent / evaluation setup

Training and evaluation tables are documented in `PPO_PLUS_RULES.md` and
`monopoly_game_engine/train.py`. ASU (`ASU_FROZEN_TEACHER`) is used only
as a live black-box opponent (`choose_action(env)`) for training/eval —
never as a source of labels, weights, or formulas. See
`tools/train_vs_target_table.py`, `tools/train_vs_3asu.py`, and
`tools/scout_vs_asu.py`.
