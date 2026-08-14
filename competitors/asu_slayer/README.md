# ASU_SLAYER — a net-worth-exact challenger

`ASU_SLAYER` is a search policy for this repository's `ppo-plus-v2` simulator,
built to challenge `ASU_FROZEN_TEACHER`. It contains no learned weights and
needs no training: the entire edge comes from optimizing the quantity the
simulator actually ranks players by.

## Measured status (seat-balanced, natural game end, Wilson 95% CI)

| Lineup | slayer-v1 | Reference |
| --- | --- | --- |
| vs fixed-b/d/e, n=2000 | **50.55%** [48.4, 52.7] | best prior agent 41.8%; parity 25% |
| vs 3x asu-value-v1, n=120, hardened | **22.50%** [16.0, 30.8] | parity 25%; pre-hardening 15.0% |

The net-worth-exact thesis wins clearly in the CAP regime (games decided by
the round-200 net-worth ranking). Against three ASU teachers, games end by
ELIMINATION, and the thesis alone was not enough: the original build scored
15.0% [9.7, 22.5] with 85% bankruptcies — while completing a color group in
64% of those games. The hardening pass below recovered +7.5pp (McNemar
p=0.035 on paired seeds) by fixing the solvency machinery; the remaining gap
to parity is an open problem, not a solved one. See SLAYER_REVIEW.md for the
full audit.

## The thesis: score in the engine's own units

The engine decides a game two ways. Either one player is left standing, or the
200-round cap is reached and `MonopolyEnv.winner()` picks the greatest
`Player.net_worth()`. That function does **not** price a deed at its list
price (`monopoly_game_engine/state.py`):

| Holding | Contribution to `net_worth()` | Cash cost |
| --- | --- | --- |
| Unmortgaged deed | `2.5 x price` | `price` |
| Deed in a complete color group | `5.0 x price` | `price` |
| `h` houses | `h x house_price x (1 + 0.5h)` | `h x house_price` |
| Hotel | `5 x house_price x 3.5` | `5 x house_price` |
| Cash | face value | — |

Two consequences drive the whole policy:

- **Every purchase is accretive.** A deed bought at list price converts $1 of
  cash into $2.50 of score, and completing a color group re-prices every deed
  already held in it from `2.5x` to `5.0x`.
- **Every liquidation is expensive, and unequally so.** Mortgaging an ordinary
  deed destroys 2.5 net worth per dollar raised; selling it to the bank
  destroys 5.0; breaking a hotel destroys 11.0. Liquidating in the wrong order
  is a large, silent loss.

`ASU_FROZEN_TEACHER` maximizes `M_assets + R_short + R_long + M_monopoly`,
where `M_assets` is list price plus development *cost* and cash is excluded by
design (`ASU_FROZEN_TEACHER/spec.py`). That objective is a defensible reading
of the ASU papers, but it is not the simulator's scoring rule: it undervalues a
completed group, undervalues development, and prices cash at zero right up to
the moment cash decides the game. The challenger scores every decision in the
engine's own units instead. The measured caveat: in elimination-decided games
the teacher's cash discipline beats raw net-worth maximization; scoring rule
exactness decides capped games, solvency decides eliminated ones.

## Policies

- `slayer-v1` (`policy.py`) — deterministic, analytic, roughly 60x faster per
  decision than `asu_value_v1`. It ranks every way to spend a dollar by the net
  worth it buys, gated by a solvency reserve set at a high quantile of the rent
  it may owe on the next turn: near zero on an empty board, and rising by
  itself as opponents develop.
- `slayer-rollout-v1` (`search.py`) — truncated common-random-number rollouts
  driven by `slayer-v1` at every seat, scored at the leaf by `equity()`
  (advantage over the strongest live opponent). It only searches contested
  shortlists, so the greedy policy handles the many decisions that have one
  sensible answer.

Both are pure functions of the environment. Search clones the environment and
restores Python, NumPy, and Torch RNG state, so it never perturbs the caller.

## What the policy does differently

| Situation | `slayer-v1` | `asu_value_v1` |
| --- | --- | --- |
| Unowned deed it can afford | Buys whenever the group-aware gain exceeds the price | Buys under its own value gates |
| Monopoly in hand | Builds to hotels; gain per dollar rises from 0.5x to 4.5x | Builds to maximize projected rent |
| Raising cash under debt | Cheapest net worth per dollar: mortgage, then houses, then bank sale | Orders by its own value function |
| Auction | Bids to a fraction of the true group-aware gain, ascending in coarse steps | Bids the largest increment under its marginal value |
| In jail | Stays while the board is expensive, leaves while it is cheap | Decides from its value function |
| Incoming trade | Accepts only when our net-worth gain beats the proposer's | Requires both parties to gain in ASU units |
| Making offers | Deed-for-deed swaps only, and only when both gain and we gain more | Offers cash or swaps under its own value gates |

### What the instrumented games changed

Two games were logged decision by decision against three `asu-value-v1` seats.
Three defects surfaced, and each is now pinned by a regression test:

- **90 cash-for-deed offers, 0 accepted.** An opponent holding a deed prices it
  above list, and no legal cash level (0.75x, 1.0x, 1.25x) reaches that. In the
  won game these offers consumed a fifth of the agent's decisions. Only swaps
  are proposed now.
- **A flat reserve was added while deeds were unowned.** It peaked on an empty
  board, which is exactly when rent is near zero and buying is safest and
  cheapest. Removed in favor of the rent quantile.
- **Mortgage capacity counted toward the reserve**, which created a trap: a
  strict reserve blocked purchases, owning nothing removed the credit, and the
  gate tightened until the agent never invested and died with zero net worth.
  Every conservative point in an earlier sweep lost all of its games this way.
  The reserve now looks only at real cash.

## Usage

The repository's seat-balanced evaluator accepts both identifiers:

```bash
python -m ASU_FROZEN_TEACHER.evaluate \
  --focus slayer-v1 \
  --opponents asu-value-v1 asu-value-v1 asu-value-v1 \
  --seeds 0 1 2 3 4 5 --pretty
```

`benchmark.py` runs a lineup suite and, for each lineup, a paired control in
which `asu-value-v1` occupies the focus seat on the same seeds:

```bash
python -m ASU_SLAYER.benchmark --seeds 0 1 2 3 --output artifacts/slayer.json
```

```python
from ASU_SLAYER import SlayerV1

agent = SlayerV1(player_id=0)
action = agent.choose_action(env)
```

`SlayerConfig` and `SearchConfig` are frozen dataclasses; use `evolve()` to
sweep a weight without mutating the shared default.

## Scope and honesty

These are `ppo-plus-v2` results. The `2.5x` / `5.0x` net-worth multipliers and
the 200-round cap are research controls in this repository, not traditional
Monopoly rules, so the margin reported in `TRAINING_RESULTS.md` measures play
against this simulator's scoring rule and does not transfer to official
Monopoly or to either ASU paper's reported figures. The frozen teacher is not
weakened or modified in any way: `ASU_FROZEN_TEACHER/spec.py` and its hash are
untouched, and the only edit to that package registers two new policy IDs in
its evaluation CLI.
