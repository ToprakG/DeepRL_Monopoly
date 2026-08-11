# Anti-ASU Oracle: independent evaluator + offline MCTS + distillation

Date: 2026-08-11
Status: approved, not yet implemented
Deadline: submission Friday 2026-08-14

## Motivation

Competition rule 5: overall win rate across all opponents (generalization) wins,
not head-to-head record against any single one. Since most entrants will
converge on cloning or lightly patching ASU, an independently-designed,
genuinely differentiated evaluator is the highest-leverage lever available.

Target: fix four specific weaknesses in ASU's frozen value function
(`ASU_FROZEN_TEACHER`, `V(s) = M_assets + R_short + R_long + M_monopoly`):

1. `M_assets` deliberately excludes cash.
2. No visible round-200 endgame-race awareness.
3. `M_monopoly` rewards own-completion but has no explicit opponent-denial term.
4. `R_long` assumes a uniform distribution over 28 deeds rather than real
   2d6 landing probabilities.

## Compliance constraints (from the competition rules)

- Rule 1: using ASU verbatim is forbidden.
- Rule 2: playing against ASU is fine; cloning ASU's outputs is forbidden.
- Rule 5: scored on overall win rate across all opponents, not most-matches.

Consequence for this design: **zero imports from `ASU_FROZEN_TEACHER`**
anywhere in the new code. Everything is built directly on
`monopoly_game_engine` primitives (the shared simulator every entrant uses,
not ASU's reconstruction).

**Known separate issue, out of scope for this work**: `asu_plus/agent.py`'s
`ASUPlusV1` subclasses `ASU_FROZEN_TEACHER.core.ASUValueV1` directly and its
value function is ASU's own `evaluate_value(...)` plus four additive terms —
this is verbatim reuse of ASU's class and formula, a real exposure under
rule 1. Flagged for awareness; not addressed by this design, which exists
specifically as the compliant alternative.

## Milestones

- **M1 (priority)**: a standalone, independently-built oracle policy
  (`OracleV1`) that plugs into the existing seat-balanced h2h eval harness
  (the same pattern as `asu_plus/eval_h2h.py`) as a new lineup entrant, and
  beats ASU by the same statistical criterion already used for `asu-plus-v1`:
  `beats_asu = wilson_95_lower_bound(oracle_win_rate) > asu_point_win_rate`.
- **M2 (stretch, only after M1 is validated)**: distill the oracle's MCTS
  search into a `ppo-plus-v2` format-3 checkpoint (same architecture as the
  existing DDQN/PPO checkpoints) so it's a real warm start — loadable by
  `tools/play_game.py`, resumable by `tools/train_and_save.py --resume`.

Compute: local machine (22 cores) now; user plans to add Colab Pro (GPU)
later. Design must not hardcode assumptions that only hold at either scale —
worker/simulation counts are config, not constants.

## Architecture

New top-level package `oracle/`, structurally parallel to `asu_plus/` but
with no dependency on `ASU_FROZEN_TEACHER`:

```
oracle/
  board_model.py   # 2d6 Markov landing-probability model
  value.py         # OracleValueV1 heuristic
  simulate.py      # deepcopy + RNG-isolation sim wrapper (independent reimpl)
  rollout.py       # Approach A: truncated-rollout search (built first)
  mcts.py          # Approach B: tree MCTS (built second, upgrades rollout)
  agent.py         # OracleV1 — the eval-harness-pluggable policy
  distill/
    generate_selfplay.py   # M2: self-play data via mcts.py
    train.py                # M2: supervised distillation training
tests/
  test_oracle_value.py
  test_oracle_board_model.py
  test_oracle_simulate.py
  test_oracle_mcts.py
```

Search strategy is staged, not built as tree-MCTS from the start: build the
value function once, wrap it first in `rollout.py` (Approach A) to get a
legal, testable, pluggable policy fast (protects against the worst outcome —
nothing shippable by Friday), then upgrade to `mcts.py` (Approach B) for the
real ask — visit-count policy targets needed for a proper M2 distillation —
if time allows. A is a valid, submittable M1 on its own if B doesn't land in
time.

## Value function (`oracle/value.py`)

```
V(p, s) = NW(p)
        + w_endgame * alpha(round) * (NW(p) - NW(best_opponent))
        + w_block   * Denial(p)
        + w_income  * Income(p)
        + w_liability * Liability(p)
```

- **`NW(p)`** = `env.players[p].net_worth()` (`monopoly_game_engine/state.py:95-97`)
  — the engine's own ground-truth win-condition formula, used directly rather
  than reapproximated. Already weights completed-monopoly holdings at 2x
  undeveloped value and folds in house/hotel development cost. Fixes the
  cash-exclusion weakness by construction, since it's the literal quantity
  that decides who wins at the round-200 cap.
- **`alpha(round) = clamp01((round - 160) / 40)`** — ramps 0 to 1 across
  rounds 160-200 (independently parameterized; not copied from `asu_plus`'s
  `endgame_start_round=120`). Blends "grow absolute net worth" (early game)
  into "maximize the lead that actually decides the round-200 cap" (late
  game).
- **`Denial(p)`** — for every color group `g` and every opponent `o` with
  `theirs_g(o) < n_g`:
  ```
  group_rent_ceiling(g) * (theirs_g(o) / n_g) * (mine_g / (n_g - theirs_g(o)))
  ```
  summed over all `(g, o)` pairs. `group_rent_ceiling(g)` is the
  hotel-level/full rent value for the group, computed from engine rent data,
  independently implemented. This generalizes the "exactly one blocker"
  special case in `asu_plus/value.py`'s `blocking_bonus` into a continuous
  proximity x my-share-of-what's-left formula that also captures partial
  denial (I hold 2 of the 3 deeds an opponent needs, not just the literal
  last one).
- **`Income(p)` / `Liability(p)`** — expected rent collected / paid, each
  square weighted by its stationary landing probability from
  `board_model.py`. That model solves the actual transition structure of
  this ruleset: 2d6 combinatorics for normal movement, plus the two
  confirmed position overrides (Go-To-Jail at square 30 -> 10, and the
  triple-doubles-to-jail override which bypasses normal dice movement).
  Chance and Community Chest are confirmed true no-ops in this ruleset
  (dead, unimported card data) and need no modeling. `Income` replaces
  ASU's uniform-over-28-deeds `R_long` assumption. `Liability` (expected
  rent I'll pay landing on opponents' developed groups) has no equivalent
  in ASU's formula — it's a new term.
- **Terminal states**: `V = NW(p)` if the player isn't bankrupt (continuous
  with the anchor term, same scale, no artificial jump at the terminal
  boundary), else a large negative constant.
- **No hand-coded bankruptcy safety gate.** ASU uses two frozen static gates
  on discretionary spending. This design deliberately omits an equivalent:
  a bankruptcy-inducing line already scores badly once its simulated
  terminal is reached (via the terminal-value rule above), so risk-avoidance
  emerges from search lookahead rather than a separate static formula. This
  only holds once real search (rollout or MCTS) is wrapped around the value
  function — noted as a dependency, not a free assumption.

## Search

### `oracle/simulate.py`

Independent reimplementation of the deepcopy-per-node + save/swap/restore
pattern needed because `monopoly_game_engine.MonopolyEnv` provides no
`copy()`/`clone()` method (confirmed — plain Python attributes, structurally
deepcopy-safe but not engine-provided) and dice rolls use the global stdlib
`random` module directly (`env.py:615`, no per-instance RNG). Each simulated
step: `copy.deepcopy(env)` once per branch, then around every `env.step()`
call, swap in a private `random` state, run the step, capture the resulting
state, restore the caller's global `random` state — identical engineering
necessity to what `ASU_FROZEN_TEACHER`'s rollout does, written from scratch
against the stdlib directly (not imported). A `preserve_global_rng()`
context manager wraps the entire search call so the live game's Python,
NumPy, and Torch RNG streams are never perturbed by search activity.

### `oracle/rollout.py` (Approach A — built first)

Value function plus multi-seed truncated rollouts to a configurable depth.
No tree bookkeeping. Fast to build and validate; gets `OracleV1` to a legal,
pluggable, testable state quickly.

### `oracle/mcts.py` (Approach B — built second)

Tree search over `get_allowed_actions(pid)` (`env.py:195-300`) — already a
cheap generative enumerator of only legal actions (typically single digits
to low tens at normal decision points, occasionally low hundreds at
trade-heavy late-game states), no 2958-wide mask ever materialized.

- **Selection**: UCB1 over children, with a running min/max rescale of
  observed values into [0,1] for the exploration term (our value scale
  isn't naturally bounded like standard UCT assumes).
- **Backup**: "maxn"-style multiplayer backup — each node accumulates a
  value **vector** across all 4 players (not a scalar), since this is a
  non-zero-sum game. Selection at a node uses the acting player's own
  component of the child's mean value vector.
- **Leaf evaluation**: `V(s)` directly, optionally preceded by a short
  truncated rollout for variance reduction at shallow depths (the general
  "truncated rollout + evaluator" pattern is from published literature
  cited in `ASU_FROZEN_TEACHER/README.md`, not ASU's specific frozen
  implementation — the pattern itself is fair game, only ASU's exact
  budget/weights/seeds are off-limits).
- **Root action selection**: most-visited child (robust child). The
  normalized visit-count distribution over root children is retained as the
  policy target for M2, for free.
- **Config**: `simulations` (200 for M1 iteration speed; 1000+ for offline
  M2 label generation — this is offline and can be arbitrarily slow per the
  original framing), `c_puct`, `rollout_depth`, `max_tree_depth`.
- **Parallelization**: across games via `multiprocessing.Pool`, mirroring
  the pattern already proven in `asu_plus/eval_h2h.py:182` (one full game
  per worker) rather than within-tree parallelism (root parallelization /
  virtual loss) — deliberately the simpler, lower-risk option given the
  Friday deadline.

## Distillation pipeline (M2, stretch)

Reuse `DDQNNetwork` (`monopoly_game_engine/networks.py:122`,
`STATE_DIM -> 1024 -> ReLU -> 512 -> ReLU -> ACTION_SPACE_SIZE`) — the exact
architecture `tools/train_and_save.py` already produces format-3
`ppo-plus-v2` checkpoints for. Self-play games run the M1 oracle at full
MCTS budget and record, at every decision:

```
(observation[300], policy_target[2958, legal-masked, from visit counts], value_target[4])
```

Train with cross-entropy (policy) + MSE (value) loss, Adam, save as a
format-3 checkpoint (`ruleset="ppo-plus-v2"`, `state_dim=300`,
`action_dim=2958`) so it's loadable by `tools/play_game.py` and resumable by
`tools/train_and_save.py --algo ddqn --resume`.

**Explicit risk**: distillation can lose fidelity relative to the raw
oracle. The resulting checkpoint must be re-validated against `beats_asu`
after training — a mediocre distillation misleads the "warm start" claim and
must not be assumed to inherit the oracle's win rate.

## Testing

- `board_model.py`: stationary distribution sums to 1.0; sanity check that
  Jail-adjacent squares (Illinois Ave / B&O-type positions) come out
  high-probability, matching the well-known qualitative Monopoly result.
- `value.py`: hand-built minimal states per term — cash-only state gives
  `V ~ cash`; a completed-monopoly state reflects the 2x multiplier baked
  into `net_worth()`; a state where an opponent is missing exactly one deed
  of a group I hold produces nonzero `Denial`.
- `simulate.py`: assert zero state mutation leaks into the caller's original
  `env`, and that global `random`/NumPy/Torch RNG state is bit-identical
  before and after a full search call.
- `mcts.py`: fixed-seed smoke test — visit counts sum consistently across
  the tree, terminal states short-circuit expansion without further
  simulation.
- **M1 acceptance test**: plug `OracleV1` into the same seat-balanced h2h
  harness and lineup used for `asu-plus-v1` (`asu_plus/eval_h2h.py`), apply
  the identical `beats_asu = wilson_lower > asu_point_rate` criterion.

## Error handling

`get_allowed_actions` already enforces legality engine-side, so unlike
ASU's fixed-policy adapters, `OracleV1` never needs an illegal-action
fallback — rollout/MCTS only ever expand from the legal set already
returned. Deepcopy-per-simulation is confirmed structurally safe
(`MonopolyEnv` has no C-extension or file-handle state) but is the likely
performance bottleneck at high simulation budgets; flagged as a stretch
optimization (custom lightweight snapshot/restore instead of full deepcopy),
not required for M1 correctness.

## Open risks

- MCTS (Approach B) may not land in time before Friday — Approach A is a
  deliberately valid fallback submission on its own.
- M2 (distillation) is explicitly lower priority than M1; if the deadline
  forces a cut, M1 ships without M2.
- Value function weights (`w_endgame`, `w_block`, `w_income`, `w_liability`,
  `alpha`'s start round) are defaults pending empirical tuning via the same
  ablation-style methodology `asu_plus` uses (independently implemented, not
  imported) — not expected to be correct on the first pass.
