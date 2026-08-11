# Anti-ASU Oracle — Milestone 1 (Rollout Search) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone, independently-designed Monopoly evaluator (`OracleV1`) that fixes ASU's four named weaknesses (cash exclusion, round-200 endgame blindness, no denial/blocking term, uniform-rent assumption), wrap it in a truncated-rollout search, and validate it beats ASU (`wilson_lower_bound(oracle_win_rate) > asu_point_win_rate`) via a seat-balanced head-to-head harness.

**Architecture:** New top-level package `oracle/`, built entirely on `monopoly_game_engine` primitives with zero dependency on `ASU_FROZEN_TEACHER`'s evaluator. Six modules, each independently testable: a Markov landing-probability model, the value heuristic, a private-simulation wrapper (deepcopy + RNG isolation), a rollout search, the policy class wrapping search, and an h2h eval harness that pits it against ASU using the repo's existing shared evaluation infrastructure (ASU is a legal *opponent*, never a code dependency of the oracle itself).

**Tech Stack:** Python 3, `numpy` (already a repo dependency), stdlib `random`/`copy`/`multiprocessing`, `unittest` (matches existing test style in `tests/test_monopoly_regression.py` and `tests/test_asu_frozen_teacher.py`), `pytest` as the test runner.

## Global Constraints

- **Zero imports from `ASU_FROZEN_TEACHER` anywhere under `oracle/`** (competition rule 1: using ASU verbatim is forbidden). The only place `ASU_FROZEN_TEACHER` may be imported is `oracle/eval_h2h.py`, and only to (a) run ASU as an *opponent* via the shared `_run_game`/`wilson_interval` infra (competition rule 2 explicitly allows playing against ASU) and (b) reuse `ASUValueV1` as that opponent instance. Never import anything from `ASU_FROZEN_TEACHER.core`/`.types` into `oracle/value.py`, `oracle/rollout.py`, `oracle/simulate.py`, or `oracle/agent.py`.
- **Acceptance criterion** (from `docs/superpowers/specs/2026-08-11-anti-asu-oracle-design.md`): `beats_asu = wilson_95_lower_bound(oracle_win_rate) > asu_point_win_rate`, computed by the same seat-balanced, seed-rotated harness pattern already proven in `asu_plus/eval_h2h.py`.
- **Deadline**: competition submission is Friday 2026-08-14. This plan covers Milestone 1 only (the rollout-search oracle); the MCTS upgrade (Approach B) and checkpoint distillation (Milestone 2) are separate follow-on plans, written only after this one ships and is validated — do not start on them from inside this plan.
- Every new source file starts with `from __future__ import annotations` and uses `@dataclass(frozen=True, slots=True)` for config/weight containers, matching the existing style in `asu_plus/value.py`.
- Run all commands from the repository root: `C:\Users\iefey\Documents\GitHub\DeepRL_Monopoly`.

---

### Task 1: Package scaffold + board landing-probability model

**Files:**
- Create: `oracle/__init__.py`
- Create: `oracle/board_model.py`
- Test: `tests/test_oracle_board_model.py`

**Interfaces:**
- Produces: `oracle.board_model.NUM_SQUARES: int` (=40), `oracle.board_model.JAIL_SQUARE: int` (=10), `oracle.board_model.landing_probability(square: int) -> float`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_oracle_board_model.py`:

```python
from __future__ import annotations

import unittest

from oracle.board_model import JAIL_SQUARE, NUM_SQUARES, landing_probability


class BoardModelTests(unittest.TestCase):
    def test_distribution_sums_to_one(self) -> None:
        total = sum(landing_probability(sq) for sq in range(NUM_SQUARES))
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_jail_square_is_the_single_most_likely_landing(self) -> None:
        jail_probability = landing_probability(JAIL_SQUARE)
        for square in range(NUM_SQUARES):
            if square == JAIL_SQUARE:
                continue
            self.assertGreater(jail_probability, landing_probability(square))

    def test_illinois_and_bo_railroad_above_uniform(self) -> None:
        # Square 24 = Illinois Ave, square 25 = B&O Railroad. Both sit in the
        # "jail feeder" cluster (roughly a typical dice roll past Jail at
        # square 10) and are above the uniform 1/40 baseline in the real
        # transition structure of this ruleset.
        uniform = 1.0 / NUM_SQUARES
        self.assertGreater(landing_probability(24), uniform)
        self.assertGreater(landing_probability(25), uniform)

    def test_square_index_out_of_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            landing_probability(40)
        with self.assertRaises(ValueError):
            landing_probability(-1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_oracle_board_model.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'oracle'`)

- [ ] **Step 3: Write the implementation**

Create `oracle/__init__.py` (empty file — marks `oracle` as a package).

Create `oracle/board_model.py`:

```python
"""Exact 2d6 landing-probability model for this ruleset's board.

Independently derived from the actual transition rules of
monopoly_game_engine (not ASU's rollout horizons): normal 2d6 movement,
the extra-roll-on-doubles mechanic, the three-consecutive-doubles-to-jail
override, and the Go-To-Jail square teleport. Chance and Community Chest
squares are confirmed no-ops in this ruleset (monopoly_game_engine/env.py
never imports the card tables) and need no special-casing — landing there
has no further effect, same as Free Parking.

Simplification, documented deliberately: a player who is "in jail" is
modeled as simply occupying square 10 for the purposes of this stationary
distribution. The pay/roll/card choice to leave jail doesn't change which
square a player resumes normal movement from (still square 10), so the
steady-state landing distribution this module computes is unaffected by
that simplification.
"""

from __future__ import annotations

import numpy as np

NUM_SQUARES = 40
GO_TO_JAIL_SQUARE = 30
JAIL_SQUARE = 10


def _turn_move_outcomes() -> list[tuple[float, int, bool]]:
    """Enumerate (probability, squares_moved, sent_to_jail) for one full turn.

    A turn is one to three dice rolls: rolling doubles grants another roll,
    but a third consecutive double sends the player directly to jail instead
    of moving by that roll's total (monopoly_game_engine/env.py:635-644).
    """

    outcomes: list[tuple[float, int, bool]] = []

    def recurse(prob: float, moved: int, doubles_count: int) -> None:
        for d1 in range(1, 7):
            for d2 in range(1, 7):
                p = prob / 36.0
                is_double = d1 == d2
                total = d1 + d2
                if is_double and doubles_count == 2:
                    outcomes.append((p, moved, True))
                elif is_double:
                    recurse(p, moved + total, doubles_count + 1)
                else:
                    outcomes.append((p, moved + total, False))

    recurse(1.0, 0, 0)
    return outcomes


_TURN_OUTCOMES = _turn_move_outcomes()


def build_transition_matrix() -> np.ndarray:
    """40x40 row-stochastic matrix: matrix[s, s'] = P(land on s' | start turn on s)."""

    matrix = np.zeros((NUM_SQUARES, NUM_SQUARES), dtype=np.float64)
    for start in range(NUM_SQUARES):
        for prob, moved, sent_to_jail in _TURN_OUTCOMES:
            if sent_to_jail:
                dest = JAIL_SQUARE
            else:
                dest = (start + moved) % NUM_SQUARES
                if dest == GO_TO_JAIL_SQUARE:
                    dest = JAIL_SQUARE
            matrix[start, dest] += prob
    return matrix


def stationary_distribution(
    matrix: np.ndarray, iterations: int = 5000, tol: float = 1e-14
) -> np.ndarray:
    """Solve for the stationary distribution via power iteration."""

    pi = np.full(NUM_SQUARES, 1.0 / NUM_SQUARES, dtype=np.float64)
    for _ in range(iterations):
        next_pi = pi @ matrix
        if np.max(np.abs(next_pi - pi)) < tol:
            pi = next_pi
            break
        pi = next_pi
    return pi / pi.sum()


_TRANSITION_MATRIX = build_transition_matrix()
_LANDING_PROBABILITY = stationary_distribution(_TRANSITION_MATRIX)


def landing_probability(square: int) -> float:
    """Steady-state probability that a player's turn ends on this square."""

    if not 0 <= square < NUM_SQUARES:
        raise ValueError(f"square must be in [0, {NUM_SQUARES - 1}]")
    return float(_LANDING_PROBABILITY[square])


__all__ = [
    "NUM_SQUARES",
    "JAIL_SQUARE",
    "GO_TO_JAIL_SQUARE",
    "build_transition_matrix",
    "stationary_distribution",
    "landing_probability",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_oracle_board_model.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add oracle/__init__.py oracle/board_model.py tests/test_oracle_board_model.py
git commit -m "$(cat <<'EOF'
Add oracle package scaffold and 2d6 landing-probability model

Independently derived stationary distribution over board squares,
replacing ASU's uniform-over-28-deeds R_long assumption with real
transition probabilities (doubles, triple-doubles-to-jail, Go-To-Jail).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Value function (`OracleValueV1`)

**Files:**
- Create: `oracle/value.py`
- Test: `tests/test_oracle_value.py`

**Interfaces:**
- Consumes: `oracle.board_model.landing_probability(square: int) -> float` (Task 1).
- Produces: `oracle.value.OracleWeights` (dataclass), `oracle.value.evaluate(env, player_id: int, weights: OracleWeights | None = None) -> float`, `oracle.value.net_worth(env, player_id: int) -> float`, `oracle.value.denial_value(env, player_id: int) -> float`, `oracle.value.income_value(env, player_id: int) -> float`, `oracle.value.liability_value(env, player_id: int) -> float`, `oracle.value.TERMINAL_BANKRUPT_VALUE: float`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_oracle_value.py`:

```python
from __future__ import annotations

import random
import unittest

from monopoly_game_engine.env import MonopolyEnv

from oracle.value import (
    OracleWeights,
    TERMINAL_BANKRUPT_VALUE,
    denial_value,
    evaluate,
    income_value,
    liability_value,
)


class OracleValueTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(7)
        self.env = MonopolyEnv(agent_ids=[0], max_rounds=5)
        self.env.turn_order = [0, 1, 2, 3]
        self.env.current_turn_idx = 0

    def give_property(self, square: int, pid: int) -> None:
        prop = self.env.properties[square]
        prop.owner = pid
        self.env.players[pid].properties.append(prop)
        self.env._update_monopolies()

    def test_cash_only_state_matches_net_worth(self) -> None:
        for player in self.env.players:
            player.cash = 1500
        flat_weights = OracleWeights(w_endgame=0.0, w_block=0.0, w_income=0.0, w_liability=0.0)
        value = evaluate(self.env, 0, flat_weights)
        self.assertAlmostEqual(value, 1500.0)

    def test_denial_value_positive_when_one_deed_from_opponent_monopoly(self) -> None:
        # Brown group is squares 1 (Mediterranean) and 3 (Baltic).
        self.give_property(1, 0)
        self.give_property(3, 1)
        self.assertGreater(denial_value(self.env, 0), 0.0)

    def test_denial_value_zero_when_opponent_has_no_partial_group(self) -> None:
        self.give_property(1, 0)
        self.assertEqual(denial_value(self.env, 0), 0.0)

    def test_income_value_positive_for_owned_property(self) -> None:
        self.give_property(1, 0)
        self.give_property(3, 0)
        self.assertGreater(income_value(self.env, 0), 0.0)

    def test_liability_value_positive_for_opponent_developed_property(self) -> None:
        self.give_property(1, 1)
        self.give_property(3, 1)
        self.env.properties[1].houses = 5
        self.assertGreater(liability_value(self.env, 0), 0.0)

    def test_bankrupt_player_gets_terminal_penalty(self) -> None:
        self.env.players[0].bankrupt = True
        self.assertEqual(evaluate(self.env, 0), TERMINAL_BANKRUPT_VALUE)

    def test_endgame_margin_ramps_toward_late_game(self) -> None:
        self.env.players[0].cash = 2000
        self.env.players[1].cash = 1000
        self.env.round = 199
        late_value = evaluate(self.env, 0)
        self.env.round = 0
        early_value = evaluate(self.env, 0)
        self.assertGreater(late_value, early_value)

    def test_invalid_player_id_raises(self) -> None:
        with self.assertRaises(ValueError):
            evaluate(self.env, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_oracle_value.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'oracle.value'`)

- [ ] **Step 3: Write the implementation**

Create `oracle/value.py`:

```python
"""OracleValueV1: independent Monopoly value heuristic.

Built entirely on monopoly_game_engine primitives. No dependency on
ASU_FROZEN_TEACHER anywhere in this module (competition rule 1: using ASU
verbatim is forbidden) — the four terms below independently fix the four
named ASU weaknesses:

  - cash exclusion       -> anchor term is Player.net_worth(), which
                             already includes cash (state.py:95-97).
  - round-200 endgame race -> endgame_margin ramps toward maximizing the
                             net-worth lead as `round` approaches 200.
  - no denial/blocking term -> denial_value generalizes "I hold the last
                             deed an opponent needs" into a continuous
                             proximity x my-share-of-remaining formula.
  - uniform rent assumption -> income_value / liability_value weight
                             expected rent by the real per-square landing
                             probability from oracle.board_model, not a
                             uniform assumption over 28 deeds.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from monopoly_game_engine.constants import COLOR_GROUPS, NUM_PLAYERS

from .board_model import landing_probability

TERMINAL_BANKRUPT_VALUE = -1_000_000.0
EXPECTED_DICE_TOTAL = 7.0  # E[2d6]


@dataclass(frozen=True, slots=True)
class OracleWeights:
    w_endgame: float = 1.0
    endgame_start_round: int = 160
    endgame_end_round: int = 200
    w_block: float = 1.0
    w_income: float = 1.0
    w_liability: float = 1.0


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def net_worth(env, player_id: int) -> float:
    return float(env.players[player_id].net_worth())


def group_rent_ceiling(env, color: str) -> float:
    """Max achievable rent for one landing on this color group.

    Uses a throwaway deep-copied Property so the live game state is never
    mutated, then queries the engine's own Property.get_rent() at maximum
    development (hotel for real estate, full ownership count for
    railroads/utilities) — reuses the engine's authoritative rent formula
    rather than re-deriving it.
    """

    squares = COLOR_GROUPS[color]
    sample = copy.deepcopy(env.properties[squares[0]])
    if color == "railroad":
        return float(sample.get_rent(num_railroads=len(squares)))
    if color == "utility":
        return float(sample.get_rent(dice_roll=EXPECTED_DICE_TOTAL, num_utilities=len(squares)))
    sample.is_monopoly = True
    sample.houses = 5
    return float(sample.get_rent())


def denial_value(env, player_id: int) -> float:
    """Value of holding deeds that deny opponents' group completions.

    For every color group and every opponent who owns part but not all of
    it: (rent ceiling of the group) x (how close the opponent already is
    to completing it) x (what fraction of their still-needed deeds I hold).
    """

    total = 0.0
    for color, squares in COLOR_GROUPS.items():
        size = len(squares)
        if size < 2:
            continue
        owners = [env.properties[sq].owner for sq in squares]
        mine = sum(1 for owner in owners if owner == player_id)
        if mine == 0:
            continue
        ceiling = group_rent_ceiling(env, color)
        for opponent_id in range(NUM_PLAYERS):
            if opponent_id == player_id or env.players[opponent_id].bankrupt:
                continue
            theirs = sum(1 for owner in owners if owner == opponent_id)
            if theirs == 0 or theirs >= size:
                continue
            proximity = theirs / size
            remaining = size - theirs
            my_share = mine / remaining
            total += ceiling * proximity * my_share
    return total


def _expected_rent(env, square: int) -> float:
    prop = env.properties[square]
    if prop.owner is None or prop.mortgaged:
        return 0.0
    owner = env.players[prop.owner]
    return float(
        prop.get_rent(
            dice_roll=EXPECTED_DICE_TOTAL,
            num_railroads=owner.railroads_owned(),
            num_utilities=owner.utilities_owned(),
        )
    )


def income_value(env, player_id: int) -> float:
    """Expected rent income from my own properties, weighted by real
    per-square landing probability (replaces ASU's uniform-lap assumption)."""

    total = 0.0
    for square, prop in env.properties.items():
        if prop.owner != player_id:
            continue
        total += landing_probability(square) * _expected_rent(env, square)
    return total


def liability_value(env, player_id: int) -> float:
    """Expected rent I'd pay landing on opponents' developed properties.

    Has no equivalent in ASU's formula, which only models prospective
    income, not prospective payments.
    """

    total = 0.0
    for square, prop in env.properties.items():
        if prop.owner is None or prop.owner == player_id:
            continue
        total += landing_probability(square) * _expected_rent(env, square)
    return total


def endgame_margin(env, player_id: int, weights: OracleWeights) -> float:
    own = net_worth(env, player_id)
    opponents = [
        net_worth(env, player.player_id)
        for player in env.players
        if player.player_id != player_id and not player.bankrupt
    ]
    best_opponent = max(opponents, default=0.0)
    span = max(1, weights.endgame_end_round - weights.endgame_start_round)
    alpha = _clamp01((float(env.round) - weights.endgame_start_round) / span)
    return alpha * (own - best_opponent)


def evaluate(env, player_id: int, weights: OracleWeights | None = None) -> float:
    if not 0 <= player_id < NUM_PLAYERS:
        raise ValueError(f"player_id must be in [0, {NUM_PLAYERS - 1}]")
    if weights is None:
        weights = OracleWeights()

    player = env.players[player_id]
    if player.bankrupt:
        return TERMINAL_BANKRUPT_VALUE
    if env.done:
        return net_worth(env, player_id)

    value = net_worth(env, player_id)
    value += weights.w_endgame * endgame_margin(env, player_id, weights)
    value += weights.w_block * denial_value(env, player_id)
    value += weights.w_income * income_value(env, player_id)
    value -= weights.w_liability * liability_value(env, player_id)
    return value


__all__ = [
    "OracleWeights",
    "TERMINAL_BANKRUPT_VALUE",
    "net_worth",
    "group_rent_ceiling",
    "denial_value",
    "income_value",
    "liability_value",
    "endgame_margin",
    "evaluate",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_oracle_value.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add oracle/value.py tests/test_oracle_value.py
git commit -m "$(cat <<'EOF'
Add OracleValueV1 heuristic: cash-inclusive, endgame-aware, denial-aware

Anchors on Player.net_worth() directly (fixes cash exclusion), ramps a
net-worth-lead term across rounds 160-200 (fixes endgame blindness),
generalizes single-blocker denial into a continuous proximity formula,
and weights rent income/liability by real landing probability from
oracle.board_model (fixes the uniform-deed assumption). No
ASU_FROZEN_TEACHER dependency.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Private simulation wrapper (deepcopy + RNG isolation)

**Files:**
- Create: `oracle/simulate.py`
- Test: `tests/test_oracle_simulate.py`

**Interfaces:**
- Produces: `oracle.simulate.PrivateSim` (class: `__init__(self, env, seed: int)`, `.env` attribute, `.step(action: int)`), `oracle.simulate.preserve_global_rng()` (context manager).

- [ ] **Step 1: Write the failing test**

Create `tests/test_oracle_simulate.py`:

```python
from __future__ import annotations

import random
import unittest

from monopoly_game_engine.actions import ActionType
from monopoly_game_engine.env import MonopolyEnv

from oracle.simulate import PrivateSim, preserve_global_rng


class PrivateSimTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(11)
        self.env = MonopolyEnv(agent_ids=[0], max_rounds=5)
        self.env.turn_order = [0, 1, 2, 3]
        self.env.current_turn_idx = 0

    def test_step_does_not_mutate_caller_env(self) -> None:
        before_cash = self.env.players[0].cash
        before_position = self.env.players[0].position
        allowed = self.env.get_allowed_actions(self.env.whose_turn())
        sim = PrivateSim(self.env, seed=1)
        sim.step(allowed[0])
        self.assertEqual(self.env.players[0].cash, before_cash)
        self.assertEqual(self.env.players[0].position, before_position)

    def test_same_seed_gives_reproducible_dice_outcome(self) -> None:
        allowed = self.env.get_allowed_actions(self.env.whose_turn())
        roll_action = int(ActionType.ROLL_DICE)
        action = roll_action if roll_action in allowed else allowed[0]

        sim_a = PrivateSim(self.env, seed=42)
        sim_a.step(action)
        sim_b = PrivateSim(self.env, seed=42)
        sim_b.step(action)

        self.assertEqual(
            sim_a.env.players[0].position, sim_b.env.players[0].position
        )

    def test_preserve_global_rng_restores_state_exactly(self) -> None:
        before = random.getstate()
        with preserve_global_rng():
            random.random()
            random.randint(1, 100)
        after = random.getstate()
        self.assertEqual(before, after)

    def test_preserve_global_rng_restores_on_exception(self) -> None:
        before = random.getstate()
        with self.assertRaises(RuntimeError):
            with preserve_global_rng():
                random.random()
                raise RuntimeError("boom")
        after = random.getstate()
        self.assertEqual(before, after)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_oracle_simulate.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'oracle.simulate'`)

- [ ] **Step 3: Write the implementation**

Create `oracle/simulate.py`:

```python
"""Private simulation wrapper: deepcopy + RNG isolation for search.

monopoly_game_engine.MonopolyEnv provides no clone()/copy() method and its
dice rolls read the global stdlib `random` module directly
(monopoly_game_engine/env.py, `random.randint(1, 6)`, no per-instance RNG).
Search needs many independent forward branches from the same live state
without mutating it or perturbing the caller's RNG stream. This is an
independent, from-scratch reimplementation of that engineering necessity —
not a reuse of ASU_FROZEN_TEACHER's rollout code.
"""

from __future__ import annotations

import contextlib
import copy
import random
from typing import Any


class PrivateSim:
    """An isolated forward-simulation branch cloned from a live env."""

    def __init__(self, env: Any, seed: int) -> None:
        self.env = copy.deepcopy(env)
        self._rng_state = random.Random(seed).getstate()

    def step(self, action: int) -> Any:
        outer_state = random.getstate()
        random.setstate(self._rng_state)
        try:
            result = self.env.step(action)
        finally:
            self._rng_state = random.getstate()
            random.setstate(outer_state)
        return result


@contextlib.contextmanager
def preserve_global_rng():
    """Snapshot and restore Python's global random state around a block."""

    outer_state = random.getstate()
    try:
        yield
    finally:
        random.setstate(outer_state)


__all__ = ["PrivateSim", "preserve_global_rng"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_oracle_simulate.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add oracle/simulate.py tests/test_oracle_simulate.py
git commit -m "$(cat <<'EOF'
Add PrivateSim: independent deepcopy + RNG-isolation search wrapper

Engine has no clone() and dice rolls hit the global random module
directly, so every search branch needs its own isolated copy + private
RNG stream that never leaks into the live game. Written from scratch
against the stdlib, not imported from ASU_FROZEN_TEACHER.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Rollout search (Approach A)

**Files:**
- Create: `oracle/rollout.py`
- Test: `tests/test_oracle_rollout.py`

**Interfaces:**
- Consumes: `oracle.simulate.PrivateSim`, `oracle.simulate.preserve_global_rng` (Task 3); `oracle.value.OracleWeights`, `oracle.value.evaluate` (Task 2).
- Produces: `oracle.rollout.RolloutConfig` (dataclass: `depth: int = 6`, `seeds: int = 3`, `base_seed: int = 0`), `oracle.rollout.choose_action(env, player_id: int, weights: OracleWeights, config: RolloutConfig | None = None) -> int`.

**Design note for the implementer:** the root decision (the action actually being chosen) is evaluated under `config.seeds` independent random seeds to capture dice variance — this is the "multi-seed" part. Each seed's continuation, and every intermediate decision made by *any* player during that continuation, uses a single deterministic one-ply-greedy lookahead (`_greedy_action`, seeded 0) purely to keep the search fast; only the root's branching captures seed variance. This is a deliberate speed/fidelity trade-off for Milestone 1, not an oversight — document it as a code comment, don't hide it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_oracle_rollout.py`:

```python
from __future__ import annotations

import random
import unittest

from monopoly_game_engine.env import MonopolyEnv

from oracle.rollout import RolloutConfig, choose_action
from oracle.value import OracleWeights


class RolloutTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(19)
        self.env = MonopolyEnv(agent_ids=[0], max_rounds=5)
        self.env.turn_order = [0, 1, 2, 3]
        self.env.current_turn_idx = 0

    def test_choose_action_returns_a_legal_action(self) -> None:
        player_id = self.env.whose_turn()
        allowed = self.env.get_allowed_actions(player_id)
        action = choose_action(
            self.env, player_id, OracleWeights(), RolloutConfig(depth=2, seeds=1)
        )
        self.assertIn(action, allowed)

    def test_choose_action_does_not_mutate_caller_env(self) -> None:
        player_id = self.env.whose_turn()
        before_cash = [p.cash for p in self.env.players]
        before_round = self.env.round
        choose_action(
            self.env, player_id, OracleWeights(), RolloutConfig(depth=3, seeds=2)
        )
        self.assertEqual([p.cash for p in self.env.players], before_cash)
        self.assertEqual(self.env.round, before_round)

    def test_choose_action_preserves_global_rng_state(self) -> None:
        player_id = self.env.whose_turn()
        before = random.getstate()
        choose_action(
            self.env, player_id, OracleWeights(), RolloutConfig(depth=2, seeds=2)
        )
        after = random.getstate()
        self.assertEqual(before, after)

    def test_single_legal_action_short_circuits(self) -> None:
        # Auction phase where the current bidder can't afford any increment
        # is the simplest guaranteed single-action (PASS-only) state; force
        # it directly instead of hunting for one via play.
        # (monopoly_game_engine/env.py:210-217: only auction_current_pid and
        # auction_high_bid affect get_allowed_actions in PHASE_AUCTION.)
        self.env.phase = "auction"
        self.env.auction_current_pid = 0
        self.env.auction_high_bid = 10_000_000  # unaffordable, only PASS legal
        allowed = self.env.get_allowed_actions(0)
        self.assertEqual(len(allowed), 1)
        action = choose_action(
            self.env, 0, OracleWeights(), RolloutConfig(depth=1, seeds=1)
        )
        self.assertEqual(action, allowed[0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_oracle_rollout.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'oracle.rollout'`)

- [ ] **Step 3: Write the implementation**

Create `oracle/rollout.py`:

```python
"""Approach A search: value function + multi-seed truncated rollouts.

No tree bookkeeping. For the decision actually being made, every legal
action is tried under several independent seeds (captures dice variance at
the root); each branch then continues for a bounded number of decisions
using a fast, single-seed one-ply-greedy policy for whoever is acting
(captures the *choice structure* of continued play cheaply, without paying
for full seed variance at every subsequent step — a deliberate Milestone-1
speed/fidelity trade-off). Every branch is scored with oracle.value.evaluate
from the root player's perspective.
"""

from __future__ import annotations

from dataclasses import dataclass

from .simulate import PrivateSim, preserve_global_rng
from .value import OracleWeights, evaluate


@dataclass(frozen=True, slots=True)
class RolloutConfig:
    depth: int = 6
    seeds: int = 3
    base_seed: int = 0


def _greedy_action(env, player_id: int, weights: OracleWeights, allowed: list[int]) -> int:
    """One-ply lookahead: try every legal action, keep the one with the best
    resulting value. Single fixed seed (common random numbers across
    candidates) since this is only used for fast rollout continuation, not
    the root decision."""

    if len(allowed) == 1:
        return allowed[0]
    best_action = allowed[0]
    best_value = float("-inf")
    for action in allowed:
        sim = PrivateSim(env, seed=0)
        sim.step(action)
        value = evaluate(sim.env, player_id, weights)
        if value > best_value:
            best_value = value
            best_action = action
    return best_action


def _play_out(env, root_player: int, weights: OracleWeights, depth: int) -> float:
    steps = 0
    while steps < depth and not env.done:
        actor = env.whose_turn()
        allowed = env.get_allowed_actions(actor)
        action = _greedy_action(env, actor, weights, allowed)
        env.step(action)
        steps += 1
    return evaluate(env, root_player, weights)


def choose_action(
    env,
    player_id: int,
    weights: OracleWeights,
    config: RolloutConfig | None = None,
) -> int:
    if config is None:
        config = RolloutConfig()
    allowed = env.get_allowed_actions(player_id)
    if not allowed:
        raise RuntimeError(f"no legal actions for player {player_id}")
    if len(allowed) == 1:
        return allowed[0]

    best_action = allowed[0]
    best_mean = float("-inf")
    with preserve_global_rng():
        for action in allowed:
            total = 0.0
            for seed_offset in range(config.seeds):
                sim = PrivateSim(env, seed=config.base_seed + seed_offset)
                sim.step(action)
                total += _play_out(sim.env, player_id, weights, config.depth)
            mean_value = total / config.seeds
            if mean_value > best_mean:
                best_mean = mean_value
                best_action = action
    return best_action


__all__ = ["RolloutConfig", "choose_action"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_oracle_rollout.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add oracle/rollout.py tests/test_oracle_rollout.py
git commit -m "$(cat <<'EOF'
Add Approach A rollout search over OracleValueV1

Multi-seed root branching plus fast greedy-continuation playouts,
isolated via PrivateSim/preserve_global_rng so search never mutates the
caller's game or RNG stream. Gets the oracle to a legal, pluggable,
testable policy ahead of the tree-MCTS upgrade.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: `OracleV1` policy

**Files:**
- Create: `oracle/agent.py`
- Test: `tests/test_oracle_agent.py`

**Interfaces:**
- Consumes: `oracle.rollout.RolloutConfig`, `oracle.rollout.choose_action` (Task 4); `oracle.value.OracleWeights` (Task 2).
- Produces: `oracle.agent.ORACLE_V1: str` (=`"oracle-v1"`), `oracle.agent.OracleV1` (class: `__init__(self, player_id: int, weights: OracleWeights | None = None, config: RolloutConfig | None = None)`, `.policy_id` class attribute, `.player_id`, `.weights`, `.config` attributes, `.choose_action(self, env) -> int`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_oracle_agent.py`:

```python
from __future__ import annotations

import random
import unittest

from monopoly_game_engine.agents_fixed import FPAgentA, FPAgentB, FPAgentC
from monopoly_game_engine.env import MonopolyEnv

from oracle.agent import ORACLE_V1, OracleV1
from oracle.rollout import RolloutConfig


class OracleAgentTests(unittest.TestCase):
    def test_policy_id_is_oracle_v1(self) -> None:
        self.assertEqual(OracleV1.policy_id, ORACLE_V1)
        self.assertEqual(ORACLE_V1, "oracle-v1")

    def test_invalid_player_id_raises(self) -> None:
        with self.assertRaises(ValueError):
            OracleV1(4)
        with self.assertRaises(ValueError):
            OracleV1(-1)

    def test_choose_action_returns_a_legal_action(self) -> None:
        random.seed(5)
        env = MonopolyEnv(agent_ids=[0], max_rounds=5)
        agent = OracleV1(0, config=RolloutConfig(depth=1, seeds=1))
        allowed = env.get_allowed_actions(env.whose_turn())
        action = agent.choose_action(env)
        self.assertIn(action, allowed)

    def test_full_game_completes_without_truncation(self) -> None:
        # Deliberately small max_rounds and a cheap rollout config: this
        # test proves the whole decision loop (buy/trade/mortgage/auction/
        # bankruptcy) works end to end without crashing or ever returning
        # an illegal action, not that OracleV1 plays well. The real
        # strength validation is the Task 7 h2h eval against ASU, run
        # separately with a much larger config.
        random.seed(31)
        env = MonopolyEnv(agent_ids=[0], max_rounds=20)
        fast_config = RolloutConfig(depth=2, seeds=1)
        agents = [
            OracleV1(0, config=fast_config),
            FPAgentA(1),
            FPAgentB(2),
            FPAgentC(3),
        ]
        decisions = 0
        while not env.done and decisions < 5000:
            actor = env.whose_turn()
            allowed = env.get_allowed_actions(actor)
            action = agents[actor].choose_action(env)
            self.assertIn(action, allowed)
            env.step(action)
            decisions += 1
        self.assertTrue(env.done)
        self.assertLess(decisions, 5000)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_oracle_agent.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'oracle.agent'`)

- [ ] **Step 3: Write the implementation**

Create `oracle/agent.py`:

```python
"""OracleV1: independent Monopoly policy, wraps oracle.rollout search."""

from __future__ import annotations

from monopoly_game_engine.constants import NUM_PLAYERS

from . import rollout
from .value import OracleWeights

ORACLE_V1 = "oracle-v1"


class OracleV1:
    """One-step-pluggable policy: search-selects a legal action per call."""

    policy_id = ORACLE_V1

    def __init__(
        self,
        player_id: int,
        weights: OracleWeights | None = None,
        config: rollout.RolloutConfig | None = None,
    ) -> None:
        if not 0 <= player_id < NUM_PLAYERS:
            raise ValueError(f"player_id must be in [0, {NUM_PLAYERS - 1}]")
        self.player_id = player_id
        self.weights = weights if weights is not None else OracleWeights()
        self.config = config if config is not None else rollout.RolloutConfig()

    def choose_action(self, env) -> int:
        return rollout.choose_action(env, self.player_id, self.weights, self.config)


__all__ = ["ORACLE_V1", "OracleV1"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_oracle_agent.py -v`
Expected: PASS (4 tests). This test is slower than the others (a real, if short, simulated game) — allow it to run for up to a minute or two.

- [ ] **Step 5: Commit**

```bash
git add oracle/agent.py tests/test_oracle_agent.py
git commit -m "$(cat <<'EOF'
Add OracleV1 policy wiring rollout search into a choose_action interface

Drives a full short game end to end against the repo's existing fixed
agents without ever returning an illegal action, proving the search +
value function integrate correctly before the ASU h2h validation.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Seat-balanced h2h eval harness vs ASU

**Files:**
- Create: `oracle/eval_h2h.py`
- Test: `tests/test_oracle_eval_h2h.py`

**Interfaces:**
- Consumes: `oracle.agent.ORACLE_V1`, `oracle.agent.OracleV1` (Task 5); `oracle.rollout.RolloutConfig` (Task 4); `oracle.simulate.preserve_global_rng` (Task 3); `oracle.value.OracleWeights` (Task 2); `ASU_FROZEN_TEACHER.core.ASUValueV1`; `ASU_FROZEN_TEACHER.evaluate.{DEFAULT_MAX_DECISIONS, RESULTS_DISCLAIMER, _run_game, wilson_interval}`; `ASU_FROZEN_TEACHER.spec.FROZEN_SPEC_HASH`; `monopoly_game_engine.agents_fixed.{FPAgentA, FPAgentB}`; `monopoly_game_engine.constants.{NUM_PLAYERS, RULESET_VERSION}`.
- Produces: `oracle.eval_h2h.DEFAULT_LINEUP: tuple[str, ...]`, `oracle.eval_h2h.run_h2h(*, games: int, seed: int, weights: OracleWeights, rollout_config: RolloutConfig, lineup: tuple[str, ...] = DEFAULT_LINEUP, max_decisions: int = DEFAULT_MAX_DECISIONS, workers: int = 1) -> dict`, with `result["oracle_vs_asu"]["beats_asu"]: bool` as the acceptance signal, and a `main(argv)` CLI entry point mirroring `asu_plus/eval_h2h.py`'s.

This file is deliberately the *only* place in `oracle/` allowed to import `ASU_FROZEN_TEACHER` — see Global Constraints above. It closely mirrors the proven structure of `asu_plus/eval_h2h.py:1-353`, substituting `OracleV1`/`oracle-v1` for `ASUPlusV1`/`asu-plus-v1` and `oracle.simulate.preserve_global_rng` for ASU's own.

- [ ] **Step 1: Write the failing test**

Create `tests/test_oracle_eval_h2h.py`:

```python
from __future__ import annotations

import unittest

from oracle.eval_h2h import DEFAULT_LINEUP, run_h2h
from oracle.rollout import RolloutConfig
from oracle.value import OracleWeights


class EvalH2HSmokeTests(unittest.TestCase):
    def test_run_h2h_completes_and_reports_beats_asu(self) -> None:
        # Tiny games count + cheap rollout config: this is a wiring smoke
        # test (does the harness run OracleV1 vs ASU end to end and
        # produce a well-formed comparison?), not a strength claim. Task 7
        # runs the real acceptance eval with a much larger configuration.
        result = run_h2h(
            games=2,
            seed=0,
            weights=OracleWeights(),
            rollout_config=RolloutConfig(depth=1, seeds=1),
            max_decisions=500,
            workers=1,
        )
        self.assertEqual(result["games"], 2)
        self.assertEqual(list(result["lineup"]), list(DEFAULT_LINEUP))
        self.assertIn("oracle_vs_asu", result)
        self.assertIn("beats_asu", result["oracle_vs_asu"])
        self.assertIsInstance(result["oracle_vs_asu"]["beats_asu"], bool)
        self.assertIn("oracle-v1", result["win_rates"])
        self.assertIn("asu-value-v1", result["win_rates"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_oracle_eval_h2h.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'oracle.eval_h2h'`)

- [ ] **Step 3: Write the implementation**

Create `oracle/eval_h2h.py`:

```python
"""Seat-balanced OracleV1 vs ASU head-to-head harness.

Reuses ASU_FROZEN_TEACHER's shared evaluation infrastructure (_run_game,
wilson_interval) and ASUValueV1 strictly as the opponent baseline that
OracleV1 is measured against. Competition rule 2 allows playing against
ASU; only cloning its outputs is forbidden. OracleV1 itself (oracle/value.py,
oracle/rollout.py, oracle/simulate.py, oracle/agent.py) has no
ASU_FROZEN_TEACHER dependency — this file is the one deliberate exception,
and only for running ASU as a table opponent.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ASU_FROZEN_TEACHER.core import ASUValueV1  # noqa: E402
from ASU_FROZEN_TEACHER.evaluate import (  # noqa: E402
    DEFAULT_MAX_DECISIONS,
    RESULTS_DISCLAIMER,
    _run_game,
    wilson_interval,
)
from ASU_FROZEN_TEACHER.spec import FROZEN_SPEC_HASH  # noqa: E402
from monopoly_game_engine.agents_fixed import FPAgentA, FPAgentB  # noqa: E402
from monopoly_game_engine.constants import NUM_PLAYERS, RULESET_VERSION  # noqa: E402

from .agent import ORACLE_V1, OracleV1
from .rollout import RolloutConfig
from .simulate import preserve_global_rng
from .value import OracleWeights

ASU_VALUE_ID = "asu-value-v1"
FIXED_A_ID = "fixed-a"
FIXED_B_ID = "fixed-b"
DEFAULT_LINEUP = (ORACLE_V1, ASU_VALUE_ID, FIXED_A_ID, FIXED_B_ID)


class _Spec:
    __slots__ = ("policy_id",)

    def __init__(self, policy_id: str) -> None:
        self.policy_id = policy_id


class _ScriptedAdapter:
    """Match evaluate.py's fixed-agent compatibility fallback."""

    def __init__(self, agent, player_id: int) -> None:
        self.agent = agent
        self.player_id = player_id
        self.fallbacks = 0

    def choose_action(self, env) -> int:
        from monopoly_game_engine.actions import ActionType

        allowed = env.get_allowed_actions(self.player_id)
        action = self.agent.choose_action(env)
        if action in allowed:
            return action
        self.fallbacks += 1
        if int(ActionType.END_TURN) in allowed:
            return int(ActionType.END_TURN)
        return allowed[0]


class _H2HFactory:
    def __init__(self, weights: OracleWeights, rollout_config: RolloutConfig) -> None:
        self.weights = weights
        self.rollout_config = rollout_config

    def build(self, spec: _Spec, player_id: int):
        if spec.policy_id == ORACLE_V1:
            return OracleV1(player_id, self.weights, self.rollout_config)
        if spec.policy_id == ASU_VALUE_ID:
            return ASUValueV1(player_id)
        if spec.policy_id == FIXED_A_ID:
            return _ScriptedAdapter(FPAgentA(player_id), player_id)
        if spec.policy_id == FIXED_B_ID:
            return _ScriptedAdapter(FPAgentB(player_id), player_id)
        raise ValueError(f"Unsupported H2H policy {spec.policy_id!r}")


def rotate_lineup(base: tuple[str, ...], game_index: int) -> tuple[_Spec, ...]:
    if len(base) != NUM_PLAYERS:
        raise ValueError(f"lineup must have {NUM_PLAYERS} policies")
    shift = game_index % NUM_PLAYERS
    rotated = base[-shift:] + base[:-shift] if shift else base
    return tuple(_Spec(policy_id) for policy_id in rotated)


def _game_job(payload: dict[str, Any]) -> dict[str, Any]:
    weights = OracleWeights(**payload["weights"])
    rollout_config = RolloutConfig(**payload["rollout_config"])
    factory = _H2HFactory(weights, rollout_config)
    specs = tuple(_Spec(policy_id) for policy_id in payload["policies"])
    with preserve_global_rng():
        return _run_game(
            specs,
            focus_seat=payload["focus_seat"],
            seed=payload["seed"],
            max_decisions=payload["max_decisions"],
            factory=factory,
        )


def summarize_policies(games: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    identifiers = sorted({policy for game in games for policy in game["policies"]})
    summaries = {}
    for identifier in identifiers:
        appearances = [
            (game, seat)
            for game in games
            if not game["truncated"]
            for seat, policy in enumerate(game["policies"])
            if policy == identifier
        ]
        wins = sum(game["winner"] == seat for game, seat in appearances)
        interval = wilson_interval(wins, len(appearances))
        rate = wins / len(appearances) if appearances else None
        summaries[identifier] = {
            "wins": wins,
            "games": len(appearances),
            "win_rate": rate,
            "win_rate_percent": None if rate is None else 100.0 * rate,
            "wilson_95": list(interval),
        }
    return summaries


def oracle_vs_asu(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    oracle = summaries.get(ORACLE_V1)
    asu = summaries.get(ASU_VALUE_ID)
    if oracle is None or asu is None:
        raise RuntimeError("H2H summary missing OracleV1 or ASU")
    oracle_rate = oracle["win_rate"] or 0.0
    asu_rate = asu["win_rate"] or 0.0
    oracle_lower = oracle["wilson_95"][0]
    return {
        "oracle_win_rate": oracle_rate,
        "asu_win_rate": asu_rate,
        "oracle_wilson_95": oracle["wilson_95"],
        "asu_wilson_95": asu["wilson_95"],
        "oracle_wilson_lower": oracle_lower,
        "beats_asu": oracle_lower > asu_rate,
        "rate_gap": oracle_rate - asu_rate,
    }


def run_h2h(
    *,
    games: int,
    seed: int,
    weights: OracleWeights,
    rollout_config: RolloutConfig,
    lineup: tuple[str, ...] = DEFAULT_LINEUP,
    max_decisions: int = DEFAULT_MAX_DECISIONS,
    workers: int = 1,
) -> dict[str, Any]:
    if games < 1:
        raise ValueError("games must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")

    jobs = []
    for index in range(games):
        specs = rotate_lineup(lineup, index)
        oracle_seat = next(
            seat for seat, spec in enumerate(specs) if spec.policy_id == ORACLE_V1
        )
        jobs.append(
            {
                "policies": [spec.policy_id for spec in specs],
                "focus_seat": oracle_seat,
                "seed": seed + index,
                "max_decisions": max_decisions,
                "weights": asdict(weights),
                "rollout_config": asdict(rollout_config),
            }
        )

    results: list[dict[str, Any] | None] = [None] * len(jobs)
    if workers == 1:
        for index, job in enumerate(jobs):
            results[index] = _game_job(job)
            if (index + 1) % max(1, min(8, games)) == 0 or index + 1 == games:
                print(f"h2h progress {index + 1}/{games}", flush=True)
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            for index, result in enumerate(pool.imap(_game_job, jobs, chunksize=1)):
                results[index] = result
                if (index + 1) % max(1, min(8, games)) == 0 or index + 1 == games:
                    print(f"h2h progress {index + 1}/{games}", flush=True)
    completed = [result for result in results if result is not None]
    if len(completed) != len(jobs):
        raise RuntimeError("H2H pool returned incomplete results")

    summaries = summarize_policies(completed)
    comparison = oracle_vs_asu(summaries)
    return {
        "ruleset": RULESET_VERSION,
        "frozen_spec_hash": FROZEN_SPEC_HASH,
        "lineup": list(lineup),
        "games": games,
        "seed": seed,
        "weights": asdict(weights),
        "rollout_config": asdict(rollout_config),
        "workers": workers,
        "truncations": sum(game["truncated"] for game in completed),
        "win_rates": summaries,
        "oracle_vs_asu": comparison,
        "game_records": completed,
        "disclaimer": RESULTS_DISCLAIMER,
    }


def _parser() -> argparse.ArgumentParser:
    weight_defaults = OracleWeights()
    rollout_defaults = RolloutConfig()
    parser = argparse.ArgumentParser(
        description="Seat-balanced OracleV1 vs ASU head-to-head on ppo-plus-v2"
    )
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-decisions", type=int, default=DEFAULT_MAX_DECISIONS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--w-endgame", type=float, default=weight_defaults.w_endgame)
    parser.add_argument(
        "--endgame-start-round", type=int, default=weight_defaults.endgame_start_round
    )
    parser.add_argument("--w-block", type=float, default=weight_defaults.w_block)
    parser.add_argument("--w-income", type=float, default=weight_defaults.w_income)
    parser.add_argument("--w-liability", type=float, default=weight_defaults.w_liability)
    parser.add_argument("--rollout-depth", type=int, default=rollout_defaults.depth)
    parser.add_argument("--rollout-seeds", type=int, default=rollout_defaults.seeds)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--save-games",
        action="store_true",
        help="Include per-game records in --output JSON (large)",
    )
    parser.add_argument("--pretty", action="store_true")
    return parser


def _public_result(result: dict[str, Any], *, save_games: bool) -> dict[str, Any]:
    if save_games or "game_records" not in result:
        return result
    trimmed = dict(result)
    trimmed.pop("game_records", None)
    return trimmed


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    weights = OracleWeights(
        w_endgame=args.w_endgame,
        endgame_start_round=args.endgame_start_round,
        w_block=args.w_block,
        w_income=args.w_income,
        w_liability=args.w_liability,
    )
    rollout_config = RolloutConfig(depth=args.rollout_depth, seeds=args.rollout_seeds)
    try:
        result = run_h2h(
            games=args.games,
            seed=args.seed,
            weights=weights,
            rollout_config=rollout_config,
            max_decisions=args.max_decisions,
            workers=args.workers,
        )
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    public = _public_result(result, save_games=args.save_games)
    payload = json.dumps(public, indent=2 if args.pretty else None, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload, flush=True)

    comparison = result["oracle_vs_asu"]
    print(
        (
            f"\nOracle {comparison['oracle_win_rate']:.3f} "
            f"(Wilson [{comparison['oracle_wilson_95'][0]:.3f}, "
            f"{comparison['oracle_wilson_95'][1]:.3f}]) "
            f"vs ASU {comparison['asu_win_rate']:.3f} "
            f"| beats_asu={comparison['beats_asu']}"
        ),
        flush=True,
    )
    return 0 if comparison["beats_asu"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_LINEUP",
    "rotate_lineup",
    "summarize_policies",
    "oracle_vs_asu",
    "run_h2h",
    "main",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_oracle_eval_h2h.py -v`
Expected: PASS (1 test). This test plays two real (short-capped) games through `_run_game`, so allow up to a couple of minutes.

- [ ] **Step 5: Commit**

```bash
git add oracle/eval_h2h.py tests/test_oracle_eval_h2h.py
git commit -m "$(cat <<'EOF'
Add OracleV1 vs ASU seat-balanced h2h eval harness

Mirrors asu_plus/eval_h2h.py's proven structure (seat rotation, Wilson
interval, beats_asu = wilson_lower > asu_rate) with OracleV1 swapped in.
The only oracle/ file allowed to import ASU_FROZEN_TEACHER, and only to
run ASU as a table opponent (competition rule 2).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Run the Milestone 1 acceptance evaluation

**Files:** none (this task runs the code from Tasks 1-6; no new source files).

This is the actual go/no-go check for Milestone 1: does `OracleV1` beat ASU by the agreed statistical criterion? Start with a cheap, fast configuration to sanity-check the whole pipeline runs cleanly end to end before committing to a slower, more decisive one — `oracle/rollout.py`'s `_greedy_action` does one full `deepcopy` per legal action considered at *every* simulated decision (not just the root), so wall-clock cost per game scales with `seeds x depth x avg_legal_actions_per_step`, and it is easy to accidentally configure something that takes hours for one game. Do not jump straight to a large `--games`/`--depth`/`--seeds` run.

- [ ] **Step 1: Fast sanity run (confirms the pipeline works, not strength)**

Run:
```bash
python -m oracle.eval_h2h \
  --games 4 --seed 0 --workers 4 \
  --rollout-depth 2 --rollout-seeds 1 \
  --max-decisions 2000 --pretty
```
Expected: completes within a few minutes, prints a `beats_asu=...` line (either value is fine here — this step only confirms nothing crashes and the harness produces a well-formed result). If it hangs or crashes, stop and debug before proceeding — do not increase the config on top of a broken run.

- [ ] **Step 2: Real acceptance run**

Run (create the output directory first if it doesn't exist):
```bash
mkdir -p artifacts_scratch
python -m oracle.eval_h2h \
  --games 64 --seed 64 --workers 8 \
  --rollout-depth 6 --rollout-seeds 3 \
  --save-games --pretty \
  --output artifacts_scratch/oracle_v1_h2h_64_seed64.json
```
This mirrors the exact `--games 64 --seed 64 --workers 8` configuration already used for the `asu-plus-v1` validation (see `artifacts_scratch/asu_plus_h2h_64_seed64.json` on `main`), so the two results are directly comparable. Expect this to take substantially longer than the ASU+ run did — `OracleV1`'s rollout search does far more simulation per decision than ASU's one-step evaluator. If it is impractically slow (many hours), stop, reduce `--rollout-depth`/`--rollout-seeds`, and re-run rather than waiting indefinitely.

- [ ] **Step 3: Interpret the result**

Check the final printed line: `Oracle <rate> (Wilson [lo, hi]) vs ASU <rate> | beats_asu=<bool>`.

- **If `beats_asu=True`:** Milestone 1 is complete. Do not proceed to MCTS (Approach B) or distillation (Milestone 2) inside this plan — those are separate follow-on plans, written fresh once this result exists, so they can be scoped against real data instead of guesses.
- **If `beats_asu=False`:** do not treat this as failure requiring a redesign. Inspect `win_rates` in the output JSON for `oracle-v1` vs `asu-value-v1` vs the two fixed agents to see whether `OracleV1` is losing to everyone (search/value bug — revisit Tasks 2/4 tests) or specifically weak against ASU while beating the fixed agents (weight tuning). For weight tuning, adjust one of `--w-endgame`, `--endgame-start-round`, `--w-block`, `--w-income`, `--w-liability` at a time and re-run Step 2 (same seed, so results are comparable) — this mirrors the ablation methodology `asu_plus/eval_h2h.py --ablate` uses, applied manually here since `oracle/eval_h2h.py` does not (yet) have an `--ablate` flag of its own.

- [ ] **Step 4: Commit the result artifact**

```bash
git add artifacts_scratch/oracle_v1_h2h_64_seed64.json
git commit -m "$(cat <<'EOF'
Add OracleV1 vs ASU h2h eval result, 64 games seed 64

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Notes

- **Spec coverage:** every section of `docs/superpowers/specs/2026-08-11-anti-asu-oracle-design.md`'s Milestone 1 scope (board model, value function, simulate wrapper, rollout search, agent, eval harness, acceptance test) has a task above. MCTS (Approach B) and distillation (Milestone 2) are explicitly out of scope for this plan per the spec's own staging and the Global Constraints section.
- **Type consistency checked:** `OracleWeights` (Task 2) is consumed identically in Tasks 4, 5, 6; `RolloutConfig` (Task 4) is consumed identically in Tasks 5, 6; `OracleV1.choose_action(self, env) -> int` (Task 5) matches the `factory.build(...).choose_action(env)` interface `_run_game` expects (Task 6, confirmed against `ASU_FROZEN_TEACHER/evaluate.py:315-354`'s existing usage).
- **No placeholders:** every step above contains complete, real code — no "TODO"/"handle edge cases" stand-ins.
