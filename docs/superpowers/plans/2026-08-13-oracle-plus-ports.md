# Oracle Plus Ports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the validated regression from the prior oracle-exploit-heuristics plan by porting two rules our friend's independent `origin/newthings` branch already validated against the real competitor field: (1) never hand away a monopoly-completing deed, and correctly decline incoming trades that would complete the requester's set or break our own; (2) prioritize buying "first monopoly race" color squares (brown/light-blue/dark-blue, or any contested/partially-owned group) using a dynamic cash-floor affordability check instead of a flat buffer.

**Architecture:** Both changes land in `oracle/rollout_policy.py::DealBuilderRollout` — the same file the prior plan touched, on the same branch/worktree (`oracle-exploit-heuristics`, already set up). Task 1 removes the broken sell-to-completer branch from `_make_trade_offer` and replaces `_should_accept_trade`'s logic with a superset that adds two new decline conditions while preserving all existing accept behavior. Task 2 adds new module-level helper functions (ported from `origin/newthings:oracle/plus_steals.py`) and layers a priority check at the top of `_should_buy`.

**Tech Stack:** Python, pytest, existing `monopoly_bench`/`monopoly_game_engine` simulator.

## Global Constraints

- Full `oracle/` test suite must pass after every task (existing tests must not regress). Two pre-existing, unrelated failures are excused: `test_each_competitor_returns_a_legal_opening_action` (known bug in an external competitor bot) and `test_clone_leaf_loads_checkpoint` (missing local artifact file).
- Every new/changed code path must fall through to sensible existing behavior when it doesn't apply — no change may leave oracle without a legal move.
- No changes outside `oracle/rollout_policy.py` and `oracle/tests/test_rollout_exploits.py`.
- The ported logic must be behaviorally faithful to `origin/newthings:oracle/plus_steals.py` (read via `git show origin/newthings:oracle/plus_steals.py` — do not modify or check out that branch; this repo's working tree must stay on `oracle-exploit-heuristics` throughout). Deliberate adaptations (e.g. using `prop.color`/`prop.square_id` attribute access to match this codebase's existing style, instead of the source's raw `PROPERTIES` dict lookups) are fine and are called out explicitly in each task below — anything else should match the source exactly.
- `TradeOffer` fields (verified in `monopoly_game_engine/env.py:50-70`): `from_player`, `to_player`, `offered_prop`, `requested_prop`, `cash_offered`, `cash_requested`. `offered_prop` = what the proposer gives us; `requested_prop` = what the proposer wants from us (i.e. one of *our* properties).
- `Property.get_rent(self, dice_roll=7, num_railroads=1, num_utilities=1)` (verified in `monopoly_game_engine/state.py:60-76`).

---

### Task 1: Remove sell-to-completer, port incoming-trade defense

**Files:**
- Modify: `oracle/rollout_policy.py` (`_should_accept_trade`, `_make_trade_offer`)
- Test: `oracle/tests/test_rollout_exploits.py` (append)

**Interfaces:**
- Consumes: `monopoly_game_engine.constants.COLOR_GROUPS` (already imported).
- Produces: `DealBuilderRollout._should_accept_trade(self, offer, env) -> bool` — same signature, now declines two additional cases before falling through to the existing accept-if-completes-ours check. `DealBuilderRollout._make_trade_offer(self, allowed, env) -> Optional[int]` — same signature, sell-to-completer branch removed entirely (first ~20 lines of the current method body).

- [ ] **Step 1: Write the failing tests**

Append to `oracle/tests/test_rollout_exploits.py`:

```python
def test_declines_incoming_trade_that_would_complete_requesters_set():
    env = SharedGame.new(600, max_rounds=50).env
    env.turn_order = [0, 1, 2, 3]
    env.current_turn_idx = 0
    sq_theirs_a, sq_theirs_b, sq_mine = LIGHT_BLUE_GROUP
    env.properties[sq_theirs_a].owner = 1
    env.properties[sq_theirs_b].owner = 1
    env.properties[sq_mine].owner = 0
    env.players[0].properties = [env.properties[sq_mine]]
    env.players[1].properties = [env.properties[sq_theirs_a], env.properties[sq_theirs_b]]
    env._update_monopolies()

    from monopoly_game_engine.env import TradeOffer

    agent = DealBuilderRollout(0)
    offer = TradeOffer(1, 0, requested_prop=env.properties[sq_mine], cash_offered=500)
    assert agent._should_accept_trade(offer, env) is False


def test_declines_incoming_trade_that_would_break_our_own_completed_set():
    env = SharedGame.new(601, max_rounds=50).env
    env.turn_order = [0, 1, 2, 3]
    env.current_turn_idx = 0
    for sq in LIGHT_BLUE_GROUP:
        env.properties[sq].owner = 0
    env.players[0].properties = [env.properties[sq] for sq in LIGHT_BLUE_GROUP]
    env._update_monopolies()

    from monopoly_game_engine.env import TradeOffer

    agent = DealBuilderRollout(0)
    offer = TradeOffer(1, 0, requested_prop=env.properties[LIGHT_BLUE_GROUP[0]], cash_offered=1000)
    assert agent._should_accept_trade(offer, env) is False


def test_still_accepts_incoming_trade_that_completes_our_set():
    env = SharedGame.new(602, max_rounds=50).env
    env.turn_order = [0, 1, 2, 3]
    env.current_turn_idx = 0
    sq_a, sq_b, sq_c = LIGHT_BLUE_GROUP
    env.properties[sq_a].owner = 0
    env.properties[sq_b].owner = 0
    env.properties[sq_c].owner = 1
    env.players[0].properties = [env.properties[sq_a], env.properties[sq_b]]
    env.players[1].properties = [env.properties[sq_c]]
    env._update_monopolies()

    from monopoly_game_engine.env import TradeOffer

    agent = DealBuilderRollout(0)
    offer = TradeOffer(1, 0, offered_prop=env.properties[sq_c])
    assert agent._should_accept_trade(offer, env) is True


def test_make_trade_offer_never_hands_away_a_completing_deed():
    env = _setup_completer_env(seed=603, target_cash=1500)
    agent = DealBuilderRollout(0)
    allowed = env.get_allowed_actions(0)

    action = agent._make_trade_offer(allowed, env)

    top_tier = _sell_trade_action_id(0, 1, LIGHT_BLUE_GROUP[0], price_idx=2)
    mid_tier = _sell_trade_action_id(0, 1, LIGHT_BLUE_GROUP[0], price_idx=1)
    low_tier = _sell_trade_action_id(0, 1, LIGHT_BLUE_GROUP[0], price_idx=0)
    assert action not in (top_tier, mid_tier, low_tier)
```

`_setup_completer_env`, `_sell_trade_action_id`, and `LIGHT_BLUE_GROUP` already exist in this file from the prior plan's Task 1 — reuse them, do not redefine.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest oracle/tests/test_rollout_exploits.py -v -k "declines_incoming or still_accepts_incoming or never_hands_away"`
Expected: `test_declines_incoming_trade_that_would_complete_requesters_set` and `test_declines_incoming_trade_that_would_break_our_own_completed_set` FAIL (current `_should_accept_trade` only ever looks at `offered_prop`, ignores `requested_prop` entirely, so it returns `False` by accident for the first test — check the actual failure; if it already returns `False` by coincidence rather than by the intended new logic, that's still a gap to fix, note it in your report). `test_still_accepts_incoming_trade_that_completes_our_set` should already PASS (existing behavior). `test_make_trade_offer_never_hands_away_a_completing_deed` FAILS (current code still has the sell-to-completer branch from the prior plan).

- [ ] **Step 3: Implement**

In `oracle/rollout_policy.py`, replace `_should_accept_trade`:

```python
    def _should_accept_trade(self, offer, env) -> bool:
        pid = self.player_id
        requested = offer.requested_prop
        if requested is not None:
            requester = offer.from_player
            if _would_complete(env, requester, requested.square_id):
                return False
            group = COLOR_GROUPS.get(requested.color, [])
            if group and all(env.properties[sq].owner == pid for sq in group):
                return False
        offered = offer.offered_prop
        if offered is None:
            return False
        color = offered.color
        if color in ("railroad", "utility"):
            return False
        group = COLOR_GROUPS.get(color, [])
        if not group:
            return False
        would_own = sum(1 for sq in group if env.properties[sq].owner == pid) + 1
        return would_own == len(group)
```

This calls `_would_complete`, a module-level helper — add it now (Task 2 also needs it, defined once here):

```python
def _would_complete(env: MonopolyEnv, pid: int, square: int) -> bool:
    prop = env.properties.get(square)
    if prop is None:
        return False
    color = prop.color
    group = COLOR_GROUPS.get(color) or ()
    if not group:
        return False
    return all(env.properties[sq].owner == pid or sq == square for sq in group)
```

Place it near the top of the module, after the existing constants block (`:22-34`), before the `DealBuilderRollout` class.

Then remove the sell-to-completer branch from `_make_trade_offer` — delete everything from the `# Sell-to-completer:` comment through the `for price_idx in (2, 1, 0):` loop's closing (the entire first block added by the prior plan, roughly 20 lines), leaving the method starting directly with:

```python
    def _make_trade_offer(self, allowed: List[int], env: MonopolyEnv) -> Optional[int]:
        pid = self.player_id

        # DealMaker monopoly tempo: bargain buy-offers, then colour exchanges.
        # Skip premium sell-spam — it is not monopoly-completing and dilutes search.
        for color, group in COLOR_GROUPS.items():
```

(i.e. restore this method to its pre-prior-plan-Task-1 shape — the `_sell_trade_action` import becomes unused; remove it from the import block too, `oracle/rollout_policy.py:12-18`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest oracle/tests/test_rollout_exploits.py -v`
Expected: all tests in the file PASS, including the 4 new ones and everything from the prior plan (except the prior plan's own sell-to-completer tests, which you are about to see fail — see Step 5).

- [ ] **Step 5: Remove the now-obsolete prior-plan tests**

The prior plan's `test_sell_to_completer_offers_top_price_tier`,
`test_sell_to_completer_falls_back_when_target_cannot_afford_top_tier`, and
`test_sell_to_completer_does_not_fire_when_group_split_across_opponents`
tests assert the exact behavior this task deliberately removes. Delete
those three test functions from `oracle/tests/test_rollout_exploits.py`
(keep `_setup_completer_env` and `_sell_trade_action_id` — Step 1's new
test still uses them). Also delete the `test_sell_trade_family_weight_is_boosted_not_penalized`
test and revert its corresponding change in `oracle/agent.py` — the prior
plan's Task 4 boosted `STRUCTURAL_BOOST["sell_trade"]` specifically to help
search find the now-removed sell-to-completer tactic; move `"sell_trade": 0.3`
back into `STRUCTURAL_PENALTY` (its original pre-prior-plan value) and
remove it from `STRUCTURAL_BOOST`. (This touches `oracle/agent.py`, which
is otherwise out of scope for this task — it's included here because it's
the direct, mechanical undo of Task 4's now-inapplicable rationale, not new
work.)

- [ ] **Step 6: Run tests to verify they still pass after cleanup**

Run: `python -m pytest oracle/tests/test_rollout_exploits.py -v`
Expected: all remaining tests PASS (the file should now have no
sell-to-completer-related tests at all, only the new defense tests plus
everything from Tasks 2-4 of the prior plan that's still valid — early
group buying and mortgage trigger tests are untouched by this task).

- [ ] **Step 7: Run the full oracle test suite for regressions**

Run: `python -m pytest oracle/ -q`
Expected: same pass count as before this task (minus the 4 deleted tests,
plus the 4 new ones — net the same or better), no new failures beyond the
2 excused pre-existing ones.

- [ ] **Step 8: Commit**

```bash
git add oracle/rollout_policy.py oracle/agent.py oracle/tests/test_rollout_exploits.py
git commit -m "Remove sell-to-completer tactic, port incoming-trade defense from newthings"
```

---

### Task 2: Port race-buy priority logic

**Files:**
- Modify: `oracle/rollout_policy.py` (imports, new constants/helpers, `_should_buy`)
- Test: `oracle/tests/test_rollout_exploits.py` (append)

**Interfaces:**
- Consumes: `_would_complete` (added in Task 1, same file).
- Produces: `DealBuilderRollout._should_buy(self, player, prop, env) -> bool` — same signature. New module-level constants `FIRST_RACE_COLOURS`, new helpers `_next_roll_threat(env, pid) -> float`, `_spend_floor(env, pid) -> float`, `_complete_floor(env, pid) -> float`, `_is_race_square(env, pid, square) -> bool`.

- [ ] **Step 1: Write the failing tests**

Append to `oracle/tests/test_rollout_exploits.py`:

```python
def test_race_buy_uses_dynamic_floor_for_first_race_colour():
    # Brown is a FIRST_RACE_COLOURS square (squares 1, 3) -- should use the
    # dynamic complete_floor/next_roll_threat check, not the flat BUY_BUFFER.
    env = SharedGame.new(700, max_rounds=50).env
    env.turn_order = [0, 1, 2, 3]
    env.current_turn_idx = 0
    agent = DealBuilderRollout(0)
    prop = env.properties[1]  # brown
    player = env.players[0]
    # No opponent owns anything near our landing squares, so next_roll_threat
    # is 0 -> spend/complete floor is 0 -> any affordable price should buy,
    # even with cash far below BUY_BUFFER's usual +100 margin.
    player.cash = prop.price + 1

    assert agent._should_buy(player, prop, env) is True


def test_race_buy_fires_on_contested_non_priority_group():
    # Light-blue is not in FIRST_RACE_COLOURS, but if an opponent already
    # owns a piece of the group it counts as a race square too.
    env = SharedGame.new(701, max_rounds=50).env
    env.turn_order = [0, 1, 2, 3]
    env.current_turn_idx = 0
    env.properties[LIGHT_BLUE_GROUP[1]].owner = 1
    env.players[1].properties = [env.properties[LIGHT_BLUE_GROUP[1]]]
    env._update_monopolies()

    agent = DealBuilderRollout(0)
    prop = env.properties[LIGHT_BLUE_GROUP[0]]
    player = env.players[0]
    player.cash = prop.price + 1

    assert agent._should_buy(player, prop, env) is True


def test_non_race_square_still_uses_buffer_logic():
    # Orange (squares 16, 18, 19) is not in FIRST_RACE_COLOURS; with nobody
    # owning any piece of it yet and this being our first, it's not a race
    # square, so the existing buffer logic (BUY_BUFFER=100) still applies.
    env = SharedGame.new(702, max_rounds=50).env
    env.turn_order = [0, 1, 2, 3]
    env.current_turn_idx = 0
    agent = DealBuilderRollout(0)
    prop = env.properties[16]  # orange, first piece, uncontested
    player = env.players[0]
    # Below BUY_BUFFER's margin -> should NOT buy under the old flat-buffer path.
    player.cash = prop.price + 5

    assert agent._should_buy(player, prop, env) is False
```

If square 1 is not "brown" or square 16 is not "orange" in this repo's
board data, or squares 16/18/19 don't form the orange group, read
`env.properties[<sq>].color` / `monopoly_game_engine.constants.COLOR_GROUPS["orange"]`
directly and adjust the literal square numbers used above accordingly —
the test's intent (a FIRST_RACE_COLOURS square vs. a non-priority,
uncontested square) is what matters, not these exact numbers.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest oracle/tests/test_rollout_exploits.py -v -k "race_buy or non_race_square"`
Expected: `test_race_buy_uses_dynamic_floor_for_first_race_colour` and
`test_race_buy_fires_on_contested_non_priority_group` FAIL (current
`_should_buy` has no race-square path, so a cash margin of only `+1` over
price fails the flat `BUY_BUFFER=100` check). `test_non_race_square_still_uses_buffer_logic`
should already PASS (existing behavior, unaffected until Step 3 is done —
verify it doesn't regress once you do implement Step 3).

- [ ] **Step 3: Implement**

In `oracle/rollout_policy.py`, add to the imports (`oracle/rollout_policy.py:12-24` region, after Task 1's edits):

```python
from monopoly_game_engine.constants import JAIL_BAIL
```

Add the new constant next to `FIRST_RACE_COLOURS`'s sibling constants (near `EARLY_GROUP_BUFFER`/`MORTGAGE_ROUND_SCALE`):

```python
# Colours that decide the first monopoly on the board -- race to buy these
# on sight. Ported from origin/newthings:oracle/plus_steals.py.
FIRST_RACE_COLOURS = ("brown", "lightblue", "darkblue")
```

Add the new helpers next to `_would_complete` (added in Task 1):

```python
def _next_roll_threat(env: MonopolyEnv, pid: int) -> float:
    """Worst published rent we can hit on a 2-12 walk from here."""

    player = env.players[pid]
    pos = int(player.position)
    worst = 0.0
    for total in range(2, 13):
        square = (pos + total) % 40
        prop = env.properties.get(square)
        if prop is None or prop.owner in (None, pid) or prop.mortgaged:
            continue
        opp = env.players[prop.owner]
        if opp.bankrupt:
            continue
        rails = sum(1 for p in opp.properties if p.color == "railroad" and not p.mortgaged)
        utils = sum(1 for p in opp.properties if p.color == "utility" and not p.mortgaged)
        rent = float(prop.get_rent(7, max(rails, 1), max(utils, 1)))
        if rent > worst:
            worst = rent
    return worst


def _spend_floor(env: MonopolyEnv, pid: int) -> float:
    return max(float(JAIL_BAIL), _next_roll_threat(env, pid))


def _complete_floor(env: MonopolyEnv, pid: int) -> float:
    return float(_next_roll_threat(env, pid))


def _is_race_square(env: MonopolyEnv, pid: int, square: int) -> bool:
    prop = env.properties.get(square)
    if prop is None:
        return False
    color = prop.color
    if color in ("railroad", "utility"):
        return False
    if color in FIRST_RACE_COLOURS:
        return True
    group = COLOR_GROUPS.get(color) or ()
    ours = sum(1 for sq in group if env.properties[sq].owner == pid)
    if ours >= 1:
        return True
    return any(env.properties[sq].owner not in (pid, None) for sq in group)
```

Replace `_should_buy`:

```python
    def _should_buy(self, player, prop, env) -> bool:
        pid = self.player_id
        square = prop.square_id
        color = prop.color
        if color not in ("railroad", "utility") and _is_race_square(env, pid, square):
            floor = _complete_floor(env, pid)
            if color not in FIRST_RACE_COLOURS and not _would_complete(env, pid, square):
                floor = _spend_floor(env, pid)
            return float(player.cash) - float(prop.price) >= floor
        # DealMaker: buy anything affordable with a small buffer, relaxed
        # further for the 1st/2nd piece of a group nobody else has touched.
        buffer = BUY_BUFFER
        if color not in ("railroad", "utility"):
            group = COLOR_GROUPS.get(color, [])
            contested = any(
                env.properties[sq].owner not in (None, self.player_id)
                for sq in group
                if sq != prop.square_id
            )
            mine_already = sum(
                1
                for sq in group
                if sq != prop.square_id and env.properties[sq].owner == self.player_id
            )
            if not contested and mine_already <= 1:
                buffer = EARLY_GROUP_BUFFER
        return player.can_afford(prop.price + buffer)
```

Note this is a deliberate adaptation, not a literal copy: the source's
`race_buy_action` and `is_race_square` read from the module-level
`PROPERTIES` dict (`PROPERTIES.get(square)["color"]`); this port uses
`env.properties[square].color` instead, matching this codebase's existing
style (already used everywhere else in this file). Behaviorally identical
— both read the same underlying color data.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest oracle/tests/test_rollout_exploits.py -v`
Expected: all tests PASS, including the 3 new ones and everything from
Task 1 and the surviving parts of the prior plan (early group buying,
mortgage trigger).

- [ ] **Step 5: Run the full oracle test suite for regressions**

Run: `python -m pytest oracle/ -q`
Expected: no new failures beyond the 2 excused pre-existing ones.

- [ ] **Step 6: Commit**

```bash
git add oracle/rollout_policy.py oracle/tests/test_rollout_exploits.py
git commit -m "Port race-buy priority logic for first-monopoly colours from newthings"
```

---

### Task 3: Revalidate against the original baseline

**Files:** none modified — validation only.

**Interfaces:** none new.

- [ ] **Step 1: Run the full oracle test suite**

Run: `python -m pytest oracle/ -q`
Expected: all tests pass except the 2 excused pre-existing failures.

- [ ] **Step 2: Run the identical original before/after eval**

This worktree should still have `oracle_labels/eval_bigtest.json` (the
original pre-any-change baseline) from the prior plan's Task 5 — verify it
exists first (`ls oracle_labels/eval_bigtest.json`); if missing, copy it
from the main checkout's `oracle_labels/eval_bigtest.json` before proceeding
(same file used throughout this session, seed 10, 40 games/lineup, sims=128,
two 4-team lineups).

Run:
```bash
python -m oracle.eval_h2h --games 40 --sims 128 --workers 22 --seed 10 \
  --lineup "oracle-rollout-v1,inncenta-heuristic,boom-hybrid,alinebidal-final;oracle-rollout-v1,slayer-v1,expo-heuristic-v1,inncenta-heuristic" \
  --checkpoint-dir oracle_labels/eval_ports_after.ckpt --resume \
  --output oracle_labels/eval_ports_after.json --pretty
```

(Includes `--checkpoint-dir`/`--resume` for resilience — background processes
in this environment have been killed unexpectedly and repeatedly throughout
this session, cause unknown, possibly OS sleep/power management. If a run
dies partway, re-run the identical command with `--resume` already included
— do not start over, do not ask permission, this is expected.)

- [ ] **Step 3: Compare against the original baseline and the ablation baseline**

```bash
python -c "
import json
before = json.load(open('oracle_labels/eval_bigtest.json'))
after = json.load(open('oracle_labels/eval_ports_after.json'))
for label, payload in (('before', before), ('after', after)):
    for field in payload['fields']:
        lineup = field['lineup']
        wr = field['oracle_win_rate']
        print(f'{label}: {lineup} -> oracle WR={wr:.3f}')
        for name, opp in field['oracle_vs_field']['opponents'].items():
            margin = opp['net_worth_margin']['mean']
            print(f'    vs {name}: WR={opp[\"win_rate\"]:.3f} margin={margin:.1f}')
"
```

Also worth a direct sanity comparison against the exact-lineup, no-tactics
ablation baseline from the earlier bisection investigation
(`oracle_labels/ablation_base.json`, if still present in this worktree — it
used the lineup `oracle-rollout-v1,slayer-v1,expo-heuristic-v1,inncenta-heuristic`
only, seed 10, 40 games, sims=128 — same conditions as this eval's second
field): oracle WR was 27.5%, inncenta WR vs oracle was 22.5%, margin -1408.8
with NO tactics active at all. This task's changes should get oracle at or
above that level against inncenta specifically — if inncenta's win rate vs
oracle in this task's field-2 result is still elevated anywhere near the
35% seen with the broken Task 1, something in Task 1/2 of this plan did not
fully fix the regression and needs investigation before declaring success.

- [ ] **Step 4: Record the result**

Edit `docs/superpowers/specs/2026-08-13-oracle-exploit-heuristics-design.md`,
under "Follow-ups (not this stage)", add a dated note (e.g. "2026-08-13,
second validation after porting newthings' incoming-trade defense and
race-buy logic: oracle WR vs {inncenta,boom,alinebidal} moved from 15% to
X%; vs {slayer,expo,inncenta} moved from 20% to Y%; inncenta WR vs oracle
specifically moved from 35% to Z%") with the actual numbers from Step 3.

- [ ] **Step 5: Preserve all logs for later use**

Per explicit instruction, do not delete any of this session's eval
artifacts. Confirm the following still exist in this worktree (do not
regenerate, just verify presence) and leave them as-is:
`oracle_labels/eval_bigtest.json`, `oracle_labels/eval_bigtest_after.json`,
`oracle_labels/eval_ports_after.json` (just written), and the four ablation
files `oracle_labels/ablation_base.json`, `ablation_task1.json`,
`ablation_task1_2.json`, `ablation_task1_2_3.json`. These are all gitignored
(`oracle_labels/` is in `.gitignore`), so they will not be committed — that's
fine, they're for reference within this session/worktree, not repo history.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-08-13-oracle-exploit-heuristics-design.md
git commit -m "Record validation results for newthings ports (sell-to-completer fix, race-buy)"
```
