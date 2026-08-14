# DECISIONS.md

Engineering choices and rationale for the competition agent. Newest phase last.

---

## D0.1 — Repository facts: what was verified, and what was wrong

The brief supplied a list of "verified, do not rediscover" repository facts.
Before building anything I re-verified them, because a wrong premise here would
invalidate every later phase. Result:

**Confirmed by direct measurement** (`competition_agent/probes/` and runtime
introspection, no source reading of the teacher):

| Claim | Status |
| --- | --- |
| 300-dim observation from `env._get_state(pid)` | confirmed — `len(...) == 300` |
| 2958-dim action space | confirmed — `actions.ACTION_SPACE_SIZE == 2958` |
| `OFFSETS` table with the named families | confirmed — 12 families, `binary`..`auction` |
| phases `pre_roll / post_roll / out_of_turn / auction` | confirmed |
| `env.get_allowed_actions(pid)`, `env.step`, `env.whose_turn()` | confirmed |
| auction state fields (all five) | confirmed present on `MonopolyEnv` |
| `ASUValueV1` / `ASURolloutV1` with `decide` + `choose_action` | confirmed |
| teacher is deterministic given the same state | confirmed — see D0.2 |

**Contradicted.** None of the six claimed "existing starting points" exist
anywhere in the repository:

    probe_teacher.py   distill_collect.py   distill_train.py
    student_policy.py  student_policy.pt    heuristic_policy.py

A `find` across the whole tree returns nothing for any of them. The brief
budgets Phase 1 as "extend `probe_teacher.py`" and Phase 3 as "retrain the
existing ~82%-agreement student", so both phases start from zero rather than
from a working artifact. This is recorded here rather than silently absorbed
because it changes the effort estimate for those two phases, not the plan.

**Partial substitutes that do exist and are being reused instead of rewritten:**

- `ASU_FROZEN_TEACHER/evaluate.py` — a seat-balanced paired-block evaluator
  exposing `_new_seeded_game(seed)` and `_ScriptedAdapter`. Used (imported, not
  read) so `bench.py` seeds games identically to the existing artifacts.
- `tools/asu_baseline.py` — a measurement-only ASU driver that establishes the
  digest/timing convention followed here.
- `tests/test_asu_phase_a.py` — 24 passing tests pinning teacher behaviour.

## D0.2 — The teacher is modified but behaviourally frozen; probing is safe

`git status` shows `ASU_FROZEN_TEACHER/core.py` with **258 added / 32 removed
uncommitted lines**. A modified teacher would undermine the whole premise of
the exercise: probe results would characterise a local variant, and the audit
trail would not describe the policy the organizer froze.

It does not, and the evidence is already in the repository. `artifacts/`
contains a pre-modification digest (`asu_baseline_locked.json`) and the current
one (`asu_baseline.json`). Every seat-0 `decide()` result is canonicalised and
SHA-256 chained per seed:

| seed | locked digest | current digest | match |
| --- | --- | --- | --- |
| 3  | `65a666333635d9f8…` | `65a666333635d9f8…` | yes |
| 7  | `deba3445cdd4eb9f…` | `deba3445cdd4eb9f…` | yes |
| 11 | `144b23efba152c4d…` | `144b23efba152c4d…` | yes |

All 24 tests in `tests/test_asu_phase_a.py` pass. The change is a
behaviour-preserving optimisation: aggregate cost fell from **0.0501 s** to
**0.0167 s** per decision (3.0×). The decision *function* is bit-identical on
every probed state, so behavioural reverse engineering targets the intended
policy.

**Decision:** probe the working tree as-is. Re-run `tools/asu_baseline.py` and
compare against `asu_baseline_locked.json` before each phase's probe batch;
a digest change invalidates that batch.

## D0.3 — `decide()` returns internals; we use only the selected action

`ASUValueV1.decide(env)` returns a `Decision` carrying, for *every* legal
action, all four value components, both safety margins, trade gains, auction
ceilings, and rejection reasons. `ASU_FROZEN_TEACHER` additionally exports
`evaluate_value`, `rent_projection`, `safety_breakdown`, `monopoly_value` and
friends as public API.

Using any of that would reduce Phase 1 from inference to transcription. It is
technically "observed behaviour" — no source is read — but it is reading the
teacher's internals through a window, and a spec derived that way would not be
the artifact the brief asks for.

**Decision:** the teacher is consumed strictly as `decide(env) -> action`.
Probes record the *selected action id only*. Value components, safety
breakdowns and rejection reasons are never read, and no `ASU_FROZEN_TEACHER`
helper other than the two policy classes is called from probe or policy code.
`competition_agent/policies.py` enforces this by construction: it imports only
`ASUValueV1`, `ASURolloutV1`, and the harness helpers.

This is the strict reading of "treat the teacher as an opaque function". If it
is ever relaxed, Phase 1 collapses to roughly a day — but the audit trail then
proves much less.

## D0.4 — Reading policy for allowed documents

`ASU_FROZEN_TEACHER/README.md` and `spec.py` are explicitly permitted reading
and both were read. They are rich: the README states the value decomposition
`V = M_assets + R_short + R_long + M_monopoly`, the 5-turn/5-lap horizons, the
`2 ** missing_deeds` monopoly discount, both safety gate inequalities, the trade
gain thresholds, and the auction ceiling rule; `spec.py` adds the numeric
constants (`minimum_cash: 200`, `terminal_utility: 1_000_000`, rollout
`8 x 8 x 32`, seed 0).

**Decision:** these are treated as *hypothesis sources*, never as evidence.
Every rule in `SPEC.md` must still cite a probe CSV that confirms it on
observed decisions, and a documented rule that probes contradict is recorded as
contradicted. Reading them makes Phase 1 efficient — it tells us which
experiments to run — but the evidentiary chain stays behavioural, which is what
makes the derived policy defensible.

`core.py` and `evaluate.py` have not been opened. `git diff` on `core.py` was
deliberately not run (`--stat` only, which reports line counts and no source).

## D0.5 — `bench.py` design

- **Seeding.** Game *k* uses `--seed + k` via `_new_seeded_game`, matching the
  existing artifacts. Stochastic policies are seeded from `(game_seed, seat)`.
  Every seed is written to the output JSON.
- **Two win rates, not one.** The engine terminates a game either by
  elimination (one solvent player) or by hitting its round cap; `--max-steps`
  adds a harness-level cap. `env.winner()` falls back to the net-worth leader
  when no one has been eliminated, so counting it as a "win" silently conflates
  two different outcomes. `bench.py` reports **leader rate** (all games) and
  **decisive rate** (elimination games only) side by side, following the
  convention the repo's own evaluator uses with `provisional_leader`. Wilson
  95% intervals on both — Wilson rather than normal-approximation because the
  per-seat counts are small and rates land near 0 and 1.
- **Scripted-agent fallback.** The fixed agents sometimes return `END_TURN`
  where only liquidation is legal. `bench.py` routes them through the repo's
  `_ScriptedAdapter` so the identical compatibility fallback applies; ASU and
  learned policies stay strictly checked and any illegal action raises.
- **Parallelism.** `--workers` uses a process pool over whole games. Games are
  independent and each is seeded from its own seed, so results are invariant to
  worker count.

## D0.6 — Measured costs, and what they imply for Phase 4

Warm per-decision cost on this machine (10 cores):

| policy | seconds / decision |
| --- | --- |
| `ASUValueV1` | 0.0015 (warm) / 0.0167 (aggregate incl. cold caches) |
| `ASURolloutV1` | 0.69 |

The rollout variant is ~460× the value variant, not the ~2048× a naive
`8 x 8 x 32` reading predicts, so it is memoising heavily across candidates.
Two consequences:

1. A 200-game rollout reference run is ~3 CPU-hours — feasible parallelised,
   which is why the reference number is being measured rather than estimated.
2. Phase 4's K/M/P budget has real headroom, but 0.69 s/move is the number to
   beat, and it must be re-measured on competition hardware rather than
   inherited from here.

Early incidental observation, not yet a probe: over the first 10 seat-0
decisions of seed 3, `ASURolloutV1` and `ASUValueV1` selected the **same
action 10/10 times**. If lookahead only rarely changes the choice, Phase 1
experiment 8 (rollout divergence) needs states chosen adversarially rather
than sampled from early play. Recorded here so Phase 1 designs for it.

**Correction to the per-decision figures above.** The 0.0015 s/decision value
figure was measured on early-game states of seed 3 and does not generalise: a
full 1200-step game of seed 1 costs 25.6 s for 258 seat-0 decisions, i.e.
**~0.10 s/decision** averaged over a developing board. Decision cost grows
with board development (more legal actions, richer monopoly planning). The
0.69 s/move rollout figure is likewise an early-game number and should be
treated as a lower bound. Phase 4's timing budget must be measured on
late-game states, not early ones.

## D0.7 — Opacity discipline reconsidered and reaffirmed

D0.3 was revisited deliberately rather than by inertia, because relaxing it
would cut Phase 1 from roughly two days to one: `decide()` returns per-action
value components, both safety margins, trade gains, auction ceilings and
rejection reasons, and the package exports `evaluate_value`,
`rent_projection`, `safety_breakdown` and `monopoly_value` directly.

**Reaffirmed unchanged.** Probes record the selected action id and nothing
else. The reasoning:

1. The brief's framing — "treat the teacher as an opaque function
   `decide(env) -> action`", "the evidentiary chain must rest on observed
   behaviour alone" — is explicit, and the competition legitimacy argument
   rests on it. A spec transcribed from returned internals would not
   demonstrate that the policy was derived from behaviour.
2. It has already proved productive rather than merely restrictive. A2–A5
   recovered the exact gate structure *and* its rent-projection term from
   flip points alone, to the dollar, including the doubles tail. Nothing was
   lost by not reading the breakdown.
3. The constraint is enforced by construction, not discipline:
   `competition_agent/policies.py` and `probe_harness.py` import only
   `ASUValueV1`, `ASURolloutV1` and the seeded-game helpers. Adding a
   breakdown-reading import would be a visible diff, not a silent slip.

This decision should not be revisited again without a stated reason recorded
here. If it ever is relaxed, every rule in `SPEC.md` derived after that point
must be tagged as internals-derived so the audit trail stays honest about
which rules are behavioural evidence and which are transcription.

## D0.8 — The 200-game reference run was killed; `bench.py` needs checkpointing

The Phase 0 rollout reference (200 games, `rollout,fixed-a,fixed-b,fixed-c`)
ran for **2h09** and was then killed to free the CPU for the audit-trail
certification, which gates all further probing. The certification is a hard
prerequisite for Phase 1 validity; the reference number is not, so the
reference lost the tie.

Two hours of compute were lost because `bench.py` accumulates results in
`pool.map` and writes JSON only at the end — a run that is interrupted
produces nothing at all. That is a design flaw for jobs of this length.

**Decision:** before the reference run is restarted, `bench.py` must stream
per-game records to disk as they complete (`imap_unordered` + append) and
support resuming by skipping seeds already present in the output file. Long
benchmarks are then interruptible at no cost, and the Phase 4/5 head-to-head
runs (≥300 and ≥500 games) inherit the same property.

## D1.1 — Phase 4 gate: RESOLVED, and the layer is rescoped to two families

The conditional gate was stated as: *Phase 4 is built only if the
adversarial-state divergence rate is statistically distinguishable from zero;
if divergence is negligible even on constructed states, Phase 4 is skipped and
its time moves to Phase 5.*

**Gate outcome: PASSED — Phase 4 proceeds.** Over 230 constructed boundary
states the rollout changed the decision in 81 cases, a rate of **35.2%**
(95% Wilson CI [29.3%, 41.6%]). Both policies are deterministic given a state,
so the null "rollout never changes the decision" is not merely rejected at some
confidence level — a single divergence falsifies it outright, and there are 81.

**But the flat reading of the gate would have produced the wrong design.**
Divergence is not spread evenly; it is almost entirely confined to two action
families:

| family | divergence | cost ratio |
| --- | --- | --- |
| auction | 94.6% | 1035× |
| build | 58.3% | 464× |
| buy | 0.0% | 112× |
| trade | 0.0% | 309× |

Wrapping *every* decision in rollout — the Phase 4 brief's default reading —
would spend 112× on buy decisions and 309× on trade decisions to reproduce an
answer the fast path already gives, in 126 of 126 cases tested.

**Decision.** The Phase 4 rollout layer is applied **selectively**, gated on
action family: auctions and build/improve decisions get lookahead; buy and
trade decisions take the hybrid's answer directly. The K/M/P budget is then
spent where it demonstrably changes outcomes, which also relieves the
per-move time limit — the expensive path runs on a minority of decisions.

This gate is re-checked, not assumed, once the clone exists: the divergence
measurement above is between the *teacher's* two variants, and the clone's
own value/rollout divergence profile could differ. `p08_rollout_divergence.py`
is written to re-run against any policy pair.

**Caveat on coverage.** The 0% buy/trade divergence is measured over 112 buy
and 14 trade boundary states. The trade sample is small; before buy/trade are
finally excluded from the rollout path, the trade-boundary population needs
widening (Experiment 6 will produce it). Until then the exclusion is provisional
and is recorded here so it is not mistaken for a settled result.

## D1.2 — Two teacher weaknesses found in Phase 1, carried to Phase 5

Recorded now so Phase 5 modules are aimed at measured gaps rather than
assumed ones.

1. **Rent projection is already sharp on the collection side** (A3–A6). The
   teacher enumerates 2d6 complete-turn landings over opponents' *actual*
   positions, including doubles-driven extra rolls, and prices deeds
   accordingly. Phase 5 module 1 should therefore target rent *paid to*
   opponents from their developed holdings, not rebuild the collection side.

2. **The auction ceiling ignores group presence it does not already have**
   (B3, B5). The teacher pays the same for the first deed of a colour group as
   for the second, and escalates only on the completing deed. An opponent can
   take the first two deeds of a group at ordinary prices and only meets
   resistance on the third — by which point it holds the blocking position.
   This is a direct opening for Phase 5 module 2 (denial-value trading), and
   it is a weakness in the *teacher we are cloning*, so the clone will inherit
   it unless the module explicitly overrides the auction ceiling.

## D1.3 — Phase 4 gate CORRECTED: trade goes back into the rollout path

D1.1 scoped the Phase 4 rollout layer to auction and build only, excluding buy
and trade on the strength of p08's 0% divergence in both. That exclusion was
recorded as **provisional for trade**, because the trade sample was 14 states
from a single narrow setup. The caveat was justified.

Experiment 6 built the real trade surface — seat 0 holding four deeds against
a rival holding five, sweeping the sweetener across the whole accept region —
and measured **50 divergences in 54 states, 92.6%**. Trade is not a family
where lookahead is redundant; it is the family where the two variants agree
*least*.

| family | p08 (narrow) | p06 (wide) | in rollout path? |
| --- | --- | --- | --- |
| auction | 94.6% (56) | — | yes |
| build | 58.3% (48) | — | yes |
| **trade** | **0.0% (14)** | **92.6% (54)** | **yes — corrected** |
| buy | 0.0% (112) | — | no (112 states, one setup) |

**Decision.** The selective rollout path covers **auction, build and trade**.
Buy remains on the fast path, but that exclusion now carries the same warning
the trade one did: 112 states is a decent sample, yet they came from a single
board configuration, and the trade case is a worked example of how badly a
one-setup sample can mislead. Before the competition entry is frozen, buy must
be re-tested on a population that varies board configuration, not just cash.

**Process lesson, recorded because it nearly shipped a wrong design.** A 0%
result on a narrow population is not evidence of absence; it is evidence about
that population. Divergence probes must vary the *board*, not only the
parameter under sweep. Both p08's trade cell and its buy cell hold board shape
fixed, which is exactly the flaw that produced the wrong conclusion.

## D1.4 — Orphaned workers: the bug that silently halved throughput for 3.5 hours

**Symptom.** After the 200-game reference run was stopped (D0.8), everything
was inexplicably slow: the teacher certification timed out twice at 10 minutes
on an apparently idle box, a full value game measured 25.6 s against an
expected ~3 s, and Experiment 6 took over half an hour.

**Cause.** The run was stopped with `pkill -f "bench.py --games 200"`. That
pattern matches the parent only. `multiprocessing` workers are spawned with a
command line of `python -c from multiprocessing.spawn import spawn_main...`,
which contains neither the script name nor its arguments, so the ten workers
never matched, were re-parented to init, and kept running. They were found
still alive **3.5 hours later** at ~55% CPU each — 911% of the machine's
1000% total — computing results that no living process would ever collect.

Two independent defects made this possible:

1. SIGTERM to the parent kills it immediately, so `with mp.Pool(...)` never
   reaches its cleanup and the children survive.
2. Even a correct pattern kill cannot match a worker's command line, so there
   was no way to clean up after the fact except by hand.

**Fix.** `competition_agent/proc.py`:

- `managed_pool(workers)` installs SIGTERM/SIGINT/SIGHUP handlers that call
  `pool.terminate()` before re-raising, and puts the parent in its own process
  group so the job can be killed as a unit. Wired into `bench.py`,
  `certify_teacher.py`, and all seven pooled probes.
- `kill_by_script(name)` resolves script → pid → process group → group kill,
  which is what a bare `pkill -f` cannot do.
- `find_orphans()` / `python3 -m competition_agent.proc orphans` lists python
  workers whose parent is init, so this smell is diagnosable in seconds rather
  than mistaken for a slow machine.

Verified: a pool running a real module-level target is SIGTERMed, and both the
child count and the count of processes matching the script drop to zero. The
first version of that test was invalid — the workers died unpickling the
target rather than running it, so it would have passed against a broken
implementation — and was rewritten against a real module file.

**What it cost.** Every timing figure taken between the kill and the cleanup
is contaminated and none should be trusted: the 25.6 s/game measurement in
D0.6, the certification's two timeouts, and Experiment 6's runtime. The
*decision* data from those runs is unaffected — the teacher is deterministic
given a state, so contention changes only wall-clock, not selected actions.
Timing-sensitive conclusions (Phase 4's per-move budget) must be re-measured
on a quiet machine.

## D2.1 — Phase 2 status: 76.4% held-out, blocked on trade proposal

`spec_policy.py` is a priority-ordered rule pipeline; every branch cites the
SPEC rule it implements. `spec_model.py` rebuilds the quantities the rules are
stated in — the 2d6 complete-turn landing enumeration (A4/A5), rent flow from
real positions (A6), liquidatable worth (D3/F3), and both safety gates (D1–D4).

**Model validation before policy work.** `gates_ok` was checked against the
probe corpus first: it reproduces **28/28** measured buy flip points within $2
(26 exact) and every gate-1 row of the build and unmortgage sweeps exactly.
Gate 2 carries a known residual, recorded in the function's docstring rather
than curve-fitted away — the clone is $21–$81 more cautious than the teacher
when opponents are heavily developed, a safe direction to err.

**Agreement (held-out, seeds 900000+, disjoint from all probe seeds):**

| family | n | rate |
| --- | --- | --- |
| ROLL_DICE | 212 | 100.0% |
| unmortgage | 11 | 100.0% |
| BUY_PROPERTY | 26 | 96.2% |
| END_TURN | 931 | 98.2% |
| auction | 153 | 90.8% |
| DECLINE_TRADE | 315 | 69.5% |
| improve_house | 37 | 27.0% |
| **exch_trade** | **281** | **0–5%** |
| buy_trade / sell_trade / mortgage | 36 | 0% |
| **TOTAL** | **2005** | **76.4%** |

Against the ≥90% target: **FAIL**. The decision families the probes covered
directly are in good shape — buy, auction, jail, roll, unmortgage all sit at
90–100%. The gap is concentrated in trade *proposal*.

**Root cause: an experiment that was never briefed.** Phase 1 mapped the trade
accept/decline surface (Experiment 6) — the reply side. It never asked what
the teacher *proposes*. That family is ~15% of all decisions (307 of 474
disagreements), so no amount of tuning the covered rules reaches 90%.

**Three attempts, all recorded because the failures are informative:**

1. *No proposal rule* — 76.4%. The clone simply ends its turn.
2. *Completion heuristic* (p09: offer the least valuable spare for the deed
   completing a group) — **75.0%**, `exch_trade` 5%. p09's narrow setup made
   this look right: 36/36 proposals there were exactly that shape. Held-out
   play refuted it — across 281 real proposals the teacher requested 23, 25,
   37, 12, 9, 31, 27, 35 and *offered valuable* deeds (13, 24, 9, 21), not
   spares. Agreement on the requested deed alone was 27/189.
3. *General two-sided +EV search* over all legal exchange pairs, scored with
   the same deed valuation used elsewhere — **73.8%**, `exch_trade` 0.4%.

Attempt 3 is structurally the right shape and still scored worst, which
locates the problem precisely: **`deed_value` is not accurate enough to rank
exchange pairs.**

> **RETRACTED (see D2.6).** This paragraph originally read "calibrated well
> enough for threshold decisions — where only its comparison against a cash
> gate matters, hence 96% on buy and 91% on auction". The buy half is wrong:
> `_buy` never calls `deed_value`. It is `gates_ok(env, pid, price)` and
> nothing else, so buy's agreement is evidence about the safety gates and says
> nothing about the valuation, in either direction. The auction half stands as
> a fact but not as support for "calibrated well enough" — D2.6 shows removing
> the monopoly term *improves* auction by 5.0 points and shrinking it by 8.2.
> The valuation is not well calibrated for thresholds either; it is merely
> outvoted there by price and rent.

**Next step, and it is a probe, not a tuning pass.** Attempts 2 and 3 were both
made without evidence to guide them, which is why each was worse than the last.
Experiment 9b must measure the teacher's *ranking* directly: fix a board, offer
a forced choice between two specific exchanges, and sweep the pair to recover
the ordering deed-by-deed. That calibrates `deed_value` on relative
comparisons instead of inferring it from thresholds. Until that exists, no
further change should be made to `_propose_trade`.

Secondary, smaller gaps once trade is solved: `improve_house` at 27% (E1's
rent ordering is right in isolation but something else outranks it in real
positions), and `DECLINE_TRADE` at 69.5% (the clone accepts offers the teacher
refuses — consistent with the same valuation weakness).

## D2.2 — Experiment 9b: ranking calibration data, and why board diversity was mandatory

The Phase 2 gap is trade, and the fault was located in `deed_value`: accurate
enough for thresholds (buy 98.4% — RETRACTED, see D2.6: `_buy` does not call
`deed_value`; auction 90.5%) but not for ranking two deeds
against each other (`exch_trade` 0.3% over 722 held-out decisions).

**Board diversity was made a design requirement of this probe, not an
afterthought.** Two narrow samples had already produced confident, wrong
conclusions: p08's 14-state trade cell reported 0% rollout divergence against
Experiment 6's 92.6% (D1.3), and p09's single board shape showed 36/36
proposals were "cheapest spare for the completing deed" — a rule that scored
5% in real play. Both looked unambiguous at the time.

p09b therefore samples 400 boards from a seeded generator randomising deed
allocation (seat 0, one rival, and a third party so the board is not
two-sided), all four positions, development level, bank house/hotel stock,
mortgage flags and every player's cash. Diversity is **reported rather than
claimed**: the sample covers 2–5 deeds a side, 0–4 development levels, 4 bank
stock levels, 7 cash levels, 23 distinct candidate-set sizes, and **all ten
colour groups on both the offered and the requested side**.

### Result: two separate defects, not one

| | |
| --- | --- |
| boards offering a real ranking choice | 400 |
| teacher proposed a trade | **118 (29.5%)** |
| teacher ended its turn instead | **282 (70.5%)** |
| our model's top-1 accuracy | **13/118 = 11.0%** |
| teacher's pick inside our top-3 | 21.2% |
| teacher's pick inside our top-5 | 32.2% |
| teacher's pick inside our top-10 | 62.7% |
| median rank our model gives its pick | 8 (mean 10.4, worst 40) |

**Defect 1 — when to propose.** The teacher proposes on fewer than a third of
boards where a legal exchange exists. `_propose_trade` fires whenever any pair
scores positive, which is most of the time. That is the source of the 200
`END_TURN` disagreements: the same fault, counted in a different row. Whatever
gate suppresses 70% of proposals is not modelled at all, and none of the
obvious board features separate the two populations — deeds held, cash,
candidate count and development are nearly identical across proposed and
ended-turn boards (3.64 vs 3.48 deeds, $1,170 vs $1,153, dev 0.05 vs 0.13).
The gate is therefore a property of the *offers available*, not of the board,
which points at a threshold on the gain itself.

**Defect 2 — which to propose.** Top-1 of 11% against a candidate set
averaging 22 is only modestly better than the 4.5% a random pick would give,
and top-10 at 62.7% says the correct action is usually somewhere in the upper
half of our ordering but rarely at the top. The ordering carries signal; it is
not calibrated.

The direction is at least sane: the teacher asks for more than it gives (mean
requested price $224 vs offered $204; requested price exceeds offered in
70/118), and it trades across all ten colours rather than favouring any.

### Next work item

Fit `deed_value` against `p09b_trade_ranking.csv` as a ranking problem —
top-1 accuracy on the 118 proposal boards is the objective, with the 282
end-turn boards as the negative class for the propose/don't-propose gate.
Both defects are measurable on this one file, so the fit can be validated
without touching held-out play, and held-out agreement stays an honest test.

No further change to `_propose_trade` until that fit exists. The two previous
attempts were both made without calibration data and both regressed.

## D2.3 — Debt/jail evaluation set: G1 refuted, and rule interaction exposed

`p10_debt_jail_eval.py` builds the population ordinary play never reaches —
250 randomised debt boards (followed through the whole liquidation chain) and
250 randomised jail boards, each swept in both phases. 1,508 decisions.

| scenario | agree | n | rate |
| --- | --- | --- | --- |
| debt | 833 | 1040 | **80.1%** |
| jail_post_roll | 157 | 218 | **72.0%** |
| jail_pre_roll | 106 | 250 | 42.4% |

Per family: `mortgage` 83.8% (980), `USE_GOOJ_CARD` 100% (53), `PAY_BAIL`
78.6%, `sell_house` 33.3%, `ROLL_DICE` 61.6%.

**F1–F5 and G2–G5 largely survive contact.** Liquidation at 80% and the jail
exit choice at 72% are the first real validation these families have had;
`USE_GOOJ_CARD` is perfect across 53 states. `sell_house` at 33% is a genuine
ordering defect within F4.

**G1 is refuted.** It was stated as "in pre_roll the teacher defers, choosing
END_TURN" on the strength of p07's 224/224. This set shows that was p07's
*setup*, not the rule: over 250 jailed pre_roll states the teacher chose
END_TURN in only 106 and spent the rest unmortgaging (63) and proposing
trades (95). Being in jail suppresses the exit decision, not every other rule.
Recorded in SPEC as a contradiction, third one on the record.

**Fixing G1 made the score worse, and that is the finding.** Letting the
pipeline fall through dropped jail_pre_roll from 42.4% to 23.2% and the total
from 72.7% to 69.5%, because the rules that now fire — trade proposal at 4%,
unmortgage at 14% — are worse than doing nothing. G1's wrong rule was
accidentally protective.

The fix is kept anyway. Reverting would be tuning to a symptom: the pipeline
would score better while containing a rule known to be false, and the debt
figures would still be carried by `mortgage` alone. It does mean **the trade
gate is a blocker, not an optimisation** — several families are held hostage
to it.

## D2.4 — Both trade fits FAIL, and they fail informatively

`fit_trade.py`, split 60/40 by board, fixed before any search (240 train / 160
held-out; 71 / 47 proposals).

**Defect 2 — ranking.** 4,000 weight searches over
(price, rent, mono, mortgaged) differences found **nothing better than the
baseline**: train top-1 stayed 12.7%, held-out 8.5%. The optimum is the
starting point.

**Defect 1 — gate.** The best threshold scores 70.4% on train — *exactly* the
never-propose baseline — and 70.6% held out. The fitted gate degenerates to
"never propose anything".

**Interpretation: the feature set is wrong, not the weights.** A search that
cannot beat its own initialisation, and a threshold that collapses to a
constant classifier, both say the same thing: these four features contain no
signal about which exchange the teacher picks or whether it proposes at all.
More search, more features of the same kind, or a smarter optimiser would all
be wasted.

**The likely omission is that every feature is one-sided.** They score the
trade from our perspective only. The published description of the teacher
requires proposer gain > 0, **recipient gain >= 0**, and *both* parties'
safety gates — and Experiment 6 already demonstrated the recipient side
behaviourally (H2: the accept region's upper edge is set by the
counterparty's ability to pay, not by our valuation). If most high-gain-for-us
candidates are infeasible for the recipient, the teacher's pick is the best
*feasible* one, which a one-sided ranking cannot reproduce at any weighting.

**Next step:** re-extract features with the recipient's valuation and both
safety gates included, then re-fit on the same fixed split. If a two-sided
feasibility filter alone lifts top-1 substantially, that confirms the
diagnosis before any weight tuning. The split and the held-out play set both
stay untouched so the check remains honest.

**Phase 2 stays open.** Neither defect is closed, `sell_house` ordering is a
known F4 defect, and G1's correction is net-negative until the gate lands.

## D2.5 — Three diagnoses, three refutations, and what they jointly imply

The trade-ranking failure has now survived three separate explanations. Each
was tested in isolation before anything was built on it, and each was refuted
by its own test rather than by a later regression.

| # | diagnosis | test | outcome |
| --- | --- | --- | --- |
| 1 | weights are miscalibrated | 4,000-iteration search on train | **refuted** — nothing beat the initialisation; train top-1 stayed 12.7% |
| 2 | features are one-sided | feasibility filter alone, no tuning | **refuted** — top-1 fell (12.7%→10.1%), and the filter discarded the teacher's own pick in 60/118 cases |
| 3 | marginals are non-separable | joint `state_value` swap delta vs difference of marginals | **refuted** — scores change by up to 23× relative, argmax identical on 40/40 boards |

Refutation 3 is the informative one. Replacing a difference of marginals with a
genuine whole-position valuation moved candidate scores by more than an order
of magnitude and **changed which candidate ranked first exactly zero times**.
That is not a small effect failing to help; it is a large effect that cannot
reach the argmax.

**What all three have in common.** Weight changes, feasibility filters and a
restructured valuation all left the same candidate on top. A ranking that
refuses to move under three independent large perturbations is being decided
by a single dominant term, and everything else is rounding error against it.

The suspect is the monopoly term. `max_group_rent(...) / 2**missing` is
hundreds to low thousands, while list price is $60–$400 and projected rent is
tens of dollars. Any candidate touching a group we have presence in therefore
outranks every candidate that does not, regardless of price, rent, recipient
gain, or group interaction — which is precisely the invariance observed.

If that is right, it also explains refutation 2 without extra assumptions:
recipient gains computed from the same dominant term would be mis-signed on
exactly the trades where group structure changes hands, which is the ~50% of
the teacher's picks the filter rejected.

**Next test, and it is a cheap ablation, not a fit.** Rank by each component
*alone* — price only, rent only, monopoly only — on the fixed split, and
compare against the combined model's 12.7%/8.5%. Three outcomes, all
informative:

- monopoly-only reproduces the combined model → the term dominates as
  suspected, and the fix is scale, not structure;
- price-only or rent-only beats the combined model → the monopoly term is
  actively harmful and should be down-weighted or dropped;
- none of them reaches 12.7% → no single component carries the signal, and the
  valuation is wrong in kind rather than in proportion, which would mean the
  teacher is not ranking trades by a state-value difference at all.

That third outcome is worth taking seriously. If it lands, the next move is
not another valuation variant but a direct probe of *what the teacher's trade
choice actually correlates with*, measured rather than assumed.

**Phase 2 remains open.** No further valuation change until this ablation runs.

## D2.6 — Component ablation on the full pool, and the threshold-vs-ranking reconciliation

Run before anything from D2.5 is allowed into `SPEC.md`, on the full 400-board
pool (7,675 candidates, 118 proposals), board-level 60/40 split, Wilson 95%
intervals on every arm.

### Part 1 — which component drives the trade ranking

| arm | train top-1 | held-out top-1 |
| --- | --- | --- |
| combined (current) | 12.7% [6.8, 22.4] | 8.5% [3.4, 19.9] |
| **monopoly only** | **12.7% [6.8, 22.4]** | **8.5% [3.4, 19.9]** |
| price only | 7.0% [3.0, 15.4] | 4.3% [1.2, 14.2] |
| rent only | 5.6% [2.2, 13.6] | 8.5% [3.4, 19.9] |
| no monopoly | 5.6% [2.2, 13.6] | 4.3% [1.2, 14.2] |
| monopoly x0.1 | 8.5% [3.9, 17.2] | 6.4% [2.2, 17.2] |
| *random-pick reference* | *4.3%* | *4.3%* (mean 23.4 candidates) |

**The dominance hypothesis is confirmed, and not statistically — identically.**
Monopoly-only reproduces the combined model exactly on both splits (9/71 and
4/47, the same boards). Dropping the term costs more than half the accuracy.
This is the same fact the 40/40 argmax invariance showed, now with the cause
named: the ordering *is* the monopoly term, and price and rent are decoration.

**But the absolute numbers cannot support a SPEC rule.** The held-out interval
for every arm contains the 4.3% random-pick reference. At 47 proposals nothing
here is distinguishable from guessing, and the arms were chosen after seeing
earlier results, so train is not clean either. The correct statement is
"monopoly-only is identical to combined", which is an identity over the same
boards and does not need statistics. Any claim about which component ranks
*better* is unsupported and is not being made.

### Part 2 — the same perturbations on auction, a threshold decision

402 randomly configured auction states.

| arm | auction agreement | Δ vs combined |
| --- | --- | --- |
| combined (current) | 78.6% [74.3, 82.3] | — |
| price only | 79.4% [75.1, 83.0] | +0.7 |
| **no monopoly** | **83.6% [79.6, 86.9]** | **+5.0** |
| **monopoly x0.1** | **86.8% [83.2, 89.8]** | **+8.2** |
| monopoly only | 68.7% [64.0, 73.0] | −10.0 |
| rent only | 36.6% [32.0, 41.4] | −42.0 |

### Reconciliation — the proposed explanation is only half right

The hypothesis was: buy and auction survive a dominant monopoly term because
they compare one candidate against a fixed gate, where a term that shifts the
ceiling far above the standing bid is harmless, whereas ranking collapses under
a dominant additive term. Tested rather than assumed, it splits:

**Confirmed for buy — but for a more basic reason than proposed.** `_buy` does
not call `deed_value` at all; it is `gates_ok(env, pid, price)` and nothing
else. Buy's 96–98% is evidence about the safety gates exclusively and can be
cited neither for nor against the valuation. The apparent paradox for half the
cases was never a paradox; it was me quoting an agreement number for a code
path that does not exist.

**Refuted in its strong form for auction.** Auction is *not* insensitive to
the term. Removing it improves agreement by 5.0 points and shrinking it to a
tenth by 8.2 — well outside the intervals. The monopoly term is actively
harmful in auction too; auction merely survives it, at 78.6%, because price
and rent carry the decision (price-only alone scores 79.4%, and stripping rent
collapses it to 36.6%).

**What is actually true.** The same defect damages both decisions, by very
different amounts, and the threshold/ranking distinction explains the
*magnitude* rather than the presence:

- in a threshold comparison the term is one addend among three, so being wrong
  costs a bounded 5–8 points;
- in a ranking it is the *only* term that separates candidates, so being wrong
  costs everything — the ordering is fully determined by it.

So the finding is not "the monopoly term is fine for thresholds and bad for
rankings". It is **"the monopoly term is wrong, and ranking is simply the
decision that has no other term to fall back on."**

### Consequences

1. **Nothing from D2.5/D2.6 goes into `SPEC.md`.** These are findings about
   *our model*, not about the teacher's behaviour. `SPEC.md` documents
   observed teacher behaviour; a defect in `spec_model.py` belongs here.
2. **An immediate, measurable improvement is available and is not speculative**:
   scaling the monopoly term to 0.1 gains +8.2 points of auction agreement on
   402 states, interval [83.2, 89.8] against [74.3, 82.3]. That is worth
   taking on its own merits, independently of trade.
3. **The trade ranking still has no working model,** and the pool is too small
   to choose between candidate models. Widening it means more *proposal*
   boards, not more boards — 400 boards yielded only 118. The next step is a
   generator biased toward states where the teacher actually proposes, so the
   118 becomes ~500, before any further model is compared.
4. `max_group_rent / 2**missing` needs re-deriving from probe evidence rather
   than from the published formula. B3/B4/B5 measured *auction ceilings*, and
   ceilings constrain the term only up to the additive company it keeps —
   which is exactly the freedom that let a wrong term reproduce them.

## D2.7 — Monopoly x0.1 measured on the real distribution: it does not transfer

D2.6 measured scaling the monopoly term to 0.1 as worth **+8.2pp of auction
agreement** on 402 randomly generated auction states, [83.2, 89.8] against
[74.3, 82.3], non-overlapping. Applied and re-measured on held-out play, with
an A/B on **identical code** (the scale is now an env var so the two arms are
not two different commits):

| | scale 1.0 | scale 0.1 | Δ |
| --- | --- | --- | --- |
| auction | 90.5% | 88.7% | **−1.8** |
| turn flow | 93.3% | 88.3% | **−5.0** |
| development order | 30.3% | 29.1% | −1.2 |
| trade reply | 78.1% | 78.4% | +0.3 |
| trade proposal | 0.2% | 1.6% | +1.4 |
| **TOTAL (5,363 decisions)** | **73.4%** | **70.7%** | **−2.7** |

Auction moves **−1.8pp, not +8.2pp**, and the overall figure falls 2.7pp.
**Default reverted to 1.0.**

**This is the same failure mode for the fourth time,** and it is worth naming
plainly rather than filing as bad luck:

| # | narrow sample | conclusion | refuted by |
| --- | --- | --- | --- |
| 1 | p08 trade cell, 14 states, one board | rollout never changes trade decisions | Exp 6: 92.6% (D1.3) |
| 2 | p09, 80 states, one board shape | offer cheapest spare for the completing deed | held-out: 27/189 on the requested deed |
| 3 | p07, 224 states, one setup | never leaves jail in pre_roll | p10: 106/250 (D2.3) |
| 4 | D2.6, 402 random auction boards | monopoly x0.1 gains +8.2pp | this A/B: −1.8pp |

Case 4 is the sharpest because the sample was *large* (402 states, tight
intervals) and still wrong. Sample size was never the problem — **the
generator's distribution was**. Random deed allocations and random positions
do not produce the auction states that arise after teacher-driven play, and no
amount of widening a synthetic generator fixes a mismatch with the target
distribution.

**Standing rule from here:** a change is accepted only when measured on
held-out *play*, not on synthetically generated states. Synthetic probes stay
useful for isolating mechanism — that is what recovered A1–A6 and D1–D5 to the
dollar — but they do not decide whether a change ships. This should have been
the rule after case 1.

## D2.8 — First head-to-head: the clone is well short of the teacher

`bench.py`, 2 seats `spec` vs 2 seats `ASUValueV1`, seat-rotated across two
arrangements (spec on 0,2 then on 1,3), 30 games each, 60 total, all decisive.

| | |
| --- | --- |
| spec wins | 16 |
| teacher wins | 44 |
| **spec win rate** | **26.7%, 95% Wilson CI [17.1, 39.0]** |
| parity | 50.0% |

The interval excludes parity, so this is a real deficit, not noise. Per-seat
net worth tells the same story: spec averages $2.0k–$7.1k against the
teacher's $10.7k–$14.2k, and goes bankrupt in 80–93% of games against 57–70%.

This is the expected consequence of 73.4% agreement concentrated in the wrong
place: trade proposal is 16% of decisions at 0.2% agreement, and the clone
either proposes badly or ends its turn where the teacher trades. Phase 2's
acceptance also asks for the spec policy to be within 5 win-rate points of the
value teacher; at 26.7% vs 73.3% it is 46.6 points short.

**Note on what was benched.** The brief asked for `HeuristicRolloutPolicy`
from `heuristic_policy.py` with an early-denial bonus, versus `ASURolloutV1`.
Neither exists: `heuristic_policy.py` is absent from the tree (it was one of
the six phantom "existing starting points" recorded in D0.1) and no
early-denial exploit was built in any of this session's commits. The nearest
real measurement was run instead and is labelled as such. `ASUValueV1` was
used rather than `ASURolloutV1` because the 200-game rollout reference has now
run over 2.5 hours without completing a single game, so a rollout head-to-head
is not feasible at this budget.

## D2.9 — Harvesting real-play trades: the signal was there all along

D2.7's rule said changes ship only on held-out play. The corollary, applied
here, is that they should be *fitted* on the target distribution too. Instead
of widening the synthetic generator to ~500 boards, `harvest_trades.py`
collects the same decisions from 60 teacher-driven games:

| | synthetic (p09b) | harvested (real play) |
| --- | --- | --- |
| decision states | 400 | **6,032** |
| proposals | 118 | **2,508** |
| candidates | 7,675 | **358,042** |
| mean candidates / decision | 23.4 | 64.8 |

**The synthetic pool was not merely small — it was misleading.** On it, every
model scored ~8.5% against a 4.3% random reference, intervals overlapping, and
D2.5's third branch ("no component carries the signal") looked live. On real
play *every single feature* beats its 1.54% random reference, by 4× to 13×.
The signal was never absent; it was absent from the generated boards.

`analyze_trades.py` measured the decision rather than proposing a model. The
chosen candidate has a clear profile against the pool it was drawn from:

| quantity | chosen | pool avg |
| --- | --- | --- |
| requested deed, projected rent to us | **29.05** | 16.55 |
| requested deed, our deeds in its group | **0.98** | 0.45 |
| offered deed, projected rent to us | **6.51** | 11.44 |
| offered deed, list price | **156** | 189 |

It asks for high-rent deeds in groups it already holds part of, and gives away
cheap low-rent deeds from groups it does not. Best single feature — rent
difference — reaches 20.85% [18.43, 23.49] on its own.

`fit_trade_v2.py` searched weights on 1,520 train proposals (split by game
seed, since decisions inside a game share a board and would leak):

    held-out top-1  29.86%  [27.09, 32.79]      (988 proposals)
    rent only       20.85%
    random           1.54%

Train scored 25.72% — *below* held-out, so nothing is overfitted. The monopoly
term is deliberately absent from these features, per D2.6.

## D2.10 — Agreement up 4.1pp, win rate down: the objectives diverge

Fitted ranking plus a gate threshold (3.92, fitted on harvest train, 60.1%
held-out against a 57.4% never-propose baseline) applied to `spec_policy`:

| category | before | after |
| --- | --- | --- |
| trade proposal | 0.2% | **8.0%** |
| turn flow | 93.3% | **96.9%** |
| development order | 30.3% | **54.5%** |
| auction | 90.5% | 90.5% |
| **TOTAL held-out agreement** | **73.4%** | **77.5%** |

The gate matters more than the ranking: without it, proposals fire on every
positive score and steal END_TURN and build decisions (turn flow falls to
85.9%, development to 17.0%, total to 72.7%) even though trade proposal itself
reaches 25.3%. With it, trade proposal drops to 8.0% but everything else
recovers and the total gains 4.1pp.

**The head-to-head went the other way:**

| | spec wins | win rate |
| --- | --- | --- |
| before the trade fit | 16/60 | 26.7% [17.1, 39.0] |
| after | 12/60 | **20.0% [11.8, 31.8]** |

The intervals overlap heavily, so this is not a significant *decline* — but it
is certainly not the improvement the agreement gain predicts. **Imitation
fidelity and playing strength are different objectives, and this is the first
direct evidence of them diverging in this project.** A clone can match more of
the teacher's decisions while losing more games, because the decisions it
still gets wrong are not weighted by how much they cost.

That is worth stating plainly because Phase 2's two acceptance criteria assume
they move together: ">= 90% agreement" and "within 5 win-rate points of the
value teacher". At 77.5% and 20.0% the agent fails both, and closing the first
is not on its own a route to the second.

**Phase 2 status: still failing.**

| criterion | target | actual |
| --- | --- | --- |
| held-out decision agreement | >= 90% | **77.5%** |
| on-policy agreement (10 fresh games) | >= 85% | not yet measured |
| win rate vs value teacher | within 5 pts | **20.0% vs 80.0%** |

Remaining known defects, in order of measured cost: trade proposal is still
8.0% on 16% of decisions; trade reply sits at 78.1% on another 15%;
liquidation order is 23.8%; unmortgage 45.7%.

## D2.11 — Covariate shift in the trade fit: flagged, not yet critical

The trade ranking and gate (D2.9) were fitted on states harvested from **60
teacher-vs-teacher games** — seat 0 was `ASUValueV1` and so were, effectively,
the dynamics that produced every board it saw.

In a real match the agent plays `spec_policy` against the teacher. Two things
differ:

1. **The opponent is different.** The boards seat 0 encounters are produced by
   a teacher reacting to *spec's* moves, not to another teacher's. Deed
   allocations, cash levels and development trajectories will drift from the
   harvest distribution.
2. **Our own trajectory is different.** The harvest recorded states reached by
   a teacher playing optimally-by-its-own-lights. A clone at 77.5% agreement
   reaches materially different positions — typically worse ones, given it
   goes bankrupt in 86–93% of head-to-head games against the teacher's 60%.

This is ordinary covariate shift, and it is exactly what DAgger (Phase 3 in
the brief) exists to fix: iterate self-play, label the states the *clone*
visits, retrain.

**Not treated as critical yet**, because the fit is validated on held-out play
agreement (77.5%, up 4.1pp) which is measured on the same teacher-driven
distribution, and because the ranking's held-out top-1 (29.86%) was measured
on games the fit never saw. But the validation shares the harvest's bias, so
it cannot detect this failure mode.

**Trigger:** if a win-rate improvement lands materially below what the
agreement gain predicts, this is the first place to look — before any further
tuning of the ranking itself. D2.10 already shows one instance of agreement
rising while win rate did not, which is consistent with covariate shift
although not diagnostic of it on its own.

**Cheap check when needed:** re-harvest with seat 0 driven by `spec_policy`
instead of the teacher (the teacher still supplies the ground-truth label at
each state), refit, and compare. That is one DAgger iteration and it directly
tests whether the distribution is the problem.

## D2.12 — Cost-weighted defect ranking: agreement was ranking the wrong things

`pinned_ablation.py` pins one family at a time to the teacher's ground truth
and measures win-rate recovery. 40 seat-rotated games per arm.

**Harness validated first:** the `all` arm returns **50.0% [35.2, 64.8]** — an
agent pinned on every family is the teacher, and scores like it. Without that
check none of the other arms would mean anything.

| pinned family | win rate | Δ vs none | fires on | agreement |
| --- | --- | --- | --- | --- |
| none (baseline) | 27.5% [16.1, 42.8] | — | — | — |
| **trade_proposal** | **37.5% [24.2, 53.0]** | **+10.0** | 2.3% | 8.0% |
| liquidation | 27.5% [16.1, 42.8] | +0.0 | 4.6% | 23.8% |
| unmortgage | 27.5% [16.1, 42.8] | +0.0 | 0.8% | 45.7% |
| auction | 27.5% [16.1, 42.8] | +0.0 | 3.2% | 90.5% |
| turn_flow | 27.5% [16.1, 42.8] | +0.0 | 67.3% | 96.9% |
| development | 25.0% [14.2, 40.2] | −2.5 | 2.3% | 54.5% |
| trade_reply | 20.0% [10.5, 34.8] | −7.5 | 3.4% | 78.1% |
| all (upper bound) | 50.0% [35.2, 64.8] | +22.5 | 100% | — |

**The two rankings disagree, and the agreement one was wrong.**

- **Liquidation, agreement's worst family at 23.8%, costs nothing.** It fires
  on 4.6% of decisions and recovers exactly 0.0pp. Unmortgage (45.7%) the
  same. Both were near the top of the "fix next" list by agreement and both
  are worthless to fix. This is precisely the hypothesis that motivated the
  ablation.
- **Trade proposal is the only family with a positive effect**, +10.0pp on
  2.3% of decisions — **+4.34pp per 1% of decisions touched**, the best rate
  in the table by a wide margin.
- The gap between `all` (+22.5) and the sum of individual arms is large, which
  suggests the remaining loss is distributed or interactive rather than
  sitting in one family.

**Power caveat, stated before any allocation decision.** At 40 games per arm,
*every* interval overlaps the baseline's [16.1, 42.8]; only `all` is
distinguishable. The negative arms are almost certainly noise — pinning a
family to ground truth should not make play worse — and four arms landing on
exactly 11/40 indicates their changed decisions rarely flip a game rather than
that their effect is precisely zero. **This ranking is suggestive, not
settled.**

A confirmatory run over `none, trade_proposal, trade_reply` at 60 games per
arrangement (360 games, seeds 930000+) is in flight, using the streaming
version so it is visible and resumable.

**Provisional allocation, pending that run:** trade proposal first. It is both
the top of the cost ranking and still the worst-agreeing family with headroom
(8.0%), so the two rankings agree on it even though they disagree elsewhere.
Liquidation, unmortgage and auction are explicitly **de-prioritised despite
their agreement numbers** — that is the whole point of measuring cost.

## D2.13 — Confirmatory ablation: trade is the whole gap, and the two families are additive

480 games, 120 per arm, fresh seeds 930000+, seat-rotated. Three times the
games of the exploratory run, which was the right call — one of its findings
inverted.

| pinned family | win rate | Δ vs none | fires on | per 1% of decisions |
| --- | --- | --- | --- | --- |
| none | 26.7% [19.6, 35.2] | — | — | — |
| trade_reply | 33.3% [25.5, 42.2] | **+6.7** | 3.6% | +1.87pp |
| trade_proposal | 40.0% [31.7, 48.9] | **+13.3** | 2.6% | **+5.21pp** |
| **trade_both** | **45.8% [37.2, 54.7]** | **+19.2** | 4.3% | +4.49pp |

### Three results

**1. The n=40 negative was noise, exactly as suspected.** `trade_reply`
measured **−7.5pp** at 40 games and **+6.7pp** at 120 — it flipped sign. The
reasoning that rejected it ("pinning a family to ground truth should not make
play worse") held, and the insistence on more games before allocating was
correct. Had we acted on the exploratory run we would have written off a
family worth +6.7pp.

**2. The two families are additive; there is no interaction.**

    trade_proposal alone   +13.3
    trade_reply alone      + 6.7
    sum if independent     +20.0
    trade_both measured    +19.2   (difference −0.8pp, inside noise)

So the `trade_both` arm answers its question in the first of the three ways it
was set up to distinguish: the proposal fix stands on its own, the reply fix
stands on its own, and neither is hostage to the other. They can be worked
sequentially without one blocking the other.

**3. Fixing trade alone is statistically indistinguishable from parity.**
`trade_both`'s interval **[37.2, 54.7] contains 50.0%**, and 50% is parity by
construction in a 2v2 of identical policies. The `all` arm — every family
pinned — measured 50.0%. So trade accounts for essentially the entire gap
between the clone and the teacher: 26.7% → 45.8% of a 50% ceiling.

### This refutes my own earlier inference

D2.12 noted the gap between `all` (+22.5) and the sum of individual arms and
speculated that "the remaining loss is distributed or interactive rather than
sitting in one family", and that per-family fixes would therefore have a
ceiling well below parity with Phase 3 being the better route. **That was
wrong**, and it was wrong because it reasoned from underpowered arms. The loss
is highly concentrated: two families covering 4.3% of decisions carry ~19 of
the ~23 points.

### Allocation and Phase 3

**Allocation: trade_proposal first, trade_reply second.** Proposal is the most
efficient per decision touched (+5.21pp per 1%) and has the larger absolute
effect. Reply follows. Everything else — liquidation (23.8% agreement),
unmortgage (45.7%), auction, development, turn flow — stays de-prioritised;
the exploratory run measured all of them at +0.0pp and nothing here changes
that.

**Phase 3 is not the next move.** The hybrid/DAgger route was attractive on
the belief that the loss was distributed. It is not. A targeted fix to two
trade families has a measured ceiling of ~parity with the value teacher, which
is what Phase 2's acceptance actually asks for. Phase 3 should be reconsidered
after the trade work lands and its realised gain is compared against the
+19.2pp this predicts.

**D2.11's covariate-shift trigger is now armed and specific.** If the trade
fixes land materially below +19.2pp, the harvest distribution is the first
suspect — the fit came from teacher-vs-teacher games, and this ablation shows
trade decisions are precisely where the clone's play is decided.

## D2.14 — Additivity is provisional, and fitted models are the reason to doubt it

D2.13 measured `trade_both` (+19.2) against `trade_proposal` (+13.3) plus
`trade_reply` (+6.7), a −0.8pp residual, and concluded the two families are
independent. That conclusion is **provisional on one measurement**, and there
is a specific mechanism by which it could fail once both fixes are real code
rather than oracles.

**Why oracle additivity need not imply fitted additivity.** The oracle arms
substitute the teacher's *correct* action. Two correct components cannot
generate compounding error — each independently removes its own loss, so
additivity is close to guaranteed and the measurement mostly confirms the
absence of a strategic interaction (e.g. "good proposals are wasted if replies
are bad").

Fitted models are different. Both sides will be **wrong in correlated ways**,
because both are scored by the same deed-valuation features on the same board.
A proposal the fitted ranker likes for the wrong reason is exactly the kind of
trade the fitted reply rule is also likely to misjudge — and unlike the oracle
case, the two errors can reinforce. Concretely: the clone proposes a trade its
own valuation overrates, the opponent accepts because it is good *for them*,
and the reply rule that should have refused the mirror-image offer shares the
misvaluation that caused the proposal.

**So additivity must be re-tested, not inherited.** When both fixes are
implemented, run the three-way comparison again with fitted models in place of
oracles:

    fitted proposal only    vs floor
    fitted reply only       vs floor
    both fitted             vs floor

If `both` falls materially short of the sum, the two share a failure mode and
the shared component — the valuation — is the thing to fix, not either rule.
That would be the fourth time the valuation has been the root cause (D2.5,
D2.6, D2.12), which is itself a reason to expect it.

Recorded now so the check is not skipped by inertia once the first fix shows a
gain.

## D2.15 — Fitted vs oracle: the model captures ~37%, and that reverses the Phase 3 decision

Measured with the fitted trade model live in `spec_policy.py`, all three points
on the same seeds (930000+) and seat rotation:

| point | | win rate |
| --- | --- | --- |
| floor — proposals disabled (`TRADE_GATE=1e9`) | 15/80 | 18.8% [11.7, 28.7] |
| **fitted** — the 29.86% top-1 model | 32/120 | **26.7% [19.6, 35.2]** |
| oracle — teacher's actual proposal | 48/120 | 40.0% [31.7, 48.9] |

    fitted real gain   +7.9pp
    oracle ceiling    +21.3pp
    CAPTURE RATIO      37.3%

**The fitted model does convert to wins.** This also resolves D2.10, which
worried that agreement rose while win rate did not: that comparison used 60
games on different seeds and was noise. On matched seeds with more games the
fit is worth +7.9pp over never proposing.

**But it is not statistically significant on its own** — floor vs fitted gives
z = 1.29, p = 0.196, intervals overlapping. The point estimate is the best
available and the direction is consistent with the oracle result, but +7.9pp
is not established. Widening this specific comparison is cheap and should
happen before the number is leaned on hard.

### This reverses D2.13's Phase 3 recommendation

D2.13 deferred Phase 3 on the grounds that oracle-pinned trade reached parity,
so targeted fixes had a ceiling of ~parity. **That ceiling assumed perfect
trade decisions.** The fitted model reaches 37% of it, and there is no reason
to expect a second hand-fitted rule to do better — it would use the same
features on the same valuation.

Projection if `trade_reply` is fitted with the same method and captures the
same share:

    trade_reply fitted   ~ +2.5pp   (37% of its +6.7 oracle)
    total                ~ 29.2%    against parity of 50%

> **THESE TWO NUMBERS ARE ESTIMATES, NOT MEASUREMENTS.** No fitted
> `trade_reply` exists; +2.5pp is 37.3% of that family's *oracle* gain, and it
> assumes the capture ratio measured for one family transfers to another. It
> may not: the reply side is a binary accept/decline against a single offer,
> not a ranking over ~65 candidates, so its features and its achievable
> capture could differ in either direction. The ~29.2% total inherits the same
> assumption and compounds it with the additivity caveat of D2.14.
>
> They are used here only to argue that hand-fitting is unlikely to reach
> parity — a conclusion that holds across a wide range of plausible capture
> ratios, since even 100% capture on `trade_reply` would give ~33.4%. Nothing
> downstream should treat +2.5pp or 29.2% as established.

So hand-fitted rules plausibly get to ~29%, not ~46%. The remaining ~21 points
are inside the trade families but **not reachable by better weights on these
features** — three independent refutations (D2.5, D2.6, D2.12) already showed
the valuation, not the weighting, is the limit.

**That is precisely what Phase 3's learned component is for.** A network over
the 300-dim observation is not restricted to the four hand-chosen features
that cap the current ranker at 29.86% top-1, and DAgger addresses the
covariate shift flagged in D2.11 at the same time.

**Revised recommendation: proceed to Phase 3, and do not fit `trade_reply` by
hand first.** The evidence for that ordering:

1. per-family hand fitting has a measured capture of 37%, giving ~29% overall;
2. the binding constraint is the feature set, which Phase 3 replaces rather
   than reweights;
3. the trade families are where decisions are made (D2.13), so a learned
   component targeted there has the largest measured headroom in the project —
   ~21 points.

D2.13's "Phase 3 deferred" is superseded. It was reasoned from the oracle
ceiling, which is exactly the substitution this measurement was run to avoid.

**Additivity check (D2.14) is deferred with it.** It only becomes meaningful
once a second real implementation exists, and the recommendation is now not to
build that second implementation as a hand-fitted rule.

## D3.1 — Phase 3 scope: the network owns trade only, and why

The brief's hybrid is "rules decide when confident, the network breaks ties /
covers uncovered states". The ablation already measured where that boundary
is, so the network is aimed rather than general:

| family | agreement | win-rate cost |
| --- | --- | --- |
| ROLL_DICE / BUY / turn flow / auction | 90–99% | **+0.0pp** |
| liquidation | 23.8% | **+0.0pp** |
| trade reply | 78.1% | +6.7pp |
| trade proposal | 8.0% | +13.3pp |

Every family outside trade recovers **zero** win rate when pinned to ground
truth, so a network covering them adds risk and latency for no measured
return. Liquidation is the sharpest case — the worst agreement number in the
project and worth nothing. `distill_collect.py` therefore records only states
offering a trade choice, and the head scores only
`buy_trade / sell_trade / exch_trade / ACCEPT / DECLINE`.

## D3.2 — Covariate shift confirmed on first contact

D2.11 flagged that the trade fit came from teacher-vs-teacher play while
matches are decided in states the *clone* reaches. The DAgger collector
measures this directly, and the distributions differ immediately:

    teacher driving (harvest, D2.9)   teacher proposes in 41.6% of states
    student driving (DAgger iter 0)   teacher proposes in 64.3% of states

A 23-point gap in the label distribution alone. The states the clone reaches
are ones where trading is *much* more often correct — consistent with it
holding worse positions. This is no longer a flagged risk; it is a measured
property of the data, and it is the reason iteration 0 uses the student as
driver rather than reproducing the old harvest.

## D3.3 — Pipeline validated, data scale is the blocker

End-to-end smoke test on 6 games (420 trade states, split by game seed):

    train top-1     93.04%
    held-out top-1  13.46%
    hand-fitted reference (D2.9)  29.86%

The gap is textbook overfitting: 316 training states against a 2,774-way
masked-softmax output. The pipeline is correct — loss falls, training accuracy
rises, the seed-level split holds — but the sample is roughly two orders of
magnitude too small, and **the learned head is currently worse than the rule it
is meant to replace.**

Scale required, from the observed ~81 trade states per game:

    60 games   ~  4,850 states
    300 games  ~ 24,250 states
    800 games  ~ 64,700 states

**No claim about the learned approach can be made from 13.46%.** It is not
evidence against D2.15's reasoning — that a network over the 300-dim
observation is not confined to four hand-chosen features — because the model
has not been given enough data to test that. The honest statement is that the
Phase 3 machinery works and is not yet trained.

Next: collect at scale (300+ games) once the widened floor-vs-fitted
measurement releases the CPU, then compare held-out top-1 against the 29.86%
hand-fitted reference before wiring anything into `hybrid_policy.py`. If the
network does not clear that bar with adequate data, D2.15's recommendation to
prefer Phase 3 over further hand-fitting is itself refuted and should be
revisited.

## D2.16 — Why 300 games per arm: the power calculation behind z ≈ 2.31

Recorded so "why 300?" has an answer later.

**What the target represents.** z ≈ 2.31 is the two-proportion z-statistic for
the **floor-vs-fitted win-rate comparison** — proposals disabled
(`TRADE_GATE=1e9`) against the fitted ranker live. It corresponds to
p ≈ 0.021, i.e. the comparison clearing the conventional 0.05 bar with a
little margin, and the two 95% Wilson intervals separating rather than
overlapping as they do now.

**Why it was needed.** At the sizes available when D2.15 was written the
comparison was 15/80 against 32/120: a +7.9pp point estimate but z = 1.29,
p = 0.196, intervals overlapping. That is not enough to carry a phase-level
decision, which is what it was being asked to do.

**The calculation.** Holding the observed rates (p₁ = 0.188, p₂ = 0.267,
difference 7.9pp) and using the pooled two-proportion standard error
`sqrt(p̄(1-p̄)·2/n)` with p̄ = 0.2275:

| n per arm | expected z | expected p |
| --- | --- | --- |
| 120 | 1.46 | 0.14 |
| 180 | 1.79 | 0.073 |
| 240 | 2.06 | 0.039 |
| **300** | **2.31** | **0.021** |

300 was chosen as the first size giving margin past 1.96 rather than landing
on it — at 240 the projected z is 2.06, close enough that ordinary run-to-run
variation could drop it back under the bar and leave the question open after
paying for the games.

**What it is conditional on.** The projection assumes the observed 7.9pp
difference is the true effect. If the real difference is smaller, 300 games
will not reach significance — and that outcome is itself informative, since it
would mean the fitted ranker's win-rate contribution is smaller than D2.15
estimated and the 37.3% capture ratio is optimistic.

**Two different bars, not to be conflated.** This z target governs the
*win-rate* comparison. The 29.86% figure is the hand-fitted ranker's
*top-1 agreement* and is the bar for the Phase 3 network's held-out top-1
(D3.3) — a different metric on a different comparison. A network beating
29.86% top-1 would still need its own win-rate measurement before any claim
about playing strength.

**Sequencing.** The widened measurement and the Phase 3 collection run
strictly one after the other, never sharing cores. Concurrent heavy jobs
starved each other earlier in this project and cost a 2h09 run (D0.8); more
importantly, win rates measured under contention are still valid but the
timings are not, and both jobs are long enough that interleaving them would
roughly double wall-clock for no gain.

## D2.17 — Widened floor-vs-fitted: significant at p = 0.030, capture 33.8%

300 games per arm, matched seeds (930000+), seat-rotated, fitted model live.

| | | win rate |
| --- | --- | --- |
| floor — proposals disabled | 55/300 | 18.3% [14.4, 23.1] |
| **fitted — the 29.86% top-1 ranker** | 77/300 | **25.7% [21.1, 30.9]** |

    difference   +7.3pp
    z             2.17   (projected 2.31)
    p             0.0301

**The comparison is significant.** The fitted trade ranker is worth +7.3pp of
win rate over never proposing, at p = 0.030. D2.15's +7.9pp point estimate
held up under 3.75x the games, landing at +7.3pp.

**But the two 95% intervals still overlap** — fitted's lower bound 21.1%
against floor's upper bound 23.1% — and the (a) condition asked for them to
separate. That is worth stating precisely rather than glossing:

Overlapping 95% CIs do **not** imply p > 0.05. The two-proportion z-test is
the correct test of a difference; the CI-overlap heuristic is a distinct and
substantially more conservative criterion, roughly equivalent to demanding
p ≈ 0.005. So the honest report is: **significant by the standard test
(p = 0.030), not separated by the stricter overlap criterion.** Reaching the
latter would need roughly 700–800 games per arm at this effect size.

z came in at 2.17 against the projected 2.31 because the realised difference
was 7.3pp rather than the 7.9pp the projection assumed — exactly the
conditional D2.16 recorded.

**Capture ratio, revised down slightly:**

    oracle ceiling  40.0% - 18.3%  = +21.7pp
    fitted gain     25.7% - 18.3%  = + 7.3pp
    CAPTURE                          33.8%   (was 37.3% on the smaller sample)

**D2.15's Phase 3 recommendation stands and is now on firmer evidence.**
Hand-fitting captures about one third of what perfect trade decisions are
worth. The remaining two thirds sit inside the trade families and are not
reachable by reweighting these features (D2.5, D2.6, D2.12). Proceeding to
Phase 3 collection at scale.

## D3.4 — Third repetition of the same instrumentation failure

`distill_collect.py` was written with `pool.map`, which holds every result in
memory and writes only on completion. A 320-game collection ran **2h11 with no
visible progress** and was killed, discarding all of it.

This is the third occurrence of one pattern:

| # | harness | cost | fix |
| --- | --- | --- | --- |
| 1 | `bench.py` | 2h09 reference run lost | streaming + resume (D0.8) |
| 2 | `pinned_ablation.py` | 1h24 opaque, nothing recoverable | streaming + resume |
| 3 | `distill_collect.py` | 2h11 lost | streaming + resume (here) |

Each fix was written, committed, and then **not carried to the next harness**.
That is a habit, not three separate oversights, and the correct reading is that
any new long-running harness in this project starts with streaming and resume
rather than acquiring them after losing a run.

**The larger cost was diagnostic, not compute.** After restarting with progress
output the same 320 games run at 15.5 games/min — about 21 minutes total. The
killed run had been going 2h11 on identical parameters, workers and a free
machine. **Why it was ~6x slower is now unknowable**, because killing it
destroyed the evidence. With progress output the anomaly would have been
visible within five minutes and diagnosable while it was happening. The lost
compute is replaceable; the lost explanation is not.

No cause is offered here. The plausible candidates (chunking under `map`,
memory pressure, a degenerate game) were not tested and inventing one would be
worse than recording the gap.

## D3.5 — The network clears the bar: 38.51% vs 29.86%

52,590 trade states from 320 student-driven games; 40,956 usable after
restricting to in-scope labels with a real choice. Split by game seed:
28,403 train / 12,553 held out.

    epoch  1   train 63.76%   held-out 35.70%
    epoch  5   train 81.39%   held-out 38.22%
    epoch 15   train 91.96%   held-out 38.49%
    epoch 30   train 97.76%   held-out 37.66%

    best held-out top-1   38.51%
    hand-fitted reference 29.86%   (D2.9)

**The learned head beats the hand-fitted ranker by +8.65pp of top-1**, and the
decision rule set in D3.3 resolves in its favour. D2.15's reasoning — that the
binding constraint was the four hand-chosen features, not their weights — is
supported: given the same task and the full 300-dim observation, a model finds
substantially more signal.

D3.3's 13.46% was a data-scale artefact, as recorded. 316 training states
became 28,403 and held-out top-1 went 13.46% → 38.51%.

**Still overfitting, and more epochs do not help.** Held-out peaks at epoch 5
(38.22%) and is flat-to-declining through epoch 30 while train climbs to
97.76%. The gap says regularisation, more data, or a smaller head — not
longer training. This has not been tuned; 38.51% is a first, untuned number.

### The win-rate implication is much smaller than the top-1 jump suggests

Interpolating between the two measured points — hand-fitted 29.86% top-1 worth
+7.3pp (D2.17), oracle 100% worth +21.7pp:

    network 38.51% top-1  ->  projected +9.1pp
    gain over hand-fitted ->  +1.8pp

**A +8.65pp gain in top-1 projects to roughly +1.8pp of win rate — a 1:4.9
dilution.** That is consistent with everything measured so far: agreement and
playing strength are different objectives (D2.10), and the capture ratio is
33.8% (D2.17).

**This projection is an assumption, not a measurement.** It presumes top-1 maps
linearly to win rate between the two anchors, which nothing establishes — the
decisions the network newly gets right may be systematically cheaper or dearer
than average. It is recorded to set expectations before the measurement, in
the same spirit as marking the trade_reply estimate (D2.15), and the actual
head-to-head is what decides.

### Next

Wire the head into `hybrid_policy.py` behind a flag — rules everywhere except
the two trade families, network there — and run the same floor/fitted/network
head-to-head. If the realised gain is near +1.8pp, the honest conclusion is
that Phase 3 bought a real but small improvement at considerable complexity,
and Phase 5's structural work is the better remaining investment.

## D3.6 — Hybrid head-to-head: +8.65pp of top-1 buys nothing, and regularisation does not move the ceiling

### The head-to-head, raw

`hybrid` (rules everywhere, learned head on the two trade families) against
`ASUValueV1`, seat-rotated, seeds 930000+ — the same base as every other arm.

| arm | win rate | n |
| --- | --- | --- |
| floor — proposals off | 18.3% [14.4, 23.1] | 300 |
| fitted — hand-fitted ranker | 25.7% [21.1, 30.9] | 300 |
| **hybrid — learned head** | **24.0% [18.6, 30.4]** | 200 |
| oracle — perfect trades | 40.0% | 120 |

    hybrid - fitted   -1.7pp     (projected +1.8pp)
    z = -0.42, p = 0.673

**The network did not beat the hand-fitted ranker.** The point estimate is
negative, the difference is not significant, and the projection was wrong in
sign. A +8.65pp improvement in held-out top-1 (29.86% → 38.51%) produced no
measurable win-rate gain.

The D3.5 projection of +1.8pp assumed top-1 maps linearly to win rate between
the two measured anchors. It was flagged as an assumption; it is now falsified.
Whatever decisions the network newly gets right are not the ones that decide
games.

### The regularisation iteration

Run before drawing any Phase 4 conclusion, since the first head overfitted
badly (train 97.76% vs held-out 38.51%, held-out peaking at epoch 5).

| config | train top-1 | **held-out top-1** |
| --- | --- | --- |
| hidden 512, no dropout, no wd | 97.76% | **38.51%** |
| hidden 256, dropout 0.3, wd 1e-4 | 81.78% | **37.98%** |
| hidden 128, dropout 0.4, wd 1e-3 | 74.85% | **38.52%** |

Regularisation works as intended on the *gap* — train accuracy falls from
97.8% to 74.9% — and **held-out top-1 does not move at all**: 38.51, 37.98,
38.52 across a 4x range of capacity and three regularisation strengths.

That is the informative result. The first head's overfitting was real but not
the binding constraint; ~38.5% is a genuine ceiling for this model class on
this data, not an artefact of capacity. More regularisation, more epochs or a
smaller head will not produce a better number.

### What this establishes, and what it does not

**Established:** the learned head reaches ~38.5% top-1 and that is worth
nothing in win rate over the hand-fitted 29.86%. Two independent objectives
have now diverged twice in this project (D2.10, and here), and the second time
was against a specific quantitative prediction.

**Not established:** that a learned approach cannot help. Untested are a
different input representation, a value-based rather than imitation objective,
and further DAgger iterations. What is refuted is the specific claim D2.15
made — that swapping four hand-chosen features for the 300-dim observation
would convert into playing strength. It converted into agreement and stopped
there.

**Phase 4 is now on solid ground.** The regularisation iteration was the
condition for that, and it came back flat. Continuing to tune this head would
be optimising a metric measured not to matter.

### Standing correction

D2.15 reversed D2.13's Phase 3 deferral on the strength of a capture-ratio
argument. That reversal produced a working pipeline and a clear negative
result — which is worth having — but its central prediction did not hold. The
project's record on this is now three for three: every time agreement was used
to predict strength (D2.10, D3.5, and the capture-ratio extrapolation itself),
the prediction failed. Win rate must be measured directly, and no further
agreement-based projection should be treated as load-bearing.

## D4.1 — Which oracle ceiling applies where: a correction

D2.17 through D3.6 quoted **40.0%** as "the oracle ceiling" for every capture
calculation. That is right for one comparison and wrong for the other.

**The ceiling has to match what the arm actually changes.**

| comparison | what it changes | correct ceiling | capture |
| --- | --- | --- | --- |
| floor → **fitted** | proposals only — `TRADE_W`/`TRADE_GATE` live only inside `_propose_trade`, and the floor (`TRADE_GATE=1e9`) disables only proposals | `trade_proposal` = **40.0%** | +7.4 / +21.7 = **34.1%** |
| floor → **hybrid** | proposals **and** replies — `HybridPolicy`'s scope is `buy/sell/exch_trade` **plus ACCEPT/DECLINE** | `trade_both` = **45.8%** | +5.7 / +27.5 = **20.7%** |

So the hand-fitted number (34.1%, previously quoted 33.8%) is essentially
unchanged, but **the hybrid's capture was overstated**: measured against its
own ceiling it captures **20.7%**, not the ~28% implied by dividing into 40.0%.

**This strengthens the negative conclusion rather than softening it.** The
learned head owns a *larger* share of the decision space than the hand-fitted
ranker and converts a *smaller* fraction of it. D3.6's finding stands and is
worse than reported.

## D4.2 — Experiment 8 was run; what it does and does not license for Phase 4

Experiment 8 (rollout-variant divergence) was completed early and is recorded
as SPEC C1–C3. `probes/p08_rollout_divergence.csv`, 230 constructed boundary
states:

| category | n | divergence | rollout cost |
| --- | --- | --- | --- |
| auction | 56 | **94.6%** | 11.37 s |
| build | 48 | **58.3%** | 22.60 s |
| buy | 112 | 0.0% | 4.78 s |
| trade | 14 | 0.0% | 8.89 s |

**Two caveats that bear directly on the Phase 4 decision.**

1. **The trade cell is stale and known wrong.** 0/14 was overturned by
   Experiment 6, which measured **92.6%** divergence over 54 trade states on a
   wider board population (D1.3). The CSV still holds the superseded number;
   SPEC C1 carries the contradiction. Any Phase 4 scoping must use 92.6%.

2. **It measures the wrong pair for Phase 4.** p08 compares `ASUValueV1`
   against `ASURolloutV1` — the *teacher's* two variants. Phase 4 would wrap
   *our* policy in rollout, and our policy is not the teacher: it agrees with
   it 77.5% overall and 8% on trade proposals. That the teacher's lookahead
   changes the teacher's decisions 35% of the time does not establish that
   lookahead over *our* leaf evaluation would help — our leaf evaluation is
   the component measured wrong three separate times (D2.5, D2.6, D2.12).

**Neither of the two questions is settled, so no Phase 4/5 decision is taken
here.** What is now on the record: the correct ceilings, the corrected capture
ratios, and the fact that Experiment 8's evidence is about the teacher's
internals rather than about our agent's headroom.

The cheap experiment that *would* settle it: wrap `spec_policy` in a truncated
rollout using its own evaluation, and measure divergence and win rate on the
same seeds. If our leaf evaluation is the weak component, rollout over it
amplifies the weakness rather than repairing it — that is a testable claim and
it costs one bench run, not a phase.

## D4.3 — The rollout test is INVALID: horizon misalignment, not a finding

The rollout probe returned **0/200 wins, 100% bankruptcy on both spec seats**.
That is not "rollout amplifies a weak valuation"; a policy that never wins a
single game in 200 is broken, and reporting it as evidence would have been
wrong.

**Diagnosis.** Tracing the divergences, `RolloutPolicy` consistently prefers
`sell_prop` / `mortgage` where `spec_policy` chooses `END_TURN`:

    spec=END_TURN   rollout=sell_prop(sq=8)
    spec=END_TURN   rollout=mortgage(sq=8)          (repeatedly)

The cause is in `_playout`, not in the valuation. Each candidate is scored by
applying it and then running a **fixed 12 plies**. But `END_TURN` passes the
turn, so those 12 plies are mostly *opponents* acting, while `sell_prop` keeps
the turn, so its 12 plies are mostly *ours*. The two leaves are sampled at
different points in the turn cycle and are not comparable. The comparison
therefore rewards any action that retains the turn, and liquidating one's own
assets retains the turn — hence a policy that mortgages and sells itself into
bankruptcy in 100% of games.

This is a textbook truncated-rollout error: the horizon must be aligned to a
common decision point (e.g. "until control returns to us N times", or a fixed
number of *our own turns*), not to a fixed number of plies.

**What this does and does not tell us.**

- It does **not** test the D4.2 claim that rollout over our leaf evaluation
  would amplify its weakness. That claim remains untested.
- It does **not** say anything about the valuation, which was never reached:
  the selection was decided by the horizon artefact before the valuation
  mattered.
- It **does** show the diagnostic was cheap and caught quickly — 200 games and
  one trace — which is the argument for running it before committing to
  Phase 4 rather than after.

**The 0.0% figure must not be cited.** It is an artefact of a broken harness
and is recorded here only so it is not mistaken later for a measurement.

**To make the test valid**, two changes are needed:

1. align the horizon — run each playout until our seat has acted N times, so
   every candidate's leaf sits at the same point in the turn cycle;
2. fix the shortlist — `_shortlist` currently pads with arbitrary legal
   actions sampled by stride, which injects candidates the rule pipeline would
   never consider. A shortlist needs a ranking, and the only ranking available
   is the valuation under suspicion, which is a genuine design problem for
   this experiment rather than a coding one.

Point 2 is the harder issue and worth stating plainly: **a rollout layer needs
a candidate ranker, and ours is the component we distrust.** That is an
argument about Phase 4's premise, and it is now concrete rather than
speculative — but it is reasoning, not the measurement, and it is labelled as
such.

## D4.4 — Rollout over our own valuation makes the agent significantly WORSE

Rerun after fixing both faults in D4.3 (horizon aligned to our own decisions;
shortlist ranked by one-ply state value instead of stride-sampled).
Seat-rotated, seeds 930000+, 200 games.

| agent | win rate | n |
| --- | --- | --- |
| floor — proposals off | 18.3% [14.4, 23.1] | 300 |
| **spec — best agent** | **25.7% [21.1, 30.9]** | 300 |
| hybrid — learned head | 24.0% [18.6, 30.4] | 200 |
| **rollout — lookahead over spec** | **14.5% [10.3, 20.0]** | 200 |
| oracle — perfect trades | 40.0% | 120 |

    rollout - spec   -11.2pp
    z = -2.99, p = 0.003
    intervals do NOT overlap (20.0 vs 21.1)
    bankruptcy 92%

**This is a real, significant, negative result** — not the broken 0/200 of the
first attempt. Bankruptcy fell from 100% to 92%, the policy plays whole games,
and the horizon artefact is gone. Rollout is now doing what it was asked to do
and the answer is that it hurts.

**Rollout is worse than the floor.** 14.5% against 18.3% for an agent that
makes no trade proposals at all. Adding lookahead is worse than removing the
feature the lookahead is searching over.

### D4.2's claim is confirmed

> If the leaf evaluation is the weak component, rollout over it does not repair
> the weakness — it amplifies it, because every playout is scored by the same
> faulty valuation.

Stated before the measurement, and now measured at p = 0.003. The mechanism is
visible in the design: `state_value` scores every leaf, and the shortlist is
*also* ranked by it, so a wrong valuation is applied twice — once to choose
what to consider and once to choose among them. Depth multiplies the error
rather than averaging it away.

### The fourth independent identification of the same root cause

| # | finding | what it identified |
| --- | --- | --- |
| D2.5 | 4,000-weight search beat nothing; filter discarded half the teacher's picks | valuation, not weights |
| D2.6 | monopoly-only reproduces combined exactly; removing it *improves* auction | valuation term wrong |
| D2.12 | 23x score change, zero argmax change over 40/40 boards | valuation dominated by one wrong term |
| **D4.4** | **lookahead over the valuation is significantly worse than no lookahead** | **valuation** |

Four independent methods, four times the same component.

### Recommendation, now on measurement rather than inference

**Phase 4 is dead as scoped.** A rollout layer needs a leaf evaluator and a
candidate ranker; ours is the same function for both, and it is measurably
harmful when trusted more deeply. No K/M/P budget fixes that — the failure is
not depth or variance, it is the thing being searched.

**Phase 5's modules are also downstream of the valuation.** Denial-value
trading needs to price what a deed denies an opponent; the endgame switch needs
to compare survival against net worth. Both are new *terms in the valuation* —
which is the right target, but they should be built and measured as valuation
changes with the win-rate harness, not as standalone "modules" layered on top
of a function measured wrong four times.

**Concrete next step:** rebuild `state_value` against evidence rather than
inheriting the published formula. B3/B4/B5 measured auction *ceilings*, which
constrain the monopoly term only up to the additive company it keeps — the
freedom that let a wrong term reproduce them (recorded in D2.6). A valuation
fitted directly to win rate, or to the oracle's own action choices across all
families, is the untried approach with the largest measured headroom: oracle
sits at 40.0% against our 25.7%.

**Standing scoreboard.** Best agent remains `spec_policy` at 25.7%. Neither a
learned head (24.0%) nor lookahead (14.5%) improved on it. Both failures point
at the same component.

## D5.1 — Denial value: positive signal, NOT proven

Phase 5 module 2 implemented as a **valuation term**, not a layer — D4.4
identified the leaf valuation as the root cause for the fourth time, and
Phase 4 demonstrated what layering on top of it costs.

A/B on identical code via `BEYOND_DENIAL`, same seed base, seat-rotated:

| | win rate | n |
| --- | --- | --- |
| `BEYOND_DENIAL=0` | 25.7% [21.1, 30.9] | 300 |
| `BEYOND_DENIAL=1` | **31.0% [25.0, 37.7]** | 200 |

    +5.3pp,  z = 1.30,  p = 0.192

**This is the first change in the project to improve on `spec_policy`.**

| attempt | delta | verdict |
| --- | --- | --- |
| learned head (Phase 3) | −1.7pp | no effect |
| rollout (Phase 4) | −11.2pp | significantly harmful |
| **denial (Phase 5-2)** | **+5.3pp** | **positive, unproven** |

**Status: positive signal, not proven.** p = 0.192 does not support a claim,
and this project's record on optimistic readings is three for three wrong
(D2.10, D3.5, and the capture-ratio extrapolation). The number is recorded as
suggestive and nothing downstream should treat +5.3pp as established.

**Why the intermediate 400-game run was cancelled.** The closed arm is fixed
at n=300, which caps the achievable z regardless of how far the open arm is
extended:

    open arm n=200 -> z=1.29
    open arm n=400 -> z=1.54
    open arm n=600 -> z=1.66

None of those decides anything. If the true effect is +5.3pp, significance
needs roughly **550 games per arm on both sides**. Spending 20 minutes to move
z from 1.29 to 1.54 buys no decision, so the run was cancelled and the budget
moved to module 3. One large measurement with both modules enabled is worth
more than two underpowered ones.

**Why this one has better prospects than the previous two attempts.** It has a
measured basis rather than an architectural argument: SPEC B3/B5 established
that the teacher pays the *same* for the first deed of a colour group as for
the second (own1/own0 = 1.00 to the dollar, 18/18 cases) and never bids
defensively for a group it has no presence in. The clone inherits that blind
spot, and this term prices exactly it.

## D5.2 — Endgame module OFF: the hypothesis survives, the implementation does not

`BEYOND_ENDGAME` defaults to `0`. The code is kept, not deleted, and this entry
exists so nobody rebuilds the same wrong version.

### Result

| configuration | win rate | n | vs baseline |
| --- | --- | --- | --- |
| baseline (both flags off) | 25.7% [21.1, 30.9] | 300 | — |
| denial only | **31.0% [25.0, 37.7]** | 200 | +5.3pp |
| denial + endgame | 22.0% [18.2, 26.3] | 400 | −3.7pp (z=−1.13, p=0.258) |

Denial alone gains +5.3pp; adding endgame gives back that gain and more. The
two do not add — they **conflict**.

### The hypothesis was NOT refuted

Bankruptcy is a real weakness and was correctly identified: **~88% for us
against ~60% for the teacher**, unchanged by this module (88% with it enabled).
Survival genuinely is where we lose. Nothing here argues otherwise.

### What failed is seeking survival by widening the safety threshold

The endgame term feeds `cushion_multiplier` into `gates_ok`, which scales the
$200 floor with board development. That makes **every discretionary purchase
harder** — and denial's entire mechanism is *"buy this deed to block an
opponent"*. The second module could not get through the first module's gate.

The prediction was recorded before the measurement:

> Denial says "take this deed, don't let the opponent have it" while endgame
> says "hold cash, don't spend". They may pull in opposite directions. If the
> result is flat, that is the first place I will look.

It came out worse than flat, and the bankruptcy rate confirms the mechanism:
the module restricted spending without buying any survival. It made the agent
poorer, not safer.

### The likely correct form, for whoever picks this up

**A change of objective, not a change of threshold.** Late in the game, score
states by *"does an opponent go bankrupt before we do"* rather than by net
worth — i.e. modify `state_value` itself, not the gate in front of it.

That is deliberately **not attempted here**. `state_value` is the component
four independent diagnoses have already shown to be wrong (D2.5, D2.6, D2.12,
D4.4), so changing it opens a diagnosis round of uncertain length. With time as
the binding constraint, confirming the one positive result the project has is
worth more than starting that.

**Do not re-implement survival as a wider cushion.** It has been measured and
it costs more than it saves.

## D5.3 — Denial validated to n=550: positive signal, NOT statistically confirmed

Both arms extended to 550 games on the same seed base (930000+), seat-rotated,
run through the **same harness** (`bench.py`) so the only difference is one
environment variable. The earlier baseline came from `pinned_ablation`, and
mixing harnesses has misled this project once already.

| | wins | win rate |
| --- | --- | --- |
| `BEYOND_DENIAL=0` | 135/550 | 24.5% [21.1, 28.3] |
| `BEYOND_DENIAL=1` | 151/550 | **27.5% [23.9, 31.3]** |

    difference  +2.9pp
    z = 1.10,  p = 0.2714      NOT significant
    intervals overlap

**The effect shrank with sample size: +5.3pp at n=200 became +2.9pp at
n=550.** That is regression to the mean — the small sample overestimated it,
which is the outcome the validation existed to detect. The pre-registered
power note said +5.3pp would land at z≈1.98, a knife edge; the realised effect
was smaller and it did not come close.

**Ships anyway, labelled.** +2.9pp remains the best measured point estimate
available, the sign has been positive in both independent measurements, and
the mechanism has a probe-established basis (SPEC B3/B5: the teacher pays the
same for the first deed of a colour group as for the second, 18/18 cases to
the dollar, and never bids defensively for a group it has no presence in). But
the honest label is:

> **positive signal, not statistically confirmed at n=550 (p = 0.271).**

Nothing downstream should treat +2.9pp as established. Confirming an effect
this size would need roughly 1,900 games per arm.

**Baseline cross-check.** The rerun baseline (24.5% [21.1, 28.3], n=550) agrees
with the independent `pinned_ablation` measurement (25.7% [21.1, 30.9], n=300)
across two harnesses and two runs. The measurement infrastructure is
consistent; the uncertainty is in the effect, not the instrument.

## D5.4 — Agent frozen

`competition_agent/final_agent.py` is the deliverable.

    spec_policy  +  BEYOND_DENIAL=1  +  BEYOND_ENDGAME=0

Selection record — every entry a head-to-head win rate against `ASUValueV1`,
seat-rotated on a common seed base, never an agreement score or a projection:

| configuration | win rate | n | outcome |
| --- | --- | --- | --- |
| floor (no trade proposals) | 18.3% [14.4, 23.1] | 300 | — |
| spec_policy baseline | 24.5% [21.1, 28.3] | 550 | — |
| + learned trade head | 24.0% [18.6, 30.4] | 200 | rejected |
| + rollout layer | 14.5% [10.3, 20.0] | 200 | rejected, p=0.003 harmful |
| + denial + endgame | 22.0% [18.2, 26.3] | 400 | rejected, modules conflict |
| **+ denial only** | **27.5% [23.9, 31.3]** | **550** | **shipped, unconfirmed** |

Known ceiling: perfect trade *proposals* reach 40.0%, perfect proposals **and**
replies 45.8%, parity is 50.0%. Trade work alone cannot reach parity, and the
leaf valuation is the identified root cause of the remainder (D2.5, D2.6,
D2.12, D4.4) — inherited from the published formula rather than fitted to
evidence. Rebuilding it is the largest untried lever and is explicitly out of
scope for this run.

Deliverables complete: `bench.py`, `probes/` (14 experiments), `SPEC.md`
(40 rules, 35 certain), `DECISIONS.md`, `spec_policy.py`, `hybrid_policy.py`,
`rollout_policy.py`, `beyond/`, `final_agent.py`, `tests/`.

## D6.1 — Absolute strength: the agent dominates the field, and Phase 6 is not needed

Single command, 600 games (300 seeds x 2 seat arrangements via the new
`--rotate`), the frozen `final_agent` against the three scripted opponents:

    335/600 = 55.8%  [51.8, 59.8]
    parity (1 seat of 4) = 25.0%
    bankruptcy 16%
    z = 17.4 vs parity, p ~ 0

**This reframes every earlier number.** The same agent that bankrupts in ~87%
of games against two ASU seats bankrupts in **16%** against ordinary
opponents, and wins at 2.2x parity. The 87% was never evidence that the agent
is broken — it is evidence that the frozen teacher is strong.

| opponent | win rate | parity |
| --- | --- | --- |
| ASU teacher, 2v2 | 27.5% [23.9, 31.3] | 50.0% |
| scripted field, 1v3 | **55.8% [51.8, 59.8]** | 25.0% |

### Decision: do not enter Phase 6

Phase 6 would rebuild `state_value` from evidence rather than the published
formula. Four independent diagnoses (D2.5, D2.6, D2.12, D4.4) identify it as
the root cause of the gap to the teacher, so the target is right. It is still
the wrong thing to start now:

1. The agent is **dominant on the field it will actually face** — 55.8%
   against a 25% parity, with the interval nowhere near it.
2. It is **absolutely sound**, not merely relatively weak: 16% bankruptcy,
   games carried to 2,463 steps on average.
3. The teacher it trails **cannot be entered** — competition rules exclude it.
   Trailing an ineligible opponent is not a competitive deficit.
4. Rebuilding the valuation is open-ended, and every prior attempt to improve
   on `spec_policy` (learned head, rollout, endgame) measured neutral or
   harmful. With time binding, shipping a measured agent beats starting an
   uncertain round.

### The caveat, stated so it is not lost

`fixed-a/b/c` are scripted policies, not the real competition field. 55.8% is
an upper bound on a weak field and 27.5% a lower bound against an unusually
strong one; the true figure sits between them and is not measured.

**The condition that would flip this decision:** if the competition field is
known to play near ASU strength, then 27.5% is the operative number, the
valuation is the binding constraint, and Phase 6 becomes necessary rather than
optional. Recorded here so the trigger is explicit rather than a matter of
later judgement.

**Status: agent frozen and delivered.** `final_agent.py`, measured, with its
limits on the record.

## D6.2 — Phase 6 Part B: the value network fails, and the stratified AUC says why

500 teacher-vs-teacher games, 499,321 states, split by game (348k train /
151k held out).

**Both label variants are worse than a constant.**

| model | held-out log-loss | AUC |
| --- | --- | --- |
| constant (p = 0.314) | **0.6177** | 0.500 |
| network, plain label | 0.7156 | 0.573 |
| network, discounted label | 0.6720 | 0.575 |

Best held-out log-loss occurs at **epoch 1** for both variants and degrades
monotonically after (plain: 0.716 → 2.754 by epoch 12). Single hand-picked
features do no better — own cash AUC 0.490, own position 0.493 — which rules
out a training bug: the network and the raw features hit the same wall.

### Stratified by remaining steps, the picture inverts

| bucket | n | win rate | AUC plain | AUC discounted |
| --- | --- | --- | --- | --- |
| last 50 steps | 7,500 | 0.448 | 0.739 | **0.766** |
| 50–150 | 15,000 | 0.434 | 0.682 | **0.709** |
| 150–300 | 22,500 | 0.355 | 0.608 | **0.633** |
| **300+** | **106,318** | 0.271 | 0.508 | 0.495 |
| all | 151,318 | 0.308 | 0.573 | 0.575 |

Read against the criterion fixed **before** the measurement — late-game AUC
above 0.65 with early-game near 0.5 means the signal exists and credit
assignment is the problem:

**That is the result.** 0.766 in the last 50 steps against 0.495 beyond 300,
decaying monotonically in between (0.77 → 0.71 → 0.63 → 0.49).

### What this establishes

**The 300-dim observation is not the limitation.** The same vector predicts the
winner well when the horizon is short. The limitation is the label: a binary
end-of-game outcome carries almost no information about a state 300+ decisions
from the end, and **70% of the corpus (106,318 of 151,318 rows) sits in that
bucket**. Training is dominated by rows that are close to pure noise, which is
why the overall figure sits at 0.57 and why log-loss cannot beat a constant.

The discounted label behaves exactly as designed — better than plain in every
late bucket (0.766/0.709/0.633 vs 0.739/0.682/0.608), worse in the 300+ bucket
where it deliberately shrinks toward the base rate. It softens the label but
does not change which rows dominate the loss, so it could not rescue the
aggregate.

### Consequence

Phase 6 is **not** closed. The alternative outcome — flat ~0.5 in every bucket,
meaning the game itself is unpredictable from an instantaneous state — did not
occur, and it is worth stating that it was the outcome expected as likely.
A learned value function is not impossible here; it was attempted with the
wrong training scheme.

The diagnosis points at three fixes, in ascending cost:

1. **Sample weighting / filtering** — drop or downweight the 300+ bucket so
   training is not dominated by unlearnable rows. Almost no new code.
2. **Short-horizon targets** — predict "ahead in n steps" rather than "wins",
   which is the regime already scoring 0.63–0.77.
3. **TD learning** — bootstrap from the next state's estimate instead of
   propagating the outcome 1,000 decisions back.

No projection is made from any AUC to win rate. Part C remains the arbiter.

## D6.3 — Candidate A (sample weighting) fails: the problem is not row dominance

D6.2 attributed the failure to 70% of the corpus sitting in the unlearnable
300+ bucket and dominating the loss. Candidate A tests that directly — drop or
downweight those rows, retrain the same network, change nothing else.

**AUC is reported on the FULL held-out set in every row.** Evaluating a
filtered model on a filtered held-out set would move the goalposts and break
comparability with the 0.575 baseline; the agent still has to act in those
states at play time.

| configuration | train n | last 50 | 50–150 | 150–300 | 300+ | **full AUC** |
| --- | --- | --- | --- | --- | --- | --- |
| all rows | 348,003 | 0.680 | 0.661 | 0.595 | 0.479 | 0.540 |
| filter < 300 remaining | 105,000 | 0.718 | 0.669 | 0.579 | 0.511 | 0.569 |
| **filter < 150 remaining** | 52,500 | **0.751** | **0.701** | 0.612 | 0.518 | **0.584** |
| weight 300+ × 0.2 | 348,003 | 0.708 | 0.646 | 0.554 | 0.474 | 0.530 |
| weight 300+ × 0.05 | 348,003 | 0.725 | 0.692 | 0.595 | 0.500 | 0.556 |

Prior anchors: plain 0.573, discounted 0.575. **Target was 0.65+.**

**Best is 0.584 — the target is missed by a wide margin,** and weighting did
worse than filtering.

### What this corrects in D6.2

D6.2's diagnosis was half right. Removing the noisy rows *does* improve the
late buckets — last-50 AUC rises 0.680 → 0.751 despite training on 85% fewer
rows — so credit assignment is a genuine problem. But the 300+ bucket stays
pinned at 0.47–0.52 in **every** configuration, and it is 70% of the held-out
set, so the full AUC cannot move.

The correct statement is therefore stronger than D6.2's: **early states are
unpredictable because of the states, not because noisy rows dominate
training.** No reweighting of the outcome signal reaches them, because the
information is not there to be reweighted.

### Consequence for Candidates B and C

Both share the assumption Candidate A just tested — that a useful valuation
can be learned from the game's outcome:

- **TD learning (C)** redistributes the variance of that same outcome signal
  through bootstrapping. It does not add information about early states.
- **Short-horizon targets (B)** predict something the decision path does not
  need; the ranker must compare candidate actions, not forecast a position n
  steps out.

Neither escapes the wall A hit, so neither is worth its cost. The next attempt
uses a different signal entirely — the teacher's revealed preferences
(Candidate D).

## D6.4 — Candidate D fails; Phase 6 closes and the agent stays frozen

> **RETRACTED IN PART by D7.2 (2026-08-12).** The 29.86% anchor used below was
> measured on the *original* harvest, not on the corpus Candidate D trained on.
> Replayed on Candidate D's own corpus and split, the hand-fitted ranker scores
> **26.07%**, and the paired difference is −0.45pp (McNemar p = 0.70).
> Candidate D **tied** the hand-fitted ranker; it did not fall 4.2 points short.
> The sections below are left as written; read them against D7.2.

Revealed-preference ranking over the teacher's own trade choices. Data
re-harvested from 120 teacher-driven games **with the 300-dim observation**
this time — the previous harvest carried only hand-picked deed features, which
is the cap D2.5 and the Phase 3 head both hit.

    proposals 4,916   train 3,589 / held-out 1,327   (split by game seed)
    mean candidates per state 68.4  ->  random top-1 1.46%

    best held-out top-1   25.62%

Listwise softmax over each state's candidate set, chosen over Bradley-Terry
because it trains exactly the operation the policy performs (argmax over one
state's candidates), because pairwise expansion is quadratic in a 68-candidate
set, and because it isolates the variable under test — the Phase 3 head used
the same loss and only its inputs differed.

### The comparison, with one anchor corrected

| model | top-1 | task |
| --- | --- | --- |
| random | 1.46% | 68-way |
| **Candidate D** | **25.62%** | 68-way ranking |
| hand-fitted ranker | 29.86% | 68-way ranking |
| Phase 3 head | 38.51% | mixed — **not comparable** |

**The 38.51% anchor should not be used here.** The Phase 3 head also scored
`ACCEPT`/`DECLINE`, which are two-way choices; including them inflates top-1
relative to a pure ~65-way ranking. The honest comparison is against the
hand-fitted ranker's 29.86%, measured on the same task — and Candidate D is
**4.2 points below it**.

Training is unstable: train climbs 20% → 54.75% while held-out oscillates
19–22%. 3,589 states over a 314-dim input overfits, and adding the observation
made the model worse rather than better, despite the observation being the
thing D2.5 blamed for the previous cap.

### Phase 6 closes

Per the pre-set rule, no fifth valuation variant is started. Every cheap route
into the valuation has now been measured:

| candidate | approach | result |
| --- | --- | --- |
| Phase 6 A | outcome label, plain / discounted | worse than a constant (log-loss 0.672 vs 0.618) |
| Phase 6 A′ | filter / downweight noisy rows | 0.584 full AUC, target 0.65 missed |
| **D** | revealed preference over trades | 25.62% vs hand-fitted 29.86% |
| B, C | short-horizon targets, TD | skipped — share the assumption A refuted |

**The closure is itself the finding.** Two independent signals were tried —
the game's outcome and the teacher's preferences — with three
representations between them, and neither beat the hand-written valuation
this project has been trying to replace since D2.5. That valuation is
demonstrably wrong (four diagnoses) and still better than everything learned
against it.

The stratified table from D6.2 remains the sharpest artefact: signal exists at
short horizon (AUC 0.766 in the last 50 steps) and vanishes beyond 300
(0.495). Anyone attacking this again should start there, and should expect the
representation not to be the binding constraint — adding the full observation
to the ranker made it worse.

### Final state

The agent stays frozen exactly as it was:

    spec_policy  +  BEYOND_DENIAL=1  +  BEYOND_ENDGAME=0

    vs ASU teacher (2v2)          27.5% [23.9, 31.3]   n=550
    vs strong field (~1252 ELO)   34.2% [30.5, 38.1]   n=600
    vs weak field  (~1103 ELO)    55.8% [51.8, 59.8]   n=600

---

# Step 0 — is bankruptcy a cause or a symptom?

## D7.1 — The survival oracle recovers nothing. Bankruptcy is a symptom.

Two weaknesses were on the table before any new module: survival (27.8%
bankruptcy against the strong field, 87% against the teacher) and trade
(+19.2pp when pinned, of which the hand-fitted ranker captures 20–34%).
Trade's value was measured; survival's was not. So survival got the same
treatment: pin it to the teacher and read the win-rate recovery.

`survival_ablation.py`, 2,000 games per arm against the strong field
(`fixed-b`, `fixed-d`, `fixed-e`, ~1252 ELO). Paired: both arms play the same
seeds with the agent in the same seat (`seat = seed % 4`, so all four seats
are covered while every game remains an independent board). The pin is on the
**state** — `env.debt_player == pid` — which is exactly the forced
debt-resolution menu: which house to sell, which deed to mortgage, in what
order, and whether to declare bankruptcy.

| arm | leader rate | decisive | bankrupt | oracle |
| --- | --- | --- | --- | --- |
| none | 759/2000 **38.0%** [35.8, 40.1] | 39.1% | 24.6% | — |
| survival | 758/2000 **37.9%** [35.8, 40.0] | 39.2% | 24.6% | fired 6,226 / overrode **1,743** |

    delta                 -0.05pp
    two-proportion z      -0.03   p = 0.9740
    McNemar (paired)      none-only 2, survival-only 1, z = -0.58, p = 0.5637
    bankruptcy rate       24.6% -> 24.6%   (z = 0.00, p = 1.0000)

### Why this is stronger than "not significant"

An unpaired z-test at n=2,000 would only bound the effect to about ±3pp. The
paired design bounds it far tighter: **1,743 debt actions were actually
overridden by a stronger player across 2,000 games, and the outcome changed in
3 of them.** Even taking the upper end of the 95% interval on that discordant
rate, the effect is bounded within ±0.44pp. The bankruptcy rate itself does not
move at all, which rules out the reading that the oracle helped survival but
the win rate failed to follow.

"Oracle fired" is reported alongside "actually overrode" deliberately. 6,226
fires sounds like a large intervention; 72% of them are decisions where the
agent already picks what the teacher picks, because most debt menus have one or
two items. The arm's real content is the 1,743, and stating only the larger
number would have overstated the test.

### Reconciliation with D5.2

D5.2 measured the `liquidation` family at +0.0pp against the *teacher* with a
*bare `spec_policy`* baseline, and that null was discounted at the time because
against a player that strong the agent's own survival may be irrelevant. All
three of those differences are now closed — strong scripted field, frozen agent
baseline, state-based rather than action-family pin — and the null is
unchanged. Two independent measurements, same answer.

### Branch taken: trade

Bankruptcy is downstream of decisions made long before the debt. Rebuilding the
endgame module would optimise the point at which the game is already decided.
**`BEYOND_ENDGAME` stays disabled and no survival module is built.** All
remaining effort goes to trade.

The hypothesis is not merely unproven — it is bounded. Anyone revisiting
survival should not re-test debt handling; they should test whether the agent
*enters* debt more than it needs to, which is a different family (purchase,
development, unmortgage) and a different measurement.

### One number changed on the way past

The `none` arm is the frozen agent against the strong field, and at n=2,000
with full four-seat rotation it scores **38.0% [35.8, 40.1]**, against the
**34.2% [30.5, 38.1]** on the record from a 600-game run that placed the agent
in only two of the four seats. Seat effects of ~7 points are documented in this
project, so the fuller rotation is the better estimate and it supersedes 34.2%.
This was not the run's purpose and the two intervals overlap; recorded because
it would otherwise look like an unexplained discrepancy later.

## D7.2 — Correction: Candidate D tied the hand-fitted ranker, it did not fail

D6.4 concluded Candidate D failed on 25.62% held-out top-1 against a
"hand-fitted anchor" of 29.86%. **Those two numbers were never measured on the
same data.** 29.86% is from D2.9, fitted and evaluated on the *original*
harvest (2,508 proposals, 60 games). 25.62% is from the 120-game re-harvest
(4,916 proposals) that Candidate D was trained on. This is the same error as
the Phase 4 oracle-ceiling mismatch — caught there, repeated here.

`rank_anchor.py` replays `spec_policy.TRADE_W` over Candidate D's corpus under
Candidate D's own split, then scores the checkpoint on the identical states:

| model | held-out top-1 | 95% CI |
| --- | --- | --- |
| hand-fitted ranker | **26.07%** | [23.78, 28.50] |
| Candidate D | **25.62%** | [23.35, 28.04] |
| random | 1.46% | — |

    paired on the same 1,327 states
      fitted-only right 127   Candidate-D-only right 121   both/neither 1,079
      McNemar z = -0.38   p = 0.7032   difference -0.45pp

A tie, not a 4.2-point loss. Note also that the fitted ranker scores 20.62% on
the *train* seeds and 26.07% on the *held-out* seeds — both out-of-sample for
it, since its weights come from a corpus that no longer exists — so the
held-out subset is simply the easier one, and Candidate D's number has to be
read against that subset rather than against the corpus average of 22.09%.

**Consequences.** Two claims in D6.4 no longer stand:

1. "Candidate D failed" becomes *Candidate D matched the hand-fitted ranker at
   this data scale*, with the training curve (train 20% → 54.75% while held-out
   sat at 19–22%) pointing at data starvation rather than at a verdict on
   revealed preference.
2. "Adding the 300-dim observation made the model worse" is **withdrawn**. 314
   input dimensions on 3,589 rows cannot test that claim; the comparison it
   rested on was against the mis-attributed 29.86%.

D6.4's closing paragraph — that anyone attacking this again should not expect
the representation to be the binding constraint — is withdrawn with it. The
data scale is the untested variable, and D7.4 tests it.

## D7.3 — Part B: the trade scorer is over-committed, not blind

`analyze_trade_errors.py` replays the fitted scorer over all 4,916 harvested
proposals, isolates the 3,830 where it puts something other than the teacher's
choice on top, and reports what separates the two picks. No feature was
proposed in advance — D2.6 records what that costs.

### Where the teacher's pick sits in our ranking

    rank 1   22.09%      rank <=5    50.65%
    rank 2   11.31%      rank <=10   66.44%
    rank 3    7.30%      rank 21+    16.21%

Half the time the teacher's choice is already in our top five out of ~68. The
scorer is directionally right with a heavy tail, not blind.

### Which term steers us wrong

The scorer is linear in six weighted terms, so the gap between our pick and the
teacher's decomposes exactly. Mean gap 1.430 over the disagreements:

| term | contribution | share of the gap |
| --- | --- | --- |
| `completes` | +0.986 | **68.9%** |
| `d_ours` | +0.308 | **21.5%** |
| `d_rent` | +0.129 | 9.0% |
| `off_mort` | +0.008 | 0.6% |
| `d_price` | −0.001 | −0.1% |
| `d_houses` | +0.000 | 0.0% |

**Ninety per cent of the error comes from two terms already in the model.**
`d_price`, `d_houses` and `off_mort` are dead weight — they contribute nothing
to any disagreement, so removing them costs nothing and adding features
alongside them is not where the gain is.

### What the picks look like, against the population

Our pick is an argmax and is therefore extreme by construction on whatever the
scorer weights, so the mean over *all* candidates in the same states is shown
as a third column. Without it the table cannot say which side is the outlier.

| quantity | our pick | teacher | all candidates | Cohen d |
| --- | --- | --- | --- | --- |
| `off_breaks_ours` (give away a deed from a group we hold ≥2 of) | 0.01 | 0.24 | 0.51 | +0.77 |
| `off_rent_if_ours` | 4.16 | 7.86 | 11.23 | +0.70 |
| `off_price` | 126 | 171 | 187 | +0.64 |
| `mutual_swap` (both sides complete) | 0.39 | 0.13 | 0.03 | −0.60 |
| `req_completes_ours` | 0.59 | 0.31 | 0.10 | −0.60 |
| `off_group_size` | 2.65 | 2.96 | 2.91 | +0.58 |
| `req_price` | 196 | 236 | 213 | +0.55 |
| `req_theirs_in_group` | 1.55 | 1.78 | 2.21 | +0.24 |

The third column changes the reading. On every one of these the teacher sits
**between** our pick and the population, and our pick sits at the extreme:
`mutual_swap` 0.39 against a 0.03 base rate, `off_price` $126 against a $187
base rate. The teacher likes the same things we like — it just does not
insist on them.

### The evidenced list

1. **Re-weight before adding.** `completes` (3.47) and `d_ours` (0.505) carry
   90% of the error and both push the same way. Shrinking them is the first
   test and it costs one fit.
2. **Absolute value surrendered.** The scorer sees only differences, so it
   cannot express "this deed is cheap". We give away $126 deeds where the
   teacher gives away $171 against a $187 field. `off_price` and
   `off_rent_if_ours` are not derivable from any existing term.
3. **`off_breaks_ours`.** Currently reachable only implicitly through `d_ours`,
   and the implicit version is far too strong: we do it in 1% of picks where
   the teacher does it in 24%.
4. **`req_theirs_in_group`.** We ask for deeds in groups opponents are *less*
   close to completing than average (1.55 vs 2.21). Denial logic says the
   opposite, and `denial.py` already prices this for purchases but nothing
   feeds it into trade.
5. **Drop `d_price`, `d_houses`, `off_mort`.** Zero contribution to any
   disagreement.

Nothing here is shipped on this evidence. Each item is a hypothesis with a
measured separation behind it rather than a plausible story, and a win rate
still decides.

### The limit of this analysis

The harvest recorded no counterparty state, so nothing above can speak to the
cash position of the player being asked — which is a live hypothesis, since a
proposal that is declined is worth nothing and the scorer has no notion of
acceptance at all. `harvest_trades.py` now records a per-seat snapshot (cash,
deeds, net worth, position) so the Part C corpus can answer it.

---

# Part C — the data-scale hypothesis, and what it overturned

## D7.5 — Candidate D was starved, not wrong. D6.4 is overturned.

`harvest_trades.py` was restructured to write one gzipped shard per game with
atomic rename and existence-based resume, then run for 1,000 teacher-driven
games (seeds 910000-910999, 9 workers, ~45 min):

    96,668 decision states with >=2 exchange candidates
    39,081 of them the teacher proposed in  (40.4%)
    70 MB gzipped, 1,000 shards

Determinism check: shard `g910000` reproduces the legacy single-file record
exactly — 31 states, every field, every candidate set. The harvest's opponents
(`TheHoarder`, `TheDealMaker`, `TheGambler`) contain no string-set iteration,
so the D7.4 hash-seed defect does not touch this corpus.

### The retrain

Same architecture, same loss, same split rule, same seed. Only the corpus size
changed — 3,589 training proposals became 27,789.

| corpus | train proposals | held-out top-1 |
| --- | --- | --- |
| 120 games | 3,589 | 25.62% |
| **1,000 games** | **27,789** | **35.82%** |

Measured on the same 11,292 held-out states as every number below, and
`fit_trade_v3.py` reports the hand-fitted models on those identical states:

| model | held-out top-1 |
| --- | --- |
| shipped `TRADE_W` | 24.90% [24.11, 25.71] |
| refit, same six features | 26.48% [25.67, 27.30] |
| refit, + D7.3's ten features | 26.56% [25.75, 27.38] |
| **Candidate D** | **35.82%** |
| random | 1.50% |

**+9.3 points over the best hand-fitted model.** D6.4's "Candidate D fails" and
D7.2's softened "tied at this data scale" are both superseded: at 8x the data
it wins outright. The claim that adding the 300-dim observation hurt, already
withdrawn in D7.2, stays withdrawn — the observation is what carries the gain.

Caveat kept on the record: `train_rank.py` checkpoints the best epoch by
held-out top-1, which is model selection on the reported set and is optimistic
by roughly a point. The final epoch's raw value is 34.82%. The conclusion does
not depend on it — 34.82% still clears 26.48% by 8.3 points.

### D7.3's feature list is refuted, its diagnosis is confirmed

D7.3 concluded "re-weight before adding" and then listed ten features to add.
At 8x data, on the same held-out states:

    refit, same six features   26.48%   +178 net vs shipped, p < 0.0001
    refit, + the ten features  26.56%   +9 net vs the refit, p = 0.6661

Re-weighting is worth 1.58 points and highly significant. The features are
worth nothing. At the small scale they had looked 3.9 points *worse*, and that
gap closed to zero rather than turning positive — so the small-scale reading
("they overfit") was right about the mechanism and wrong to treat the sign as
informative.

A methodological point that cost a wrong conclusion first time round: a
maximum-likelihood fit is **not** a refit of `TRADE_W`. The shipped weights
came from a direct top-1 search, and LBFGS on listwise log-likelihood converges
to weights that score significantly *worse* top-1 (24.37% vs 24.90%,
p = 0.0258). Both objectives are reported in `fit_trade_v3.py` so the surrogate
gap stays visible.

## D7.6 — Three fields. The headline is not "positive everywhere".

Candidate D wired into `_propose_trade` behind `TRADE_RANKER`, with its gate
re-derived on its own logit scale under the same accuracy-max convention the
shipped 3.92 turns out to follow (`calibrate_rank_gate.py`; the grid search
recovers 3.8942 for the shipped weights, which is how the convention was
identified). Every arm paired by seed against the frozen agent.

| field | parity | shipped | Candidate D | delta | p | games changed |
| --- | --- | --- | --- | --- | --- | --- |
| ASU, 2v2 | 50% | 30.33% [26.79, 34.13] | **36.50%** [32.74, 40.43] | +6.17pp | 0.0065 | 185/600 |
| strong ~1252 ELO | 25% | 37.95% [35.85, 40.10] | 38.40% [36.29, 40.55] | +0.45pp | 0.0389 | **19/2000** |
| weak ~1103 ELO | 25% | 57.75% [55.57, 59.90] | **65.65%** [63.54, 67.70] | +7.90pp | <0.0001 | 464/2000 |

**Against the strong field there is no measurable effect.** 19 of 2,000 games
changed, and at three tests Bonferroni requires p < 0.017, which +0.45pp does
not meet. That field is the closest available proxy for the tournament; ASU
cannot be entered, and the weak field is near-random. The honest summary is
**large gains against ASU and the weak field, nothing on the strong field** —
not "positive on all three".

What is nonetheless solid: the pre-set criterion was no weak-field regression
and the weak field gained 7.90 points; bankruptcy against ASU fell from 69.2%
to 63.5%; and this is the first change in this project to beat the frozen agent
on any field at all. Both win-rate conventions agree (decisive-only: weak
78.55% -> 84.02%, strong 39.11% -> 39.34%).

### The mechanism is measured, not inferred

| field | proposals | accepted, shipped | accepted, Candidate D |
| --- | --- | --- | --- |
| ASU | 23,560 / 11,861 | 2.9% | **8.6%** |
| weak | 100,479 / 108,890 | 2.9% | **5.2%** |
| strong | 166,240 / 123,862 | **0.06%** | 0.02% |

The strong field accepts 98 of 166,240 proposals. No ranking of proposals can
pay off through a counterparty that declines essentially everything, which is
why the strong-field result is not a failure of the ranker so much as a
property of that field. It also means `_trade_reply` — the accept side — is
where any strong-field gain would have to come from.

Note against the earlier reading: the propose rate moves in *different*
directions by field (ASU 39.3 -> 19.8 per game, weak 50.2 -> 54.5), so
"proposes less" cannot be a general explanation of the gain. D7.7 measures
that directly.

## D7.7 — Gate versus ranking: the gate hypothesis is refuted

The ranker proposes half as often as the shipped agent against ASU, so the
+6.17pp was attributable to the ranking and the gate jointly. If the gain came
from proposing less, the fix would be one number rather than a 7 MB network.

Calibrating the gate on the corpus does not work: the ranker proposes at a
*higher* corpus rate (29.3% vs 22.9%) yet half as often in ASU play. The
distribution the agent reaches is different, so `gate_probe.py` plays ASU games
with the frozen agent and scores every trade-legal decision under **both**
scorers without acting on either, giving the in-play score distribution. On
those matched states:

    shipped gate  3.92   -> propose 16.3% = 35.1/game
    ranker gate -20.25   -> propose 19.2% = 41.2/game

On the same states the ranker's gate proposes *more*, not less. The realised
19.8/game is therefore downstream of the ranking changing the trajectory, not
set by the threshold. Calibrating on propose rate rather than win rate keeps
this off the test set.

**567 of 600 paired games** (stopped by request once the direction was
settled), gate raised to 4.0077 with the frozen six-term scorer otherwise
untouched:

    shipped     173/567  30.51%  [26.86, 34.42]
    gate only   132/567  23.28%  [19.99, 26.93]
    delta -7.23pp   discordant 53 (shipped-only 47, gate-only 6)
                    z -5.63   p < 0.0001
    propose/game 39.4 -> 29.2   accepted 2.9% -> 3.0%
    bankrupt     69.1% -> 76.4%

Raising the gate **loses** 6.8 points, and does not improve proposal quality at
all (acceptance 2.8% -> 2.9%, against the ranker's 8.6%). The gate hypothesis
is refuted in the direction opposite to the one it predicted. It also implies
the ranking contribution exceeds +6.17pp, since Candidate D achieves that while
also carrying a propose-rate reduction that is worth about -7pp on its own —
but additivity is an assumption, recorded as such, and the rate-matched ranking
arm would have measured it directly. **That arm was not run** — the run was
stopped once the gate direction was settled, so the size of the ranking
contribution remains inferred rather than measured, and is labelled as such
wherever it appears. What is measured is that the gate cannot explain the
gain.

Calibration drift is on the record: the arm was aimed at 19.8 proposals/game
and realised 29.2, because changing the gate changes the trajectory. That makes
this an under-estimate of the gate penalty, not an over-estimate.

Bankruptcy moves the same way — 69.1% to 76.4% — so the gate arm is not merely
neutral-but-quieter; withholding proposals leaves the agent measurably worse
off. Combined with acceptance staying flat (2.9% -> 3.0%) while the ranker
tripled it, the two levers are doing different things entirely.

**Ship decision: the network, not the number.** The one-number alternative was
tested and is 7.2 points worse than doing nothing.

Not measured, and open: the teacher's own win rate against the strong field.
`field_ref.py` was written and queued for it (200 games, ~37 min at 9 workers,
4 games already recorded) and stopped with the rest. Until it is run, 37.95%
has no ceiling to be read against, and whether the strong-field null is a
defect or a property of that field is unresolved.

## D7.8 — The regime-switched scorer was not built, and why

Two measurements, taken before building, closed it.

**1. `state_value` is dead code in the shipped agent.**

    spec_policy calls state_value : 0
    spec_policy calls swap_delta  : 0
    spec_policy calls deed_value  : 2   (both inside _trade_reply)

`state_value`'s only callers are `swap_delta` (used by `test_joint_ranking.py`
alone) and `rollout_policy` (a separate policy, measured at -11.2pp and not
shipped). Switching it by regime would have changed nothing measurable.
`final_agent.py`'s docstring calling it "the leaf valuation" is wrong and is
corrected there.

**2. D6.2's slice table does not survive a solvency control.**

`regime_probe.py` reproduces D6.2 exactly on the held-out split — 0.766 /
0.709 / 0.633 / 0.495 on n = 7,500 / 15,000 / 22,500 / 106,318 — and then adds
the column that matters: the same slices restricted to states where **all four
seats are still solvent**.

| remaining steps | n | AUC all | n solvent | AUC solvent |
| --- | --- | --- | --- | --- |
| 0-50 | 7,500 | 0.766 | **78** | 0.927 |
| 50-150 | 15,000 | 0.709 | 1,912 | 0.745 |
| 150-300 | 22,500 | 0.633 | 9,200 | **0.556** |
| 300+ | 106,318 | 0.495 | 96,842 | 0.477 |

Only 78 of the 7,500 states in the strongest slice have four solvent seats: the
0.766 is very largely the network reading bankruptcies off a board where the
game is already decided, and a scorer that is accurate once nothing can be
changed has no decision value. At the 150-300 boundary proposed as the switch
point, 0.633 becomes 0.556 on live boards — indistinguishable from the 0.495
that closed Phase 6. Saturation is not the problem (sd of p on solvent boards
is 0.218 / 0.168 / 0.153 / 0.127); discrimination is.

The one regime where the network survives the control is 50-150 remaining steps
(0.745 on 1,912 states, 1.3% of held-out).

An in-sample version of this table was computed first and reported AUC 0.916 /
0.899 / 0.864 / 0.748. That was an error — the network overfits hard, as D6.2
recorded, and in-sample AUC runs ~0.15 high. Noted because the corrected table
is the one that refutes the premise, and the wrong one would have supported it.

**Q1 was answered on the way past.** Observable proxies, Spearman against
actual remaining steps over 499,321 states:

    round_frac       -0.7644     <- obs[278], = env.round / max_rounds
    deeds_owned      -0.7260
    houses_on_board  -0.7111
    monopoly_deeds   -0.6982
    n_bankrupt       -0.5552
    cash_spread      +0.0654

`round_frac` is the best proxy and is directly readable from `env.round` at
decision time. Recorded for whoever needs a game-progress signal later; it is
not used by anything today.

## D7.9 — Shipped. The agent is no longer the frozen v1.

The decision in D7.7 was "the network, not the number", and until now it was
only a decision: `TRADE_RANKER` defaulted off, so `FinalAgent` still ran the
frozen v1 configuration and every measured gain required an environment
variable set by hand.

    FinalAgent.config = {"BEYOND_DENIAL": "1",
                         "BEYOND_ENDGAME": "0",
                         "TRADE_RANKER": "probes/rank_gate_1000.json"}

Three defects were fixed on the way, all of which would have shipped:

1. **The checkpoint path was absolute.** `rank_gate_1000.json` carried
   `/Users/.../competition_agent/rank_head_1000.pt`, which resolves on exactly
   one machine. Now stored as a bare name and resolved by `spec_policy._resolve`
   against the package directory, with the absolute form still honoured when it
   exists.
2. **A missing checkpoint would have cost the whole agent, not the trade
   branch.** `_load_ranker` raised, `SpecPolicy.choose_action` propagated, and
   `FinalAgent` catches by returning the first legal action — applied to every
   decision in the game, not just trades. It now returns None on any failure,
   latches `TRADE_RANKER_FAILED`, and the linear scorer takes over.
3. **`final_agent.py` named `state_value` as the leaf valuation and root
   cause.** It is dead code in this agent (D7.8); corrected in place.

Regression check, with no environment variable set: 500 weak-field games
reproduce the measured arm on **500 of 500 seeds**, identical outcome and
identical game length. The shipped default is byte-for-byte the configuration
that was measured.

    old frozen v1   275/500   55.00%
    shipped v2      331/500   66.20%

That +11.20pp is the first 500 seeds only and is **not** a new headline; the
figure of record is +7.90pp at n=2,000. It is reported here solely as evidence
that the default now runs what was measured.

### The shipped agent, as measured

| opponent | v1 (frozen) | v2 (shipped) | parity |
| --- | --- | --- | --- |
| ASU teacher, 2v2 | 30.33% | **36.50%** [32.74, 40.43] | 50% |
| strong field ~1252 ELO | 37.95% | 38.40% [36.29, 40.55] | 25% |
| weak field ~1103 ELO | 57.75% | **65.65%** [63.54, 67.70] | 25% |

The strong-field figure is unchanged in any way that survives correction, and
that is the field closest to the tournament. This is an improvement against
opponents that trade and no improvement against opponents that do not.

### Open, and stated so it is not mistaken for finished

- **The teacher's own strong-field win rate is unmeasured.** `field_ref.py` is
  written and 4 games are recorded; until it runs, 37.95% has no ceiling to be
  read against and the strong-field null cannot be attributed.
- **The ranking contribution's size is inferred, not measured.** The
  rate-matched arm was not run. What is measured is that the gate cannot
  explain the gain, and that raising it alone costs 7.23pp.
- **`_trade_reply` is untouched** and is where a strong-field gain would have
  to come from, since that field accepts 0.06% of what we propose.

## D7.10 — The teacher answers an offer by making one of its own

Run as a screening probe before committing to `_trade_reply` work, to check
that the accept side fires at all against a field that accepts 0.06% of what we
propose. It fires constantly, and the probe found something else.

    strong field, 1,279 states where ACCEPT and DECLINE are both legal

      teacher chose DECLINE_TRADE   716   56.0%
      teacher chose exch_trade      488   38.2%   <- a counter-proposal
      teacher chose sell_trade       45    3.5%
      teacher chose ACCEPT_TRADE     21    1.6%
      teacher chose buy_trade         9    0.7%

    weak field, 379 states
      DECLINE 74.9%   exch_trade 16.1%   buy_trade 9.0%

Offered a trade, the teacher responds with a proposal of its own **42.4%** of
the time on the strong field. Our agent does this in **0 of 974** observed
states: it always answers yes or no.

### The cause is rule ordering, not valuation

`SpecPolicy.choose_action` consults `_trade_reply` before `_propose_trade`, and
the first rule to return an action wins. So the moment ACCEPT/DECLINE are legal
the pipeline halts at the reply and the proposal branch is never reached. The
change to test is one line: when `_trade_reply` would decline, return `None`
and let `_propose_trade` try, declining only if nothing clears the gate.

No network, no training, no new parameter. It is the cheapest untested change
on the board and it targets the branch that D7.6 identified as the only route
to a strong-field gain.

### How it was found, because the route matters

The probe first reported 6.7% / 1.2% acceptance with 62.3% agreement. That is
arithmetically impossible: if both sides only ever answer ACCEPT or DECLINE,
agreement cannot fall below 100 - 6.7 - 1.2 = 92.1%. Chasing the contradiction
is what surfaced the third action. A menu check confirmed it — of 974 such
states, **none** had a menu of exactly {ACCEPT, DECLINE}; 384 were
{ACCEPT, DECLINE, END_TURN} and the rest ran to 18-25 options, all in
`out_of_turn`. Our agent picked a reply in all 974.

**SPEC H1-H4 are built on an assumption that does not hold.** They describe the
incoming-offer decision as accept-or-decline. The teacher treats it as a full
decision point. The rules are not wrong about what they cover; they are
incomplete about what the decision *is*, and the confidence tag should read
that way rather than staying at `certain`.
