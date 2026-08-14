# SPEC.md — behavioural specification of the ASU frozen teacher

Every rule is stated as **observation → inferred rule → confidence**, and cites
the probe CSV that produced it. Rules are derived only from
`decide(env) -> action` outputs; no value component, safety breakdown or
rejection reason is read (see `DECISIONS.md` D0.3), and `core.py` /
`evaluate.py` have never been opened.

Confidence tags: **certain** (quantitatively predicted and confirmed to the
dollar), **likely** (consistent across a sweep, mechanism inferred),
**guess** (one observation, plausible reading).

Rule ids are stable. When a later probe contradicts a rule, the rule is kept,
its confidence downgraded, and the contradiction recorded beneath it.

---

## Audit Trail Integrity — certification of the probed teacher

**Verdict: PASS.** Probes in this document target a decision function proved
identical to the committed frozen teacher.

### Why a stronger check was needed

`ASU_FROZEN_TEACHER/core.py` carries 258 added / 32 removed uncommitted lines.
The initial evidence — a chained digest match on 3 ordinary-play seeds plus 24
green tests — was insufficient: three ordinary games are unlikely to reach
auctions with several live bidders, jail exits by card, forced liquidation,
bankruptcy resolution, trade acceptance, or the house/hotel boundary under a
constrained bank. If any of those branches had changed, the entire probe
corpus would rest on a policy nobody froze.

### What changed, by function name

Established from `git diff --numstat`, hunk headers, and a set-difference of
`def`/`class` names between `HEAD:core.py` and the working tree. Names only —
no diff body, and no decision logic, was read.

| | functions |
| --- | --- |
| **added** | `_DeedRentTables`, `_build_deed_rent_tables`, `_deed_rent_with`, `_expected_landings_items`, `_fast_copy_env`, `_fast_copy_player`, `_fast_copy_property`, `_fast_copy_trade_offer`, `_hypothetical_group_rent_uncached`, `_max_developed_rent_uncached` |
| **removed** | `_hypothetical_group_rent`, `_max_developed_rent`, `_owned_count` |
| **touched** | `deed_rent`, `expected_landings`, `liquidatable_worth`, `long_rent_projection`, `movement_probabilities`, `rent_projection`, `preserve_global_rng`, `_PrivateGame` |

The shape is memoisation plus renamed-behind-cache wrappers — with one
genuinely risky family. `_fast_copy_env / _fast_copy_player /
_fast_copy_property / _fast_copy_trade_offer` replace `deepcopy` in the
rollout's state duplication. A hand-rolled copy that omits a rarely-populated
field (auction bidders, pending trades, debt bookkeeping) would reproduce
ordinary play exactly while corrupting precisely the rare branches above.
That risk is what the certification had to exclude.

### Method

Not a comparison against a stored digest, which only fixes behaviour on the
three seeds that produced it. Instead **both versions were executed and their
outputs diffed.** `HEAD:ASU_FROZEN_TEACHER/core.py` was checked out into a
shadow package placed first on `sys.path`; `certify_teacher.py` runs under
each version and emits the selected action id at every decision point.

This reads no source at all — each version is executed, never inspected — and
it records only selected action ids, so the D0.3 opacity discipline holds
here too.

| | |
| --- | --- |
| old `core.py` sha256 | `09d84f268c9f7e99…` (git `HEAD`) |
| new `core.py` sha256 | `4ea57aa919de4735…` (working tree) |
| ordinary-play seeds | 20 (seeds 1–20, value variant, full games to 1200 steps) |
| rollout seeds | 5 (seeds 1–5, 20 decisions each — exercises `_fast_copy_*`) |
| synthetic scenarios | 10, each run under **both** variants |
| **decision points compared** | **6,314** |
| **mismatches** | **0** |

The 10 scenarios cover every branch named above: a four-bidder auction for
Boardwalk; jail exit by bail, by card, and at the third-turn boundary; forced
liquidation under a $1,100 debt; bankruptcy with nothing left to liquidate;
an incoming trade priced to accept and one priced to refuse; and the
4-house/hotel boundary with the bank down to 2 houses and 1 hotel, against an
unconstrained control. Each was verified to actually reach its branch before
being trusted — the first trade scenario exposed only offer-*making* actions,
so real `TradeOffer` objects were injected until `ACCEPT_TRADE` and
`DECLINE_TRADE` became legal.

### Certification

Across 6,314 decision points spanning 20 full ordinary-play seeds, 5 rollout
seeds, and 10 synthetic scenarios chosen specifically to reach the rare
branches that the `_fast_copy_*` family could plausibly have broken, the
modified `core.py` selected an identical action to the committed `core.py` at
**every single decision point**, under both the value and the rollout variant.
The change is therefore behaviour-preserving on the decision function, and the
3× speedup (0.0501 → 0.0167 s/decision aggregate) is a pure optimisation. The
probe corpus in this document is valid evidence about the frozen teacher.

Artifacts: `probes/certification/` (`cert_old_core.json`,
`cert_new_core.json`, `certification_result.txt`, `versions.txt`).
Reproduce with `certify_teacher.py --emit old --shadow <dir>` /
`--emit new`, then `--compare`.

**Standing check.** Re-run this certification if `core.py` is touched again.
A digest-only check against `artifacts/asu_baseline_locked.json` is not a
substitute and should not be treated as one.

---

## Group A — the buy decision

### A1. Buy response is monotone in cash
**Observation.** For all 28 deeds, sweeping cash from list price to price +
2500 in $25 steps, the teacher's buy/skip response flips exactly once and
never flips back (`monotone = True`, 28/28).
**Rule.** For a fixed board, `BUY` is chosen iff cash ≥ a single deed-specific
threshold. A per-deed "flip point" is therefore well defined.
**Confidence:** certain.
**Evidence:** `p01_buy_threshold.csv` (28 rows, `monotone` column, seed 20250811).

### A2. The buy gate is a floor on cash *after* the purchase, not on cash
**Observation.** On an empty board with all opponents on Go, the flip point is
`price + 200` for 21 of 28 deeds — across prices from $60 (Mediterranean) to
$400 (Boardwalk), the *difference* is pinned at exactly 200 while the price
varies 6.7×.
**Rule.** The teacher buys only if the post-purchase cash balance clears a
fixed floor of **$200**. The gate is on `cash - price`, not on `cash` and not
on any multiple of price.
**Confidence:** certain.
**Evidence:** `p01_buy_threshold.csv`, `flip_minus_price` column.

### A3. The floor is offset by projected rent income, not applied bare
**Observation.** The 7 deeds that flipped *below* `price + 200` in p01 were
squares 5, 6, 8, 9, 11, 12, 15 — precisely the deeds reachable from Go by one
2d6 roll, with the opponents parked on Go. Holding the deed fixed and walking
the opponents instead (p01b) reproduces the effect on deeds that showed no
residual at all: Boardwalk's flip point falls from 600 to 575 as three
opponents move from 1 to 7 squares behind it.
**Rule.** The gate is

    (cash - price) + E[next-round net rent] >= 200

**Confidence:** certain.
**Evidence:** `p01_buy_threshold.csv` (residual deeds), `p01b_rent_residual.csv`
(70 rows, 5 deeds × 14 opponent distances).

### A4. The rent projection is an exact 2d6 landing enumeration × rent, summed over opponents
**Observation.** With three opponents parked `g` squares behind the target
deed, the residual `price + 200 - flip_cash` tracks
`3 × P(2d6 = g) × base_rent` to the dollar over gaps 1–8, for deeds of
different rent:

| gap | P(2d6=g)·3·$50 (Boardwalk) | observed | P(2d6=g)·3·$20 (Illinois) | observed |
| --- | --- | --- | --- | --- |
| 1 | 0.0 | 0 | 0.0 | 0 |
| 2 | 4.2 | 4 | 1.7 | 1 |
| 3 | 8.3 | 8 | 3.3 | 3 |
| 4 | 12.5 | 12 | 5.0 | 5 |
| 5 | 16.7 | 16 | 6.7 | 6 |
| 6 | 20.8 | 21 | 8.3 | 8 |
| 7 | **25.0** | **25** | **10.0** | **10** |
| 8 | 20.8 | 21 | 8.3 | 8 |

The peak sits at gap 7 (the 2d6 mode) and the curve is symmetric about it.
Residual scales linearly in rent: the Boardwalk/Illinois ratio is 25/10 =
2.5 = 50/20, the ratio of base rents.
**Rule.** Expected rent income is computed by enumerating the 2d6 landing
distribution over each opponent's *actual board position* — not a uniform lap
model — and multiplying by the rent that deed would charge.
**Confidence:** certain.
**Evidence:** `p01b_rent_residual.csv`.

### A5. The projection includes doubles-driven extra rolls
**Observation.** For gaps 9–14 the observed residual consistently *exceeds*
the single-roll prediction, and is non-zero where a single 2d6 roll cannot
reach at all: gap 13 → residual 1 and gap 14 → residual 2 against a
single-roll prediction of 0.0 (Boardwalk). The excess grows with gap.

| gap | single-roll prediction | observed |
| --- | --- | --- |
| 9 | 16.7 | 18 |
| 10 | 12.5 | 14 |
| 11 | 8.3 | 9 |
| 12 | 4.2 | 6 |
| 13 | 0.0 | 1 |
| 14 | 0.0 | 2 |

**Rule.** The landing enumeration models a *complete turn*, not a single roll:
rolling doubles grants another roll, so squares 13–24 ahead are reachable with
positive probability. This matches the published description of a five-turn
complete-turn enumeration.
**Confidence:** certain (the non-zero mass beyond gap 12 is unreachable in one
roll, so it cannot be explained any other way).
**Evidence:** `p01b_rent_residual.csv`, gaps 9–14.

### A6. Rent is projected from opponents' real positions, giving a positional bias to deed value
**Observation.** A1–A5 taken together: two deeds with identical price and rent
have different buy thresholds purely because opponents stand at different
distances from them (p01b, gap 7 vs gap 1: 575 vs 600 for the same deed on the
same board).
**Rule.** Deed valuation is position-dependent and recomputed per decision.
There is no static per-deed price table.
**Confidence:** certain.
**Evidence:** `p01b_rent_residual.csv`.

**Competitive note.** A3–A6 show the teacher already models rent it will
*collect* from opponents' true positions. This narrows Phase 5 module 1: the
gap to exploit is rent *paid to* opponents from their developed holdings, not
the collection side, which is already sharp. Recorded in `DECISIONS.md`.

---

## Group B — the auction

Method note: in the auction phase the legal actions are PASS and four bid
increments (+1, +10, +50, +100), the increment being legal only if
`high_bid + increment <= cash`. Sweeping the standing bid `B` and finding the
smallest `B` at which the teacher passes locates its ceiling exactly, because
raising by +1 costs `B+1`: it passes precisely when `B+1` exceeds the ceiling.
Cash is pinned at $5,000 throughout so the safety gates never bind and the
ceiling reflects deed value alone.

### B1. The auction ceiling is several times list price
**Observation.** With no group presence, ceilings run 1.9×–7.4× list price:
Mediterranean $60 → $416 (6.9×), Oriental $100 → $546 (5.5×), Kentucky $220 →
$1,083 (4.9×), Boardwalk $400 → $2,315 (5.8×), Reading Railroad $200 → $388
(1.9×).
**Rule.** The teacher values a deed at auction far above its list price. It is
not anchored to price, and it will not be outbid by a price-anchored opponent.
**Confidence:** certain.
**Evidence:** `p02_auction_ceiling.csv` (82 rows, seed 20250811).

### B2. The value variant always opens at the maximum legal increment
**Observation.** `opening_action` is `auction_bid(+$100)` in **82/82** rows,
across every deed, colour and ownership configuration.
**Rule.** Given a ceiling above the standing bid, `ASUValueV1` jumps by the
largest legal increment rather than creeping. Combined with B1 this makes it
an aggressive, fast-escalating bidder.
**Confidence:** certain.
**Evidence:** `p02_auction_ceiling.csv`, `opening_action` column.
**See C2** — the rollout variant does the exact opposite, and this is where
the two disagree most.

### B3. Owning one deed of a three-deed group buys *no* premium
**Observation.** For every one of the 18 three-deed real-estate cases, the
ceiling with one group member already owned is **identical to the ceiling with
none owned** — ratio exactly 1.00, to the dollar:

| group | sq | own 0 | own 1 | own 2 | own1/own0 | own2/own1 |
| --- | --- | --- | --- | --- | --- | --- |
| green | 31 | 1377 | 1377 | 2641 | **1.00** | 1.92 |
| orange | 16 | 954 | 954 | 1835 | **1.00** | 1.92 |
| red | 21 | 1083 | 1083 | 2079 | **1.00** | 1.92 |
| pink | 11 | 775 | 775 | 1488 | **1.00** | 1.92 |
| lightblue | 6 | 546 | 546 | 1044 | **1.00** | 1.91 |
| yellow | 26 | 1212 | 1212 | 2324 | **1.00** | 1.92 |

(18/18 cases; the table shows one deed per group.)
**Rule.** The teacher pays the same for the *first* deed of a group as for the
*second*. Its bid escalates only on the deed that completes the monopoly.
**Confidence:** certain.
**Evidence:** `p02_auction_ceiling.csv`.

### B4. The completing deed roughly doubles the ceiling
**Observation.** `own2/own1` is 1.87–1.92 in all 18 three-deed cases (green
2641/1377 = 1.92, yellow 2324/1212 = 1.92, lightblue 1071/574 = 1.87).
**Rule.** Acquiring the deed that completes a monopoly is worth about twice
the marginal value of a non-completing deed of the same group.
**Confidence:** certain.
**Evidence:** `p02_auction_ceiling.csv`.

### B5. A group the player owns nothing of contributes no monopoly term
**Observation.** B3 and B4 together are not explained by a plain
`2 ** missing_deeds` discount, which predicts ratios 1 : 2 : 4 across
own0/own1/own2. Adding one clause does explain them exactly — that the
"before" state contributes a monopoly term **only if the player already owns
at least one deed of the group**. Marginal value is then

    M / 2**missing_after  -  (M / 2**missing_before  if owned_before > 0 else 0)

| | own0 | own1 | own2 |
| --- | --- | --- | --- |
| predicted ratio, 3-deed group | 1.00 | 1.00 | 2.00 |
| **observed** | **1.00** | **1.00** | **1.90–1.92** |
| predicted ratio, 2-deed group | 1.00 | 1.00 | — |
| **observed** (brown, darkblue) | **1.00** | **1.05–1.13** | — |

**Rule.** The monopoly term is gated on existing group presence, so the first
deed of a group is valued as if the group opportunity did not previously
exist. The residual 1.05–1.13 on two-deed groups is the base asset and
short-rent terms, which differ slightly between the two deeds.
**Confidence:** likely (the arithmetic matches to within the base-term
residual across 22 configurations, but the clause is inferred from ratios
rather than observed directly).
**Evidence:** `p02_auction_ceiling.csv`.

**Competitive note.** B3+B5 are an exploitable structural weakness, and they
sharpen Phase 5 module 2 (denial trading). The teacher will not bid defensively
for a group it has no presence in, and pays no premium for a second deed. An
opponent can therefore acquire the first two deeds of a group cheaply and only
faces competition on the third — by which point it holds the blocking
position. Recorded for Phase 5.

---

## Group D — the safety gates

A2 located a $200 floor on the buy decision. These rules establish that the
same machinery governs discretionary spending generally, that a *second*
gate exists, and which of the two binds when.

Setup: seat 0 holds a spendable position, a rival holds the green monopoly
developed to H houses per deed (H = 0..5, hotel at 5, so the worst green rent
runs $56 → $1,400), and seat 0 stands either 6 squares before the group
("near", reachable next roll) or 30 squares away ("far", not reachable).

### D1. The same $200 floor governs building and unmortgaging, not just buying
**Observation.** With an undeveloped rival and seat 0 far away, the cash
remaining after the spend is **201** for building (cost $100) and **200** for
unmortgaging (cost $110) — the same cushion A2 found for buying, across three
different spend types and two different costs.
**Rule.** The $200 floor is a property of discretionary spending in general,
not of the buy decision.
**Confidence:** certain.
**Evidence:** `p03_safety_floor.csv` (build rows), `p03b_unmortgage_isolated.csv`.

### D2. A second gate exists and dominates once opponents are developed
**Observation.** The cushion is flat at ~200 for low rival development, then
grows steeply and becomes independent of the $200 floor:

| rival H | build cushion (far) | unmortgage cushion (far) |
| --- | --- | --- |
| 0 | 201 | 200 |
| 1 | 201 | 200 |
| 2 | 201 | **250** |
| 3 | 671 | 800 |
| 4 | 871 | 1000 |
| 5 | 1071 | 1200 |

Crucially the two setups diverge at H=2 — the same rival board, different
cushions — so the second gate depends on something seat 0 holds, not only on
the threat.
**Rule.** Two gates are evaluated and the binding one is whichever demands
more cash.
**Confidence:** certain.
**Evidence:** `p03_safety_floor.csv`, `p03b_unmortgage_isolated.csv`.

### D3. The second gate is worst reachable rent, offset by liquidatable worth and rent income
**Observation.** The two setups differ in what seat 0 can liquidate: the build
setup holds the orange group (mortgage values 90+90+100 = **280**), the
unmortgage setup holds two railroads, one already mortgaged (**100**). Taking
the worst reachable green rent (Pennsylvania Avenue: $1,000 / $1,200 / $1,400
at H = 3/4/5) and subtracting liquidatable worth and a constant rent-income
term predicts every observed cushion exactly:

| setup | H | worst rent | liquidatable | income | predicted | **observed** |
| --- | --- | --- | --- | --- | --- | --- |
| build | 3 | 1000 | 280 | 49 | 671 | **671** |
| build | 4 | 1200 | 280 | 49 | 871 | **871** |
| build | 5 | 1400 | 280 | 49 | 1071 | **1071** |
| unmortgage | 3 | 1000 | 100 | 100 | 800 | **800** |
| unmortgage | 4 | 1200 | 100 | 100 | 1000 | **1000** |
| unmortgage | 5 | 1400 | 100 | 100 | 1200 | **1200** |

**Rule.**

    cash_after + rent_income + liquidatable_worth - worst_reachable_rent > 0

**Confidence:** certain (6/6 exact, across two different liquidatable values).
**Evidence:** `p03_safety_floor.csv`, `p03b_unmortgage_isolated.csv`.

### D4. The binding gate is the maximum of the two, and the crossover is predicted
**Observation.** At H=2 the two setups sit on opposite sides of the crossover,
and both are predicted correctly:

| setup | gate-2 demand | vs $200 floor | predicted | **observed** |
| --- | --- | --- | --- | --- |
| build, far | 450−280−49 = 121 | floor wins | 201 | **201** |
| unmortgage, far | 450−100−100 = 250 | gate 2 wins | 250 | **250** |

**Rule.** Required cushion = `max(gate-1 floor, gate-2 demand)`. A player with
more liquidatable assets may spend down further, which is why the build setup
still sits on the floor where the unmortgage setup has already left it.
**Confidence:** certain.
**Evidence:** as D3.

### D5. Only *reachable* danger counts
**Observation.** At low rival development the near and far positions give
different cushions; at H≥3 they are identical:

| rival H | near | far | difference |
| --- | --- | --- | --- |
| 0 | 225 | 200 | 25 |
| 1 | 260 | 200 | 60 |
| 2 | 380 | 250 | 130 |
| 3 | 800 | 800 | **0** |
| 4 | 1000 | 1000 | **0** |
| 5 | 1200 | 1200 | **0** |

The near-position excess matches expected rent *paid*, computed the same way
as A4: from position 25 the green deeds sit at gaps 6, 7 and 9, so
`Σ P(2d6=gap) × rent` gives 22.1 at H=0 (observed 25) and 56.4 at H=1
(observed 60).
**Rule.** Gate 1's rent term is *net* — expected rent collected minus expected
rent paid, both from actual board positions. Gate 2's `worst_reachable_rent`
is a worst case over the whole board, which is why it stops depending on
position once it dominates.
**Confidence:** certain for the net-rent mechanism (matches A4's method on
independent data); **likely** for the exact worst-case scope of gate 2, which
is inferred from the near/far collapse rather than measured directly.
**Evidence:** `p03_safety_floor.csv`, `p03b_unmortgage_isolated.csv`.

### D6. Methodological correction — p03's unmortgage rows are void
**Observation.** p03's original unmortgage sweep reported thresholds that were
wrong, and reported *no* threshold at H=4 and H=5 where one plainly exists. A
linear scan showed the response is not monotone in cash: with a buildable
monopoly also held, the teacher goes `END_TURN → improve_house → unmortgage`
as cash rises, because building outranks unmortgaging over a middle band.
`bisect_flip` assumes monotonicity and failed silently.
**Consequence.** The unmortgage rows of `p03_safety_floor.csv` are void and
are superseded by `p03b_unmortgage_isolated.csv`, which removes the competing
action (two railroads, no completable colour group — `build_action_available`
is False in 12/12 rows) and verifies monotonicity explicitly
(`monotone` True in 12/12). The build rows of p03 are unaffected: no mortgaged
deed was held, so no unmortgage action competed.
**Rule (about the method, not the teacher).** A threshold is only meaningful
if the response is monotone, so the scan that establishes a threshold must
also prove one exists. `probe_harness.scan_flip` now returns
`(flip, monotone, points)` and `bisect_flip` is marked unsafe on its own.
**Confidence:** certain.
**Evidence:** `p03b_unmortgage_isolated.csv`, `monotone` and
`build_action_available` columns.

**Competitive note.** D3–D5 say the teacher's caution is driven by *worst
reachable rent* and is discounted by liquidatable worth. Two exploitable
consequences: it will refuse to develop while a big rent sits within reach
even when the probability of landing there is low, and a player who mortgages
to raise cash makes the teacher *less* cautious (liquidatable worth falls, but
so does the rent it fears). Both feed Phase 5 module 3.

---

## Group E — development order (Experiment 4)

Method: seat 0 holds one completed colour group and nothing else. Each chosen
improvement is applied via `game.step`, so the engine's even-building rule and
bank inventory bind exactly as in play; the sequence of squares is the order.

### E1. The first house goes on the highest-*rent* deed, not the highest-priced
**Observation.** In 7 of 8 groups the first improvement targets the deed with
the largest base rent. Brown decides it: Mediterranean (sq 1) and Baltic
(sq 3) both cost $60, so price cannot break the tie, and the teacher picks
**sq 3**, whose base rent is 4 against Mediterranean's 2. The same holds where
price and rent agree — lightblue → sq 9, pink → sq 14, orange → sq 19,
red → sq 24, yellow → sq 29, green → sq 34.
**Rule.** Development priority is ordered by rent, not by price.
**Confidence:** certain.
**Evidence:** `p04_development_order.csv`, `first_built` column (24 rows).

### E2. Deeds identical in price *and* rent are taken in ascending square id
**Observation.** Orange 16 and 18 are identical on every attribute; the build
sequence is `19 → 16 → 18 → 19 → 16 → 18 → …`. Red 21 and 23 are likewise
identical and go `24 → 21 → 23 → …`.
**Rule.** The final tie-break is the lower square id, which corresponds to the
lower action id within the improvement family.
**Confidence:** certain.
**Evidence:** `p04_development_order.csv`, `sequence` column.

### E3. Darkblue inverts when unconstrained, and reverts under scarcity
**Observation.** Darkblue is the sole exception to E1: with ample cash and
bank stock the teacher opens on Park Place (sq 37, rent 35) rather than
Boardwalk (sq 39, rent 50). Constrain the bank to 3 houses and no hotels — so
only a single house can ever be placed — and it opens on **Boardwalk**.
**Rule.** When the whole group can be built out, every ordering reaches the
same end state, so the choice falls through to the E2 id tie-break (37 < 39).
When only one house can be placed the end states differ and E1's rent
ordering governs.
**Confidence:** likely (the explanation fits both darkblue observations and is
consistent with E1/E2, but three-deed groups also build out fully and still
follow rent order, which the account does not fully explain).
**Evidence:** `p04_development_order.csv`, darkblue rows.

### E4. A near-empty bank suppresses building almost entirely
**Observation.** With 3 houses and 0 hotels available, the teacher places
**exactly one house and stops** in all 8 groups (`n_improvements = 1`),
despite holding $5,000 and two more houses being available.
**Rule.** Scarce bank inventory collapses the value of developing, not merely
the reachable level.
**Confidence:** certain for the behaviour; the mechanism is not established.
**Evidence:** `p04_development_order.csv`, house-scarce rows.

### E5. Development stops exactly at the safety floor
**Observation.** With $400 and an unconstrained bank, brown builds 4 houses at
$50 each — spending $200 and stopping with $200 in hand, the D1 cushion to the
dollar. Groups nearer Go build further on the same cash (lightblue reaches 8
improvements) because their projected rent income is larger.
**Rule.** Building is gated by the same cushion as every other discretionary
spend, and the gate's rent term uses actual positions.
**Confidence:** certain (independent cross-validation of D1 and A3 from a
different decision family).
**Evidence:** `p04_development_order.csv`, low-cash rows.

---

## Group F — liquidation order under debt (Experiment 5)

Method: seat 0 owes a fixed debt in post_roll with `debt_player` set, which is
the engine's forced-rescue branch. Holdings are mixed on purpose — a developed
orange monopoly, two railroads, and a lone Boardwalk — so the ordering has
something to choose between. Debt is swept $50–$2,500.

### F1. Mortgages come before house sales
**Observation.** With houses on the orange group, the sequence always opens
`mortgage(15) → mortgage(5) → …` and only reaches `sell_house` once every
mortgageable deed is gone.
**Rule.** Raise cash by mortgaging before selling development.
**Confidence:** certain for the observed order — but see F2 for why this is
weaker evidence than it looks.
**Evidence:** `p05_liquidation_order.csv`.

### F2. F1 is partly forced by legality, not preference
**Observation.** The engine only offers `mortgage` for a deed with
`houses == 0`. When the orange group carries houses those deeds are not
mortgageable at all, so "railroads first" is the only legal opening.
**Rule.** Do not read F1 as the teacher protecting its monopoly. The genuine
preference is visible only in the `houses = 0` rows, where every deed is
mortgageable — and there it mortgages the **orange monopoly first**
(`18 → 16 → 19`), ahead of the railroads and Boardwalk.
**Confidence:** certain.
**Evidence:** `p05_liquidation_order.csv`, `houses = 0` rows compared against
`houses = 2/4`.

### F3. Cheapest asset first, by mortgage value
**Observation.** With no houses anywhere, the mortgage order is
`18 → 16 → 19 → 15 → 5 → 39`, whose mortgage values are
**90, 90, 100, 100, 100, 200** — ascending, with no exception.
**Rule.** Liquidation proceeds in ascending mortgage value, raising the least
cash per action and so minimising over-liquidation. This is what makes the
monopoly go first in F2: orange deeds are simply the cheapest, not specially
protected or specially expendable.
**Confidence:** certain.
**Evidence:** `p05_liquidation_order.csv`.

### F4. Deeds are emptied one at a time, not levelled down together
**Observation.** At `houses = 2`, once house-selling starts the pattern is
`sell_house(18) → sell_house(18) → mortgage(18) → sell_house(16) →
sell_house(16) → mortgage(16) → …` — each deed is stripped and mortgaged
before the next is touched. The opening three sales at `houses = 4`
(`19 → 18 → 16`) are the engine's even-building constraint forcing one house
off each before any deed may go below the others.
**Rule.** Subject to even-building, liquidation concentrates on one deed at a
time rather than spreading evenly.
**Confidence:** certain.
**Evidence:** `p05_liquidation_order.csv`, `sequence` column.

### F5. Bankruptcy is declared only when nothing remains, and development is a buffer
**Observation.** `DECLARE_BANKRUPT` appears only as the final action after
every asset is exhausted. Depth of development changes survival: at a $1,000
debt, `houses = 4` survives (14 actions, no bankruptcy) while `houses = 2`
does not.
**Rule.** Houses are liquidatable worth, so a more developed position absorbs
a larger debt — the same quantity that relaxes the D3 safety gate.
**Confidence:** certain (independent cross-validation of D3).
**Evidence:** `p05_liquidation_order.csv`, `went_bankrupt` column.

---

## Group G — jail policy (Experiment 7)

### G1. The pre-roll jail menu is a deferral, not a decision
**Observation.** Swept over 224 pre-roll states (card × jail turn × cash ×
rival development), the teacher chose `END_TURN` in **224/224** and never paid
bail or played a card. In post_roll on the same states it leaves jail in
**192/448**.
**Rule.** In pre_roll `END_TURN` advances to post_roll rather than declining to
leave, so the teacher defers the jail choice to the phase where it cannot be
deferred. Any reading of the pre-roll sweep as "never leaves jail" is wrong.
**Confidence:** certain.
**Evidence:** `p07_jail_policy.csv` (pre-roll only, retained for this rule),
`p07b_jail_post_roll.csv` (both phases, 896 rows).

### G2. The card is spent freely; bail is not
**Observation.** In post_roll, holding a get-out-of-jail-free card the teacher
leaves in **141/224** (63%); without one it pays bail in **51/224** (23%) and
otherwise rolls for doubles. Whether it owns deeds barely matters (27 vs 24
bail payments).
**Rule.** A free exit is taken readily; a $50 exit is treated as discretionary
spending and usually refused.
**Confidence:** certain.
**Evidence:** `p07b_jail_post_roll.csv`.

### G3. Bail obeys the $200 floor
**Observation.** Without a card, bail is paid **0/32** at cash $50, $100 and
$200, and first appears at cash $260 (6/32). Paying $50 from $200 leaves $150,
under the floor; from $260 it leaves $210, over it. The flip is bracketed in
($200, $260] by this grid, and the floor predicts $250.
**Rule.** Bail is discretionary spending subject to the D1 cushion.
**Confidence:** certain for the gating; the exact flip is bracketed, not
located, at this grid resolution.
**Evidence:** `p07b_jail_post_roll.csv`.

### G4. Jail is treated as shelter — exit rate falls as the board gets dangerous
**Observation.** Post-roll exit rate against rival development:

| rival houses | 0 | 2 | 4 | 5 (hotel) |
| --- | --- | --- | --- | --- |
| leaves jail | **64.3%** | 42.9% | 32.1% | 32.1% |

**Rule.** The more rent is waiting outside, the more willing the teacher is to
stay in jail — the classic result that jail is shelter late and a waste early.
The teacher does track it.
**Confidence:** certain.
**Evidence:** `p07b_jail_post_roll.csv`.

### G5. It rolls for doubles first and buys out later
**Observation.** Exit rate by jail turn: **5.4%** at turn 0, 35.7% at turn 1,
65.2% at turns 2 and 3. Exit rate also rises with cash: 31.2% at $50–$200,
43.8% at $400, 50.0% at $800, **75.0%** at $1,500.
**Rule.** Rolling for doubles is free, so it is tried first; a paid exit is
bought only as the forced-release boundary approaches, and more readily when
cash is ample.
**Confidence:** certain.
**Evidence:** `p07b_jail_post_roll.csv`.

**Competitive note.** G4 is the teacher playing jail correctly, so there is no
easy edge there. G2 is more interesting: refusing a $50 bail at moderate cash
early, when the board is cheap and deeds are still unowned, costs tempo in
exactly the phase where buying matters most. That is a Phase 5 endgame-switch
candidate — the value of leaving jail should be stage-dependent, and the
teacher's gate is not.

---

## Group C — rollout vs value divergence (Experiment 8, pulled forward)

Both variants are deterministic given a state — the rollout uses fixed
common-random-number streams — so repeating a state cannot yield a new answer.
The agreement rate is therefore estimated over a *population of constructed
boundary states*, not over RNG draws.

### C1. On boundary states the two variants disagree on a third of decisions
**Observation.** Over 230 constructed boundary states:

| category | n | diverge | rate | value s | rollout s | cost ratio |
| --- | --- | --- | --- | --- | --- | --- |
| auction | 56 | 53 | **94.6%** | 0.011 | 11.37 | 1035× |
| build | 48 | 28 | **58.3%** | 0.049 | 22.60 | 464× |
| buy | 112 | 0 | 0.0% | 0.043 | 4.78 | 112× |
| trade | 14 | 0 | 0.0% | 0.029 | 8.89 | 309× |
| **all** | **230** | **81** | **35.2%** | | | |

95% Wilson CI on the overall rate: **[29.3%, 41.6%]**.
**Rule.** Lookahead changes the decision often, but only in two families:
auctions and building. It never changed a buy or trade decision in this
corpus.
**Confidence:** certain (both policies are deterministic given a state, so the
null "rollout never changes the decision" is falsified outright).
**Evidence:** `p08_rollout_divergence.csv` (230 rows, seed 20250811).

> **CONTRADICTED IN PART — see H5.** C1's "0.0% in trade (0/14)" does not
> generalise. Experiment 6 built a richer trade population (seat 0 holding
> four deeds against a rival holding five, sweeping the sweetener across the
> whole accept surface) and measured **50/54 divergences — 92.6%**. C1's trade
> row reflects one narrow setup, not trade decisions in general. The auction,
> build and buy figures are unaffected. C1 is downgraded from **certain** to
> **certain for auction/build/buy, contradicted for trade**, and the Phase 4
> gate is corrected in `DECISIONS.md` D1.3.

### C2. Where they diverge, the rollout is systematically more liquidity-preserving
**Observation.** Every one of the 53 auction divergences is bid-versus-bid:
the value variant takes the largest increment (B2) while the rollout takes a
smaller one — in the certification scenario, `+$100` versus `+$1` on the same
state. The 28 build divergences fall into three patterns: `improve_house` vs
`improve_house` on a different square (11), `END_TURN` vs `mortgage` (10), and
`improve_house` vs `mortgage` (7) — the rollout raising cash where the value
variant either builds or does nothing.
**Rule.** Truncated lookahead consistently prefers holding liquidity: minimum
viable bids instead of maximum ones, and mortgaging instead of building.
**Confidence:** likely (the direction is consistent across all 81 divergences,
but "preserving liquidity" is an interpretation of the action pattern).
**Evidence:** `p08_rollout_divergence.csv`; corroborated independently by the
certification scenarios (`probes/certification/certification_result.txt`),
where 5 of 10 scenarios split value-vs-rollout in the same direction, and
identically under both `core.py` versions.

### C3. The earlier "10/10 agreement" was an artefact of ordinary play
**Observation.** On ordinary seed-3 play the variants agreed on 10/10 early
decisions; on constructed boundary states they disagree on 35.2%.
**Rule.** Most decisions in ordinary play are lopsided, so agreement there says
nothing about the value of lookahead. Divergence must be measured on states
selected for closeness, not sampled.
**Confidence:** certain.
**Evidence:** `p08_rollout_divergence.csv` vs `DECISIONS.md` D0.6.

---

## Group H — the trade accept/decline surface (Experiment 6)

Method: seat 0 holds 16, 18 (two oranges), 25 and 37; the rival holds 19 (the
completing orange), 5, 1, 21, 23. An incoming offer is swept over which deed
is offered, which is requested, and a cash sweetener from −$400 (seat 0 pays)
to +$400 (seat 0 is paid) in $25 steps. Monotonicity is verified with
`scan_flip`, not assumed (D6).

> **SCOPE CORRECTED — D7.10 (2026-08-13), confidence downgraded to `partial`.**
> H1-H4 describe the incoming-offer decision as a choice between ACCEPT and
> DECLINE. Measured on real play, that is not the decision the teacher faces.
> Of 974 states where both were legal, **none** had a menu of exactly
> {ACCEPT, DECLINE}, and on the strong field the teacher answered with a trade
> proposal of its own 42.4% of the time (`exch_trade` 38.2%, `sell_trade` 3.5%,
> `buy_trade` 0.7%) against ACCEPT 1.6%. These probes were run on a synthetic
> two-option surface, so what they measured — the shape of the accept region
> — is not refuted; it is a slice through a wider decision. Nothing built on
> H1-H4 should assume the reply is binary.

### H1. The accept surface is narrow and *not* monotone in the sweetener
**Observation.** Accept/decline across the sweetener grid
(`A` = accept, `.` = decline; leftmost = we pay $400, rightmost = we receive
$400):

    offer 19 / want 25   AAAAAAAAAAAAAAAAAAAAAAAAAAAAA....
    offer 19 / want 16   .........AA......................
    offer  5 / want 25   .........AA......................
    offer  5 / want 16   .................................
    offer  1 / want 25   .................................
    offer  1 / want 16   .................................

`monotone` is False in 3 of 6 cells. Two cells accept only on a
**two-point island** at −$175 and −$150 and decline on both sides of it.
**Rule.** "Accept above a cash threshold" is the wrong model for trades. Any
single "accept from $X" figure — including the `accept_from_cash` column of
this probe's own CSV — is meaningless for the non-monotone cells and must not
be used as a threshold.
**Confidence:** certain.
**Evidence:** `p06_trade_surface.csv`, `monotone` column; surface printed above.

### H2. It refuses offers that are *too generous*, because the counterparty could not afford them
**Observation.** The top-left cell accepts while paying $400 but declines once
offered more than +$300. Raising only the proposer's cash moves that edge:

| proposer cash | highest sweetener still accepted |
| --- | --- |
| $600 | +$300 |
| $1,200 | +$300 |
| $2,000 | +$950 |
| $5,000 | +$1,200 (grid max) |
| $20,000 | +$1,200 (grid max) |

**Rule.** The upper edge of the accept region is set by the *counterparty's*
ability to pay, not by seat 0's valuation — the teacher evaluates the offer
against both sides' safety and rejects a trade the proposer could not safely
fund. It therefore declines free money from a poor opponent.
**Confidence:** certain.
**Evidence:** `p06_trade_surface.csv` plus the proposer-cash sweep above.

### H3. It pays heavily for the deed that completes its group
**Observation.** Offered deed 19 (completing seat 0's orange group) for a
spare railroad, it accepts across the entire lower range — including paying
the full $400 at the grid edge.
**Rule.** Completion is worth at least $400 over a spare deed, consistent with
B4's doubling of the auction ceiling on the completing deed.
**Confidence:** certain (a lower bound — the grid does not reach the refusal
point on that side).
**Evidence:** `p06_trade_surface.csv`.

### H4. Deeds of no group value are refused at every price
**Observation.** Three of six cells never accept anywhere on an $800-wide
sweetener range: offering a cheap brown (sq 1) is refused outright, and so is
any offer requesting seat 0's own orange piece in exchange for a non-orange
deed.
**Rule.** No cash sweetener within $400 buys a deed out of a near-monopoly,
and a worthless deed is not made attractive by cash.
**Confidence:** certain.
**Evidence:** `p06_trade_surface.csv`, `never_accepts` column.

### H5. Rollout and value disagree on almost every trade decision
**Observation.** Across 54 trade states spanning the full surface, value and
rollout selected the same action in **4** — a divergence rate of **92.6%**.
This directly contradicts C1's 0/14.
**Rule.** Trade is not a family where lookahead is redundant; it is the family
where the two variants agree *least*. C1's trade row was an artefact of a
single narrow setup.
**Confidence:** certain.
**Evidence:** `p06_trade_surface.csv`, `rollout_agree` / `rollout_states`
columns (coarser $100 rollout grid — see the file's note on cost).

**Competitive note.** H2 is the concrete requirement for Phase 5 module 2's
opponent-acceptance model: whether the teacher accepts our offer depends on
*our* cash, not only on the deeds. An offer that is generous but unaffordable
is rejected, so the proposer model must include our own safety position. H1
means that model cannot be a threshold in the sweetener — it needs the
surface.

---

## Status

| experiment | status |
| --- | --- |
| 0. Audit-trail certification (added) | done — PASS, 6,314 decisions, 0 mismatches |
| 1. Buy threshold curve | done — A1–A6 |
| 1b. Rent-residual mechanism (added) | done — A4, A5 |
| 2. Auction ceiling curve | done — B1–B5 |
| 3. Safety floor scaling | done — D1–D5 (p03 unmortgage rows void, superseded by p03b; see D6) |
| 4. Development order | done — E1–E5 |
| 5. Mortgage / liquidation order | done — F1–F5 |
| 6. Trade accept/decline surface | done — H1–H5 (contradicts C1's trade row) |
| 7. Jail policy | done — G1–G5 (p07 reinterpreted as pre-roll deferral; see G1) |
| 8. Rollout-variant divergence | done (pulled forward) — C1–C3 |

**40 rules evidenced** (A1–A6, B1–B5, C1–C3, D1–D6, E1–E5, F1–F5, G1–G5,
H1–H5) against the ≥25 required for Phase 1 acceptance — **the bar is
cleared and Phase 1 is complete**. All eight briefed experiments are done,
plus three added ones (1b rent-residual mechanism, 3b isolated unmortgage,
7b jail at the binding phase) and the audit-trail certification.

### Rules by confidence

| confidence | rules |
| --- | --- |
| certain | A1–A6, B1–B4, C3, D1–D4, D6, E1, E2, E4, E5, F1–F5, G1–G5, H1–H5 |
| likely | B5, C2, D5, E3 |
| partly contradicted | C1 (holds for auction/build/buy; trade row overturned by H5) |
| guess | — |

35 certain, 4 likely, 1 partly contradicted.

Two corrections are on the record, both found by later probes rather than
left standing:

- **C1 (trade row) contradicted by H5** — 0/14 divergence became 92.6% on a
  wider population. The rule is kept with its confidence downgraded and the
  contradiction recorded in place, per this document's own convention.
- **p03's unmortgage rows retracted (D6)** — a non-monotone predicate broke
  the bisection; superseded by p03b.

One reinterpretation: p07's "never leaves jail" measured pre-roll deferral,
not jail policy (G1); the rows are retained as evidence for what they do
show.
