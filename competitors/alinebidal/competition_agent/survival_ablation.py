"""Step 0 — is bankruptcy a cause or a symptom?

The question
------------
The frozen agent goes bankrupt in 27.8% of games against the strong field and
in 87% against the teacher. The endgame module aimed at that number and was
disabled because its implementation widened `gates_ok` and blocked the very
purchases `denial` depends on — but its *hypothesis* was never tested.

Before any survival module is rebuilt, the hypothesis gets the same treatment
trade got in D5.x: pin the family to the teacher's ground truth and read the
win-rate recovery. If handing every debt-resolution decision to a strictly
stronger player buys nothing, then bankruptcy is downstream of decisions made
long before the debt — a symptom — and survival work is misdirected.

There is already direct reason to expect that. The `liquidation` family
recovered **+0.0pp** when pinned in D5.2, despite having the worst agreement in
the project (23.8%). This run asks the same question in the setting that
matters, and with the differences that could plausibly have hidden an effect
last time closed off.

What is different from the D5.2 liquidation arm
-----------------------------------------------
1. **Opponents.** D5.2 pinned against `ASUValueV1` on the other three seats.
   Against a player that strong the agent's own survival may be irrelevant to
   the result. This runs against the strong scripted field (~1252 ELO), the
   closer proxy for the tournament.
2. **Baseline.** D5.2's baseline was bare `spec_policy`. This one is the
   *frozen agent* — `spec_policy` + `BEYOND_DENIAL=1` — so the measurement
   applies to the thing that would actually be modified.
3. **Pin definition.** D5.2 assigned a decision to `liquidation` by the
   teacher's *chosen action*, which also catches voluntary mortgaging outside
   debt. Here the pin is on the *state*: `env.debt_player == pid`, i.e. exactly
   the forced debt-resolution decisions — which house to sell, which deed to
   mortgage, in what order, and whether to declare bankruptcy. That is the
   family a survival module would own.

Design
------
Paired. Both arms play the same seed set with the agent in the same seat, so
board luck cancels between arms; a seed's seat is `seed % 4`, which spreads the
agent over all four positions without making games within an arm correlated
(each game is a distinct seed). Seat effects are real here — identical policies
have differed by 7 points across arrangements.

Reported per arm: leader rate (the metric the 34.2% figure uses) with a Wilson
interval, decisive rate, bankruptcy rate, and how often the oracle path fired
as a fraction of decisions. Between arms: a two-proportion z-test and the
paired McNemar test, which is the more powerful of the two under this design.

    python3 competition_agent/survival_ablation.py --games 1200 --workers 9
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ASU_FROZEN_TEACHER import ASUValueV1  # noqa: E402
from ASU_FROZEN_TEACHER.evaluate import _new_seeded_game  # noqa: E402

from competition_agent.policies import build_policy  # noqa: E402
from competition_agent.proc import ensure_hash_seed, managed_pool  # noqa: E402

# ~1252 ELO per the user-supplied tournament logs. Not the weak
# fixed-a/b/c field, which the agent already beats at 55.8%.
STRONG_FIELD = ("fixed-b", "fixed-d", "fixed-e")

# `none2` is an exact duplicate of `none`. With PYTHONHASHSEED pinned it must
# come out byte-identical, which is the harness's own self-test; with the seed
# left random it measures how many games the OPPONENTS alone flip, i.e. the
# noise floor any paired comparison on this field sits on. Without that number
# "3 of 2,000 games changed" cannot be attributed to the pin.
ARM_PINS = {"none": False, "none2": False, "survival": True}
ARMS = ("none", "survival")


class SurvivalPinned:
    """The frozen agent, except that under debt the teacher decides.

    `env.debt_player == pid` is the engine's own marker for "this seat owes
    money and must raise it or fold". Everything downstream of that flag —
    mortgage order, house-sale order, the bankruptcy call — is what a survival
    module would replace, so it is exactly what gets pinned.
    """

    def __init__(self, player_id: int, pin: bool, rng_seed: int) -> None:
        self.pid = player_id
        self.pin = pin
        self.agent = build_policy("final", player_id, rng_seed)
        self.teacher = ASUValueV1(player_id) if pin else None
        self.total = 0
        self.debt_decisions = 0
        self.fired = 0
        self.changed = 0

    def choose_action(self, env) -> int:
        self.total += 1
        in_debt = getattr(env, "debt_player", None) == self.pid
        if in_debt:
            self.debt_decisions += 1
            if self.pin:
                self.fired += 1
                t = int(self.teacher.choose_action(env))
                # "Fired" overstates the intervention: most debt menus are one
                # or two items and the agent already picks what the teacher
                # picks. What the arm actually tests is the subset where the
                # two differ, so that subset is counted. The agent is consulted
                # on the live env only to record this; `t` is what is returned,
                # and the teacher's answer was verified to be independent of
                # whether the agent ran first.
                a = int(self.agent.choose_action(env))
                if a != t:
                    self.changed += 1
                return t
        return int(self.agent.choose_action(env))


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - m), min(1.0, c + m)


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_prop_z(k1, n1, k2, n2):
    """Unpooled-difference / pooled-variance two-proportion z, two-sided p."""
    if not n1 or not n2:
        return 0.0, 1.0
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p2 - p1) / se
    return z, 2 * (1 - _phi(abs(z)))


def mcnemar(pairs):
    """pairs: list of (baseline_win, pinned_win). Returns (b, c, z, p).

    b = baseline won and pinned lost, c = pinned won and baseline lost. Only
    discordant pairs carry information; concordant ones cancel, which is
    precisely the board luck this design pairs away.
    """
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    if b + c == 0:
        return b, c, 0.0, 1.0
    z = (c - b) / math.sqrt(b + c)
    return b, c, z, 2 * (1 - _phi(abs(z)))


def _game(job):
    seed, arm, max_steps = job
    seat = seed % 4
    game = _new_seeded_game(seed)
    env = game.env

    agents = {}
    field = list(STRONG_FIELD)
    for s in range(4):
        if s == seat:
            agents[s] = SurvivalPinned(s, ARM_PINS[arm], seed * 4 + s)
        else:
            agents[s] = build_policy(field.pop(0), s, seed * 4 + s)

    steps = 0
    t0 = time.perf_counter()
    while not env.done and steps < max_steps:
        actor = env.whose_turn()
        game.step(int(agents[actor].choose_action(env)))
        steps += 1

    active = [p.player_id for p in env.players if not p.bankrupt]
    decisive = len(active) == 1
    me = agents[seat]
    return {
        "arm": arm, "seed": seed, "seat": seat,
        "steps": steps, "decisive": decisive,
        "leader_win": env.winner() == seat,
        "decisive_win": decisive and active[0] == seat,
        "bankrupt": bool(env.players[seat].bankrupt),
        "net": float(env.players[seat].net_worth()),
        "total": me.total, "debt": me.debt_decisions, "fired": me.fired,
        "changed": me.changed,
        "seconds": time.perf_counter() - t0,
    }


def report(rows, arms=ARMS):
    by = {a: [r for r in rows if r["arm"] == a] for a in arms}
    print(f"\n{'arm':<10}{'leader rate':>26}{'decisive':>11}"
          f"{'bankrupt':>11}{'oracle fired':>20}")
    print("-" * 78)
    stat = {}
    for a in arms:
        sub = by[a]
        n = len(sub)
        k = sum(1 for r in sub if r["leader_win"])
        p, lo, hi = wilson(k, n)
        dn = sum(1 for r in sub if r["decisive"])
        dk = sum(1 for r in sub if r["decisive_win"])
        bk = sum(1 for r in sub if r["bankrupt"])
        fired = sum(r["fired"] for r in sub)
        changed = sum(r.get("changed", 0) for r in sub)
        debt = sum(r["debt"] for r in sub)
        tot = sum(r["total"] for r in sub)
        stat[a] = (k, n, bk, p)
        print(f"{a:<10}"
              f"{f'{k}/{n}  {100*p:5.1f}% [{100*lo:4.1f},{100*hi:4.1f}]':>26}"
              f"{f'{100*dk/max(dn,1):5.1f}%':>11}"
              f"{f'{100*bk/max(n,1):5.1f}%':>11}"
              f"{f'{fired}/{tot} = {100*fired/max(tot,1):.2f}%':>20}")
        if ARM_PINS[a]:
            print(f"{'':<10}{'':>26}{'':>11}{'':>11}"
                  f"{f'(actually overridden: {changed} = '
                    f'{100*changed/max(fired,1):.1f}% of fires)':>20}")
        else:
            print(f"{'':<10}{'':>26}{'':>11}{'':>11}"
                  f"{f'(debt states seen: {debt})':>20}")

    base, test = arms[0], arms[-1]
    if base == test:
        return
    (k0, n0, b0, p0), (k1, n1, b1, p1) = stat[base], stat[test]
    z, pv = two_prop_z(k0, n0, k1, n1)
    print(f"\ndelta ({test} - {base})   {100*(p1-p0):+.2f}pp")
    print(f"  two-proportion z = {z:+.2f}   p = {pv:.4f}   "
          f"{'SIGNIFICANT' if pv < 0.05 else 'not significant'}")

    idx = {r["seed"]: r for r in by[base]}
    pairs = [(idx[r["seed"]]["leader_win"], r["leader_win"])
             for r in by[test] if r["seed"] in idx]
    b, c, zm, pm = mcnemar(pairs)
    print(f"  McNemar (paired, n={len(pairs)}): {base}-only {b}, "
          f"{test}-only {c}, z = {zm:+.2f}, p = {pm:.4f}   "
          f"{'SIGNIFICANT' if pm < 0.05 else 'not significant'}")

    zb, pb = two_prop_z(b0, n0, b1, n1)
    print(f"\nbankruptcy rate  {base} {100*b0/max(n0,1):.1f}%  -> "
          f"{test} {100*b1/max(n1,1):.1f}%   "
          f"(z = {zb:+.2f}, p = {pb:.4f})")
    print("  If the oracle does not even lower the bankruptcy rate, the pin "
          "is not reaching the mechanism.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1200,
                    help="games per arm (distinct seeds; seat = seed %% 4)")
    ap.add_argument("--seed-base", type=int, default=960000)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--arms", type=str, default=",".join(ARMS),
                    help="two arms to compare; `none,none2` is the null "
                         "control")
    ap.add_argument("--hash-seed", type=str, default="0",
                    help="PYTHONHASHSEED to pin. Pass 'random' to leave it "
                         "randomised, which is what the noise-floor control "
                         "needs.")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    # The strong field contains `fixed-d`, whose colour-target set iterates in
    # hash order. Pin it before any game is played.
    ensure_hash_seed(args.hash_seed)
    arms = tuple(a.strip() for a in args.arms.split(","))

    stem = "survival_ablation" + (f"_{args.tag}" if args.tag else "")
    out = Path(__file__).resolve().parent / "probes" / f"{stem}.json"
    partial = out.with_suffix(".partial.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    jobs = [(args.seed_base + k, arm, args.max_steps)
            for arm in arms for k in range(args.games)]

    done = {}
    if partial.exists() and not args.no_resume:
        for line in partial.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["arm"], r["seed"])] = r
        if done:
            print(f"resuming: {len(done)} game(s) already recorded")

    rows = [r for r in done.values() if r["seed"] - args.seed_base < args.games]
    todo = [] if args.report_only else [j for j in jobs
                                        if (j[1], j[0]) not in done]
    print(f"field: agent vs {list(STRONG_FIELD)}   "
          f"{len(todo)} game(s) to play, {len(rows)} reused")

    t0 = time.time()
    if todo:
        with partial.open("a") as sink, managed_pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_game, todo), 1):
                rows.append(r)
                sink.write(json.dumps(r) + "\n")
                sink.flush()
                if i % 100 == 0 or i == len(todo):
                    el = (time.time() - t0) / 60
                    rate = i / max(el, 1e-9)
                    print(f"  {i}/{len(todo)} games  {rate:.0f} g/min  "
                          f"ETA {(len(todo)-i)/max(rate,1e-9):.1f} min",
                          flush=True)

    rows = [r for r in rows if r["arm"] in arms]
    out.write_text(json.dumps(rows, indent=1))
    report(rows, arms)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
