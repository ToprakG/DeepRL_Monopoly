"""First-monopoly probe. How strong agents beat the focus policy.

Not a search. After every H2H step we record who completed the first real-estate
set and who first reached three houses. That is the race this field is decided by.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ASU_FROZEN_TEACHER.core import preserve_global_rng
from ASU_FROZEN_TEACHER.evaluate import DEFAULT_MAX_DECISIONS, _new_seeded_game
from monopoly_game_engine.constants import COLOR_GROUPS, NUM_PLAYERS

from oracle.agent import OracleConfig
from oracle.eval_h2h import GOAT_ID, _H2HFactory, _Spec, _focus_seat, rotate_lineup
from oracle.plus_steals import REAL_COLOURS

DEFAULT_OURS = GOAT_ID
STRONG = ("slayer-v1", "underdog-v1", "inncenta-heuristic", "alinebidal-final")


def _fields(ours: str, *, strong_only: bool) -> tuple[tuple[str, tuple[str, ...], int], ...]:
    if strong_only:
        return (
            ("sui", (ours, "slayer-v1", "underdog-v1", "inncenta-heuristic"), 30),
            ("sua", (ours, "slayer-v1", "underdog-v1", "alinebidal-final"), 30),
            ("sia", (ours, "slayer-v1", "inncenta-heuristic", "alinebidal-final"), 30),
        )
    return (
        ("strong", (ours, "slayer-v1", "underdog-v1", "inncenta-heuristic"), 96),
        ("vs_slayer", (ours, "slayer-v1", "fixed-b", "fixed-c"), 64),
        ("vs_inncenta", (ours, "inncenta-heuristic", "fixed-b", "fixed-c"), 64),
        ("vs_underdog", (ours, "underdog-v1", "fixed-b", "fixed-c"), 64),
        ("vs_alinebidal", (ours, "alinebidal-final", "fixed-b", "fixed-c"), 64),
    )


def _mono_of(env, pid: int) -> str | None:
    for color in REAL_COLOURS:
        group = COLOR_GROUPS[color]
        if all(env.properties[sq].owner == pid for sq in group):
            return color
    return None


def _three_of(env, pid: int) -> str | None:
    for color in REAL_COLOURS:
        for sq in COLOR_GROUPS[color]:
            prop = env.properties[sq]
            if prop.owner == pid and int(getattr(prop, "houses", 0) or 0) >= 3:
                return color
    return None


def _owned_in(env, pid: int, color: str) -> int:
    return sum(env.properties[sq].owner == pid for sq in COLOR_GROUPS[color])


def _event(seat: int, policies: list[str], color: str, env) -> dict[str, Any]:
    return {
        "seat": seat,
        "policy": policies[seat],
        "color": color,
        "round": int(env.round),
        "cash": float(env.players[seat].cash),
    }


def _probe_one(payload: dict[str, Any]) -> tuple[str, int, dict[str, Any]]:
    config = OracleConfig(**payload["config"])
    factory = _H2HFactory(config, seed=payload["seed"])
    specs = tuple(_Spec(policy_id) for policy_id in payload["policies"])
    policies = [spec.policy_id for spec in specs]
    our_seat = _focus_seat(specs)
    with preserve_global_rng():
        game = _new_seeded_game(payload["seed"])
        agents = [factory.build(spec, seat) for seat, spec in enumerate(specs)]
        env = game.env
        decisions = 0
        first_mono = None
        first_three = None
        our_deeds = None
        max_decisions = int(payload["max_decisions"])
        while not env.done and decisions < max_decisions:
            actor = env.whose_turn()
            allowed = env.get_allowed_actions(actor)
            action = agents[actor].choose_action(env)
            if action not in allowed:
                raise RuntimeError(
                    f"{policies[actor]} illegal {action} seat {actor}"
                )
            ours_before = (
                {color: _owned_in(env, our_seat, color) for color in REAL_COLOURS}
                if first_mono is None
                else None
            )
            game.step(action)
            decisions += 1
            if first_mono is None:
                for pid in range(NUM_PLAYERS):
                    color = _mono_of(env, pid)
                    if color is not None:
                        first_mono = _event(pid, policies, color, env)
                        our_deeds = 0 if ours_before is None else ours_before[color]
                        break
            if first_three is None:
                for pid in range(NUM_PLAYERS):
                    color = _three_of(env, pid)
                    if color is not None:
                        first_three = _event(pid, policies, color, env)
                        break
            if first_mono is not None and first_three is not None:
                # Still play out — winner can differ from first-set owner.
                pass
        winner = env.winner() if env.done else None
        nw = [float(p.net_worth()) for p in env.players]
        rec = {
            "field": payload["field"],
            "index": payload["index"],
            "seed": payload["seed"],
            "policies": policies,
            "our_seat": our_seat,
            "winner": winner,
            "winner_policy": None if winner is None else policies[winner],
            "rounds": int(env.round),
            "truncated": not env.done,
            "our_nw": nw[our_seat],
            "our_bust": nw[our_seat] <= 0.0,
            "first_mono": first_mono,
            "first_three": first_three,
            "our_deeds_on_first_mono": our_deeds,
            "we_won": winner == our_seat,
        }
    return payload["field"], int(payload["index"]), rec


def _jobs(
    seed0: int,
    config: OracleConfig,
    fields: tuple[tuple[str, tuple[str, ...], int], ...],
) -> list[dict[str, Any]]:
    jobs = []
    offset = 0
    for name, lineup, games in fields:
        for index in range(games):
            specs = rotate_lineup(lineup, index)
            jobs.append(
                {
                    "field": name,
                    "index": index,
                    "policies": [spec.policy_id for spec in specs],
                    "seed": seed0 + offset + index,
                    "max_decisions": DEFAULT_MAX_DECISIONS,
                    "config": asdict(config),
                }
            )
        offset += games
    return jobs


def _pct(n: int, d: int) -> str:
    if d <= 0:
        return "n/a"
    return f"{100.0 * n / d:5.1f}% ({n}/{d})"


def analyze(records: list[dict[str, Any]], ours: str) -> str:
    by_field: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_field[rec["field"]].append(rec)
    lines = ["FIRST-MONOPOLY PROBE", ""]
    loss_deeds: Counter[int] = Counter()
    loss_colors: Counter[str] = Counter()
    loss_first_them = 0
    loss_first_us = 0
    loss_n = 0
    three_them_on_loss = 0
    for name in sorted(by_field):
        games = by_field.get(name, [])
        if not games:
            continue
        wins = sum(1 for g in games if g["we_won"])
        bust = sum(1 for g in games if g["our_bust"])
        lines.append(f"== {name}  n={len(games)}  our WR {_pct(wins, len(games))}  bust {_pct(bust, len(games))}")
        losers = [g for g in games if not g["we_won"]]
        winners = [g for g in games if g["we_won"]]
        for label, bucket in (("LOSS", losers), ("WIN", winners)):
            if not bucket:
                continue
            mono_us = sum(
                1
                for g in bucket
                if g["first_mono"] and g["first_mono"]["policy"] == ours
            )
            mono_none = sum(1 for g in bucket if not g["first_mono"])
            three_us = sum(
                1
                for g in bucket
                if g["first_three"] and g["first_three"]["policy"] == ours
            )
            colors = Counter(
                g["first_mono"]["color"]
                for g in bucket
                if g["first_mono"] and g["first_mono"]["policy"] != ours
            )
            deeds = Counter(
                g["our_deeds_on_first_mono"]
                for g in bucket
                if g["first_mono"] and g["first_mono"]["policy"] != ours
                and g["our_deeds_on_first_mono"] is not None
            )
            who = Counter(
                g["first_mono"]["policy"]
                for g in bucket
                if g["first_mono"]
            )
            lines.append(
                f"  {label} n={len(bucket)}  first-set-us {_pct(mono_us, len(bucket))}  "
                f"first-3h-us {_pct(three_us, len(bucket))}  no-set {_pct(mono_none, len(bucket))}"
            )
            if who:
                parts = ", ".join(f"{k}={v}" for k, v in who.most_common())
                lines.append(f"    first-set owner: {parts}")
            if colors:
                parts = ", ".join(f"{k}={v}" for k, v in colors.most_common(6))
                lines.append(f"    their first-set colour: {parts}")
            if deeds:
                parts = ", ".join(f"{k} deeds={v}" for k, v in sorted(deeds.items()))
                lines.append(f"    our pieces of that colour: {parts}")
        for g in losers:
            if not g["first_mono"]:
                continue
            loss_n += 1
            if g["first_mono"]["policy"] == ours:
                loss_first_us += 1
            else:
                loss_first_them += 1
                loss_colors[g["first_mono"]["color"]] += 1
                if g["our_deeds_on_first_mono"] is not None:
                    loss_deeds[int(g["our_deeds_on_first_mono"])] += 1
            if g["first_three"] and g["first_three"]["policy"] != ours:
                three_them_on_loss += 1
        lines.append("")

    lines.append("== SHARED (all losses with a first set)")
    if loss_n:
        lines.append(f"  they completed first: {_pct(loss_first_them, loss_n)}")
        lines.append(f"  we completed first and still lost: {_pct(loss_first_us, loss_n)}")
        lines.append(f"  they hit 3-houses first (on a loss): {_pct(three_them_on_loss, loss_n)}")
        if loss_colors:
            parts = ", ".join(f"{k}={v}" for k, v in loss_colors.most_common())
            lines.append(f"  colour they completed: {parts}")
        if loss_deeds:
            parts = ", ".join(f"{k} deeds={v}" for k, v in sorted(loss_deeds.items()))
            lines.append(f"  our pieces of their set: {parts}")
            zero = loss_deeds.get(0, 0)
            raced = sum(v for k, v in loss_deeds.items() if k >= 1)
            tot = zero + raced
            if tot:
                lines.append(
                    f"  never contested (0 deeds): {_pct(zero, tot)}  "
                    f"in the colour and lost the race: {_pct(raced, tot)}"
                )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts_scratch/first_mono_probe"))
    parser.add_argument("--seed", type=int, default=9_800_000)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument(
        "--ours",
        default=DEFAULT_OURS,
        help="Focus policy id (default: toprakthegoat-v1)",
    )
    parser.add_argument(
        "--strong-only",
        action="store_true",
        help="90 games vs slayer/underdog/inncenta/alinebidal; no fixed agents",
    )
    parser.add_argument(
        "--only-seeds",
        default="",
        help="Comma-separated game seeds to run (from a prior probe jsonl)",
    )
    args = parser.parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "games.jsonl"
    report_path = out / "report.txt"
    config = OracleConfig()
    ours = str(args.ours)
    fields = _fields(ours, strong_only=args.strong_only)
    jobs = _jobs(args.seed, config, fields)
    if str(args.only_seeds).strip():
        keep = {int(part) for part in str(args.only_seeds).split(",") if part.strip()}
        jobs = [job for job in jobs if int(job["seed"]) in keep]
        if not jobs:
            raise SystemExit(f"no jobs matched --only-seeds {sorted(keep)}")
    total = len(jobs)
    done = 0
    records: list[dict[str, Any]] = []
    print(f"probe start ours={ours} fields={len(fields)} games={total} workers={args.workers}", flush=True)
    workers = int(args.workers)
    pool = None
    with jsonl.open("w", encoding="utf-8") as fh:
        try:
            if workers == 1:
                iterator = (_probe_one(job) for job in jobs)
            else:
                pool = mp.get_context("spawn").Pool(workers)
                iterator = pool.imap_unordered(_probe_one, jobs, chunksize=1)
            for field, index, rec in iterator:
                done += 1
                records.append(rec)
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
                fh.flush()
                won = "W" if rec["we_won"] else "L"
                mono = rec["first_mono"]["policy"] if rec["first_mono"] else "-"
                color = rec["first_mono"]["color"] if rec["first_mono"] else "-"
                print(
                    f"probe {done}/{total} {field}#{index} {won} "
                    f"first={mono}/{color} rounds={rec['rounds']}",
                    flush=True,
                )
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    text = analyze(records, ours)
    report_path.write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    print(f"wrote {jsonl} {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
