"""Replay the 24-game plus field and say how the losses actually happened."""

from __future__ import annotations

import json
import multiprocessing as mp
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from ASU_FROZEN_TEACHER.core import preserve_global_rng
from ASU_FROZEN_TEACHER.evaluate import DEFAULT_MAX_DECISIONS, _new_seeded_game
from monopoly_game_engine.actions import ActionType
from monopoly_game_engine.constants import COLOR_GROUPS, NUM_PLAYERS, PROPERTIES

from oracle.agent import OracleConfig
from oracle.eval_h2h import ORACLE_PLUS_ID, _H2HFactory, _Spec, _focus_seat, rotate_lineup
from oracle.plus_loop import OPENING_RACE, active_colour
from oracle.plus_steals import REAL_COLOURS, full_sets

LINEUP = (ORACLE_PLUS_ID, "slayer-v1", "underdog-v1", "inncenta-heuristic")
SEED0 = 9710000
GAMES = 24
OURS = ORACLE_PLUS_ID
OUT = Path("artifacts_scratch/plus_loss_replay.json")


def _owners(env):
    return {sq: env.properties[sq].owner for sq in env.properties}


def _owned(env, pid: int, color: str) -> int:
    return sum(env.properties[sq].owner == pid for sq in COLOR_GROUPS[color])


def _mono(env, pid: int) -> str | None:
    for color in REAL_COLOURS:
        if all(env.properties[sq].owner == pid for sq in COLOR_GROUPS[color]):
            return color
    return None


def _three(env, pid: int) -> str | None:
    for color in REAL_COLOURS:
        for sq in COLOR_GROUPS[color]:
            prop = env.properties[sq]
            if prop.owner == pid and int(getattr(prop, "houses", 0) or 0) >= 3:
                return color
    return None


def _houses_of(env, pid: int) -> int:
    return sum(int(p.houses or 0) for p in env.players[pid].properties)


def _colour_map(env, pid: int) -> dict[str, dict[str, int | bool]]:
    out: dict[str, dict[str, int | bool]] = {}
    for color in REAL_COLOURS:
        group = COLOR_GROUPS[color]
        n = sum(env.properties[sq].owner == pid for sq in group)
        houses = sum(
            int(env.properties[sq].houses or 0)
            for sq in group
            if env.properties[sq].owner == pid
        )
        mort = sum(
            1
            for sq in group
            if env.properties[sq].owner == pid and env.properties[sq].mortgaged
        )
        out[color] = {
            "n": n,
            "need": len(group),
            "houses": houses,
            "mortgaged": mort,
            "complete": n == len(group),
        }
    return out


def _opening_touch(cmap: dict) -> list[str]:
    return [c for c in ("brown", "lightblue", "pink") if cmap[c]["n"] > 0]


def _snapshot(env, pid: int) -> dict:
    cmap = _colour_map(env, pid)
    return {
        "cash": float(env.players[pid].cash),
        "nw": float(env.players[pid].net_worth()),
        "round": int(env.round),
        "sets": int(full_sets(env, pid)),
        "houses": _houses_of(env, pid),
        "colours": sum(1 for c in REAL_COLOURS if cmap[c]["n"] > 0),
        "opening": _opening_touch(cmap),
        "opening_deeds": sum(cmap[c]["n"] for c in OPENING_RACE),
        "plan": active_colour(env, pid),
        "cmap": cmap,
        "rails": sum(
            1
            for sq in COLOR_GROUPS.get("railroad", ())
            if env.properties[sq].owner == pid
        ),
    }


def _landed(env, pid: int) -> dict | None:
    pos = int(env.players[pid].position)
    prop = env.properties.get(pos)
    if prop is None:
        return {"square": pos, "name": None, "color": None, "houses": 0, "owner": None}
    owner = prop.owner
    return {
        "square": pos,
        "name": PROPERTIES[pos]["name"],
        "color": prop.color,
        "houses": int(prop.houses or 0),
        "owner": None if owner is None else int(owner),
        "mortgaged": bool(prop.mortgaged),
        "list": float(PROPERTIES[pos]["price"]),
    }


def _replay(payload: dict) -> tuple[int, dict]:
    config = OracleConfig(**payload["config"])
    factory = _H2HFactory(config, seed=payload["seed"])
    specs = tuple(_Spec(policy_id) for policy_id in payload["policies"])
    policies = [spec.policy_id for spec in specs]
    our = _focus_seat(specs)
    with preserve_global_rng():
        game = _new_seeded_game(payload["seed"])
        agents = [factory.build(spec, seat) for seat, spec in enumerate(specs)]
        env = game.env
        acquires: list[dict] = []
        spend = defaultdict(float)
        first_mono = None
        first_three = None
        our_mono = None
        our_three = None
        death = None
        last_rent = None
        at_weapon = None
        peak_colours = 0
        peak_opening = 0
        plans: list[str] = []
        decisions = 0
        while not env.done and decisions < int(payload["max_decisions"]):
            actor = env.whose_turn()
            allowed = env.get_allowed_actions(actor)
            action = agents[actor].choose_action(env)
            if action not in allowed:
                raise RuntimeError(f"illegal {action} {policies[actor]}")
            before_own = _owners(env)
            before_cash = [float(p.cash) for p in env.players]
            before_bust = bool(env.players[our].bankrupt)
            plan = active_colour(env, our)
            if plan is not None and (not plans or plans[-1] != plan):
                plans.append(plan)
            cmap_now = _colour_map(env, our)
            peak_colours = max(
                peak_colours, sum(1 for c in REAL_COLOURS if cmap_now[c]["n"] > 0)
            )
            peak_opening = max(peak_opening, len(_opening_touch(cmap_now)))
            result = game.step(action)
            info = result[3] if isinstance(result, tuple) and len(result) > 3 else {}
            decisions += 1
            after_own = _owners(env)
            for sq, owner in after_own.items():
                if owner != our or before_own.get(sq) == our:
                    continue
                data = PROPERTIES[sq]
                color = data["color"]
                spent = max(0.0, before_cash[our] - float(env.players[our].cash))
                via = "trade"
                if actor == our and action == int(ActionType.BUY_PROPERTY):
                    via = "buy"
                elif before_own.get(sq) is None:
                    via = "auction"
                spend[color] += spent
                acquires.append(
                    {
                        "round": int(env.round),
                        "square": sq,
                        "name": data["name"],
                        "color": color,
                        "via": via,
                        "spent": spent,
                        "cash": float(env.players[our].cash),
                        "plan": plan,
                        "complete": _owned(env, our, color)
                        == len(COLOR_GROUPS.get(color, ())),
                    }
                )
            if actor == our and info.get("rent_paid"):
                land = _landed(env, our)
                last_rent = {
                    "round": int(env.round),
                    "paid": float(info["rent_paid"]),
                    "debt": float(getattr(env, "debt_amount", 0) or 0),
                    "creditor": None
                    if land is None
                    else (
                        None
                        if land["owner"] is None
                        else policies[land["owner"]]
                    ),
                    "land": land,
                }
            if first_mono is None:
                for pid in range(NUM_PLAYERS):
                    color = _mono(env, pid)
                    if color is None:
                        continue
                    first_mono = {
                        "seat": pid,
                        "policy": policies[pid],
                        "color": color,
                        "round": int(env.round),
                        "cash": float(env.players[pid].cash),
                        "our": _snapshot(env, our),
                    }
                    break
            if first_three is None:
                for pid in range(NUM_PLAYERS):
                    color = _three(env, pid)
                    if color is None:
                        continue
                    first_three = {
                        "seat": pid,
                        "policy": policies[pid],
                        "color": color,
                        "round": int(env.round),
                        "cash": float(env.players[pid].cash),
                        "houses": _houses_of(env, pid),
                        "our": _snapshot(env, our),
                    }
                    if pid != our and at_weapon is None:
                        at_weapon = first_three
                    break
            if our_mono is None:
                color = _mono(env, our)
                if color is not None:
                    our_mono = {
                        "color": color,
                        "round": int(env.round),
                        "cash": float(env.players[our].cash),
                    }
            if our_three is None:
                color = _three(env, our)
                if color is not None:
                    our_three = {
                        "color": color,
                        "round": int(env.round),
                        "cash": float(env.players[our].cash),
                        "houses": _houses_of(env, our),
                    }
            if not before_bust and env.players[our].bankrupt and death is None:
                death = {
                    "round": int(env.round),
                    "rent": last_rent,
                    "debt": float(getattr(env, "debt_amount", 0) or 0),
                    "creditor": getattr(env, "debt_creditor", None),
                    "our": _snapshot(env, our),
                    "land": _landed(env, our),
                }
        winner = env.winner() if env.done else None
        nw = [float(p.net_worth()) for p in env.players]
        rec = {
            "index": payload["index"],
            "seed": payload["seed"],
            "policies": policies,
            "our_seat": our,
            "winner_policy": None if winner is None else policies[winner],
            "we_won": winner == our,
            "our_nw": nw[our],
            "our_bust": nw[our] <= 0.0 or bool(env.players[our].bankrupt),
            "rounds": int(env.round),
            "truncated": not env.done,
            "first_mono": first_mono,
            "first_three": first_three,
            "our_mono": our_mono,
            "our_three": our_three,
            "death": death,
            "acquires": acquires,
            "spend": dict(spend),
            "plans": plans,
            "peak_colours": peak_colours,
            "peak_opening": peak_opening,
            "final": _snapshot(env, our),
        }
    return payload["index"], rec


def _jobs() -> list[dict]:
    config = asdict(OracleConfig())
    jobs = []
    for index in range(GAMES):
        specs = rotate_lineup(LINEUP, index)
        jobs.append(
            {
                "index": index,
                "policies": [spec.policy_id for spec in specs],
                "seed": SEED0 + index,
                "max_decisions": DEFAULT_MAX_DECISIONS,
                "config": config,
            }
        )
    return jobs


def _pct(n: int, d: int) -> str:
    if d <= 0:
        return "n/a"
    return f"{100.0 * n / d:.0f}% ({n}/{d})"


def _fmt_cmap(cmap: dict, min_n: int = 1) -> str:
    parts = []
    for color in REAL_COLOURS:
        row = cmap[color]
        if row["n"] < min_n and not row["complete"]:
            continue
        tag = f"{color} {row['n']}/{row['need']}"
        if row["complete"]:
            tag += f" set {row['houses']}h"
        if row["mortgaged"]:
            tag += f" mort{row['mortgaged']}"
        parts.append(tag)
    return ", ".join(parts) or "(none)"


def report(records: list[dict]) -> str:
    lines = ["PLUS LOSS REPLAY  seed 9710000  n=24  plus vs slayer/underdog/inncenta", ""]
    losses = [g for g in records if not g["we_won"]]
    wins = [g for g in records if g["we_won"]]
    lines.append(
        f"wins {len(wins)}/24  bust {_pct(sum(1 for g in records if g['our_bust']), 24)}"
    )
    lines.append("")

    def _bucket(name: str, games: list[dict]) -> None:
        if not games:
            return
        lines.append(f"== {name} n={len(games)}")
        them_mono = sum(
            1
            for g in games
            if g["first_mono"] and g["first_mono"]["policy"] != OURS
        )
        us_mono = sum(
            1 for g in games if g["first_mono"] and g["first_mono"]["policy"] == OURS
        )
        them_3 = sum(
            1
            for g in games
            if g["first_three"] and g["first_three"]["policy"] != OURS
        )
        us_3 = sum(
            1
            for g in games
            if g["first_three"] and g["first_three"]["policy"] == OURS
        )
        we_set = sum(1 for g in games if g["our_mono"])
        we_3h = sum(1 for g in games if g["our_three"])
        lines.append(
            f"  first-set them {_pct(them_mono, len(games))}  us {_pct(us_mono, len(games))}  "
            f"we ever complete a set {_pct(we_set, len(games))}  we ever 3-house {_pct(we_3h, len(games))}"
        )
        lines.append(
            f"  first-3h them {_pct(them_3, len(games))}  us {_pct(us_3, len(games))}"
        )
        colors = Counter(
            g["first_mono"]["color"]
            for g in games
            if g["first_mono"] and g["first_mono"]["policy"] != OURS
        )
        if colors:
            lines.append(
                "  colour they completed first: "
                + ", ".join(f"{k}={v}" for k, v in colors.most_common())
            )
        deeds = Counter()
        opening_at = Counter()
        spread = 0
        cash_at = []
        for g in games:
            fm = g["first_mono"]
            if not fm or fm["policy"] == OURS:
                continue
            our = fm["our"]
            deeds[our["cmap"][fm["color"]]["n"]] += 1
            n_open = len(our["opening"])
            opening_at[n_open] += 1
            if n_open >= 2 and our["sets"] == 0:
                spread += 1
            cash_at.append(our["cash"])
        if deeds:
            lines.append(
                "  our deeds of their first-set: "
                + ", ".join(f"{k}={v}" for k, v in sorted(deeds.items()))
            )
        if opening_at:
            lines.append(
                "  opening colours we already touched (brown/lb/pink) at their first set: "
                + ", ".join(f"{k}={v}" for k, v in sorted(opening_at.items()))
            )
        n_them = sum(
            1 for g in games if g["first_mono"] and g["first_mono"]["policy"] != OURS
        )
        if n_them:
            lines.append(
                f"  spread thin at their first set (>=2 opening colours, 0 sets): {_pct(spread, n_them)}"
            )
        if cash_at:
            lines.append(
                f"  our cash at their first set: mean ${sum(cash_at)/len(cash_at):.0f}  "
                f"min ${min(cash_at):.0f}  max ${max(cash_at):.0f}"
            )
        peak_open = Counter(g["peak_opening"] for g in games)
        lines.append(
            "  peak opening colours touched in the whole game: "
            + ", ".join(f"{k}={v}" for k, v in sorted(peak_open.items()))
        )
        spend_open = []
        spend_else = []
        for g in games:
            so = sum(g["spend"].get(c, 0.0) for c in OPENING_RACE)
            se = sum(v for k, v in g["spend"].items() if k not in OPENING_RACE)
            spend_open.append(so)
            spend_else.append(se)
        if games:
            lines.append(
                f"  cash spent acquiring: opening-race mean ${sum(spend_open)/len(games):.0f}  "
                f"everything else mean ${sum(spend_else)/len(games):.0f}"
            )
        kills = Counter()
        kill_h = Counter()
        for g in games:
            d = g.get("death") or {}
            land = (d.get("rent") or {}).get("land") or d.get("land") or {}
            color = land.get("color")
            if color:
                kills[color] += 1
                kill_h[int(land.get("houses") or 0)] += 1
        if kills:
            lines.append(
                "  lethal colour: "
                + ", ".join(f"{k}={v}" for k, v in kills.most_common())
            )
            lines.append(
                "  lethal houses (0=bare monopoly, 3=3h, 5=hotel): "
                + ", ".join(f"{k}h={v}" for k, v in sorted(kill_h.items()))
            )
        lines.append("")

    _bucket("LOSS", losses)
    _bucket("WIN", wins)

    lines.append("== each loss")
    for g in losses:
        fm = g["first_mono"]
        ft = g["first_three"]
        d = g.get("death")
        acq = g["acquires"]
        opening_acq = [a for a in acq if a["color"] in OPENING_RACE]
        other_acq = [a for a in acq if a["color"] not in OPENING_RACE]
        who = g["winner_policy"]
        head = (
            f"g{g['index']:02d} r{g['rounds']}  {who}  "
            f"plans={'->'.join(g['plans']) or '-'}  "
            f"peak_open={g['peak_opening']} colours={g['peak_colours']}"
        )
        lines.append(head)
        if fm:
            our = fm["our"]
            lines.append(
                f"  first set r{fm['round']} {fm['policy']} {fm['color']}  "
                f"our cash ${our['cash']:.0f} sets={our['sets']}  "
                f"held { _fmt_cmap(our['cmap']) }"
            )
        if ft:
            marker = "US" if ft["policy"] == OURS else ft["policy"]
            lines.append(
                f"  first 3h r{ft['round']} {marker} {ft['color']}  "
                f"our cash ${ft['our']['cash']:.0f} houses={ft['our']['houses']} sets={ft['our']['sets']}"
            )
        if g["our_mono"]:
            om = g["our_mono"]
            ot = g["our_three"]
            extra = "" if ot is None else f"  3h r{ot['round']} {ot['color']}"
            lines.append(f"  we complete r{om['round']} {om['color']} cash ${om['cash']:.0f}{extra}")
        else:
            lines.append("  we never complete a set")
        if opening_acq:
            bits = [
                f"{a['via'][0]} {a['color']} ${a['spent']:.0f} (plan {a['plan']})"
                for a in opening_acq
            ]
            lines.append("  opening acquires: " + "; ".join(bits))
        if other_acq:
            bits = [f"{a['via'][0]} {a['color']} ${a['spent']:.0f}" for a in other_acq]
            lines.append("  other acquires: " + "; ".join(bits))
        so = sum(g["spend"].get(c, 0.0) for c in OPENING_RACE)
        se = sum(v for k, v in g["spend"].items() if k not in OPENING_RACE)
        lines.append(f"  spend opening ${so:.0f}  other ${se:.0f}")
        if d:
            land = (d.get("rent") or {}).get("land") or d.get("land") or {}
            rent = d.get("rent") or {}
            paid = rent.get("paid")
            cred = rent.get("creditor")
            h = land.get("houses")
            lines.append(
                f"  BUST r{d['round']} on {land.get('name')} ({land.get('color')} {h}h)  "
                f"rent ${paid} to {cred}  leftover { _fmt_cmap(d['our']['cmap']) }"
            )
        elif g["our_bust"]:
            lines.append("  BUST (no land snapshot)")
        else:
            lines.append(f"  survived to cap  our NW ${g['our_nw']:.0f}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    jobs = _jobs()
    records = [None] * len(jobs)
    with mp.get_context("spawn").Pool(6) as pool:
        for index, rec in pool.imap_unordered(_replay, jobs, chunksize=1):
            records[index] = rec
            print(f"replay {sum(x is not None for x in records)}/{len(jobs)}", flush=True)
    assert all(records)
    OUT.write_text(json.dumps(records, indent=2))
    text = report(records)
    Path("artifacts_scratch/plus_loss_replay.txt").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
