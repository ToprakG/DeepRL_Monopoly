"""Find the gate values that equalise propose RATE between the two scorers.

Why this cannot be done on the corpus
-------------------------------------
`calibrate_gate.py` sets thresholds on the harvested corpus. That corpus is
the teacher playing the weak field, and the propose rate a gate produces there
does not carry over: against ASU the ranker's accuracy-max gate yields a 29.3%
corpus rate but only 19.8 proposals per game, against the shipped scorer's
22.9% and 39.3. The state distribution the agent actually reaches is different,
so the gate has to be calibrated on that distribution.

Method
------
Play ASU-field games with the frozen agent and, at every decision where at
least two exchanges are legal, record the max score under BOTH scorers without
acting on either. One run then gives the full score distribution for each, and
a gate for any target propose rate is a quantile of it.

This is first-order: changing a gate changes the trajectory, so the achieved
rate will not match exactly. The arms report their realised propose rate, and
that is what the decomposition is read against.

Calibrating on propose rate is not fitting to the test set — the outcome metric
is the win rate, and no win rate is consulted here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from monopoly_game_engine.actions import OFFSETS  # noqa: E402
from monopoly_game_engine.constants import (  # noqa: E402
    COLOR_GROUPS, PROPERTIES, PROPERTY_IDS,
)

from competition_agent.field_ab import FIELDS, seats_for  # noqa: E402
from competition_agent.policies import build_policy  # noqa: E402
from competition_agent.proc import ensure_hash_seed, managed_pool  # noqa: E402
from competition_agent.spec_model import (  # noqa: E402
    SHORT_TURNS, multi_turn_landings, rent_for,
)
from competition_agent.spec_policy import TRADE_W  # noqa: E402
from competition_agent.train_rank import RankHead, cand_features  # noqa: E402

N = len(PROPERTY_IDS)
OUT = Path(__file__).resolve().parent / "probes" / "gate_probe.json"


def facts(env, pid, sq, cache):
    if sq in cache:
        return cache[sq]
    prop = env.properties[sq]
    group = COLOR_GROUPS[PROPERTIES[sq]["color"]]
    saved = prop.owner
    prop.owner = pid
    try:
        rent = 0.0
        for opp in env.players:
            if opp.player_id == pid or opp.bankrupt:
                continue
            for land, p in multi_turn_landings(opp.position, SHORT_TURNS):
                if land == sq:
                    rent += p * rent_for(env, sq)
    finally:
        prop.owner = saved
    cache[sq] = {
        "price": PROPERTIES[sq]["price"], "rent": rent,
        "base_rent": PROPERTIES[sq]["rent"][0],
        "ours": sum(1 for t in group if env.properties[t].owner == pid),
        "theirs": sum(1 for t in group
                      if env.properties[t].owner not in (None, pid)),
        "size": len(group), "mort": prop.mortgaged, "houses": prop.houses,
    }
    return cache[sq]


def lin_score(fr, fo):
    return (TRADE_W["d_rent"] * (fr["rent"] - fo["rent"])
            + TRADE_W["d_price"] * ((fr["price"] - fo["price"]) / 100.0)
            + TRADE_W["completes"] * (1.0 if fr["ours"] == fr["size"] - 1
                                      else 0.0)
            + TRADE_W["d_ours"] * (fr["ours"] - fo["ours"])
            + TRADE_W["off_mort"] * (1.0 if fo["mort"] else 0.0)
            + TRADE_W["d_houses"] * (fr["houses"] - fo["houses"]))


def _play(job):
    seed, field, max_steps, ckpt = job
    import torch
    from ASU_FROZEN_TEACHER.evaluate import _new_seeded_game
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = RankHead(ck.get("hidden", 256), ck.get("dropout", 0.2))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    torch.set_num_threads(1)

    mine = seats_for(field, seed)
    game = _new_seeded_game(seed)
    env = game.env
    agents, fill = {}, list(FIELDS[field])
    for s in range(4):
        agents[s] = (build_policy("final", s, seed * 4 + s) if s in mine
                     else build_policy(fill.pop(0), s, seed * 4 + s))

    lin, net, steps = [], [], 0
    others_of = {s: [i for i in range(4) if i != s] for s in mine}
    while not env.done and steps < max_steps:
        actor = env.whose_turn()
        if actor in mine:
            legal = [int(a) for a in env.get_allowed_actions(actor)]
            ex = [a for a in legal
                  if OFFSETS["exch_trade"] <= a < OFFSETS["auction"]]
            if len(ex) >= 2:
                cache, rows, ls = {}, [], []
                for a in ex:
                    loc = a - OFFSETS["exch_trade"]
                    p_idx = loc // (N * (N - 1))
                    rem = loc % (N * (N - 1))
                    off_i = rem // (N - 1)
                    rr = rem % (N - 1)
                    req_i = rr if rr < off_i else rr + 1
                    if p_idx >= len(others_of[actor]):
                        continue
                    fo = facts(env, actor, PROPERTY_IDS[off_i], cache)
                    fr = facts(env, actor, PROPERTY_IDS[req_i], cache)
                    ls.append(lin_score(fr, fo))
                    rows.append(cand_features({
                        "req": {"price": fr["price"], "rent_if_ours": fr["rent"],
                                "ours_in_group": fr["ours"],
                                "theirs_in_group": fr["theirs"],
                                "group_size": fr["size"], "mortgaged": fr["mort"],
                                "houses": fr["houses"],
                                "base_rent": fr["base_rent"]},
                        "off": {"price": fo["price"], "rent_if_ours": fo["rent"],
                                "ours_in_group": fo["ours"],
                                "theirs_in_group": fo["theirs"],
                                "group_size": fo["size"], "mortgaged": fo["mort"],
                                "houses": fo["houses"],
                                "base_rent": fo["base_rent"]}}))
                if rows:
                    obs = np.asarray(env._get_state(actor), np.float32)
                    cf = np.asarray(rows, np.float32)
                    x = np.concatenate(
                        [np.repeat(obs[None, :], len(cf), 0), cf], 1)
                    with torch.no_grad():
                        s = model(torch.from_numpy(x)).numpy()
                    lin.append(float(max(ls)))
                    net.append(float(s.max()))
        game.step(int(agents[actor].choose_action(env)))
        steps += 1
    return lin, net


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="asu")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--seed-base", type=int, default=970000)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--ckpt", type=str, default=str(
        Path(__file__).resolve().parent / "rank_head_1000.pt"))
    args = ap.parse_args()
    ensure_hash_seed()

    jobs = [(args.seed_base + k, args.field, args.max_steps, args.ckpt)
            for k in range(args.games)]
    L, Nn = [], []
    with managed_pool(args.workers) as pool:
        for i, (a, b) in enumerate(pool.imap_unordered(_play, jobs), 1):
            L += a
            Nn += b
            if i % 5 == 0:
                print(f"  {i}/{len(jobs)} games, {len(L)} decisions",
                      flush=True)
    L, Nn = np.asarray(L), np.asarray(Nn)
    per_game = len(L) / args.games
    print(f"\n{len(L)} trade-legal decisions over {args.games} games "
          f"({per_game:.1f}/game)")

    # shipped gate 3.92 -> the rate we must reproduce with the ranker
    ship_rate = float((L >= 3.92).mean())
    print(f"shipped gate 3.92     -> propose {100*ship_rate:.1f}% "
          f"= {ship_rate*per_game:.1f}/game")
    rank_acc = -20.2549
    rank_rate = float((Nn >= rank_acc).mean())
    print(f"ranker gate {rank_acc:.2f}   -> propose {100*rank_rate:.1f}% "
          f"= {rank_rate*per_game:.1f}/game")

    g_rank_matched = float(np.quantile(Nn, 1.0 - ship_rate))
    g_lin_matched = float(np.quantile(L, 1.0 - rank_rate))
    print(f"\nranker gate matched to shipped rate : {g_rank_matched:.4f}")
    print(f"shipped gate matched to ranker rate : {g_lin_matched:.4f}")
    # The realised propose rates the two ARMS produced differ far more than
    # these matched gates do (39.3 vs 19.8 per game), which already says the
    # gap is downstream of the ranking changing the trajectory, not set by the
    # threshold. The full quantile table is saved so a gate for any target
    # rate can be read off without replaying.
    targets = [0.02, 0.04, 0.06, 0.08, 0.092, 0.10, 0.12, 0.15, 0.163, 0.19,
               0.20, 0.25, 0.30]
    print(f"\n  {'target/game':>12}{'rate':>8}{'shipped gate':>15}"
          f"{'ranker gate':>14}")
    table = {}
    for t in targets:
        gl = float(np.quantile(L, 1.0 - t))
        gn = float(np.quantile(Nn, 1.0 - t))
        table[f"{t:.3f}"] = {"lin": gl, "net": gn}
        print(f"  {t*per_game:>12.1f}{t:>8.3f}{gl:>15.4f}{gn:>14.4f}")

    OUT.write_text(json.dumps({
        "quantiles": table,
        "field": args.field, "games": args.games,
        "decisions_per_game": per_game,
        "shipped_rate": ship_rate, "ranker_rate": rank_rate,
        "rank_gate_rate_matched": g_rank_matched,
        "lin_gate_rate_matched": g_lin_matched,
    }, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
