#!/usr/bin/env python3
"""Cheap BC diagnosis: PPO baseline vs fixed, greedy clone vs fixed, label spot-check."""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from monopoly_bench.adapters import ActionDecision, FixedAdapter, PPOAdapter, unwrap
from monopoly_bench.arena import balanced_single_seats, play_game, summarize
from monopoly_bench.engine import ACTION_SPACE_SIZE, legal_mask
from monopoly_bench.model import MonopolyZeroNet
from monopoly_game_engine.actions import ActionType, action_to_description
from monopoly_game_engine.agents_fixed import FPAgentA, FPAgentB, FPAgentC
from oracle.resample_hybrid_labels import bucket_for_action

ROOT = Path(__file__).resolve().parents[1]
CLONE = ROOT / "artifacts_scratch/oracle_hybrid_bc_resampled_5k/snapshots/hybrid_clone_0000.pt"
PPO = ROOT / "artifacts/ppo_plus/ppo_hybrid_2000_v2.pt"
LABELS = ROOT / "artifacts_scratch/oracle_hybrid_merged/labels_hybrid_resampled.npz"
OUT = ROOT / "artifacts_scratch/diagnose_bc_step1.json"
FIXED = (FPAgentA, FPAgentB, FPAgentC)
GAMES = 8
SEED = 5_300_000
WORKERS = 8


class GreedyZeroAdapter:
    def __init__(self, model: MonopolyZeroNet):
        self.model = model
        self.model.eval()

    def choose_action(self, game, player_id: int, decision_seed: int) -> ActionDecision:
        del decision_seed
        started = time.perf_counter()
        env = unwrap(game)
        legal = tuple(env.get_allowed_actions(player_id))
        device = next(self.model.parameters()).device
        state = torch.as_tensor(
            env._get_state(player_id), dtype=torch.float32, device=device
        ).unsqueeze(0)
        mask = torch.as_tensor(legal_mask(legal), dtype=torch.bool, device=device).unsqueeze(0)
        with torch.inference_mode():
            logits, _ = self.model(state, mask)
            action = int(logits.argmax(dim=1).item())
        return ActionDecision(action, time.perf_counter() - started)


def _run_match(jobs: list[dict], worker_fn, workers: int) -> dict:
    seats = [frozenset({job["champion_seat"]}) for job in jobs]
    results: list = [None] * len(jobs)
    started = time.perf_counter()
    if workers == 1:
        for job in jobs:
            gid, result = worker_fn(job)
            results[gid] = result
            print(
                f"  progress {gid + 1}/{len(jobs)} winner={result.winner} "
                f"seat={job['champion_seat']}",
                flush=True,
            )
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            done = 0
            for gid, result in pool.imap_unordered(worker_fn, jobs, chunksize=1):
                results[gid] = result
                done += 1
                print(
                    f"  progress {done}/{len(jobs)} game={gid} winner={result.winner} "
                    f"crashes={result.crashes}",
                    flush=True,
                )
    summary = summarize(results, seats)
    wins = sum(
        1
        for result, seat in zip(results, seats)
        if result.completed and result.winner in seat
    )
    return {
        "summary": summary.as_dict(),
        "wins": wins,
        "wall_seconds": time.perf_counter() - started,
        "per_game": [
            {
                "game_id": r.game_id,
                "clone_seat": next(iter(seats[r.game_id])),
                "winner": r.winner,
                "completed": r.completed,
                "crashes": r.crashes,
                "illegal_actions": r.illegal_actions,
                "error": r.error,
            }
            for r in results
        ],
    }


def _ppo_job(job: dict) -> tuple[int, object]:
    champion = PPOAdapter(job["checkpoint"], device="cpu", stochastic=False)
    opponents = [FixedAdapter(cls) for cls in FIXED]
    seat = int(job["champion_seat"])
    policies = {seat: champion}
    opp_i = 0
    for s in range(4):
        if s == seat:
            continue
        policies[s] = opponents[opp_i]
        opp_i += 1
    result = play_game(
        game_id=int(job["game_id"]),
        seed=int(job["seed"]),
        policies=policies,
        max_rounds=200,
        record_seats=set(),
    )
    return int(job["game_id"]), result


def _greedy_job(job: dict) -> tuple[int, object]:
    model = MonopolyZeroNet.load_inference(job["checkpoint"], device="cpu")
    champion = GreedyZeroAdapter(model)
    opponents = [FixedAdapter(cls) for cls in FIXED]
    seat = int(job["champion_seat"])
    policies = {seat: champion}
    opp_i = 0
    for s in range(4):
        if s == seat:
            continue
        policies[s] = opponents[opp_i]
        opp_i += 1
    result = play_game(
        game_id=int(job["game_id"]),
        seed=int(job["seed"]),
        policies=policies,
        max_rounds=200,
        record_seats=set(),
    )
    return int(job["game_id"]), result


def _make_jobs(checkpoint: str) -> list[dict]:
    seats = balanced_single_seats(GAMES)
    return [
        {
            "game_id": index,
            "seed": SEED + index,
            "champion_seat": next(iter(target)),
            "checkpoint": checkpoint,
        }
        for index, target in enumerate(seats)
    ]


def spot_check_labels(*, n_per: int = 8, seed: int = 0) -> dict:
    with np.load(LABELS, allow_pickle=False) as payload:
        states = np.asarray(payload["states"])
        masks = np.asarray(payload["legal_masks"])
        selected = np.asarray(payload["selected_actions"])
    model = MonopolyZeroNet.load_inference(CLONE, device="cpu")
    rng = np.random.default_rng(seed)
    buckets = {
        "buy": int(ActionType.BUY_PROPERTY),
        "accept": int(ActionType.ACCEPT_TRADE),
        "decline": int(ActionType.DECLINE_TRADE),
    }
    report: dict = {"by_bucket": {}, "overall_match": {}}
    device = next(model.parameters()).device
    for name, action_id in buckets.items():
        idx = np.flatnonzero(selected == action_id)
        if len(idx) == 0:
            report["by_bucket"][name] = {"n": 0}
            continue
        take = rng.choice(idx, size=min(n_per, len(idx)), replace=False)
        rows = []
        match = 0
        for i in take:
            state = torch.as_tensor(states[i], dtype=torch.float32, device=device).unsqueeze(0)
            mask = torch.as_tensor(masks[i], dtype=torch.bool, device=device).unsqueeze(0)
            with torch.inference_mode():
                logits, values = model(state, mask)
                probs = torch.softmax(logits, dim=1)[0]
                pred = int(logits.argmax(dim=1).item())
            teacher = int(selected[i])
            ok = pred == teacher
            match += int(ok)
            top = torch.topk(probs, k=min(3, int(mask.sum())))
            rows.append(
                {
                    "teacher": action_to_description(teacher),
                    "pred": action_to_description(pred),
                    "match": ok,
                    "p_teacher": float(probs[teacher]),
                    "p_pred": float(probs[pred]),
                    "top": [
                        {
                            "action": action_to_description(int(a)),
                            "p": float(p),
                        }
                        for a, p in zip(top.indices.tolist(), top.values.tolist())
                    ],
                    "value": [float(x) for x in values[0].tolist()],
                }
            )
        report["by_bucket"][name] = {
            "n_available": int(len(idx)),
            "n_checked": len(take),
            "match_rate": match / len(take),
            "samples": rows,
        }

    # Broader match by family on 2048 random rows
    n = len(selected)
    sample = rng.choice(n, size=min(2048, n), replace=False)
    correct = 0
    by_family = Counter()
    by_family_ok = Counter()
    for i in sample:
        state = torch.as_tensor(states[i], dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.as_tensor(masks[i], dtype=torch.bool, device=device).unsqueeze(0)
        with torch.inference_mode():
            logits, _ = model(state, mask)
            pred = int(logits.argmax(dim=1).item())
        teacher = int(selected[i])
        fam = bucket_for_action(teacher)
        by_family[fam] += 1
        ok = pred == teacher
        correct += int(ok)
        if ok:
            by_family_ok[fam] += 1
    report["overall_match"] = {
        "rows": int(len(sample)),
        "match_rate": correct / len(sample),
        "by_family": {
            fam: {
                "n": by_family[fam],
                "match_rate": by_family_ok[fam] / by_family[fam],
            }
            for fam in sorted(by_family)
        },
    }
    return report


def main() -> int:
    report = {
        "clone": str(CLONE),
        "ppo": str(PPO),
        "labels": str(LABELS),
        "games": GAMES,
        "seed": SEED,
        "prior_search_clone_vs_fixed": {
            "seed_5100000": {"wins": 0, "games": 8},
            "seed_5200000": {"wins": 0, "games": 8},
            "note": "resampled clone + sims=32 search, already measured",
        },
    }

    print("=== A) PPO hybrid bootstrap vs fixed-a/b/c ===", flush=True)
    report["ppo_vs_fixed"] = _run_match(
        _make_jobs(str(PPO.resolve())), _ppo_job, WORKERS
    )
    print(
        f"PPO wins {report['ppo_vs_fixed']['wins']}/{GAMES} "
        f"WR={report['ppo_vs_fixed']['summary']['win_rate']:.3f}",
        flush=True,
    )

    print("=== B) Greedy BC clone vs fixed-a/b/c ===", flush=True)
    report["greedy_clone_vs_fixed"] = _run_match(
        _make_jobs(str(CLONE.resolve())), _greedy_job, WORKERS
    )
    print(
        f"Greedy clone wins {report['greedy_clone_vs_fixed']['wins']}/{GAMES} "
        f"WR={report['greedy_clone_vs_fixed']['summary']['win_rate']:.3f}",
        flush=True,
    )

    print("=== C) Label spot-check (clone greedy vs teacher) ===", flush=True)
    report["spot_check"] = spot_check_labels()
    print(json.dumps(report["spot_check"]["overall_match"], indent=2), flush=True)

    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
