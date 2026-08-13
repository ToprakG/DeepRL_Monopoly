"""Best-response self-play against the real tournament field.

Phase 1 measures the current oracle net (32-sim SearchAdapter) vs the actual
opponents, with a per-opponent breakdown. Phase 2 probes that every pool agent
loads through the same arena contract the training loop uses. Phase 3
warm-starts from the clone and trains against mixed 4-player tables drawn from
the real field + ASU + fixed agents + frozen past-selves. Promotion is gated
only on field win rate vs the tournament three, never vs ASU.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import pickle
import random
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from competitors.factory import COMPETITOR_IDS, EXPO_HEURISTIC_ID
from monopoly_bench.adapters import ASUAdapter, CompetitorAdapter, FixedAdapter, SearchAdapter
from monopoly_bench.arena import balanced_single_seats, play_game, wilson_lower
from monopoly_bench.config import SearchConfig
from monopoly_bench.contracts import GameResult
from monopoly_bench.engine import NUM_PLAYERS, SharedGame
from monopoly_bench.model import MonopolyZeroNet
from monopoly_bench.storage import ReplayBuffer, _atomic_json
from monopoly_bench.training import _device, _load_model, _seed_all, train_step
from monopoly_game_engine.agents_fixed import FPAgentA, FPAgentB, FPAgentC
from oracle.leaves import DEFAULT_CLONE_CHECKPOINT

LEARNER_ID = "clone"
TOURNAMENT_OPPONENTS = ("alinebidal-final", "slayer-v1", "inncenta-heuristic")
FOUR_REAL_OPPONENTS = (*TOURNAMENT_OPPONENTS, EXPO_HEURISTIC_ID)
ROBUST_IDS = ("asu-value-v1", "fixed-a", "fixed-b", "fixed-c")
FIXED_CLASSES = {
    "fixed-a": FPAgentA,
    "fixed-b": FPAgentB,
    "fixed-c": FPAgentC,
}
PAST_SELF_PREFIX = "past-self:"
STRONG_IDS = TOURNAMENT_OPPONENTS

POOL_WEIGHTS = {
    "alinebidal-final": 4,
    "slayer-v1": 4,
    "inncenta-heuristic": 5,
    EXPO_HEURISTIC_ID: 2,
    "asu-value-v1": 2,
    "fixed-a": 1,
    "fixed-b": 1,
    "fixed-c": 1,
}


def resolve_clone(path: str | Path | None = None) -> Path:
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    candidates.extend(
        (
            DEFAULT_CLONE_CHECKPOINT,
            Path("/content/hybrid_clone_0000.pt"),
            Path("hybrid_clone_0000.pt"),
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    tried = ", ".join(str(item) for item in candidates)
    raise FileNotFoundError(
        "Oracle clone checkpoint not found. Upload hybrid_clone_0000.pt or pass --clone. "
        f"Tried: {tried}"
    )


def rotate_names(base: tuple[str, ...], game_index: int) -> tuple[str, ...]:
    if len(base) != NUM_PLAYERS:
        raise ValueError(f"lineup must have {NUM_PLAYERS} policies")
    shift = game_index % NUM_PLAYERS
    return base[-shift:] + base[:-shift] if shift else base


def _atomic_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    os.replace(temporary, path)


def build_named_adapter(
    name: str,
    *,
    learner: MonopolyZeroNet | None = None,
    search: SearchConfig | None = None,
    snapshot_cache: dict[str, MonopolyZeroNet] | None = None,
    self_play: bool = False,
):
    if name == LEARNER_ID:
        if learner is None or search is None:
            raise ValueError("Learner adapter needs the live net and search config")
        return SearchAdapter(learner, search, self_play=self_play)
    if name.startswith(PAST_SELF_PREFIX):
        if search is None:
            raise ValueError("Past-self adapter needs search config")
        path = name[len(PAST_SELF_PREFIX) :]
        cache = snapshot_cache if snapshot_cache is not None else {}
        cache.setdefault(path, MonopolyZeroNet.load_inference(path))
        return SearchAdapter(cache[path], search, self_play=False)
    if name == "asu-value-v1":
        return ASUAdapter()
    if name in FIXED_CLASSES:
        return FixedAdapter(FIXED_CLASSES[name])
    if name in COMPETITOR_IDS:
        return CompetitorAdapter(name)
    raise ValueError(f"Unknown pool policy {name!r}")


def probe_pool(
    policy_ids: tuple[str, ...] = FOUR_REAL_OPPONENTS + ROBUST_IDS,
    *,
    seed: int = 0,
    max_rounds: int = 8,
) -> dict[str, Any]:
    """Confirm each pool agent loads and returns legal actions in the arena."""

    game = SharedGame.new(seed, max_rounds=1)
    actor = game.env.whose_turn()
    legal = tuple(game.env.get_allowed_actions(actor))
    reports = []
    for policy_id in policy_ids:
        try:
            adapter = build_named_adapter(policy_id)
            decision = adapter.choose_action(game, actor, seed + 17)
            action = int(getattr(decision, "action", decision))
            opening_ok = action in legal
            error = None
        except Exception as exc:
            opening_ok = False
            action = None
            error = f"{type(exc).__name__}: {exc}"
        short = None
        if opening_ok:
            fillers = ("fixed-a", "fixed-b", "fixed-c")
            policies = {actor: adapter}
            for seat, filler in zip(
                (seat for seat in range(NUM_PLAYERS) if seat != actor),
                fillers,
            ):
                policies[seat] = build_named_adapter(filler)
            result = play_game(
                game_id=0,
                seed=seed + 1,
                policies=policies,
                max_rounds=max_rounds,
                record_seats=set(),
            )
            short = {
                "completed": result.completed,
                "crashes": result.crashes,
                "illegal_actions": result.illegal_actions,
                "fallbacks": result.fallbacks,
                "decisions": result.decisions,
                "error": result.error,
            }
        reports.append(
            {
                "policy_id": policy_id,
                "opening_ok": opening_ok,
                "opening_action": action,
                "error": error,
                "short_game": short,
                "ok": opening_ok
                and short is not None
                and short["crashes"] == 0
                and short["illegal_actions"] == 0,
            }
        )
    failed = [row["policy_id"] for row in reports if not row["ok"]]
    return {
        "ok": not failed,
        "failed": failed,
        "reports": reports,
        "policy_ids": list(policy_ids),
    }


def _weighted_sample(rng: random.Random, names: list[str], k: int) -> list[str]:
    remaining = list(names)
    picked = []
    for _ in range(min(k, len(remaining))):
        weights = [POOL_WEIGHTS.get(name, 2) for name in remaining]
        total = float(sum(weights))
        draw = rng.random() * total
        cursor = 0.0
        chosen = remaining[-1]
        for name, weight in zip(remaining, weights):
            cursor += weight
            if draw <= cursor:
                chosen = name
                break
        remaining.remove(chosen)
        picked.append(chosen)
    return picked


def mixed_table_jobs(
    *,
    generation: int,
    games: int,
    seed_base: int,
    snapshots: list[str],
) -> list[dict[str, Any]]:
    """One learning seat + three pool seats. Mix tournament tables with robustness."""

    if games < 1:
        raise ValueError("games must be positive")
    tournament_n = max(1, round(games * 0.50)) if games >= 4 else games
    real_n = max(0, round(games * 0.25)) if games >= 4 else 0
    mixed_n = games - tournament_n - real_n
    if mixed_n < 0:
        tournament_n = games // 2
        real_n = (games - tournament_n) // 2
        mixed_n = games - tournament_n - real_n

    past = [f"{PAST_SELF_PREFIX}{path}" for path in snapshots]
    mixed_pool = list(FOUR_REAL_OPPONENTS) + list(ROBUST_IDS) + past
    jobs = []
    for index in range(games):
        rng = random.Random(seed_base + generation * 1_000_003 + index)
        if index < tournament_n:
            category = "tournament"
            opponents = list(TOURNAMENT_OPPONENTS)
        elif index < tournament_n + real_n:
            category = "four_real"
            opponents = _weighted_sample(rng, list(FOUR_REAL_OPPONENTS), 3)
        else:
            category = "mixed"
            opponents = _weighted_sample(rng, mixed_pool, 3)
        rng.shuffle(opponents)
        learner_seat = index % NUM_PLAYERS
        policies = [""] * NUM_PLAYERS
        policies[learner_seat] = LEARNER_ID
        opp_iter = iter(opponents)
        for seat in range(NUM_PLAYERS):
            if policies[seat]:
                continue
            policies[seat] = next(opp_iter)
        jobs.append(
            {
                "category": category,
                "game_id": generation * 100_000 + index,
                "seed": seed_base + generation * games + index,
                "learner_seat": learner_seat,
                "policies": policies,
                "index": index,
            }
        )
    return jobs


def _record_from_result(
    result: GameResult,
    *,
    policies: list[str],
    learner_seat: int,
    category: str | None = None,
) -> dict[str, Any]:
    return {
        "game_id": result.game_id,
        "seed": result.seed,
        "learner_seat": learner_seat,
        "policies": list(policies),
        "category": category,
        "winner": result.winner,
        "completed": result.completed,
        "crashes": result.crashes,
        "illegal_actions": result.illegal_actions,
        "fallbacks": result.fallbacks,
        "decisions": result.decisions,
        "final_net_worth": None if result.final_net_worth is None else list(result.final_net_worth),
        "error": result.error,
    }


def summarize_field_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-opponent win rates on mixed 4-player tables (learner vs the field)."""

    completed = [row for row in records if row.get("completed") and row.get("crashes", 0) == 0]
    learner_wins = sum(row["winner"] == row["learner_seat"] for row in completed)
    games = len(completed)
    opponents: dict[str, dict[str, Any]] = {}
    identifiers = sorted(
        {
            name
            for row in completed
            for name in row["policies"]
            if name != LEARNER_ID and not str(name).startswith(PAST_SELF_PREFIX)
        }
    )
    for identifier in identifiers:
        appearances = []
        for row in completed:
            if identifier not in row["policies"]:
                continue
            seat = row["policies"].index(identifier)
            appearances.append((row, seat))
        wins = sum(game["winner"] == seat for game, seat in appearances)
        n = len(appearances)
        rate = wins / n if n else None
        margins = []
        for game, seat in appearances:
            worth = game.get("final_net_worth")
            learner = game["learner_seat"]
            if worth is None:
                continue
            margins.append(float(worth[learner]) - float(worth[seat]))
        mean = sum(margins) / len(margins) if margins else None
        se = None
        if margins and len(margins) > 1 and mean is not None:
            variance = sum((value - mean) ** 2 for value in margins) / (len(margins) - 1)
            se = math.sqrt(variance) / math.sqrt(len(margins))
        opponents[identifier] = {
            "wins": wins,
            "games": n,
            "win_rate": rate,
            "rate_gap": None if rate is None or games == 0 else (learner_wins / games) - rate,
            "net_worth_margin_mean": mean,
            "net_worth_margin_se": se,
            "wilson_95": list(_wilson(wins, n)),
        }

    bankruptcies = 0
    for row in completed:
        worth = row.get("final_net_worth")
        seat = row["learner_seat"]
        if worth is not None and float(worth[seat]) <= 0.0:
            bankruptcies += 1

    winner_counts: dict[str, int] = {}
    for row in completed:
        if row["winner"] is None:
            continue
        name = row["policies"][int(row["winner"])]
        winner_counts[name] = winner_counts.get(name, 0) + 1

    wr = learner_wins / games if games else 0.0
    strong = {
        name: opponents[name]
        for name in STRONG_IDS
        if name in opponents
    }
    return {
        "games_scheduled": len(records),
        "completed": games,
        "crashes": sum(int(row.get("crashes") or 0) > 0 for row in records),
        "learner_wins": learner_wins,
        "learner_win_rate": wr,
        "wilson_95": list(_wilson(learner_wins, games)),
        "wilson_lower": wilson_lower(learner_wins, games) if games else 0.0,
        "bankruptcies": bankruptcies,
        "winner_counts": winner_counts,
        "opponents": opponents,
        "strong_opponents": strong,
        "records": records,
    }


def _wilson(wins: int, games: int) -> tuple[float, float]:
    if games <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    proportion = wins / games
    denominator = 1 + z * z / games
    centre = proportion + z * z / (2 * games)
    margin = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * games)) / games)
    return (centre - margin) / denominator, (centre + margin) / denominator


def print_field_table(summary: dict[str, Any], *, title: str) -> None:
    wr = summary.get("learner_win_rate")
    print(f"\n=== {title} ===", flush=True)
    print(
        (
            f"clone WR={wr:.3f} ({summary.get('learner_wins')}/{summary.get('completed')}) "
            f"Wilson {summary.get('wilson_95')} "
            f"crashes={summary.get('crashes')} bankruptcies={summary.get('bankruptcies')}"
        ),
        flush=True,
    )
    winners = summary.get("winner_counts") or {}
    if winners:
        ranked = sorted(winners.items(), key=lambda item: item[1], reverse=True)
        print("who won the table: " + ", ".join(f"{name} {wins}" for name, wins in ranked), flush=True)
    for name, row in (summary.get("opponents") or {}).items():
        rate = row.get("win_rate")
        rate_s = "n/a" if rate is None else f"{rate:.3f}"
        gap = row.get("rate_gap")
        gap_s = "n/a" if gap is None else f"{gap:+.3f}"
        mean = row.get("net_worth_margin_mean")
        se = row.get("net_worth_margin_se")
        margin_s = "n/a" if mean is None else f"{mean:.1f}"
        if se is not None:
            margin_s += f" ± SE {se:.1f}"
        marker = "  <-- strong" if name in STRONG_IDS else ""
        print(
            (
                f"  vs {name}: their WR={rate_s} ({row.get('wins')}/{row.get('games')}) "
                f"our_gap={gap_s} net-worth margin={margin_s}{marker}"
            ),
            flush=True,
        )


def _field_worker(payload: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    current = _load_model(None, payload["model_state"])
    search = SearchConfig(**payload["search_config"])
    snapshot_cache: dict[str, MonopolyZeroNet] = {}
    job = payload["job"]
    adapters: dict[int, object] = {}
    for seat, name in enumerate(job["policies"]):
        adapters[seat] = build_named_adapter(
            name,
            learner=current,
            search=search,
            snapshot_cache=snapshot_cache,
            self_play=bool(payload.get("self_play", False)) and name == LEARNER_ID,
        )
    result = play_game(
        game_id=int(job["game_id"]),
        seed=int(job["seed"]),
        policies=adapters,
        max_rounds=int(payload["max_rounds"]),
        record_seats=set(payload.get("record_seats", ())),
        outcome_kind=str(payload.get("outcome_kind", "winner")),
    )
    record = _record_from_result(
        result,
        policies=list(job["policies"]),
        learner_seat=int(job["learner_seat"]),
        category=job.get("category"),
    )
    return {"record": record, "result": result if payload.get("keep_result") else None}


def _run_jobs(
    *,
    model: MonopolyZeroNet,
    search: SearchConfig,
    jobs: list[dict[str, Any]],
    workers: int,
    max_rounds: int,
    checkpoint_dir: Path | None,
    resume: bool,
    record_seats_from_job: bool,
    outcome_kind: str,
    self_play: bool,
    keep_result: bool,
) -> tuple[list[dict[str, Any]], list[GameResult]]:
    cached_records: dict[int, dict[str, Any]] = {}
    cached_results: dict[int, GameResult] = {}
    if resume and checkpoint_dir is not None and checkpoint_dir.exists():
        for path in sorted(checkpoint_dir.glob("game_*.pkl")):
            payload = pickle.loads(path.read_bytes())
            record = payload["record"]
            cached_records[int(record["game_id"])] = record
            if keep_result and payload.get("result") is not None:
                cached_results[int(record["game_id"])] = payload["result"]
        if cached_records:
            print(f"resume: skipping {len(cached_records)} completed games", flush=True)

    pending = [job for job in jobs if int(job["game_id"]) not in cached_records]
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    payloads = [
        {
            "model_state": state,
            "search_config": asdict(search),
            "max_rounds": max_rounds,
            "record_seats": [job["learner_seat"]] if record_seats_from_job else [],
            "outcome_kind": outcome_kind,
            "self_play": self_play,
            "keep_result": keep_result,
            "job": job,
        }
        for job in pending
    ]

    records = dict(cached_records)
    results = dict(cached_results)
    done = len(records)

    def _accept(payload_out: dict[str, Any]) -> None:
        nonlocal done
        record = payload_out["record"]
        records[int(record["game_id"])] = record
        if payload_out.get("result") is not None:
            results[int(record["game_id"])] = payload_out["result"]
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            _atomic_pickle(
                checkpoint_dir / f"game_{int(record['game_id']):06d}.pkl",
                {"record": record, "result": payload_out.get("result") if keep_result else None},
            )
            public = {key: value for key, value in summarize_field_records(list(records.values())).items() if key != "records"}
            _atomic_json(checkpoint_dir / "summary.json", public)
        done += 1
        print(f"field progress {done}/{len(jobs)}", flush=True)

    if not payloads:
        ordered_records = [records[int(job["game_id"])] for job in jobs]
        ordered_results = [results[int(job["game_id"])] for job in jobs if int(job["game_id"]) in results]
        return ordered_records, ordered_results

    if workers <= 1:
        for payload in payloads:
            _accept(_field_worker(payload))
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            for payload_out in pool.imap_unordered(_field_worker, payloads, chunksize=1):
                _accept(payload_out)

    ordered_records = [records[int(job["game_id"])] for job in jobs]
    ordered_results = [results[int(job["game_id"])] for job in jobs if int(job["game_id"]) in results]
    return ordered_records, ordered_results


def evaluate_vs_field(
    model: MonopolyZeroNet,
    *,
    opponents: tuple[str, ...] = TOURNAMENT_OPPONENTS,
    games: int,
    seed: int,
    search: SearchConfig,
    workers: int,
    max_rounds: int = 200,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    title: str = "field eval",
) -> dict[str, Any]:
    if len(opponents) != NUM_PLAYERS - 1:
        raise ValueError("evaluate_vs_field expects three opponents")
    seats = balanced_single_seats(games)
    jobs = []
    for index, target in enumerate(seats):
        learner_seat = next(iter(target))
        names = [""] * NUM_PLAYERS
        names[learner_seat] = LEARNER_ID
        rotated = opponents[index % len(opponents) :] + opponents[: index % len(opponents)]
        opp_iter = iter(rotated)
        for seat in range(NUM_PLAYERS):
            if names[seat]:
                continue
            names[seat] = next(opp_iter)
        jobs.append(
            {
                "category": "eval",
                "game_id": index,
                "seed": seed + index,
                "learner_seat": learner_seat,
                "policies": names,
                "index": index,
            }
        )
    records, _ = _run_jobs(
        model=model,
        search=search,
        jobs=jobs,
        workers=workers,
        max_rounds=max_rounds,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        record_seats_from_job=False,
        outcome_kind="winner",
        self_play=False,
        keep_result=False,
    )
    summary = summarize_field_records(records)
    print_field_table(summary, title=title)
    return summary


def measure_gap(
    clone: str | Path,
    *,
    games: int = 40,
    expo_games: int = 8,
    sims: int = 32,
    seed: int = 0,
    workers: int | None = None,
    max_rounds: int = 200,
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    clone_path = resolve_clone(clone)
    model = MonopolyZeroNet.load_inference(clone_path)
    search = SearchConfig(simulations=sims)
    workers = max(1, workers or min(4, os.cpu_count() or 1))
    root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    tournament = evaluate_vs_field(
        model,
        opponents=TOURNAMENT_OPPONENTS,
        games=games,
        seed=seed,
        search=search,
        workers=workers,
        max_rounds=max_rounds,
        checkpoint_dir=None if root is None else root / "tournament",
        resume=resume,
        title=f"gap vs tournament three @ {sims} sims, n={games}",
    )
    expo = None
    if expo_games > 0:
        expo = evaluate_vs_field(
            model,
            opponents=("alinebidal-final", "slayer-v1", EXPO_HEURISTIC_ID),
            games=expo_games,
            seed=seed + 10_000,
            search=search,
            workers=workers,
            max_rounds=max_rounds,
            checkpoint_dir=None if root is None else root / "expo",
            resume=resume,
            title=f"expo smoke @ {sims} sims, n={expo_games}",
        )
    payload = {
        "clone": str(clone_path),
        "sims": sims,
        "workers": workers,
        "tournament_opponents": list(TOURNAMENT_OPPONENTS),
        "four_real_opponents": list(FOUR_REAL_OPPONENTS),
        "tournament": _public_summary(tournament),
        "expo": None if expo is None else _public_summary(expo),
    }
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        _atomic_json(root / "gap.json", payload)
    return payload


def _public_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "records"}


def promotion_beats_incumbent(incumbent_wr: float, candidate_wr: float) -> bool:
    return candidate_wr > incumbent_wr


def _append_replay(replay: ReplayBuffer, results: list[GameResult]) -> int:
    positions = [position for result in results for position in result.positions]
    if positions:
        replay.append_many(positions)
    return len(positions)


def run_train(
    clone: str | Path,
    run_dir: str | Path,
    *,
    generations: int = 3,
    games_per_generation: int = 32,
    promotion_games: int = 40,
    updates_per_generation: int = 1000,
    batch_size: int = 256,
    sims: int = 32,
    seed: int = 0,
    workers: int | None = None,
    max_rounds: int = 200,
    device: str = "auto",
    resume: bool = True,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    gradient_clip: float = 1.0,
    replay_capacity: int = 100_000,
    incumbent_field_wr: float | None = None,
) -> dict[str, Any]:
    clone_path = resolve_clone(clone)
    run_path = Path(run_dir)
    for name in ("checkpoints", "candidates", "snapshots", "reports", "eval", "collect"):
        (run_path / name).mkdir(parents=True, exist_ok=True)
    workers = max(1, workers or min(4, os.cpu_count() or 1))
    search = SearchConfig(simulations=sims)
    torch_device = _device(device)
    status_path = run_path / "status.json"
    history_path = run_path / "reports" / "field_history.json"

    replay = ReplayBuffer(
        run_path / "replay",
        replay_capacity,
        create=not (run_path / "replay" / "metadata.json").exists(),
    )
    _seed_all(seed)
    model = MonopolyZeroNet.load_inference(clone_path, device=torch_device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=torch_device.type == "cuda")

    status: dict[str, Any]
    if resume and status_path.exists():
        status = json.loads(status_path.read_text())
        ckpt = run_path / "checkpoints" / f"generation_{int(status['generation']):04d}.pt"
        if ckpt.exists():
            payload = torch.load(ckpt, map_location=torch_device, weights_only=False)
            model.load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            if payload.get("scaler") is not None and torch_device.type == "cuda":
                scaler.load_state_dict(payload["scaler"])
            print(f"resume train from generation {status['generation']}", flush=True)
    else:
        incumbent = run_path / "snapshots" / "incumbent_0000.pt"
        model.save_inference(incumbent, {"generation": 0, "source": str(clone_path)})
        status = {
            "generation": 0,
            "incumbent": str(incumbent),
            "incumbent_field_wr": None,
            "snapshots": [str(incumbent)],
            "promotions": [],
        }

    history = json.loads(history_path.read_text()) if history_path.exists() else []
    if status["incumbent_field_wr"] is None:
        if incumbent_field_wr is not None:
            status["incumbent_field_wr"] = float(incumbent_field_wr)
            status["incumbent_field"] = {
                "learner_win_rate": float(incumbent_field_wr),
                "source": "gap_json",
            }
            print(
                f"using gap tournament WR={incumbent_field_wr:.3f} as the promotion bar",
                flush=True,
            )
            _atomic_json(status_path, status)
        else:
            print("baseline field eval of the warm-start clone (promotion bar)", flush=True)
            eval_model = MonopolyZeroNet.load_inference(status["incumbent"])
            baseline = evaluate_vs_field(
                eval_model,
                games=promotion_games,
                seed=seed + 50_000,
                search=search,
                workers=workers,
                max_rounds=max_rounds,
                checkpoint_dir=run_path / "eval" / "incumbent_0000",
                resume=resume,
                title="incumbent clone vs tournament field",
            )
            status["incumbent_field_wr"] = baseline["learner_win_rate"]
            status["incumbent_field"] = _public_summary(baseline)
            history.append({"generation": 0, "kind": "incumbent", **_public_summary(baseline)})
            _atomic_json(status_path, status)
            _atomic_json(history_path, history)

    start_gen = int(status["generation"]) + 1
    end_gen = generations
    if start_gen > end_gen:
        print(f"already completed {end_gen} generations", flush=True)
        return {"status": status, "history": history, "run_dir": str(run_path)}

    for generation in range(start_gen, end_gen + 1):
        print(f"\n--- generation {generation}/{end_gen} collect ---", flush=True)
        model.eval()
        snapshots = list(status.get("snapshots") or [])
        jobs = mixed_table_jobs(
            generation=generation,
            games=games_per_generation,
            seed_base=seed + 100_000,
            snapshots=snapshots,
        )
        collect_dir = run_path / "collect" / f"gen_{generation:04d}"
        records, results = _run_jobs(
            model=model,
            search=search,
            jobs=jobs,
            workers=workers,
            max_rounds=max_rounds,
            checkpoint_dir=collect_dir,
            resume=resume,
            record_seats_from_job=True,
            outcome_kind="net_worth_margin",
            self_play=True,
            keep_result=True,
        )
        collect_summary = summarize_field_records(records)
        print_field_table(collect_summary, title=f"generation {generation} collect (train tables)")
        ingest_path = collect_dir / "ingested.json"
        if ingest_path.exists():
            print("resume: replay already ingested this generation", flush=True)
            n_pos = int(json.loads(ingest_path.read_text()).get("positions") or 0)
        else:
            n_pos = _append_replay(replay, results)
            _atomic_json(ingest_path, {"positions": n_pos, "games": len(results)})
        print(f"replay +{n_pos} positions, size={len(replay)}", flush=True)
        if len(replay) < 1:
            raise RuntimeError("No replay positions collected; cannot train")

        print(f"--- generation {generation} train_step x{updates_per_generation} ---", flush=True)
        rng = np.random.default_rng(seed + generation)
        last = {}
        model.train()
        for update in range(updates_per_generation):
            batch = replay.sample(min(batch_size, len(replay)), rng)
            last = train_step(model, optimizer, scaler, batch, gradient_clip)
            if (update + 1) % 100 == 0 or update == 0:
                print(
                    f"  update {update + 1}/{updates_per_generation} loss={last['loss']:.4f}",
                    flush=True,
                )

        candidate = run_path / "candidates" / f"generation_{generation:04d}.pt"
        model.save_inference(
            candidate,
            {"generation": generation, "loss": last, "collect": _public_summary(collect_summary)},
        )
        ckpt = run_path / "checkpoints" / f"generation_{generation:04d}.pt"
        torch.save(
            {
                "generation": generation,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if torch_device.type == "cuda" else None,
            },
            ckpt,
        )

        print(f"--- generation {generation} field gate @ {sims} sims ---", flush=True)
        eval_model = MonopolyZeroNet.load_inference(candidate)
        field = evaluate_vs_field(
            eval_model,
            games=promotion_games,
            seed=seed + 60_000 + generation * 1_000,
            search=search,
            workers=workers,
            max_rounds=max_rounds,
            checkpoint_dir=run_path / "eval" / f"generation_{generation:04d}",
            resume=resume,
            title=f"candidate gen {generation} vs tournament field",
        )
        incumbent_wr = float(status["incumbent_field_wr"])
        candidate_wr = float(field["learner_win_rate"])
        promoted = promotion_beats_incumbent(incumbent_wr, candidate_wr)
        strong = field.get("strong_opponents") or {}
        inn = (strong.get("inncenta-heuristic") or {}).get("win_rate")
        print(
            (
                f"gate: candidate {candidate_wr:.3f} vs incumbent {incumbent_wr:.3f} "
                f"-> {'PROMOTE' if promoted else 'reject'} "
                f"(Inncenta their WR={inn})"
            ),
            flush=True,
        )
        report = {
            "generation": generation,
            "promoted": promoted,
            "incumbent_field_wr": incumbent_wr,
            "candidate_field_wr": candidate_wr,
            "collect": _public_summary(collect_summary),
            "field": _public_summary(field),
            "train": last,
        }
        _atomic_json(run_path / "reports" / f"generation_{generation:04d}.json", report)
        history.append({"generation": generation, "kind": "candidate", **report})
        _atomic_json(history_path, history)

        if promoted:
            snapshot = run_path / "snapshots" / f"promoted_{generation:04d}.pt"
            shutil.copy2(candidate, snapshot)
            status["incumbent"] = str(snapshot)
            status["incumbent_field_wr"] = candidate_wr
            status["incumbent_field"] = _public_summary(field)
            status.setdefault("snapshots", []).append(str(snapshot))
            status.setdefault("promotions", []).append(generation)
        status["generation"] = generation
        _atomic_json(status_path, status)

    return {"status": status, "history": history, "run_dir": str(run_path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Field gap + best-response vs the real opponents")
    sub = parser.add_subparsers(dest="cmd", required=True)

    probe = sub.add_parser("probe", help="Load every pool agent through the training harness")
    probe.add_argument("--output", type=Path)
    probe.add_argument("--max-rounds", type=int, default=8)

    gap = sub.add_parser("gap", help="Oracle net vs tournament field at 32 sims (~40 games)")
    gap.add_argument("--clone", type=Path)
    gap.add_argument("--games", type=int, default=40)
    gap.add_argument("--expo-games", type=int, default=8)
    gap.add_argument("--sims", type=int, default=32)
    gap.add_argument("--seed", type=int, default=0)
    gap.add_argument("--workers", type=int, default=0)
    gap.add_argument("--max-rounds", type=int, default=200)
    gap.add_argument("--checkpoint-dir", type=Path)
    gap.add_argument("--resume", action="store_true")
    gap.add_argument("--output", type=Path)

    train = sub.add_parser("train", help="Best-response self-play vs the real field")
    train.add_argument("--clone", type=Path)
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--generations", type=int, default=3)
    train.add_argument("--games-per-generation", type=int, default=32)
    train.add_argument("--promotion-games", type=int, default=40)
    train.add_argument("--updates", type=int, default=1000)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--sims", type=int, default=32)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--max-rounds", type=int, default=200)
    train.add_argument("--device", default="auto")
    train.add_argument("--no-resume", action="store_true")
    train.add_argument(
        "--gap-json",
        type=Path,
        help="Reuse tournament WR from a prior gap run as the promotion bar.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.cmd == "probe":
        report = probe_pool(max_rounds=args.max_rounds)
        text = json.dumps(report, indent=2, sort_keys=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text, flush=True)
        if not report["ok"]:
            print(f"PROBE FAILED: {report['failed']}", flush=True)
            return 2
        print("probe ok — every pool agent loaded and played", flush=True)
        return 0

    if args.cmd == "gap":
        workers = None if args.workers <= 0 else args.workers
        payload = measure_gap(
            args.clone,
            games=args.games,
            expo_games=args.expo_games,
            sims=args.sims,
            seed=args.seed,
            workers=workers,
            max_rounds=args.max_rounds,
            checkpoint_dir=args.checkpoint_dir,
            resume=args.resume,
        )
        text = json.dumps(payload, indent=2, sort_keys=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text, flush=True)
        return 0

    workers = None if args.workers <= 0 else args.workers
    incumbent_wr = None
    if args.gap_json is not None:
        gap_payload = json.loads(args.gap_json.read_text())
        incumbent_wr = float(gap_payload["tournament"]["learner_win_rate"])
    payload = run_train(
        args.clone,
        args.run_dir,
        generations=args.generations,
        games_per_generation=args.games_per_generation,
        promotion_games=args.promotion_games,
        updates_per_generation=args.updates,
        batch_size=args.batch_size,
        sims=args.sims,
        seed=args.seed,
        workers=workers,
        max_rounds=args.max_rounds,
        device=args.device,
        resume=not args.no_resume,
        incumbent_field_wr=incumbent_wr,
    )
    print(json.dumps({"status": payload["status"]}, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
