"""Behavioral clone from hybrid oracle labels (soft visits + backed-up values)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from monopoly_bench.model import MonopolyZeroNet
from monopoly_bench.training import _relative_outcomes

DEFAULT_PPO = Path("artifacts/ppo_plus/ppo_hybrid_2000_v2.pt")


def load_hybrid_examples(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = (
            "states",
            "legal_masks",
            "actors",
            "selected_actions",
            "values",
            "outcomes",
            "policy_actions",
            "policy_weights",
        )
        missing = [name for name in required if name not in payload.files]
        if missing:
            raise ValueError(f"{path} missing {missing}; run oracle.merge_hybrid_labels")
        return {name: np.asarray(payload[name]).copy() for name in required}


def hybrid_expert_train_step(
    model: MonopolyZeroNet,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    batch: dict[str, np.ndarray],
    gradient_clip: float,
) -> dict[str, float]:
    """Policy CE on visit weights; value CE on actor-relative backed-up vectors."""

    model.train()
    device = next(model.parameters()).device
    states = torch.as_tensor(batch["states"], dtype=torch.float32, device=device)
    masks = torch.as_tensor(batch["legal_masks"], dtype=torch.bool, device=device)
    actors = torch.as_tensor(batch["actors"], dtype=torch.long, device=device)
    values_t = torch.as_tensor(batch["values"], dtype=torch.float32, device=device)
    policy_actions = torch.as_tensor(batch["policy_actions"], dtype=torch.long, device=device)
    policy_weights = torch.as_tensor(batch["policy_weights"], dtype=torch.float32, device=device)
    value_targets = _relative_outcomes(values_t, actors)

    optimizer.zero_grad(set_to_none=True)
    amp = device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        logits, values = model(states, masks)
        log_probs = torch.log_softmax(logits, dim=1)
        # Sparse soft CE over visit mass.
        safe_actions = policy_actions.clamp_min(0)
        gathered = log_probs.gather(1, safe_actions)
        gathered = torch.where(policy_weights > 0, gathered, torch.zeros_like(gathered))
        policy_loss = -(policy_weights * gathered).sum(dim=1).mean()
        value_loss = -(value_targets * values.clamp_min(1e-8).log()).sum(dim=1).mean()
        loss = policy_loss + value_loss
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    scaler.step(optimizer)
    scaler.update()
    return {
        "loss": float(loss.detach()),
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "gradient_norm": float(gradient_norm),
    }


def train_hybrid_clone(
    *,
    examples_path: Path,
    run_dir: Path,
    bootstrap_ppo: Path = DEFAULT_PPO,
    updates: int = 2_000,
    batch_size: int = 256,
    gradient_clip: float = 1.0,
    lr: float = 3e-4,
    seed: int = 0,
    device: str = "auto",
) -> dict:
    examples = load_hybrid_examples(examples_path)
    n = len(examples["states"])
    if device == "auto":
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "snapshots").mkdir(exist_ok=True)
    (run_dir / "reports").mkdir(exist_ok=True)

    model = MonopolyZeroNet().to(device_t)
    model.load_ppo_actor(bootstrap_ppo)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(enabled=device_t.type == "cuda")
    rng = np.random.default_rng(seed)

    model.freeze_trunk()
    head_updates = max(1, updates // 4)
    history = []
    t0 = time.time()
    last = {}
    for update in range(updates):
        if update == head_updates:
            model.unfreeze_all()
        indices = rng.choice(n, size=min(batch_size, n), replace=False)
        batch = {name: values[indices] for name, values in examples.items()}
        last = hybrid_expert_train_step(model, optimizer, scaler, batch, gradient_clip)
        if update % 50 == 0 or update + 1 == updates:
            row = {"update": update, **last, "wall_s": time.time() - t0}
            history.append(row)
            print(
                f"update={update}/{updates} loss={last['loss']:.4f} "
                f"policy={last['policy_loss']:.4f} value={last['value_loss']:.4f}",
                flush=True,
            )
            (run_dir / "reports" / "train_status.json").write_text(
                json.dumps(row, indent=2) + "\n"
            )

    model.unfreeze_all()
    ckpt = run_dir / "snapshots" / "hybrid_clone_0000.pt"
    model.save_inference(
        ckpt,
        {
            "generation": 0,
            "bootstrap": "oracle-hybrid-distill",
            "n_labels": n,
            "updates": updates,
            "examples": str(examples_path),
        },
    )
    report = {
        "checkpoint": str(ckpt),
        "n_labels": n,
        "updates": updates,
        "device": str(device_t),
        "final": last,
        "history_tail": history[-10:],
        "wall_seconds": time.time() - t0,
    }
    (run_dir / "reports" / "hybrid_bootstrap.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def policy_match(
    model: MonopolyZeroNet,
    examples: dict[str, np.ndarray],
    *,
    max_rows: int = 4096,
    seed: int = 0,
    batch_size: int = 256,
) -> dict[str, float]:
    """Fraction of rows where greedy clone action matches oracle selected_action."""

    model.eval()
    device = next(model.parameters()).device
    rng = np.random.default_rng(seed)
    n = len(examples["states"])
    take = min(max_rows, n)
    indices = rng.choice(n, size=take, replace=False)
    correct = 0
    for start in range(0, take, batch_size):
        sl = indices[start : start + batch_size]
        states = torch.as_tensor(examples["states"][sl], dtype=torch.float32, device=device)
        masks = torch.as_tensor(examples["legal_masks"][sl], dtype=torch.bool, device=device)
        target = examples["selected_actions"][sl]
        with torch.no_grad():
            logits, _ = model(states, masks)
            pred = logits.argmax(dim=1).cpu().numpy()
        correct += int((pred == target).sum())
    return {"rows": float(take), "match_rate": correct / take if take else 0.0}


def eval_clone_vs_asu(
    checkpoint: Path,
    *,
    games: int = 48,
    seed_base: int = 4_200_000,
) -> dict:
    """Seat-balanced neural clone vs ASU (3× ASU seats) via MonopolyZero ladder."""

    from monopoly_bench.adapters import ASUAdapter
    from monopoly_bench.config import BenchmarkConfig
    from monopoly_bench.ladder import evaluate_baseline

    config = BenchmarkConfig()
    summary = evaluate_baseline(
        checkpoint,
        ASUAdapter(),
        games=games,
        seed_base=seed_base,
        config=config,
    )
    return {
        "games": int(summary.games),
        "seed_base": seed_base,
        "candidate_win_rate": float(summary.win_rate),
        "completed": int(summary.completed),
        "wilson_lower": float(summary.wilson_lower),
        "latency_p95_s": float(summary.latency_p95_s),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples",
        type=Path,
        default=Path("artifacts_scratch/oracle_hybrid_merged/labels_hybrid_merged.npz"),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("monopoly_bench/runs/oracle_hybrid_bc"),
    )
    parser.add_argument("--bootstrap-ppo", type=Path, default=DEFAULT_PPO)
    parser.add_argument("--updates", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-games", type=int, default=48)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args(argv)
    report = train_hybrid_clone(
        examples_path=args.examples,
        run_dir=args.run_dir,
        bootstrap_ppo=args.bootstrap_ppo,
        updates=args.updates,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    examples = load_hybrid_examples(args.examples)
    model = MonopolyZeroNet.load_inference(report["checkpoint"])
    report["policy_match"] = policy_match(model, examples, seed=args.seed + 1)
    if not args.skip_eval:
        report["h2h_vs_asu"] = eval_clone_vs_asu(
            Path(report["checkpoint"]), games=args.eval_games
        )
    (args.run_dir / "reports" / "hybrid_bootstrap.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
