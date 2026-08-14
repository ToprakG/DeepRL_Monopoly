"""
Distil EXPO into a neural policy.

The student is the shape REPO_STUDY_NOTES.md section 6 describes for this
repo: 300 observation features -> 1024 -> 512 -> 2958 action logits.

Two things make this supervised distillation rather than plain
classification:

* **Legal masking is part of the model, not a post-process.** Illegal
  logits are driven to -inf before the softmax, both in the training loss
  and at play time. A student trained without it spends probability mass on
  the ~2,900 actions that are illegal in any given state and its argmax
  regularly lands outside the legal set.
* **Agreement is not the metric that matters.** Top-1 match with the
  teacher is reported because it is cheap and diagnostic, but the honest
  measure is the student's own win rate in real games, which
  ``StudentAgent`` below exists to measure. A student can hit high
  agreement on common forced rolls and still play badly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import EXPOSURE_HEURISTIC  # noqa: F401
from monopoly_game_engine.actions import ACTION_SPACE_SIZE, ActionType
from monopoly_game_engine.state import STATE_DIM

NEG_INF = -1e9


class StudentNet(nn.Module):
    """The repo's 1024/512 ReLU trunk with a single policy head."""

    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_SPACE_SIZE):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
        )
        self.head = nn.Linear(512, action_dim)

    def forward(self, x, mask=None):
        logits = self.head(self.body(x))
        if mask is not None:
            logits = logits.masked_fill(~mask, NEG_INF)
        return logits


def _load(path: Path):
    data = np.load(path)
    obs = torch.from_numpy(data["obs"]).float()
    act = torch.from_numpy(data["act"]).long()
    mask = torch.from_numpy(
        np.unpackbits(data["mask"], axis=1)[:, :ACTION_SPACE_SIZE].astype(bool)
    )
    return obs, act, mask


def evaluate(model, obs, act, mask, device, batch=4096):
    """Top-1 agreement with the teacher, and how often argmax is legal."""
    model.eval()
    correct = legal = total = 0
    with torch.inference_mode():
        for i in range(0, len(obs), batch):
            o = obs[i:i + batch].to(device)
            m = mask[i:i + batch].to(device)
            a = act[i:i + batch].to(device)
            pred = model(o, m).argmax(dim=1)
            correct += (pred == a).sum().item()
            legal += m.gather(1, pred.unsqueeze(1)).sum().item()
            total += len(a)
    return correct / max(1, total), legal / max(1, total)


def train(data_dir: Path, epochs=8, batch=512, lr=1e-3, device="cpu"):
    train_obs, train_act, train_mask = _load(data_dir / "expo_teacher_train.npz")
    val = _load(data_dir / "expo_teacher_val.npz")
    test = _load(data_dir / "expo_teacher_test.npz")

    device = torch.device(device)
    model = StudentNet().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    best_val, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_obs))
        running = 0.0
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            o = train_obs[idx].to(device)
            a = train_act[idx].to(device)
            m = train_mask[idx].to(device)
            loss = loss_fn(model(o, m), a)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running += loss.item() * len(idx)
        agree, legality = evaluate(model, *val, device)
        history.append({
            "epoch": epoch,
            "train_loss": round(running / len(order), 4),
            "val_agreement": round(agree, 4),
            "val_legality": round(legality, 4),
        })
        print(json.dumps(history[-1]), flush=True)
        if agree > best_val:
            best_val = agree
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_agree, test_legal = evaluate(model, *test, device)
    return model, {
        "history": history,
        "test_agreement": round(test_agree, 4),
        "test_legality": round(test_legal, 4),
        "train_rows": len(train_obs),
    }


class StudentAgent:
    """Play the distilled policy: masked deterministic argmax."""

    policy_id = "expo-student-v1"

    def __init__(self, player_id: int, model=None, checkpoint=None,
                 device="cpu"):
        self.player_id = player_id
        self.device = torch.device(device)
        if model is None:
            model = StudentNet()
            payload = torch.load(checkpoint, map_location=self.device)
            model.load_state_dict(payload["state_dict"])
        self.model = model.to(self.device).eval()

    def choose_action(self, env) -> int:
        allowed = env.get_allowed_actions(self.player_id)
        if len(allowed) == 1:
            return allowed[0]
        obs = torch.from_numpy(
            env._get_state(self.player_id)
        ).float().unsqueeze(0).to(self.device)
        mask = torch.zeros(1, ACTION_SPACE_SIZE, dtype=torch.bool,
                           device=self.device)
        mask[0, allowed] = True
        with torch.inference_mode():
            action = int(self.model(obs, mask).argmax(dim=1).item())
        # Masking makes this unreachable, but fail closed rather than
        # silently emitting an illegal id.
        return action if action in allowed else int(ActionType.DO_NOTHING)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Distil EXPO into a network")
    parser.add_argument("--data", type=Path, default=Path("artifacts/expo_corpus"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts/expo_student.pt"))
    args = parser.parse_args(argv)

    model, report = train(args.data, args.epochs, args.batch, args.lr,
                          args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "state_dim": STATE_DIM,
        "action_dim": ACTION_SPACE_SIZE,
        "teacher": "expo-heuristic-v1",
        "report": report,
    }, args.out)
    print(json.dumps({k: v for k, v in report.items() if k != "history"},
                     indent=2))
    print(f"checkpoint -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
