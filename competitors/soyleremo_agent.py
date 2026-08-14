"""soyleremo3 A96 frozen pure-PPO (github.com/soyleremo3/monopoly-champion-agent).

Pinned to experiment 034's promoted champion. Loads with our engine — the
ActorNetwork at submodule SHA afd92057 is unchanged here — and remaps
property-owner one-hots to actor-relative order per their 2026-08-12 fix,
without monkeypatching the global ``build_state_vector``.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path

import numpy as np
import torch

from monopoly_game_engine.actions import ACTION_SPACE_SIZE
from monopoly_game_engine.agent_ppo import PPOAgent
from monopoly_game_engine.constants import NUM_PLAYERS, PROPERTY_IDS

SOYLEREMO_ID = "soyleremo-a96"
A96_CHECKPOINT_FILENAME = "candidate_ppo_80_lr_ablation_A_lr1e-4_96.pt"
A96_CHECKPOINT_SHA256 = "78585ed4e2400d024633ee2878d8b88243f7f4a9f498d9019a73e44b3a830f51"
A96_ACTOR_SHA256 = "2bd1e9bad3d6e0100e033507f72f15b5088d8f8fb2f04dba1ae9d759b5a34a40"
CHECKPOINT_PATH_ENV_VAR = "A96_CHECKPOINT_PATH"

_ROOT = Path(__file__).resolve().parents[1]
_CANDIDATES = (
    _ROOT / "artifacts" / "soyleremo_a96" / "a96_champion.pt",
    Path.home() / "Downloads" / "a96_champion.pt",
    _ROOT / "artifacts" / "soyleremo_a96" / A96_CHECKPOINT_FILENAME,
    _ROOT / "Monoply" / "dist" / "a96_friend_match" / "a96_champion.pt",
    _ROOT / "Monopoly" / "dist" / "a96_friend_match" / "a96_champion.pt",
    _ROOT / "dist" / "a96_friend_match" / "a96_champion.pt",
    _ROOT
    / "artifacts_scratch"
    / "soyleremo3_champion"
    / "dist"
    / "a96_friend_match"
    / "a96_champion.pt",
    _ROOT
    / "artifacts_scratch"
    / "soyleremo3_champion"
    / "artifacts"
    / "monopolyzero_pure_ppo_learnability_gate"
    / A96_CHECKPOINT_FILENAME,
)

_OWNER_SECTION_START = 4 * NUM_PLAYERS
_OWNER_STRIDE = 8
_ACTOR = None
_LOCK = threading.Lock()


def resolve_checkpoint_path(checkpoint_path: Path | str | None = None) -> Path:
    if checkpoint_path is not None:
        return Path(checkpoint_path)
    env_value = os.environ.get(CHECKPOINT_PATH_ENV_VAR)
    if env_value:
        return Path(env_value)
    for candidate in _CANDIDATES:
        if candidate.is_file():
            return candidate
    return _CANDIDATES[0]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _full_actor_sha256(actor) -> str:
    digest = hashlib.sha256()
    state_dict = actor.state_dict()
    for key in sorted(state_dict):
        digest.update(key.encode("utf-8"))
        digest.update(state_dict[key].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def actor_relative_state(env, agent_id: int) -> np.ndarray:
    """Seat-relative owner one-hots; identity for ``agent_id == 0``."""
    state = np.array(env._get_state(agent_id), dtype=np.float32, copy=True)
    order = [agent_id] + [i for i in range(NUM_PLAYERS) if i != agent_id]
    for prop_index, square_id in enumerate(PROPERTY_IDS):
        owner = env.properties[square_id].owner
        start = _OWNER_SECTION_START + prop_index * _OWNER_STRIDE
        state[start : start + 5] = 0.0
        if owner is not None:
            state[start + order.index(owner)] = 1.0
    return state


def _load_actor():
    global _ACTOR
    with _LOCK:
        if _ACTOR is not None:
            return _ACTOR
        path = resolve_checkpoint_path()
        if not path.is_file():
            raise FileNotFoundError(
                f"soyleremo A96 checkpoint not found at {path}. "
                f"Set {CHECKPOINT_PATH_ENV_VAR} or place "
                f"{A96_CHECKPOINT_FILENAME} (sha256 {A96_CHECKPOINT_SHA256}) "
                f"under artifacts/soyleremo_a96/."
            )
        actual = _file_sha256(path)
        if actual != A96_CHECKPOINT_SHA256:
            raise RuntimeError(
                f"soyleremo A96 checkpoint sha256 mismatch: got {actual}, "
                f"expected {A96_CHECKPOINT_SHA256}"
            )
        agent = PPOAgent(player_id=0, hybrid=False, hidden_dim=256, device="cpu")
        agent.load(str(path))
        agent.actor.eval()
        if agent.hybrid is not False or bool(agent.fixed_action_mask.any()):
            raise RuntimeError("soyleremo A96 is not a pure hybrid=False actor")
        actor_hash = _full_actor_sha256(agent.actor)
        if actor_hash != A96_ACTOR_SHA256:
            raise RuntimeError(
                f"soyleremo A96 actor sha256 mismatch: got {actor_hash}, "
                f"expected {A96_ACTOR_SHA256}"
            )
        _ACTOR = agent.actor
        return _ACTOR


def masked_argmax_action(actor, state, legal_action_ids) -> int:
    legal = [int(a) for a in legal_action_ids]
    if not legal:
        raise ValueError("soyleremo A96 received no legal actions")
    state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
    mask_t = torch.zeros(1, ACTION_SPACE_SIZE, dtype=torch.bool)
    mask_t[0, legal] = True
    with torch.inference_mode():
        log_probs = actor(state_t, mask_t)
    chosen = int(torch.argmax(log_probs, dim=-1).item())
    if chosen not in legal:
        raise RuntimeError(f"soyleremo A96 chose illegal action {chosen}")
    return chosen


class SoyleremoA96Agent:
    def __init__(self, player_id: int):
        self.player_id = int(player_id)
        self._actor = _load_actor()

    def choose_action(self, env) -> int:
        legal = env.get_allowed_actions(self.player_id)
        state = actor_relative_state(env, self.player_id)
        return masked_argmax_action(self._actor, state, legal)
