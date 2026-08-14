"""Play the distilled student in real games and report its win rate."""
from __future__ import annotations
import argparse, json, math, os, random, time
from multiprocessing import Pool
from pathlib import Path

import EXPOSURE_HEURISTIC  # noqa: F401
from monopoly_game_engine.actions import ActionType
from monopoly_game_engine.agents_fixed import FP_AGENT_CLASSES
from monopoly_game_engine.constants import NUM_PLAYERS
from monopoly_game_engine.env import MonopolyEnv
from EXPOSURE_HEURISTIC.agent import ExpoHeuristicAgent

FIXED = {f"fixed-{c}": cls for c, cls in zip("abcdef", FP_AGENT_CLASSES)}
_MODEL = None


def _student(seat, checkpoint):
    global _MODEL
    from EXPOSURE_HEURISTIC.student import StudentAgent, StudentNet
    import torch
    if _MODEL is None:
        _MODEL = StudentNet()
        _MODEL.load_state_dict(torch.load(checkpoint, map_location="cpu")["state_dict"])
        _MODEL.eval()
        torch.set_num_threads(1)
    return StudentAgent(seat, model=_MODEL)


def _build(pid, seat, checkpoint):
    if pid == "student":
        return _student(seat, checkpoint), True
    if pid == "expo":
        return ExpoHeuristicAgent(seat), True
    if pid in ("asu-value-v1", "asu-rollout-v1"):
        from ASU_FROZEN_TEACHER import ASURolloutV1, ASUValueV1
        cls = ASUValueV1 if pid == "asu-value-v1" else ASURolloutV1
        return cls(player_id=seat), True
    return FIXED[pid](seat), False


def _block(job):
    seed, focus, opponents, checkpoint = job
    out = []
    for focus_seat in range(NUM_PLAYERS):
        seats = [None] * NUM_PLAYERS
        seats[focus_seat] = focus
        for seat, opp in zip([s for s in range(NUM_PLAYERS) if s != focus_seat], opponents):
            seats[seat] = opp
        random.seed(seed)
        env = MonopolyEnv(agent_ids=[0], max_rounds=200)
        agents, strict = zip(*[_build(p, s, checkpoint) for s, p in enumerate(seats)])
        n = 0
        while not env.done and n < 20000:
            a = env.whose_turn(); allowed = env.get_allowed_actions(a)
            act = agents[a].choose_action(env)
            if act not in allowed:
                if strict[a]:
                    raise RuntimeError(f"illegal from {seats[a]}")
                act = int(ActionType.END_TURN) if int(ActionType.END_TURN) in allowed else allowed[0]
            env.step(act); n += 1
        out.append({"seats": seats, "focus_seat": focus_seat,
                    "winner": env.winner() if env.done else None,
                    "rounds": env.round,
                    "bankrupt": [p.bankrupt for p in env.players]})
    return out


def wilson(w, n):
    if not n: return (0.0, 1.0)
    z = 1.959963985; r = w / n; d = 1 + z * z / n
    c = (r + z * z / (2 * n)) / d
    rad = z * math.sqrt(r * (1 - r) / n + z * z / (4 * n * n)) / d
    return max(0, c - rad), min(1, c + rad)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--focus", default="student")
    ap.add_argument("--checkpoint", default="artifacts/expo_student.pt")
    ap.add_argument("--opponents", nargs=3, default=["fixed-b", "fixed-d", "fixed-e"])
    ap.add_argument("--blocks", type=int, default=30)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args(argv)
    jobs = [(s, a.focus, tuple(a.opponents), a.checkpoint) for s in range(a.blocks)]
    t0 = time.time()
    with Pool(a.workers) as pool:
        games = [g for blk in pool.map(_block, jobs, chunksize=1) for g in blk]
    wins = sum(1 for g in games if g["winner"] == g["focus_seat"])
    bank = sum(1 for g in games if g["bankrupt"][g["focus_seat"]])
    lo, hi = wilson(wins, len(games))
    print(json.dumps({"focus": a.focus, "games": len(games), "wins": wins,
                      "win_rate": round(wins / len(games), 4),
                      "wilson_95": [round(lo, 4), round(hi, 4)],
                      "bankrupt_rate": round(bank / len(games), 4),
                      "elapsed_s": round(time.time() - t0, 1)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
