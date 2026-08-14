# EXPO-v1 — a net-worth-gradient heuristic for `ppo-plus-v2`

EXPO is a hand-built heuristic policy for the mentor's `ppo-plus-v2`
simulator. It is **not** derived from `ASU_FROZEN_TEACHER`: it does not
import it, does not imitate its action labels, and does not reuse its value
decomposition. ASU is used only as an *opponent* in the arena.

## Layout

| File | Purpose |
|---|---|
| `agent.py` | `ExpoHeuristicAgent` — the policy |
| `board_model.py` | Stationary landing distribution for this board |
| `arena.py` | Seat-balanced, paired-seed evaluation harness |

`__init__.py` binds `monopoly_game_engine` to its directory without running
the package `__init__`, which eagerly imports torch. EXPO only touches the
rules half of the engine, so it runs on a plain numpy install.

## Running it

```bash
# vs the classic training trio
python3.13 -m EXPOSURE_HEURISTIC.arena \
  --focus expo --opponents fixed-a fixed-b fixed-c --seeds 0 1 2 3 4

# head to head with the ASU teacher
python3.13 -m EXPOSURE_HEURISTIC.arena \
  --focus expo --opponents asu-value-v1 fixed-b fixed-c --seeds 0 1 2
```
