"""Attribute a value/rollout disagreement to its source.

Two synthetic scenarios showed `ASUValueV1` and `ASURolloutV1` selecting
different actions under the modified teacher. There are two very different
explanations, and they matter for opposite reasons:

  intrinsic  — the one-step value and the truncated rollout genuinely rank
               these states differently. This is a property of the frozen
               policy design and is *evidence for* Phase 4 being worth
               building.

  introduced — the rollout's answer changed because the optimisation altered
               it, e.g. `_fast_copy_env` failing to carry a field that only
               matters in this branch. This would be a certification failure
               and would invalidate probes touching that branch.

The 2x2 (old|new) x (value|rollout) separates them:

  old_value == new_value and old_rollout == new_rollout
      -> the optimisation changed nothing; the disagreement is intrinsic.
  old_rollout != new_rollout
      -> the optimisation changed the rollout's choice: introduced.

Usage:
    python3 competition_agent/analyze_divergence_source.py OLD.json NEW.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from monopoly_game_engine.actions import action_to_description  # noqa: E402


def fmt(a):
    if isinstance(a, int):
        return f"{a} ({action_to_description(a)})"
    return str(a)


def main() -> int:
    old = json.loads(Path(sys.argv[1]).read_text())
    new = json.loads(Path(sys.argv[2]).read_text())

    names = list(new["scenarios"])
    print(f"{'scenario':<30} {'old_val':>8} {'new_val':>8} "
          f"{'old_roll':>9} {'new_roll':>9}  verdict")
    print("-" * 92)

    introduced, intrinsic = [], []
    for name in names:
        o, n = old["scenarios"].get(name, {}), new["scenarios"][name]
        ov, nv = o.get("value"), n.get("value")
        orl, nrl = o.get("rollout"), n.get("rollout")

        version_stable = (ov == nv) and (orl == nrl)
        disagrees_new = nv != nrl
        disagrees_old = ov != orl

        if not version_stable:
            verdict = "OPTIMISATION CHANGED BEHAVIOUR"
            introduced.append(name)
        elif disagrees_new:
            verdict = ("intrinsic value/rollout split"
                       if disagrees_old else "intrinsic (new only?)")
            intrinsic.append(name)
        else:
            verdict = "value == rollout"

        print(f"{name:<30} {str(ov):>8} {str(nv):>8} "
              f"{str(orl):>9} {str(nrl):>9}  {verdict}")

    print()
    for name in intrinsic:
        n = new["scenarios"][name]
        o = old["scenarios"][name]
        print(f"{name}:")
        print(f"    value   old={fmt(o.get('value'))}   new={fmt(n.get('value'))}")
        print(f"    rollout old={fmt(o.get('rollout'))}   new={fmt(n.get('rollout'))}")
        print(f"    legal actions: {n.get('value_n_legal')}")

    print()
    if introduced:
        print(f"FAIL — optimisation changed behaviour in: {introduced}")
        return 1
    print("All scenarios version-stable: every value/rollout disagreement is "
          "intrinsic to the frozen policy, not introduced by the "
          "optimisation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
