#!/usr/bin/env python3
"""Pull oracle hybrid ckpt parts from Colab VMs into artifacts_scratch/colab_shards/.

Safe to run alongside the existing monitor (does not stop anything).
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MONITOR_PY = Path(__file__).resolve().parent / "monitor_oracle_hybrid.py"


def load_monitor():
    spec = importlib.util.spec_from_file_location("monitor_oracle_hybrid", MONITOR_PY)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def once(mod) -> None:
    for name, seed, games in mod.SHARDS:
        try:
            n = mod.sync_ckpt_parts(name, seed, games)
            mod.log(f"{name}: sync pulled {n} file(s)")
        except Exception as exc:  # noqa: BLE001
            mod.log(f"{name}: sync ERROR {type(exc).__name__}: {exc}")


def main() -> int:
    mod = load_monitor()
    loop = "--loop" in sys.argv
    mod.log("ckpt syncer start -> artifacts_scratch/colab_shards/<stem>.ckpt/")
    once(mod)
    if not loop:
        return 0
    while True:
        time.sleep(60)
        once(mod)


if __name__ == "__main__":
    raise SystemExit(main())
