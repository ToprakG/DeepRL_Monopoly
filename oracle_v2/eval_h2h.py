"""Head-to-head using the cached oracle (same lineup protocol as v1)."""

from __future__ import annotations

import sys

from oracle.eval_h2h import main as _v1_main

from .agent import ORACLE_V2


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--deadline-s" not in args:
        args = ["--deadline-s", "0", *args]
    if "--early-stop-lead" not in args:
        args = ["--early-stop-lead", "0", *args]
    if "--lineup" not in args:
        args = [
            "--lineup",
            f"{ORACLE_V2},asu-value-v1,fixed-a,fixed-b",
            *args,
        ]
    return _v1_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
