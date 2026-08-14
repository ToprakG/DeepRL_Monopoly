"""Process-pool management that does not orphan workers.

Why this exists
---------------
A 200-game `bench.py` run was stopped with `pkill -f "bench.py --games 200"`.
That matched only the parent: multiprocessing workers are spawned with a
command line of `python -c from multiprocessing...`, which contains neither
the script name nor its arguments. The ten workers were re-parented to init
and kept running for **3.5 hours**, saturating every core and producing
nothing, while every later probe and the teacher certification silently ran at
a fraction of the available CPU.

Two failure modes, both fixed here:

1. **Signals do not reach workers.** SIGTERM to the parent kills it outright,
   so `with mp.Pool(...)` never runs its cleanup and the children survive.
   `managed_pool` installs handlers that terminate the pool first.
2. **Pattern kills miss workers.** Even a correct `pkill` on the script name
   cannot match a worker's command line. `managed_pool` puts the parent in its
   own process group so the whole job can be killed as a unit, and
   `kill_process_group` does that.

Usage
-----
    from competition_agent.proc import managed_pool

    with managed_pool(10) as pool:
        results = pool.map(job, items)

To stop such a job from outside, prefer the group kill:

    python3 -c "from competition_agent.proc import kill_by_script; \
                kill_by_script('bench.py')"
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import signal
import subprocess
import sys


def ensure_hash_seed(value: str = "0") -> None:
    """Re-exec with a fixed PYTHONHASHSEED so games are reproducible.

    `monopoly_game_engine/agents_fixed.py:616` iterates `_TARGET_COLORS`, a
    class-level **set of strings**, inside `TheBuilder` (`fixed-d`). Python
    randomises string hashing per process, so that set's iteration order — and
    therefore TheBuilder's choice among its two target colours — differs
    between runs of the same seed. Measured on seed 960127: the same game
    yielded 744 or 748 decisions depending only on the interpreter's hash seed.

    Only `TheBuilder` is affected; a static scan of the module found no other
    iteration over a string set. That still covers the whole strong field
    (`fixed-b`, `fixed-d`, `fixed-e`), which is what every field benchmark and
    the paired survival ablation run against, so it has to be pinned rather
    than noted.

    The seed is fixed at interpreter start and cannot be changed from inside
    the process, so this re-execs once. Pool workers are forked and inherit the
    environment, so they need no separate handling.

    Call it from `main()`, not at import time: re-executing as a side effect of
    an import would break any caller that imports the module for one function,
    and `python -c` cannot be re-executed at all because the source is not in
    `sys.argv`. In that case it warns rather than failing.
    """
    if os.environ.get("PYTHONHASHSEED") == value:
        return
    if not sys.argv or not os.path.isfile(sys.argv[0]):
        print(f"warning: PYTHONHASHSEED is not {value} and cannot be pinned "
              f"from this entry point; games involving fixed-d will not be "
              f"reproducible. Re-run as a script or set it in the environment.",
              file=sys.stderr)
        return
    os.environ["PYTHONHASHSEED"] = value
    os.execv(sys.executable, [sys.executable] + sys.argv)


@contextlib.contextmanager
def managed_pool(workers: int, **kwargs):
    """A multiprocessing Pool whose workers die with the parent.

    Installs SIGTERM/SIGINT handlers that terminate the pool before exiting,
    and detaches the parent into its own process group so the job can be
    killed as a unit.
    """
    try:
        # New process group: killing -PGID reaches every worker.
        if hasattr(os, "setpgrp") and os.getpgrp() == os.getppid():
            pass  # already a group leader in some shells; harmless either way
        os.setpgid(0, 0)
    except (OSError, AttributeError):
        pass  # not fatal — the signal handlers below still apply

    pool = mp.Pool(workers, **kwargs)
    previous = {}

    def _terminate(signum, _frame):
        try:
            pool.terminate()
            pool.join()
        finally:
            # Restore and re-raise so the exit status is honest.
            handler = previous.get(signum, signal.SIG_DFL)
            signal.signal(signum, handler)
            os.kill(os.getpid(), signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, _terminate)
        except (ValueError, OSError):
            pass  # not in main thread, or unsupported

    try:
        yield pool
    finally:
        try:
            pool.terminate()
            pool.join()
        except Exception:  # noqa: BLE001
            pass
        for sig, handler in previous.items():
            with contextlib.suppress(ValueError, OSError):
                signal.signal(sig, handler)


def kill_process_group(pgid: int, sig: int = signal.SIGTERM) -> None:
    """Signal an entire process group (parent + all workers)."""
    os.killpg(pgid, sig)


def kill_by_script(script: str, sig: int = signal.SIGTERM) -> int:
    """Kill every process group whose leader runs `script`.

    Resolves script name -> pid -> process group -> group kill, which is what
    a bare `pkill -f <script>` cannot do. Returns the number of groups
    signalled.
    """
    try:
        out = subprocess.run(
            ["pgrep", "-f", script], capture_output=True, text=True, check=False
        ).stdout.split()
    except FileNotFoundError:
        return 0

    groups, me = set(), os.getpid()
    for pid_s in out:
        pid = int(pid_s)
        if pid == me:
            continue
        try:
            groups.add(os.getpgid(pid))
        except OSError:
            continue

    killed = 0
    for pgid in groups:
        if pgid in (0, os.getpgid(me)):
            continue
        with contextlib.suppress(OSError):
            os.killpg(pgid, sig)
            killed += 1
    return killed


def find_orphans(marker: str = "multiprocessing") -> list:
    """Report python workers whose parent is init — the smell of this bug.

    Returns (pid, etime, command) triples so a stale run can be spotted before
    it is blamed on something else.
    """
    out = subprocess.run(
        ["ps", "-ax", "-o", "pid=,ppid=,etime=,command="],
        capture_output=True, text=True, check=False,
    ).stdout.splitlines()
    orphans = []
    for line in out:
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid, ppid, etime, cmd = parts
        if ppid == "1" and "python" in cmd and marker in cmd:
            orphans.append((int(pid), etime, cmd[:80]))
    return orphans


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "orphans":
        found = find_orphans()
        for pid, etime, cmd in found:
            print(f"{pid:>8}  {etime:>12}  {cmd}")
        print(f"{len(found)} orphaned worker(s)")
    elif len(sys.argv) > 2 and sys.argv[1] == "kill":
        print(f"signalled {kill_by_script(sys.argv[2])} process group(s)")
    else:
        print(__doc__)
