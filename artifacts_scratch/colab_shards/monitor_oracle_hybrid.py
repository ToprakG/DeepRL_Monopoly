#!/usr/bin/env python3
"""Self-healing monitor for oracle hybrid Colab shards + local leftover.

- Probes with hard wall timeouts (never hang forever).
- If a Colab session dies or the label job dies, recreates High-RAM VM and restarts that shard.
- Downloads finished .npz shards.
- Does NOT touch a healthy local job; will restart local only if it dies before npz exists.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHARD_DIR = ROOT / "artifacts_scratch" / "colab_shards"
MONITOR_LOG = SHARD_DIR / "oracle_hybrid_monitor.log"
LAUNCH_PY = SHARD_DIR / "launch_oracle_hybrid.py"
COLAB = str(Path.home() / ".local" / "bin" / "colab")
POLL_SECONDS = 60
PROBE_TIMEOUT = 100
STALL_SECONDS = 45 * 60  # relaunch if no progress for 45 min
RELAUNCH_COOLDOWN = 180  # don't thrash

SHARDS = (
    ("ora0", 10000, 705),
    ("ora1", 10705, 705),
)
LOCAL_GAMES = 391
LOCAL_SEED = 19000
LOCAL_LOG = SHARD_DIR / "labels_seed19000_19390.log"
LOCAL_NPZ = SHARD_DIR / "labels_seed19000_19390.npz"
LOCAL_OUT = SHARD_DIR / "labels_seed19000_19390.json"

# name -> (last_done, last_change_ts, last_relaunch_ts)
_state: dict[str, tuple[int, float, float]] = {}


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with MONITOR_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(
    cmd: list[str],
    *,
    timeout: float | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=False,
        text=True,
        input=input_text,
        capture_output=True,
        timeout=timeout,
        cwd=str(ROOT),
    )


def out_stem(seed: int, games: int) -> str:
    return f"labels_seed{seed}_{seed + games - 1}"


def local_npz_path(seed: int, games: int) -> Path:
    return SHARD_DIR / f"{out_stem(seed, games)}.npz"


def shard_done(seed: int, games: int) -> bool:
    path = local_npz_path(seed, games)
    return path.exists() and path.stat().st_size > 0


def session_exists(name: str) -> bool:
    proc = run([COLAB, "status", "-s", name], timeout=60)
    text = (proc.stdout or "") + (proc.stderr or "")
    if "not found" in text.lower() or "appears to be lost" in text.lower():
        return False
    return proc.returncode == 0 and name in text


def probe_colab(name: str, seed: int, games: int) -> str:
    stem = out_stem(seed, games)
    code = f"""
import os, subprocess, re
from pathlib import Path
pid_path=Path('/content/{name}.pid')
log_path=Path('/content/{name}.run.log')
local_npz=Path('/content/{stem}.npz')
pid=pid_path.read_text().strip() if pid_path.exists() else '?'
alive=False
if pid.isdigit():
    alive=subprocess.call(['kill','-0',pid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)==0
done=0; total={games}; kind=''; labels=''
if log_path.exists():
    lines=log_path.read_text().splitlines()
    for ln in reversed(lines[-200:]):
        m=re.search(r'label progress (\\d+)/(\\d+) labels=(\\d+) kind=(\\w+)', ln)
        if m:
            done,total=int(m.group(1)),int(m.group(2))
            labels,kind=m.group(3),m.group(4)
            break
        if 'wrote ' in ln or 'throughput' in ln or 'Traceback' in ln or 'Error' in ln:
            kind=ln[:180]; break
print(f'pid={{pid}} alive={{alive}} done={{done}}/{{total}} last_labels={{labels}} kind={{kind}} npz={{local_npz.stat().st_size if local_npz.exists() else 0}}')
"""
    try:
        proc = run(
            [COLAB, "exec", "-s", name, "--timeout", "60"],
            timeout=PROBE_TIMEOUT,
            input_text=code,
        )
    except subprocess.TimeoutExpired:
        return "PROBE_TIMEOUT"
    except Exception as exc:  # noqa: BLE001
        return f"PROBE_FAIL {exc}"
    text = (proc.stdout or "").strip() or (proc.stderr or "").strip()
    if "appears to be lost" in text or "not found" in text.lower():
        return "SESSION_LOST"
    for line in reversed(text.splitlines()):
        if line.startswith("pid="):
            return line
    if proc.returncode != 0:
        return f"PROBE_FAIL rc={proc.returncode} {text[-200:]}"
    return text.splitlines()[-1] if text else "PROBE_FAIL empty"


def parse_done(status: str) -> int | None:
    m = re.search(r"done=(\d+)/(\d+)", status)
    return int(m.group(1)) if m else None


def job_alive(status: str) -> bool:
    return "alive=True" in status


def maybe_download(name: str, seed: int, games: int, status: str) -> None:
    stem = out_stem(seed, games)
    match = re.search(r"npz=(\d+)", status)
    if not match or int(match.group(1)) <= 0:
        return
    for ext in (".npz", ".json", ".jsonl"):
        local = SHARD_DIR / f"{stem}{ext}"
        if local.exists() and local.stat().st_size > 0:
            continue
        remote = f"/content/{stem}{ext}"
        run([COLAB, "download", "-s", name, remote, str(local)], timeout=600)
        if local.exists():
            log(f"{name}: downloaded {local.name} ({local.stat().st_size} bytes)")


def sync_ckpt_parts(name: str, seed: int, games: int) -> int:
    """Pull part_*.npz / manifest.json from VM ckpt dirs into local shard folder.

    Searches both /content/{stem}.ckpt and the Drive-style path used when mount fails
    as a plain directory under /content/drive/...
    """
    stem = out_stem(seed, games)
    local_dir = SHARD_DIR / f"{stem}.ckpt"
    local_dir.mkdir(parents=True, exist_ok=True)
    # List remote part files via a short exec, then download each missing/newer one.
    list_code = f"""
from pathlib import Path
cands = []
for base in [
    Path('/content/{stem}.ckpt'),
    Path('/content/drive/MyDrive/oracle_hybrid_labels/{stem}.ckpt'),
]:
    if not base.exists():
        continue
    for p in sorted(base.iterdir()):
        if p.is_file() and (p.name.startswith('part_') or p.name == 'manifest.json'):
            cands.append(f'{{p}}|{{p.stat().st_size}}')
print('CKPT_LIST ' + ';'.join(cands))
"""
    try:
        proc = run(
            [COLAB, "exec", "-s", name, "--timeout", "60"],
            timeout=PROBE_TIMEOUT,
            input_text=list_code,
        )
    except subprocess.TimeoutExpired:
        log(f"{name}: ckpt list timeout")
        return 0
    text = (proc.stdout or "") + (proc.stderr or "")
    line = ""
    for ln in text.splitlines():
        if ln.startswith("CKPT_LIST "):
            line = ln[len("CKPT_LIST ") :].strip()
    if not line or line == "":
        return 0
    pulled = 0
    for item in line.split(";"):
        if not item or "|" not in item:
            continue
        remote, size_s = item.rsplit("|", 1)
        try:
            size = int(size_s)
        except ValueError:
            continue
        fname = Path(remote).name
        local = local_dir / fname
        if local.exists() and local.stat().st_size == size and size > 0:
            continue
        run([COLAB, "download", "-s", name, remote, str(local)], timeout=600)
        if local.exists() and local.stat().st_size > 0:
            pulled += 1
            log(f"{name}: pulled ckpt {fname} ({local.stat().st_size} bytes) -> {local}")
    return pulled


def load_launcher():
    spec = importlib.util.spec_from_file_location("launch_oracle_hybrid", LAUNCH_PY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {LAUNCH_PY}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def relaunch_colab(name: str, seed: int, games: int) -> None:
    """Recreate one High-RAM session and restart its label job. Never touches local."""
    now = time.time()
    prev = _state.get(name, (0, now, 0.0))
    if now - prev[2] < RELAUNCH_COOLDOWN:
        log(f"{name}: relaunch cooldown ({int(RELAUNCH_COOLDOWN - (now - prev[2]))}s left)")
        return
    _state[name] = (prev[0], prev[1], now)
    log(f"{name}: RELAUNCHING high-mem shard seed={seed} games={games}")
    # Stop only this named session if present (dead / empty). Never touch other work.
    if session_exists(name):
        log(f"{name}: stopping dead/idle session before recreate")
        run([COLAB, "stop", "-s", name], timeout=120)
        time.sleep(2)
    launch = load_launcher()
    tarball = launch.make_tarball(SHARD_DIR / "oracle_hybrid_src.tgz")
    launch.ensure_session(name)
    launch.setup_and_start(name, seed, games, tarball, mount_drive=False)
    # Confirm
    status = probe_colab(name, seed, games)
    log(f"{name}: after relaunch → {status}")


def needs_relaunch(name: str, status: str, seed: int, games: int) -> bool:
    if shard_done(seed, games):
        return False
    if status in {"SESSION_LOST", "PROBE_TIMEOUT"} or status.startswith("PROBE_FAIL"):
        return True
    if "not found" in status.lower() or "appears to be lost" in status.lower():
        return True
    if not job_alive(status):
        # finished remotely?
        if re.search(r"npz=([1-9]\d*)", status):
            return False
        return True
    done = parse_done(status)
    now = time.time()
    if done is None:
        last_done, last_change, last_relaunch = _state.get(name, (0, now, 0.0))
        _state[name] = (last_done, last_change, last_relaunch)
        return (now - last_change) > STALL_SECONDS
    last_done, last_change, last_relaunch = _state.get(name, (done, now, 0.0))
    if done > last_done:
        _state[name] = (done, now, last_relaunch)
        return False
    _state[name] = (last_done, last_change, last_relaunch)
    if (now - last_change) > STALL_SECONDS:
        log(f"{name}: stalled at {done}/{games} for >{STALL_SECONDS // 60}m")
        return True
    return False


def probe_local() -> str:
    done = total = LOCAL_GAMES
    last = ""
    if LOCAL_LOG.exists():
        lines = LOCAL_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        for ln in reversed(lines[-200:]):
            m = re.search(r"label progress (\d+)/(\d+) labels=(\d+) kind=(\w+)", ln)
            if m:
                done, total = int(m.group(1)), int(m.group(2))
                last = f"labels={m.group(3)} kind={m.group(4)}"
                break
            if "wrote " in ln or "throughput" in ln or "Traceback" in ln:
                last = ln[:180]
                break
    alive = False
    try:
        proc = run(
            [
                "pgrep",
                "-f",
                f"oracle.label_gen --mode hybrid --calibrate --games {LOCAL_GAMES} --seed {LOCAL_SEED}",
            ],
            timeout=30,
        )
        alive = bool((proc.stdout or "").strip())
    except Exception:  # noqa: BLE001
        alive = False
    npz = LOCAL_NPZ.stat().st_size if LOCAL_NPZ.exists() else 0
    return f"alive={alive} done={done}/{total} {last} npz={npz}".strip()


def relaunch_local() -> None:
    if LOCAL_NPZ.exists() and LOCAL_NPZ.stat().st_size > 0:
        return
    now = time.time()
    prev = _state.get("local", (0, now, 0.0))
    if now - prev[2] < RELAUNCH_COOLDOWN:
        return
    _state["local"] = (prev[0], prev[1], now)
    log("local: RELAUNCHING leftover shard")
    # Append to existing log; if partial progress lost, full 391 rerun (no mid-shard checkpoint).
    cmd = (
        f"source {ROOT}/.venv/bin/activate && "
        f"PYTHONUNBUFFERED=1 PYTHONPATH={ROOT} caffeinate -dims "
        f"python -m oracle.label_gen --mode hybrid --calibrate "
        f"--games {LOCAL_GAMES} --seed {LOCAL_SEED} --workers 6 "
        f"--output {LOCAL_OUT} "
        f"2>&1 | tee -a {LOCAL_LOG}"
    )
    subprocess.Popen(
        ["zsh", "-lc", cmd],
        cwd=str(ROOT),
        start_new_session=True,
    )
    time.sleep(3)
    log(f"local: after relaunch → {probe_local()}")


def main() -> int:
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    # Append mode — don't wipe history on restart
    log("oracle hybrid SELF-HEALING monitor start")
    log(
        f"poll={POLL_SECONDS}s probe_timeout={PROBE_TIMEOUT}s "
        f"stall={STALL_SECONDS}s → {MONITOR_LOG}"
    )
    now = time.time()
    for name, _, _ in SHARDS:
        _state.setdefault(name, (0, now, 0.0))
    _state.setdefault("local", (0, now, 0.0))

    while True:
        try:
            done_count = 0
            for name, seed, games in SHARDS:
                if shard_done(seed, games):
                    log(f"{name}: DONE (local {local_npz_path(seed, games).name})")
                    done_count += 1
                    continue
                if not session_exists(name):
                    log(f"{name}: session missing → relaunch")
                    try:
                        relaunch_colab(name, seed, games)
                    except Exception as exc:  # noqa: BLE001
                        log(f"{name}: relaunch ERROR {type(exc).__name__}: {exc}")
                    continue
                status = probe_colab(name, seed, games)
                log(f"{name}: {status}")
                try:
                    n = sync_ckpt_parts(name, seed, games)
                    if n:
                        log(f"{name}: synced {n} ckpt file(s)")
                except Exception as exc:  # noqa: BLE001
                    log(f"{name}: ckpt sync ERROR {type(exc).__name__}: {exc}")
                maybe_download(name, seed, games, status)
                if shard_done(seed, games) or (
                    re.search(r"npz=([1-9]\d*)", status) and not job_alive(status)
                ):
                    maybe_download(name, seed, games, status)
                    if shard_done(seed, games):
                        done_count += 1
                        continue
                if needs_relaunch(name, status, seed, games):
                    try:
                        relaunch_colab(name, seed, games)
                    except Exception as exc:  # noqa: BLE001
                        log(f"{name}: relaunch ERROR {type(exc).__name__}: {exc}")

            local_status = probe_local()
            log(f"local: {local_status}")
            if LOCAL_NPZ.exists() and LOCAL_NPZ.stat().st_size > 0:
                done_count += 1
            elif "alive=False" in local_status:
                log("local: dead before npz → relaunch")
                relaunch_local()

            if done_count >= len(SHARDS) + 1:
                log("ALL SHARDS DONE")
                return 0
        except Exception as exc:  # noqa: BLE001
            log(f"monitor loop ERROR {type(exc).__name__}: {exc}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
