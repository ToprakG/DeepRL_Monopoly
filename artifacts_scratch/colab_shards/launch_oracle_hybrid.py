#!/usr/bin/env python3
"""Launch 3 Colab Pro high-RAM CPU shards for hybrid oracle labeling.

You: 470 games × 3 sessions on seeds 10000–11409.
Local leftover 19000–19390 is started after Colabs are up.

Each shard writes labels_seed{start}_{end}.{json,npz,jsonl} under
  /content/drive/MyDrive/oracle_hybrid_labels/  (if Drive mounts)
and always under /content/ as backup (downloaded by --probe / monitor).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHARD_DIR = ROOT / "artifacts_scratch" / "colab_shards"
COLAB = str(Path.home() / ".local" / "bin" / "colab")
DRIVE_DIR = "/content/drive/MyDrive/oracle_hybrid_labels"
PACKAGES = (
    "oracle",
    "monopoly_bench",
    "monopoly_game_engine",
    "ASU_FROZEN_TEACHER",
    "asu_plus",
)

# name, seed_start, games → seeds [start, start+games)
# 2× High-RAM Colab sessions covering 10000–11409 (1410 games).
SHARDS = (
    ("ora0", 10000, 705),
    ("ora1", 10705, 705),
)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def run(
    cmd: list[str],
    *,
    check: bool = True,
    input_text: str | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    log("$ " + " ".join(cmd))
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        input=input_text,
        cwd=str(ROOT),
        capture_output=capture,
    )


def make_tarball(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with tarfile.open(path, "w:gz") as archive:
        for name in PACKAGES:
            archive.add(ROOT / name, arcname=name)
    log(f"packed {path} ({path.stat().st_size} bytes)")
    return path


def ensure_session(name: str) -> None:
    # CPU + --high-mem = Colab Pro/Pro+ high-RAM CPU (biggest CPU shape CLI supports).
    run([COLAB, "new", "-s", name, "--high-mem"], check=True)


def setup_and_start(
    name: str,
    seed: int,
    games: int,
    tarball: Path,
    *,
    mount_drive: bool = False,
) -> None:
    end = seed + games - 1
    out_stem = f"labels_seed{seed}_{end}"
    remote_tgz = "/content/oracle_hybrid_src.tgz"
    run([COLAB, "upload", "-s", name, str(tarball), remote_tgz], check=True)

    # Upload local resume state so --resume skips finished seeds on a fresh VM.
    local_ckpt = SHARD_DIR / f"{out_stem}.ckpt"
    remote_ckpt_tgz = f"/content/{out_stem}.ckpt.tgz"
    has_resume = local_ckpt.exists() and any(local_ckpt.iterdir())
    if has_resume:
        ckpt_tgz = SHARD_DIR / f"{out_stem}.ckpt.tgz"
        if ckpt_tgz.exists():
            ckpt_tgz.unlink()
        with tarfile.open(ckpt_tgz, "w:gz") as archive:
            archive.add(local_ckpt, arcname=out_stem + ".ckpt")
        run([COLAB, "upload", "-s", name, str(ckpt_tgz), remote_ckpt_tgz], check=True)
        log(f"{name}: uploaded resume ckpt ({ckpt_tgz.stat().st_size} bytes)")
    else:
        remote_ckpt_tgz = ""
        log(f"{name}: no local ckpt to resume")

    if mount_drive:
        log(f"{name}: mounting Drive — grant in browser, then press Enter in this terminal")
        run([COLAB, "drivemount", "-s", name, "/content/drive"], check=False)
    else:
        log(f"{name}: skipping CLI drivemount (bootstrap will try google.colab.drive)")

    resume_extract = ""
    if remote_ckpt_tgz:
        resume_extract = f"""
resume_tgz = Path('{remote_ckpt_tgz}')
if resume_tgz.exists():
    with tarfile.open(resume_tgz, 'r:gz') as t:
        t.extractall('/content', filter='data')
    print(f'RESUME_EXTRACTED {{resume_tgz}}')
"""

    bootstrap = f"""
import os, shutil, subprocess, sys, tarfile, time
from pathlib import Path

root = Path('/content/DeepRL_Monopoly')
if root.exists():
    shutil.rmtree(root)
root.mkdir(parents=True)
with tarfile.open('{remote_tgz}', 'r:gz') as t:
    t.extractall(root, filter='data')

ram_gib = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1024**3
cpus = os.cpu_count() or 2
print(f'HOST cpu={{cpus}} ram_gib={{ram_gib:.1f}}')

subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'numpy'])

{resume_extract}

drive_root = Path('{DRIVE_DIR}')
drive_ok = False
try:
    from google.colab import drive as gdrive
    gdrive.mount('/content/drive', force_remount=False)
    drive_root.mkdir(parents=True, exist_ok=True)
    probe = drive_root / '.write_probe'
    probe.write_text('ok')
    probe.unlink()
    drive_ok = os.path.ismount('/content/drive') or os.path.ismount('/content/drive/MyDrive')
    if not drive_ok:
        print('DRIVE_NOT_MOUNTED falling back to /content ckpt')
except Exception as exc:
    print(f'COLAB_DRIVE_MOUNT_FAIL {{type(exc).__name__}}: {{exc}}')

content_json = Path('/content/{out_stem}.json')
content_ckpt = Path('/content/{out_stem}.ckpt')
content_ckpt.mkdir(parents=True, exist_ok=True)
# Always checkpoint under /content so CLI sync can pull parts even without Drive.
ckpt_dir = content_ckpt
if drive_ok:
    drive_ckpt = drive_root / '{out_stem}.ckpt'
    drive_ckpt.mkdir(parents=True, exist_ok=True)
    if content_ckpt.exists():
        for p in content_ckpt.iterdir():
            if p.is_file():
                shutil.copy2(p, drive_ckpt / p.name)
    ckpt_dir = drive_ckpt

workers = max(1, cpus - 1)
cmd = [
    sys.executable, '-u', '-m', 'oracle.label_gen',
    '--mode', 'hybrid', '--calibrate',
    '--games', str({games}), '--seed', str({seed}),
    '--workers', str(workers),
    '--output', str(content_json),
    '--checkpoint-every', '25',
    '--checkpoint-dir', str(ckpt_dir),
    '--resume',
]
env = os.environ.copy()
env['PYTHONPATH'] = str(root)
env['PYTHONUNBUFFERED'] = '1'
log_path = Path('/content/{name}.run.log')
pid_path = Path('/content/{name}.pid')
with log_path.open('w') as logf:
    proc = subprocess.Popen(cmd, cwd=str(root), env=env, stdout=logf, stderr=subprocess.STDOUT)
pid_path.write_text(str(proc.pid))
Path('/content/{name}.meta.txt').write_text(
    f'pid={{proc.pid}}\\nworkers={{workers}}\\nseed={seed}\\ngames={games}\\n'
    f'out={{content_json}}\\nckpt={{ckpt_dir}}\\ndrive_ok={{drive_ok}}\\n'
)
print(
    f'STARTED pid={{proc.pid}} workers={{workers}} seed={seed} games={games} '
    f'out={{content_json}} ckpt={{ckpt_dir}} drive_ok={{drive_ok}}'
)

if drive_ok and str(ckpt_dir) != str(content_ckpt):
    mirror_code = (
        "import shutil, time\\n"
        "from pathlib import Path\\n"
        "src = Path('{DRIVE_DIR}/{out_stem}.ckpt')\\n"
        "dst = Path('/content/{out_stem}.ckpt')\\n"
        "dst.mkdir(parents=True, exist_ok=True)\\n"
        "while True:\\n"
        "    if src.exists():\\n"
        "        for p in src.iterdir():\\n"
        "            if p.is_file():\\n"
        "                shutil.copy2(p, dst / p.name)\\n"
        "    time.sleep(60)\\n"
    )
    mlog = open('/content/{name}.mirror.log', 'w')
    mproc = subprocess.Popen([sys.executable, '-u', '-c', mirror_code], stdout=mlog, stderr=subprocess.STDOUT)
    print(f'MIRROR pid={{mproc.pid}}')
"""
    run([COLAB, "exec", "-s", name, "--timeout", "300"], check=True, input_text=bootstrap)



def probe(name: str, seed: int, games: int) -> str:
    end = seed + games - 1
    out_stem = f"labels_seed{seed}_{end}"
    code = f"""
import os, subprocess
from pathlib import Path
pid_path=Path('/content/{name}.pid')
log_path=Path('/content/{name}.run.log')
meta_path=Path('/content/{name}.meta.txt')
local_npz=Path('/content/{out_stem}.npz')
drive_npz=Path('{DRIVE_DIR}/{out_stem}.npz')
pid=pid_path.read_text().strip() if pid_path.exists() else '?'
alive=False
if pid.isdigit():
    alive=subprocess.call(['kill','-0',pid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)==0
tail=''
if log_path.exists():
    lines=log_path.read_text().splitlines()
    for ln in reversed(lines[-150:]):
        if 'label progress' in ln or 'throughput' in ln or 'wrote' in ln or 'Error' in ln or 'Traceback' in ln:
            tail=ln[:220]; break
    if not tail and lines:
        tail=lines[-1][:220]
meta=meta_path.read_text().replace('\\n',' | ').strip() if meta_path.exists() else ''
print(
    f'pid={{pid}} alive={{alive}} local_npz={{local_npz.stat().st_size if local_npz.exists() else 0}} '
    f'drive_npz={{drive_npz.stat().st_size if drive_npz.exists() else 0}} '
    f'meta={{meta!r}} last={{tail!r}}'
)
"""
    proc = subprocess.run(
        [COLAB, "exec", "-s", name, "--timeout", "90"],
        input=code,
        text=True,
        capture_output=True,
        check=False,
    )
    text = (proc.stdout or "").strip() or (proc.stderr or "").strip()
    for line in reversed(text.splitlines()):
        if line.startswith("pid="):
            return line
    return text[-500:] if text else f"PROBE_FAIL rc={proc.returncode}"


def maybe_download(name: str, seed: int, games: int) -> None:
    end = seed + games - 1
    out_stem = f"labels_seed{seed}_{end}"
    for ext in (".npz", ".json", ".jsonl"):
        remote = f"/content/{out_stem}{ext}"
        local = SHARD_DIR / f"{out_stem}{ext}"
        if local.exists() and local.stat().st_size > 0:
            continue
        proc = run(
            [COLAB, "download", "-s", name, remote, str(local)],
            check=False,
            capture=True,
        )
        if local.exists():
            log(f"{name}: downloaded {local.name} ({local.stat().st_size} bytes)")
        elif proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()[-200:]
            if err:
                log(f"{name}: download miss {remote}: {err}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--skip-new", action="store_true")
    parser.add_argument(
        "--mount-drive",
        action="store_true",
        help="Interactive drivemount (browser grant + Enter). Default: skip.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Stop existing ora* sessions and create fresh high-mem CPUs.",
    )
    args = parser.parse_args()
    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    if args.probe_only or args.download:
        for name, seed, games in SHARDS:
            status = probe(name, seed, games)
            log(f"{name}: {status}")
            if args.download:
                maybe_download(name, seed, games)
        return 0

    tarball = make_tarball(SHARD_DIR / "oracle_hybrid_src.tgz")

    if args.recreate:
        for name, _, _ in SHARDS:
            run([COLAB, "stop", "-s", name], check=False)
            time.sleep(1)

    if not args.skip_new or args.recreate:
        for name, _, _ in SHARDS:
            ensure_session(name)
            time.sleep(2)

    for name, seed, games in SHARDS:
        log(f"launch {name} seed={seed} games={games} end={seed + games - 1}")
        setup_and_start(
            name, seed, games, tarball, mount_drive=args.mount_drive
        )
        log(f"launched {name}")

    log("all Colab shards started; probing")
    for name, seed, games in SHARDS:
        log(f"{name}: {probe(name, seed, games)}")
    log(f"Drive folder (if mounted): {DRIVE_DIR}/labels_seed{{start}}_{{end}}.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
