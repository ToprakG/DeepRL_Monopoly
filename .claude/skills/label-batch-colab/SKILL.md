---
name: label-batch-colab
description: Run a long/unattended compute job on Google Colab via the `colab-cli` tool, with automatic session-death recovery, checkpoint-based resume, GPU-shape CPU-core boosting, and safe result delivery. Use when the user wants something to run for hours on Colab without manual babysitting, when Colab sessions keep dying and losing progress, or when asked to "run this on Colab resiliently / in the background". Built from hard-won experience running a multi-hour, multi-session Colab job — Colab sessions provisioned this way die unpredictably (every 25 min to 2+ hours), so everything here exists to survive that.
---

# label-batch-colab

Pairs with `label-batch-local` (the same kind of batch-labeling workflow, run on a local
machine instead of Colab).

Colab sessions provisioned headlessly via `colab-cli` die unpredictably — anywhere from 25
minutes to a couple hours in, with no reliable warning. The failure looks like a clean
`404/401` ("session appears to be lost") on the CLI side. Root cause was never fully
pinned down (tried: token-refresh bugs, WSL DNS flakiness, Colab's abuse-detection for
headless/unattended sessions) — treat it as a fact of life, not a bug to fix. This skill
is the accumulated defense against that fact.

**Core principle: assume every session will die. Never build a plan whose only path to
success is "the session survives."** Make death cheap (checkpointing), detection fast
(polling), recovery cheap (resume), and result loss impossible (deliver early and often).

## 0. Setup

```bash
uv tool install google-colab-cli
```

The bundled `jupyter-kernel-client` dependency has a version-compat bug (`AttributeError:
module 'jupyter_kernel_client' has no attribute 'KernelClient'` when its installed version
is >=1.0). Pin it down:

```bash
uv tool install --with 'jupyter-kernel-client<1.0' google-colab-cli --force
```

Verify: `colab new -s probe && colab exec -s probe <<< "print(1)"` should just work, no
traceback. `colab stop -s probe` after.

Auth is via ADC/OAuth and is usually already cached in the environment — `colab new`
typically just works without any browser flow. `colab drivemount` / `google.colab.drive
.mount()` is separately broken in this CLI (hangs or raises `ValueError: mount failed`
even after successful OAuth) — don't rely on it. See §6 for the actual delivery path.

## 1. The job must support checkpoint + resume — this is non-negotiable

Because sessions die and take everything in `/content` with them, whatever script you run
**must** persist partial progress to disk *incrementally*, not just write output once at
the end. If the target script doesn't already do this, add it before starting, in
whatever shape fits the workload:

- Flush a small, self-contained output file every N units of work (games/rows/batches/
  epochs — pick N so a loss is annoying, not catastrophic; 10–25 units worked well for a
  workload doing tens of units/minute).
- Write/update a manifest (e.g. `manifest.json` with `completed_ids: [...]`) alongside the
  flushed parts, so a re-run can be told "skip what's already done" (e.g. a `--resume`
  flag that filters the work queue against the manifest before starting).
- Each flushed part should be independently valid/usable — don't require all parts present
  to use any of them.

Without this, every death is a 100% loss of that session's progress, no matter how fast
you notice and relaunch.

## 2. Bringing up one session

```bash
colab new -s <name>                       # or --gpu L4 / --gpu A100 — see §5
colab upload -s <name> <local-archive> /content/<archive>
echo "
import tarfile, os
with tarfile.open('/content/<archive>', 'r:gz') as a:
    a.extractall('/content', filter='data')
print(os.path.isdir('/content/<expected-dir>'))   # VERIFY, don't assume
" | timeout 30 colab exec -s <name> --timeout 20
colab install -s <name> <packages...>
```

**Verify the extraction succeeded** (`print(os.path.isdir(...))`) before moving on —
`colab exec` can silently no-op on transient connection blips and leave you launching
against a half-set-up VM. This happened repeatedly; always check.

If resuming a session that has prior checkpoint state (see §4), upload and restore the
checkpoint directory *before* launching the job.

## 3. Launching the actual job — detached, not blocking

`colab exec`'s own client-side timeout (default 30s, override with `--timeout`) is way too
short for a real job, and a long-running foreground `colab exec` call can itself hang the
CLI for minutes even when the underlying kernel is fine (observed: exec calls "timing out"
while the job kept running server-side, or the reverse — a stuck kernel). Don't run the job
as the exec payload itself. Launch it as a detached background process and let `colab exec`
return immediately:

```python
import subprocess, sys
log = open('/content/job.log', 'w')
proc = subprocess.Popen(
    [sys.executable, '-m', 'your_job', '--resume', ...],
    cwd='/content/your_repo', stdout=log, stderr=subprocess.STDOUT,
    stdin=subprocess.DEVNULL, start_new_session=True,
)
open('/content/job.pid', 'w').write(str(proc.pid))
print('launched pid', proc.pid, flush=True)
```

`start_new_session=True` + fully redirected/closed stdio matters — without it the child
can inherit descriptors tied to the kernel session and hang around oddly, or die with it.

Poll status cheaply afterward with short `colab exec` calls that just read the log tail /
check `ps -p <pid>`. This same detached-launch pattern is also what you need locally if
*you* (the orchestrating agent) need a background process to survive across your own tool
calls — e.g. a keep-alive daemon or a `cat fifo | some-interactive-cmd &` pattern. Verify
survival empirically (check the PID is still alive in a *separate*, later tool call) before
trusting it for anything long.

## 4. Monitoring loop

Run a persistent watcher (5–15 min interval; lean toward 5 min once you have real
throughput data, since it bounds how much unmirrored progress can be lost to a death) that,
per session:

1. Tars the checkpoint dir on the VM (`tarfile.open(..., 'w:gz')`) if it has content.
2. `colab download`s that tarball to a local mirror directory.
3. Extracts it, so the local mirror always has the latest known-good checkpoint state.
4. Reports the last progress line and whether the session responded at all.

```bash
while true; do
  for s in session1 session2 session3; do
    out=$(echo "<tar-and-check-python>" | timeout 25 colab exec -s "$s" --timeout 15 2>&1)
    echo "[$s] $out"
    # if tar existed: colab download, extract into local mirror
    [[ "$out" == *"not found"* ]] && echo "[$s] DEAD - relaunch from mirror"
  done
  sleep 300
done
```

Run this via a persistent background monitor so its output becomes a stream of
notifications rather than something you have to poll for.

**A session reporting "not found" is unambiguous — a session reporting a *stale, byte-
identical* progress line across two consecutive polls is not dead but may be *hung*
(deadlocked, not crashed) — treat that as equally actionable.** Check `ps -o pid,etimes,
%cpu` (and worker CPU% — see §7) before assuming "still running" means "still working."

## 5. The GPU-for-CPU-cores trick

For CPU-bound embarrassingly-parallel workloads (no GPU compute actually needed), request
a GPU-shaped VM anyway — the accelerator sits idle, but you get the much beefier host
machine that ships with it:

| Shape | vCPUs | RAM |
|---|---|---|
| Plain CPU (`colab new -s x`) | 2 | ~13 GB |
| `--gpu L4` | 12 | ~53 GB |
| `--gpu A100` | 12 | ~83 GB |

Verify empirically for the account/tier in question (`os.cpu_count()`), don't assume these
numbers hold everywhere. This was a 6x parallelism jump for a multiprocessing-Pool-based
workload, for free.

**GPU concurrency has its own, separate, tighter quota — per accelerator type, not one
shared GPU pool.** Observed: 2 concurrent L4 sessions was the ceiling (a 3rd L4 hit
`TooManyAssignmentsError` even with A100 slots free); A100 was a separate pool that
succeeded when L4 was maxed out. If one shape is exhausted, try a different one before
concluding you're out of GPU quota entirely. Plain CPU shapes are a separate pool again —
fall back there if all GPU pools are full rather than blocking.

## 6. Session death: detect, clean up *precisely*, relaunch with resume

**Never blanket-cleanup.** A generic "unassign everything currently listed" script will
also kill sessions you didn't create — including the user's own manual/interactive
sessions on the same account, which won't be locally named/tracked and are otherwise
indistinguishable from your own orphans in a bare `colab sessions` listing. This happened
and terminated a user's active GPU session. Always target cleanup by the *exact endpoint
ID* you yourself created (record it when you `colab new`), never by "not in my local
state":

```python
from colab_cli.common import state
TARGETS = {'<exact-endpoint-id-you-created>'}
sessions, assignments = state.sync_sessions()
for a in assignments:
    if a.endpoint in TARGETS:
        state.client.unassign(a.endpoint)
```

(`colab stop -s <name>` only works for sessions whose token is still in *your* local
`~/.config/colab-cli/sessions.json` — once a session dies, that entry is gone, and you
must unassign by endpoint ID directly against the account-level client as above, which
doesn't need the dead session's own token.)

Recovery sequence per dead session:
1. Identify which endpoint is actually yours (cross-reference against what you created —
   don't touch anything you can't positively identify).
2. Unassign that specific endpoint.
3. `colab new` a fresh session (same or different shape).
4. Upload + extract the repo archive, install deps (§2).
5. Upload + extract your **local mirror** of that session's checkpoint dir (§4) into the
   new VM at the same path the job expects.
6. Launch with `--resume` (§3). Confirm via the log ("resume: skipping N completed") that
   it picked up where it left off, and confirm via `ps ... %cpu` shortly after that it's
   actually computing, not stalled.

If a session is *hung* rather than dead (§4), you can often skip steps 2–4 — just kill and
relaunch the process on the *same* still-responsive VM. But see §7 first.

## 7. The zombie-worker trap

If you kill a hung job's main process with `kill -9 <pid>` and just relaunch, **you will
likely leave its multiprocessing worker children running**, orphaned, still burning CPU —
`kill -9` on a parent does not clean up its `multiprocessing.Pool` children. The new
relaunch then has to compete with a full set of zombie workers for the same cores, and can
itself appear to hang as a result (observed: new workers stuck at ~50% CPU instead of
~97%, no progress for many minutes, looked identical to a second hang).

Before relaunching on the same VM after killing a hung process, clean up thoroughly:

```python
import subprocess
subprocess.run(['pkill', '-9', '-f', 'your_job_module'])
subprocess.run(['pkill', '-9', '-f', 'multiprocessing-fork'])  # or whatever the worker cmdline pattern is
```

Verify zero matching processes remain before relaunching. When in doubt, check any
session's `ps aux` for suspiciously many `--multiprocessing-fork` processes at the wrong
CPU% (should be ~100% each when healthy, competing/low when zombies are present) —
including sessions you *didn't* just kill-and-restart, if you're unsure whether they were
ever touched this way.

## 8. Delivering results — don't route large files through your own context

If you (the agent) need to hand off accumulated output (to the user, to cloud storage,
etc.), **do not** read large/binary files into your own context to re-emit them as a tool
call parameter (e.g. base64-encoding an `.npz` into a Drive-upload tool's inline-content
field). Binary/base64 content tokenizes far worse than its byte size suggests — a 27 KB
file cost on the order of 30K+ tokens to merely read back. Doing this per-checkpoint over
a multi-hour job will exhaust context.

Instead:
- Periodically merge/consolidate accumulated checkpoint parts into fewer, larger files
  locally (pure local file operation, zero token cost).
- Hand those files to the user via a direct file-transfer mechanism that doesn't route
  bytes through your context (e.g. this environment's `SendUserFile`), not via a tool call
  that requires inline text/base64 content.
- If the destination truly must be pushed through an inline-content API (no direct-
  transfer option available), do it rarely (at completion, not per-checkpoint) and expect
  the token cost.

## 9. Local mirroring is not enough on its own

The local checkpoint mirror (§4) only protects against the *remote* (Colab) side dying —
it does nothing if the orchestrating session itself goes away (terminal closed, machine
loses power, job/session deleted). Anything living only in a local job-scoped scratch
directory is at risk from that angle. Get accumulated results into the user's actual
possession (§8) periodically, not just onto local disk, if the run is expected to span
a timeframe where "the operator's machine stays up the whole time" isn't a safe
assumption.

## 10. Quick reference — symptom → likely cause → action

| Symptom | Likely cause | Action |
|---|---|---|
| `colab exec` → `Session 'X' appears to be lost (404/401)` | Session genuinely died | §6 recovery |
| `colab new` → `TooManyAssignmentsError` | Account/accelerator-type concurrency cap hit | Try a different accelerator type (§5) or free a slot; don't blanket-cleanup to fix this |
| Identical progress line across 2+ polls, but `ps` shows the PID alive | Hung (deadlock), not dead | Check worker `%cpu` (§7); kill+cleanup+relaunch on same VM if truly stalled |
| Relaunched job stuck at low `%cpu` right after a same-VM restart | Zombie workers from the previous kill | §7 — pkill the worker pattern, verify clean, relaunch |
| `colab exec` returns empty output once, session fine on retry | Transient connection blip | Don't overreact to a single empty/odd response — verify with one cheap follow-up call before treating it as a death |
| A URL from `colab url -s X` doesn't seem to connect the browser to the right kernel | This mechanism is unreliable for browser-side reconnection | Don't rely on it for user-facing "go check your session in a browser" — prefer reporting live state you've already queried yourself |
