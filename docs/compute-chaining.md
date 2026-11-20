# Editable sequential Slurm queue

## Safe output-driven continuation

Automatic continuation is disabled by default. In a Slurm QE task's history,
open **Geometry optimization and run statistics → Automatic coordinate
continuation**, review the task's input/resources, set a maximum number of further
jobs (1–10), and authorize. Enable/resume the sequential chain. This authorization
applies when that task is the active predecessor; it does not scan historical jobs
or independent-mode jobs. For a historical/downloaded run, prepare a new input as
described below, preview/queue it normally, then authorize that new task if desired.

The dispatcher classifies output as converged, incomplete, failed or uncertain.
Only a scheduler-confirmed timeout or a completed allocation with exit code zero
and `JOB DONE` can continue automatically, and only for incomplete fixed-cell
`relax` output with completed SCF samples and valid complete coordinates.
QE errors, failed SCF cycles, a truncated latest coordinate card, cancellation,
missing output, and other scheduler failures require review. A total-force value
alone never proves convergence. Converged calculations create no successor.

Every successor uses a new task directory and an empty `Outputs` directory.
Only coordinates and original constraints are reused; no `.save`, density,
wavefunction or BFGS checkpoint files are copied. Input explicitly sets
`restart_mode='from_scratch'`, `startingpot='atomic'` and
`startingwfc='atomic+random'`, overriding explicit file-based electronic settings.
This intentionally gives up electronic/BFGS restart acceleration. QE documents
that electronic restart requires a clean stop and matching processor layout;
file presence alone does not establish integrity:
[QE restart_mode reference](https://www.quantum-espresso.org/Doc/INPUT_PW.html#restart_mode).

Successors preserve process count, runtime, memory, timeout, partition/account/QOS
and calculation settings. They enter the end of the existing editable queue.
Their parent ID and decremented allowance are committed with admission. Recovery
searches persisted QE input provenance, including manually queued continuations,
before creating another successor. An accepted or canceled successor consumes
that parent transition; recovery never creates a replacement automatically.

Changes to the admitted source input, pseudopotentials or image require a new
reviewed task. Changes to Python pipeline code require review and explicit
reauthorization. A blocked continuation records its reason on the task and chain;
unrelated new jobs can still be queued. Set the parent's allowance to zero and
Recover to skip its continuation and advance waiting work. Disabling a parent's
policy does not cancel an already accepted child or change its inherited policy;
pause the chain and edit/cancel that child explicitly.

To inspect downloaded output without a dashboard or Slurm connection:

```bash
.venv/bin/python -m dashboard.app.continuation /path/to/downloaded-task
.venv/bin/python -m dashboard.app.continuation /path/to/downloaded-task --output /tmp/continued.in
```

The second command refuses to overwrite a file and only prepares output classified
as incomplete with valid coordinates. It submits nothing. Place the input in the
configured input-version directory for its candidate/system (for example
`qe_inputs/500/complex.continued.in`), then preview and queue it through the usual
run controls. Do not run it in the downloaded checkpoint directory.

The API equivalent of task authorization is
`POST /api/tasks/{task_id}/continuation` with `{"limit": 2}`; use zero to disable.
The allowance and review reason are included in queue/history task records.

## Optional periodic recovery

Run the same recovery operation without starting the web server:

```bash
/path/to/dashboard/python -m dashboard.app.recover --config /path/to/saved-dispatcher/config.json
```

Use an existing dispatcher's immutable `config.json` from its recorded scheduler
directory (`.dashboard/slurm-<token>/config.json`), retained for the monitor's
lifetime. It identifies the shared artifact database and runtime paths. Recovery
reconstructs candidate identities from saved task records and preserves local
task state. It respects pause/disable settings and uncertain-submission blocks,
and may submit the next compute job or a replacement dispatcher when appropriate.
It records the attempt in the chain audit; blocked recovery exits nonzero.

[`scripts/hpc/chain-monitor.scrontab`](../scripts/hpc/chain-monitor.scrontab)
provides a three-hour template. Replace its paths and verify cluster resource and
account settings, manually test the recovery command, then add its entry using
`scrontab -e` while preserving existing entries. Nothing installs it automatically.
The cluster must enable Slurm cron and allow these allocations to query accounting
and submit jobs. Scheduled checks remain subject to queue delays, so three hours
is an intended interval rather than a guaranteed deadline. See the
[official scrontab reference](https://slurm.schedmd.com/scrontab.html).

The dashboard can advance its existing Slurm task queue through short Slurm
`afterany` dispatcher allocations. Enable this mode explicitly in **Job queue →
Sequential Slurm chain**. Independent Slurm submission remains the default for
existing installations. Local preparation and diagram tasks keep their existing
execution behavior.

## Create, edit and start

1. Enable a **paused** chain before adding work. Enabling chaining requires that
   existing independent Slurm submissions have finished. This prevents overlap.
2. In the candidate or batch controls, choose Slurm, preview the inputs and resource
   requirements, and queue the computations. Each accepted task retains its own
   CPU, memory, walltime, partition, account and QOS settings.
3. In the waiting list, move jobs up/down or move a job to a numbered position.
   Newly added work goes at the end; move it to insert it elsewhere. Remove a
   waiting task to cancel it while preserving history. To change an input or
   resource request, remove it and submit a newly reviewed task.
4. Resume the chain to submit the first computation and its dispatcher. Only that
   computation is submitted; the remaining tasks stay editable. After it ends,
   the dispatcher reads the current waiting order, submits the next task and its
   successor dispatcher, then exits.

Edits are shared by browser sessions using the same artifact directory. Reordering
sends the displayed order as an optimistic concurrency check. If another session
adds, removes or selects work, refresh and try again. A stale removal is refused
if its task has already been submitted. A submitted task can be stopped using
its explicit **Stop job** control, but cannot be reordered or edited.

The chain is scoped to the dashboard artifact directory, not a browser session.
All browser users act as the dashboard server's Unix account; recorded owner names
are informational. Keep the dashboard behind the existing SSH tunnel. Separate
Unix accounts and multi-cluster ownership are not supported.

## Execution, pause and cancellation

Each computation uses the existing direct Slurm/Apptainer/QE script. Its short
successor dispatcher uses `--dependency=afterany:<compute_id>`, so success,
failure, timeout and cancellation all release it. Calculation artifacts determine
the task result; a Slurm script's exit status alone is not proof of QE convergence.
The dispatcher reserves the next task under the shared queue lock before
submitting anything. There is no resident dispatcher service.

**Pause chain** prevents further submissions and lets the active computation
finish. Resume performs reconciliation and continues. Chain pause is independent
of the local queue pause. To safely stop the chain, pause first; optionally use
**Stop job** on the active computation. Disabling chaining restores independent
submission and is allowed only once the active compute/dispatcher pair has
finished. Clearing the existing queue retains its existing cancellation semantics;
use individual waiting-list removal when you only want to edit unscheduled work.

The default policy continues queued work after terminal compute outcomes without
retrying failed calculations. Explicitly authorized coordinate continuation adds
bounded fresh jobs under the policy above. Review other unsuccessful batch jobs with the
existing retry control, or open the candidate, preview and submit a new attempt.
Queued/running remain the existing task statuses; submitting/pending and terminal
scheduler outcomes are recorded in the task's Slurm record. A paused chain holds
waiting tasks without changing their task status.

## Monitoring and recovery

The chain panel displays the active task, next task, waiting count, dispatcher ID,
last advancement attempt and orchestration errors. Job history retains calculation
outcomes, resource requests, full task IDs and logs. Dispatcher failures are
orchestration errors; they do not mean the calculation itself failed.
Dispatcher attempts remain associated with their computation, and recent recovery
actions are shown in the chain panel.

Use **Recover chain** whenever advancement appears stalled. Recovery serializes
with dispatch and queue edits, queries both live Slurm state and accounting,
reconnects known submission tokens, and checks the active pair before acting:

- A healthy compute/dispatcher pair is left alone.
- A missing dispatcher can be submitted for a known compute allocation.
- A dispatcher known to have terminated can be replaced or its queue advanced.
- An empty chain becomes idle. After adding new work, use Resume or Recover.
- An uncertain submission blocks replacement until its identity is resolved.

Submission intent is committed before `sbatch`. If Slurm accepts a job but the
process dies before saving its ID, recovery looks up its unique job name in
`squeue` and `sacct`. Absence from accounting is never sufficient proof of
rejection. Recover again after accounting catches up. For a compute submission
that the operator has independently confirmed was rejected, the existing
**I confirmed Slurm rejected this submission** action closes that attempt; review
and retry explicitly. Do not delete uncertain records to force a new submission.
For an uncertain dispatcher that you have independently confirmed was rejected,
use **I confirmed Slurm rejected this dispatcher**, then Recover. That action
checks the displayed submission token and refuses if Slurm has a matching job.
It permits a replacement dispatcher while preserving the existing computation.

Recovery is safe to invoke repeatedly. It does not retry computations or submit
replacement dispatchers whose existing submission outcome remains uncertain.
The dashboard can be closed while jobs and dispatchers continue. The same shared
artifact directory, Python installation, repository and runtime paths must remain
available to compute nodes. When offline work encounters an error, the next
manual recovery reconciles the persisted state.

## Deployment requirements and limits

The server and dispatcher must share a filesystem with reliable POSIX `flock`
and SQLite rollback-journal locking. Confirm this with cluster administrators
before enabling chains on a shared HPC filesystem. Do not copy or independently
edit a live queue database. The process-shared lock file accompanies
`tasks.sqlite`; removing/replacing a live lock file breaks serialization.

The dispatcher needs the dashboard Python dependencies and Slurm commands on a
compute node, shared paths, and permission to submit allocations. Its allocation
requests one CPU, 256 MiB and five minutes, inheriting the computation's
partition/account/QOS. Check that those values satisfy cluster minimums.
`sbatch`, monitoring/accounting access and cancellation all run as the submitting
Unix identity. Recovery needs the same access. GPU requests, parallel chains,
electronic checkpoint restarts, unbounded retries, priorities, DAGs and resource
optimization are outside this implementation.

Grouped RAM measurements retain their existing shared-allocation behavior in
independent Slurm mode. They are unavailable in chain mode: deselect grouped
measurement and queue one process count per task. Finish or remove pending
grouped batches before enabling a chain. This keeps each editable queue position
an independent computation without silently changing a grouped batch's resources
or cancellation scope.

Slurm documents `afterany` as eligible after the predecessor terminates:
[Slurm sbatch dependency reference](https://slurm.schedmd.com/sbatch.html#OPT_dependency).
The repository and login-node `command -v sbatch` check establish neither
compute-node submission permission nor filesystem suitability.

## Target-cluster acceptance check

Local tests mock Slurm and execute generated compute scripts. A real PACE run
remains necessary. From the configured cluster environment, first confirm
`command -v sbatch squeue sacct scancel` and that the dashboard's Python and shared
artifact directory are visible in a small compute allocation.

Use an isolated artifact directory and the normal dashboard launcher/configuration.
Enable a paused chain and queue three short, low-resource computations with
valid prepared inputs. Use the smallest partition-approved CPU/memory allocation
and walltime settings. Resume and record both compute and dispatcher IDs from the
queue panel. While A runs, reorder B/C and insert another task. Close the dashboard
and check from the login node that only one compute job is pending/running at a
time and each dispatcher has the expected `afterany` dependency:

```bash
squeue --me --format='%.18i %.64j %.12T %.30E %.30R'
sacct --starttime today --format=JobIDRaw,JobName%64,State,ExitCode
```

Reopen with the same artifact directory and verify persisted history/order. Pause,
cancel one active compute using Stop, and resume: its cancellation should be
recorded and the next task should run. Cancel a dispatcher with `scancel` and
invoke Recover twice: one successor should be created, with no duplicate compute.
Also check an empty chain receiving new work. Inspect the dispatcher's log if
compute-node `sbatch`, Python imports or shared-file locking fail. Record actual
cluster results before treating this deployment as validated.

## Local validation record

On 2026-11-14, the fast repository suite passed **230 tests** (8 slow tests
excluded). New tests cover sequential advancement, live edits, terminal outcomes,
pause/resume, crashes around submission/persistence, repeated recovery, uncertain
submissions, operator-confirmed rejection, stale browser actions, cross-process
locking and grouped-RAM compatibility. A separate-process integration test runs
the generated compute and dispatcher scripts against filesystem-backed Slurm
commands after closing the parent manager; it verifies persisted advancement and
the successor dependency.

Dashboard Ruff lint/format and type checks pass, as do frontend TypeScript/build,
Node tests and formatting checks. These tests do not establish actual PACE policy,
Slurm availability from compute nodes, or shared-filesystem locking semantics.
Target-cluster acceptance remains pending; only login-node `sbatch` availability
has been confirmed by the user.

## Paired RAM estimation and shared allocations

In the batch controls, select **Estimate RAM · Candidate + TFA complex** to
measure both prepared systems in one Slurm allocation. The checks run
sequentially. You can use one process count or the existing grouped process-count
measurements. Each system/count has its own input snapshot, status, logs and RAM
result. A failed check does not prevent the remaining checks from running.

The allocation reserves the largest CPU and memory request among its sequential
checks, rather than adding their requests together. All checks share the
allocation's walltime limit. Stopping an allocation cancels its unfinished checks;
completed results remain available. Retrying an unsuccessful batch rechecks both
systems and skips matching successful estimates unless explicitly rerun.

The queue's allocation overview shows one shared allocation with its individual
checks nested below it. Submitted groups show the common Slurm job ID; waiting
groups show their batch association until submitted. Detailed task history remains
available separately.

Paired and grouped RAM allocations currently require independent Slurm mode.
The sequential chain still accepts individual-system, individual-process-count
tasks. Finish or remove pending paired/grouped batches before enabling a chain.
