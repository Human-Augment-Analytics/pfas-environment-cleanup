# Compute chaining architecture assessment

## Existing capabilities

`dashboard/core/jobs/tasks.py:Manager` accepts immutable input snapshots, validates
per-job resources and preview tokens, owns cancellation, and exposes queue/history
JSON. `jobs/persistence.py:Store` stores task JSON, batches and queue settings in
`.dashboard/tasks.sqlite`. Waiting order is currently insertion order. Settings
and tasks are shared by browser sessions using the same artifact directory, under
the dashboard process's Unix identity; this is not an authenticated multi-user
service. Its RLocks protect threads only, and constructing Store performs local
process/batch recovery.

`slurm/scheduler.py:Scheduler` submits ordinary direct scripts produced by
`slurm/export.py`, polls squeue/sacct, imports calculation artifacts, and cancels
allocations. It persists unique submission tokens BEFORE sbatch, searches live
and accounting job names after uncertain replies, and deliberately refuses to
retry ambiguous submissions. Parent Slurm/MPI/SBATCH variables are filtered so
submission from an allocation requests independent resources. This is the
existing compute-node spawning mechanism. There is no dependency dispatcher;
the server thread currently submits all waiting jobs (grouping RAM estimates).
Slurm jobs survive server shutdown, but queue advancement does not.

`dashboard/app/server.py` exposes task, batch, queue settings and cancellation
APIs. `web/frontend/src/QueueView.tsx` polls every two seconds and already provides
pause, queue clearing, history, logs and batch retry review. Candidate and batch
controls already accept computations with distinct resource requests.

## Changes and implementation strategy

Keep independent submission behavior available for compatibility. Add an explicit
sequential-chain mode for the existing Slurm waiting queue, with persistent
start/pause state. Reuse task records and scheduler records rather than a second
queue. Extend Store with persistent ordering, chain metadata and a process-shared
mutation lock; all dashboard/dispatcher writers must cooperate with that lock.
Keep compute scripts unchanged. A short Python dispatcher is justified because
it must read the latest queue and reuse the existing Python submission and result
reconciliation code; it exits after one advancement and is not a compute worker.

Add `core/slurm/chaining.py` and `app/dispatch.py`: serialize reconciliation and
selection, submit one computation, and submit its afterany successor. Persist
compute and dispatcher intent before external calls; recover unknown IDs by the
existing token lookup. Never interpret absence from accounting as proof that an
uncertain submission was rejected. Recovery blocks safely on uncertainty and
recreates only dispatchers known to have terminated. Existing scheduler polling
continues to import results while online. Dispatchers open Store without running
server-local recovery. Explicit recovery is available through the queue API/UI.

Adapt Manager queue views and settings, server routes, QueueView and data types
for start/recover, editable positions and dispatcher diagnostics. Preserve preview
validation, existing task history, cancellation and manual retry review. Queue
edits apply only before a scheduler claim. Use stale-order validation for
conflicting browser edits. Record compute failures separately from orchestration
errors; continue after terminal compute outcomes without automatic retry.

Grouped RAM measurements keep their independent-mode shared-allocation contract.
Chain mode rejects grouped batches and requires one process count per queued
task. Existing pending groups must finish or be removed before mode changes.

## Constraints and validation

No new database, resident daemon, scheduler framework or chemistry execution
layer is needed. SQLite and the process lock require a shared artifact filesystem
with reliable locking visible from login and compute nodes. Python, repository,
input snapshots, runtime/image and Slurm commands must be accessible in dispatcher
allocations. PACE partition/account/QOS permissions, compute-node sbatch access,
dependency support, shared-filesystem SQLite suitability and resource minimums
cannot be proven from this repository. Cluster administrators/policy and a short
real-allocation smoke test must establish them. Existing CPU chemistry requests
have no GPU resource contract; GPU scheduling remains unsupported rather than
silently promising GPU allocations. All browser users act as the server Unix
account; independent Unix-account ownership/authorization is outside this v1.

Validate sequential order, live edits, pause, failures/timeouts/cancellation,
submission uncertainty, dead dispatchers, repeated recovery and conflicting
writers with mocked Slurm. Run existing dashboard regression tests, lint/type
checks and frontend build/tests. Real HPC validation requires access to the target
cluster and must be reported separately if unavailable.
