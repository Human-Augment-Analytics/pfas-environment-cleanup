# Dashboard architecture and simplification

The dashboard is a browser interface to the existing chemistry pipeline and to
Slurm. It should not introduce a second chemistry workflow or require a Python
workflow engine on a compute node just to start `pw.x`.

## What changed

New Slurm submissions now write an ordinary batch script that starts the
calculation directly:

```text
browser → HTTP app → sbatch → Apptainer → image MPI → pw.x
                       ↑                         ↓
                squeue/sacct              output and exit code
                       └──── dashboard reads ────┘
```

For a RAM batch, that same script runs the selected inputs at 16, 4, and 1 MPI
processes sequentially. A short shell function records the start time and exit
code and optionally applies `timeout`. It continues after an individual failed
check. There is no compute-side dashboard task manager, Python process monitor,
JSON task manifest, or result-serialization protocol in new submissions.

The dashboard reads the output after an exit-code file appears. This preserves
QE's initialization-only exit code 255, checks the actual MPI process count,
and distinguishes a completed program from a converged calculation. Slurm still
owns resource allocation, job lifetime, cancellation, and scheduling.

Also removed: the separate `run_calculation` execution wrapper, its executor
protocol/callback plumbing, and the worker-specific configuration models. Local
execution uses its existing process launcher; local and scheduled calculations
share output-reading functions rather than another executor.

Constants are imported as module namespaces, such as `execution.TaskStatus`.
Frontend consumers likewise use `execution.DEFAULT_SLURM_PROCESSES`. Unused
constants from the worker implementation were removed.

## Where the existing pipeline is reused

`core/chemistry/preparation.py` calls these existing functions instead of
reimplementing their chemistry:

- `qespresso_pipeline/run_adsorption_case.py`: `get_mode_settings`,
  `prepare_from_smiles`, `build_molecular_complex_cif`, and `patch_qe_input`.
- `qespresso_pipeline/smiles_to_qe.py`: `run_cif2cell`, together with the
  conversion functions already used by `prepare_from_smiles`.

The existing dashboard Slurm exporter supplies the single-node container MPI
command. The live submission path uses that same `mpi_command` helper.

The full `run_adsorption_case.py` CLI prepares and runs an entire adsorption
case. Calling it for every dashboard QE action would combine stages the user
has deliberately separated, potentially regenerate edited inputs, and add
Python/Conda orchestration to a job that already has an input file. Its
`run_pwscf` helper ultimately launches a command with that input and redirects
output. Starting Apptainer/`pw.x` directly preserves that execution model without
rerunning preparation. No new scientific pipeline was added here.

## Folder and entry-point rules

```text
dashboard/
  __main__.py                 public python -m dashboard entry point
  slurm_worker.py             temporary compatibility entry point
  app/                       HTTP application and command-line adapters
  core/
    main.py                  startup workflow
    config.py                environment/configuration
    execution.py             namespaced execution constants
    models.py                record types
    candidates.py            candidate data
    snapshot.py              static export
    chemistry/               existing-pipeline adapter and output readers
    jobs/                    queue, batches, persistence, local processes
    runtimes/                native/container commands and local limits
    slurm/                   direct batch scripts, ZIP export, monitoring
  test/                      dashboard tests
```

Each CLI adapter constructs its parser in `get_parser()`. Its `main()` parses
arguments and calls a keyword-only `main(*, ...)` in core. Core never parses the
process command line. Importing diagram code no longer generates a diagram.
The public `python -m dashboard` command stays the same. FastAPI's factory is
now `dashboard.app.server:create_app`; CI, tests, and the Slurm launcher use the
new paths.

## Why the remaining pieces exist

These are feature requirements, not claims that every component is essential
to a minimal submit button.

| Component | Concrete reason to retain it |
| --- | --- |
| HTTP app and React UI | Browse candidates, inspect inputs, and explicitly request actions. |
| Candidate loading, filtering, diagrams | Existing discovery UI. Diagram generation needs RDKit, so it stays outside the web server process. |
| Preparation adapter | Connect candidate selections to the existing chemistry functions and keep preparation separate from execution. |
| Input and geometry readers | Preview edited inputs, construct initialization-only copies, and show coordinates without executing chemistry. |
| Immutable input/pseudopotential copies and checksums | A queued calculation must run the input reviewed by the user, even if the source changes while it waits. A direct shell checksum check provides this without a worker. |
| Apptainer command construction | Put the image's tools and MPI on the command line and bind the attempt directory. On Slurm there is no extra cgroup supervisor. |
| Slurm submission and polling | Support the requested independent allocations, pending reasons, cancellation, and reconnection when the dashboard restarts. These functions submit and observe jobs; they do not run chemistry themselves. |
| Submission intent and unique job name | A lost `sbatch` reply must not cause a duplicate expensive calculation. Persist before submission and reconcile by job name. |
| Start/exit marker files | Import each RAM check as it finishes, including failures, while the rest of the shared Slurm allocation continues. These are two small text files, not a compute-side object model. |
| QE output readers | Obtain RAM reports and scientific completion evidence. Scheduler success alone does not establish convergence. |
| SQLite queue/history | Preserve accepted tasks, job IDs, selections' batch history, and results across dashboard restarts. Compute jobs never open SQLite. |
| Batch review/retry | Existing bulk actions and the requested grouped RAM estimates; avoid rerunning successful rank/input combinations. |
| Local process/CPU helper | Preserve Stop, timeout, logs, and responsiveness when calculations run inside the dashboard allocation or on a workstation. Independent Slurm jobs bypass this helper. |
| Workstation cgroup supervisor | Preserve enforced per-job memory limits outside Slurm. It is not used by the PACE calculation path. |
| ZIP/static exporters | Preserve the existing portable cluster-batch and read-only browsing workflows. |
| Types and constants | Describe the records already persisted by the UI and keep resource values/environment names explicit. They do not implement another execution engine. |

The local concurrency/cgroup features, ZIP export, static snapshots, and rich
history are optional product features. Removing those would simplify the
application further, but would remove established behavior beyond the requested
PACE refactor. The scheduler adapter is needed only because independent jobs,
cancellation, and restart recovery were requested; it would be unnecessary if
the UI only downloaded a shell script.

## Temporary compatibility

Previously submitted scripts invoke `python -m dashboard.slurm_worker` with a
manifest. Removing that module immediately would break those pending jobs. Its
small compatibility entry point now translates that old manifest into the same
direct shell script. It is never used for new submissions. The reader also
accepts old completed JSON result files so existing history remains usable.

After confirming no queued/running jobs invoke the old worker, remove
`dashboard/slurm_worker.py`, `app/legacy_worker.py`, and
`core/slurm/legacy_worker.py`. The legacy result-reading branch can be removed
when no unimported old jobs remain. No database migration is required.

## Validation boundary

Tests execute generated batch scripts using fake QE/MPI executables, including
16/4/1 checks, partial failure, timeout, changed checksums, submission recovery,
and restart behavior. Other tests cover pipeline preparation, the HTTP API,
imports, and the `get_parser()` → keyword-only core dispatch contract.
Opt-in real QE/container checks and a PACE acceptance run are still required to
validate the actual cluster software and MPI configuration.

## Readability changes

The local `execute()` function now only manages log lifetime and delegates.
Launch, bounded stream draining, process-tree RSS, cgroup accounting, stop
conditions, and process termination are separate named functions. `OutputStream`
and `MemoryState` are small data records that replace implicit nested/nonlocal
state; they do not schedule or execute workflows. Observer exceptions are covered
by tests proving process and pipe cleanup.

The same decomposition was applied to queue admission, input previews, batch
eligibility and submission, restart recovery, input/geometry parsing, Slurm
memory sizing, ZIP packaging, and startup/export. Validation, filesystem writes,
and persistence now have distinct functions instead of being embedded in large
loops. Meaningful stages have leading comments.

Public option-heavy signatures and a few linear command/record declarations can
exceed the approximate 20-line guideline after type annotations and formatting.
Their docstrings identify the compatibility or auditability reason for keeping
those declarations together; they are not used to justify nested execution
logic. The project guidance and running preferences record state the short
function and block-comment rules explicitly.
