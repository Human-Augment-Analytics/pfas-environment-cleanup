# Dashboard architecture

**Draft for review: proposed organization, not yet implemented.** This document
walks through the intended modules from the inside out: first the work they do,
then the machinery supporting that work, and finally the interfaces around it.
The paths below describe the proposed layout. No code has moved as part of this
README update. The [current implementation writeup](../docs/dashboard-architecture.md)
provides background on the preceding refactor.

## Start with the work: four job kinds

The dashboard lets a user generate diagrams, prepare chemistry inputs, estimate
memory, and run Quantum ESPRESSO. Each job kind gets one named Python module in
`core/jobs`. The user requests these actions separately; preparation does not
automatically trigger estimation or production calculations.

### Diagram generation — `core/jobs/diagram.py`

The core operation takes a candidate SMILES string and an output path and
renders a molecule image using RDKit. Its output is an image the browser can
show alongside the candidate.

This needs the chemistry environment, so the operation runs in a subprocess
rather than importing that environment into the HTTP server. The surrounding
job code supplies the candidate, chooses the destination, and registers the
image as an artifact. On PACE, the subprocess runs within the dashboard's
allocation.

### Input preparation — `core/jobs/preparation.py`

The core operation turns a candidate description into prepared structures and
QE inputs for the isolated candidate, candidate–TFA complex, and TFA reference.
It delegates the scientific preparation to the existing chemistry pipeline:

- `run_adsorption_case.py`: `get_mode_settings`, `prepare_from_smiles`,
  `build_molecular_complex_cif`, and `patch_qe_input`.
- `smiles_to_qe.py`: `run_cif2cell` and the conversion functions already used
  by `prepare_from_smiles`.

This module is the dashboard's adapter to those functions. It does not
reimplement their chemistry. Its inputs include the candidate SMILES,
pseudopotentials, preparation settings, and an optional reusable TFA reference.
Its outputs are prepared input files, structures, and preparation provenance.

The supporting job code creates the attempt directory, starts the chemistry
subprocess, and registers the resulting files. Preparation runs locally or
inside the dashboard allocation. It ends with inspectable inputs, allowing the
user to review or edit them before choosing the next action.

### RAM estimation — `core/jobs/ram_estimation.py`

The core operation takes a prepared QE input, makes an initialization-only copy,
and interprets the memory estimates from QE's output. It retains the original
input separately from the transformed input used for estimation.

A usable result requires the expected memory report and no disqualifying QE
error. Initialization-only exit codes, including 255, need this interpretation;
they cannot be classified using the exit code alone. For multi-rank Slurm
estimates, the reported MPI process count must match the requested count.

The module owns that input transformation and result validation. It reuses the
QE launcher rather than defining a second execution engine. The surrounding
batch code can request several MPI counts and associate each result with the
input and rank count that produced it.

On Slurm, grouped checks normally run sequentially at 16, 4, and 1 MPI processes
in one allocation sized for the largest count. Each check has its own result.
A failed check does not prevent later checks, and retry selects unsuccessful
input/rank combinations.

### Production QE — `core/jobs/qe.py`

The core operation runs `pw.x` against an already prepared input and interprets
its scientific completion evidence. This module owns QE command construction
and the checks for completion, convergence, and energy evidence. RAM estimation
reuses its launcher construction.

A successful process or scheduler exit does not by itself establish a converged
calculation. Output interpretation distinguishes program completion from SCF
convergence and, where applicable, geometry optimization completion.

The surrounding code supplies captured inputs, pseudopotentials, resource
requests, and the selected runtime. It launches the command locally or submits
it to Slurm, then passes the output back for interpretation.

Running a prepared input does not invoke the complete adsorption CLI. That CLI
combines preparation and execution and could regenerate an edited input. The
dashboard preserves those separate actions and does not automatically aggregate
adsorption energies.

## Shared chemistry support

The jobs need to agree on what an input contains before they can preview,
capture, or run it.

`core/chemistry/inputs.py` owns shared input discovery, validation,
pseudopotential-name parsing, and hashing. RAM-specific transformations and
report interpretation belong to the RAM job module. Generic utilities move
here from preparation so callers can use them without depending on that job.

`core/chemistry/geometry.py` reads coordinates for input previews and result
inspection. It interprets existing files without executing scientific work.

Both modules support job behavior and read-only browsing. Neither owns a queue,
launches a workflow at import time, or duplicates scientific preparation.

## Turning job behavior into recorded attempts

The four job modules describe the work. Shared infrastructure gives each
requested attempt a lifecycle, an execution location, and a durable record.

### Task manager and batches

`core/jobs/manager.py` accepts requests, validates previews, captures reviewed
inputs, and coordinates task state. It replaces the less descriptive
`tasks.py` name. Job-specific commands and interpretation move into the four
job modules; shared admission and lifecycle behavior stay here.

Captured input and pseudopotential copies let a queued calculation use exactly
what the user reviewed, even if source files change while it waits. Hashes
identify those copies and let estimate selection check provenance.

`core/jobs/batches.py` applies existing review, submission, cancellation, and
retry behavior to selections of tasks. A batch groups user actions; it does
not automatically chain preparation, estimation, and production into a new
scientific pipeline.

### Runtimes and local execution

A job kind identifies the work. A runtime identifies its executable environment:
native tools or Apptainer. An execution target identifies whether the dashboard
manages the process locally or submits an independent Slurm allocation.

`core/runtimes/local.py` constructs runtime commands and determines local
resource limits. `core/runtimes/cpu.py` applies CPU affinity to local
subprocesses. `core/runtimes/container.py` verifies and supervises workstation
container memory isolation.

`core/jobs/processes.py` launches local commands, drains timestamped logs,
samples memory, and handles Stop, timeouts, and cleanup. These functions preserve
responsiveness and release resources when a process or observer fails.
Diagram and preparation jobs on PACE use this local path inside the dashboard
allocation. Independent Slurm QE jobs bypass the local process monitor and
workstation cgroup supervisor.

### Independent Slurm jobs

`core/slurm/scheduler.py` submits and observes independent allocations.
`core/slurm/export.py` constructs ordinary job scripts and the existing portable
ZIP exports. Live and exported jobs reuse the container MPI launcher.

```text
accepted task → sbatch → Apptainer → image MPI → pw.x
                  ↑                              ↓
             squeue/sacct                logs and exit markers
                  └──── dashboard imports ───────┘
```

Native jobs use the corresponding native QE command. New submissions start the
calculation directly: they do not launch a dashboard Python worker or access
SQLite on the compute node. Slurm owns scheduling, allocation, job lifetime,
and resource enforcement.

The script checks captured input checksums and writes small start/finish
markers. The dashboard imports output after an attempt finishes. Markers also
allow it to import individual RAM checks while the shared allocation continues.

Submission intent and a unique job name are persisted before invoking `sbatch`.
If its response is lost, the dashboard searches for that job instead of
submitting another expensive calculation. Independent jobs can continue while
the dashboard is stopped and be reconnected on restart.

### Results and persistence

`core/jobs/results.py` assembles common exit/error records and provenance logs.
Scientific interpretation stays in `qe.py` and `ram_estimation.py`, keeping a
single dispatcher from becoming the home for every job's rules.

`core/jobs/persistence.py` stores task history, queue settings, batches,
artifacts, and scheduler identifiers in SQLite. Recovery reconnects independent
Slurm work; previously running local attempts are interrupted and require an
explicit retry. Runtime state remains under the configured artifact directory,
with repository `.dashboard/` as the default.

`core/models.py` describes shared records. `core/execution.py` provides
namespaced constants and status enums. `core/config.py` owns explicit paths
and configuration. These modules describe the existing system rather than
introducing another execution framework.

## Interfaces around the work

### Python HTTP and command-line adapters

`app/web/server.py` owns the FastAPI application factory, HTTP request models,
route adapters, and error translation. It calls core functions for validation,
chemistry, task management, and results. Core job modules do not import the
HTTP server or frontend.

`app/web/cli.py` parses serve/export options. `app/jobs` contains diagram and
preparation subprocess adapters plus temporary legacy-worker compatibility.
`app/runtimes` contains the CPU and container-supervisor adapters.

Each CLI constructs its parser in `get_parser()` and passes parsed options as
explicit keyword arguments into a keyword-only core `main(*, ...)`. Core never
parses command-line arguments or reads `sys.argv`. A direct `pw.x` command needs
no additional Python job CLI.

### Frontend and browsing

The React interface lives under `web/frontend`. Candidate browsing, filtering,
selection, geometry inspection, task history, and run controls call the HTTP
API. Candidate loading lives in `core/candidates.py`; the web interface presents
that data and the records produced by the jobs.

Reading data does not queue calculations. Diagram generation is a separate
lightweight job for missing images. Preparation, RAM estimation, and production
QE require their own requested actions.

All TSX, TypeScript, CSS, HTML, npm configuration, and frontend tests stay in
frontend directories, outside the Python `app` and `core` trees.

### Web startup, snapshots, and exports

`core/web/main.py` orchestrates startup and static export.
`core/web/snapshot.py` assembles read-only browser data from existing candidate
records and artifacts. Neither starts chemistry work to populate an export.

- Development starts Vite and the Python API, with Vite proxying API requests.
- Live serving builds the frontend and serves it through FastAPI alongside
  the API routes.
- Static export builds snapshot mode and assembles the exported dataset with
  the page. Execution controls are unavailable in that mode.
- Pages CI packages the snapshot frontend and dataset without a running server.

The ZIP exporter serves a different purpose: it packages selected prepared
inputs, pseudopotentials, and submission scripts for manual cluster submission.
Downloading a ZIP does not submit jobs.

## Proposed file tree

The tree makes the four job kinds visible while keeping their shared execution
support and interfaces separate. Package initializer files are omitted below.

```text
dashboard/
  README.md
  __main__.py                  public python -m dashboard entry point
  slurm_worker.py              temporary compatibility for old submitted jobs

  app/                         Python interface adapters
    jobs/
      diagram.py
      preparation.py
      legacy_worker.py
    runtimes/
      cpu.py
      container.py
    web/
      cli.py
      server.py

  core/                        Python implementation
    jobs/
      diagram.py               molecule image generation
      preparation.py           adapter to existing preparation pipeline
      ram_estimation.py        initialization-only inputs and memory results
      qe.py                    QE commands and scientific completion checks
      manager.py               shared task admission and lifecycle
      batches.py               bulk actions and selective retry
      persistence.py           SQLite records and recovery
      processes.py             local subprocess monitoring and cleanup
      results.py               common results and provenance logs
    chemistry/
      inputs.py                shared input validation, discovery, and hashes
      geometry.py              coordinate parsing and previews
    runtimes/
      local.py                 native/container commands and local limits
      cpu.py                   CPU-affinity implementation
      container.py             workstation memory supervisor
    slurm/
      scheduler.py             submission, polling, cancellation, recovery
      export.py                direct shell scripts and portable ZIPs
      legacy_worker.py         old manifest translation
    web/
      main.py                  live startup and static-export workflow
      snapshot.py              read-only browser dataset
    candidates.py
    config.py
    execution.py
    models.py

  web/                         frontend source and exported web assets
    frontend/
      src/                     React, TypeScript, and CSS
      tests/                   frontend tests
      index.html
      package.json
      package-lock.json
      tsconfig.json
      vite.config.js
    snapshot/                  exported dataset and images
    public/                    assembled standalone static site

  test/                        Python regression and integration tests
```

### Source versus generated state

Paths below are relative to `dashboard` unless explicitly marked otherwise.

| Location | Contents and treatment |
| --- | --- |
| `app/`, `core/`, `test/` | Python source and tests. |
| `web/frontend/src/`, `web/frontend/tests/` | Frontend source and tests. |
| Frontend package manifests and build configuration | Tracked dependency/build definitions. |
| `web/frontend/node_modules/`, `web/frontend/dist/`, caches | Ignored installed dependencies and generated builds. |
| `web/snapshot/` | Exported browser dataset; preserve existing tracking/ignore policy during the move. |
| `web/public/` | Ignored generated standalone site; an explicit export destination can override this location. |
| Repository `.dashboard/` by default | Runtime database, captured inputs, logs, and artifacts; configurable with `PFAS_ARTIFACTS`. |

Frontend moves do not change database schemas, configured artifact paths, or
existing task-directory layouts.

## Why the supporting pieces remain

| Supporting component | Reason it exists |
| --- | --- |
| Shared input and geometry readers | Jobs and browsing need consistent interpretation of the same files. |
| Manager and batches | Accepted tasks need queueing, resource admission, bulk actions, and selective retry. |
| Persistence | Tasks and scheduler IDs must survive dashboard restarts. |
| Local process helpers | Local work needs cancellation, timeouts, logs, and cleanup. |
| Runtime helpers | The same job can use native tools or an image without duplicating its behavior. |
| Slurm adapter | Independent jobs need submission, status reporting, cancellation, and reconciliation. |
| Result helpers | Attempts share exit/provenance fields while their scientific checks differ. |
| HTTP and frontend adapters | Users need to inspect inputs and explicitly request work. |
| ZIP and static exporters | Preserve existing manual-submission and read-only publication workflows. |

Local resource management, ZIP export, and static publication are existing
product features rather than requirements for a minimal Slurm submit button.
The proposed organization retains them without adding new layers of execution.

## Compatibility and adoption

The proposed changes move repository-level `web` into `dashboard/web` and rename
internal Python modules. Update imports, subprocess paths, static serving,
export destinations, npm commands, ignore rules, CI, and the PACE launcher
together when implementing them.

Preserve `python -m dashboard`, existing HTTP routes and payloads, persisted
job kinds (`diagram`, `prepare`, `estimate_ram`, `qe`), and runtime artifacts.
The proposed Uvicorn factory becomes
`dashboard.app.web.server:create_app`. Internal imports adopt the new names;
there is no forwarding module for every moved file and no database migration.

Old submitted scripts may still invoke `python -m dashboard.slurm_worker`.
Retain that entry point and its legacy adapter until those queued/running jobs
have drained. Retain legacy result reading until no unimported old results
remain. New jobs continue to use direct shell launches.

## Adding code within this organization

Start with the behavior being added. For a new job kind, put its specific input,
command, and result rules in one clearly named `core/jobs` module. Connect it to
the existing lifecycle and reuse the runtime and scheduler functions. Do not
create a new queue or duplicate scientific preparation.

Add an `app/jobs` parser only when a subprocess entry point is necessary. For
an HTTP action, put request/response adaptation in `app/web`, call core for the
work, and add frontend controls under `web/frontend/src`.

Keep functions short and typed, constants namespaced, and logical phases
explained with leading comments. The architecture walkthrough should introduce
each module's core work before explaining its dependencies and wrappers.

## Validation before adopting the layout

- Run the existing Python suite, lint, formatting checks, and type checks.
- Verify moved parsers dispatch keywords correctly and imports start no work.
- Check the PACE application-factory import and launcher shell syntax.
- Run frontend tests, formatting checks, and live/snapshot builds.
- Verify live static serving, snapshot export, and Pages artifact locations.
- Search maintained source and configuration for obsolete paths.
- Preserve coverage for grouped RAM checks, partial failures, timeouts,
  checksums, cancellation, and submission/restart recovery.

Local checks cannot establish the actual PACE runtime and MPI configuration;
those still require a real cluster allocation. The
[user guide](../docs/dashboard.md) and [PACE guide](../docs/pace-dashboard.md)
remain the operational references and will need their paths updated when the
proposed code moves are implemented.
