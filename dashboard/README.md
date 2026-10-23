# Dashboard state

Updated: 2026-10-16. This document describes the current implementation and the
remaining deployment work. See the [full user guide](../docs/dashboard.md) for
detailed controls and workflows.

## Current capabilities

| Area | Implemented behavior |
| --- | --- |
| Candidate browser | All 1,000 clusters, representative diagrams, tiles and rows, ID filters, descriptor search, sorting, and shared selection. |
| RAM display | Separate candidate and complex values from successful one-process QE estimates; sorting and filtering by estimates. |
| Preparation | Generate isolated-candidate, candidate–TFA complex, and neutral TFA inputs without automatically running QE. |
| Geometry | Inspect prepared inputs and run geometries. |
| Input versions | Discover external input versions for individual previews and runs. |
| Local execution | Preview and queue native or Apptainer preparation, QE, and RAM-estimation tasks. |
| Queue | Persistent attempts and batches, resource budgets, pause, stop, cancellation, retry, log viewing, and artifact downloads. |
| SLURM export | Download selected candidate and complex calculations, copied pseudopotentials, a manifest, and submission scripts as a ZIP. |
| Static snapshot | Export a browsable snapshot without execution controls. |

Browsing does not start calculations. Preparation, RAM estimation, and production
QE runs are separate actions. The dashboard does not automatically orchestrate
all stages or aggregate adsorption energies.

## Run locally

For a PACE-hosted dashboard with a local browser, use the
[Slurm launch and SSH port-forwarding guide](../docs/pace-dashboard.md).

Run these commands from the repository root, with uv and Node/npm installed:

```bash
# Browsing, diagrams, and the web server:
uv run --no-default-groups --group web python -m dashboard

# Include the chemistry tools needed for preparing inputs:
uv run --no-default-groups --group web --group preparation python -m dashboard
```

Open `http://localhost:8000`. The launcher installs the locked frontend
dependencies, builds the frontend, and starts FastAPI. `--dev` starts Vite for
frontend development. Restart the backend after Python changes; rebuild/reload
the frontend when using the production launcher.

For local container execution, build and configure the image explicitly:

```bash
mkdir -p .container-build
apptainer build --fakeroot .container-build/chemistry.sif containers/chemistry.def
export PFAS_APPTAINER_IMAGE="$PWD/.container-build/chemistry.sif"
uv run --no-default-groups --group web python -m dashboard
```

The definition builds QE 7.6 with Open MPI. Its image test performs a small
two-process initialization. Local Apptainer jobs require verified memory-limit
support; native jobs have no hard memory cap. Native jobs execute one at a time,
although an individual QE job can use multiple MPI processes. Verified container
jobs can overlap within the queue's CPU and memory budgets.

## SLURM workflow currently implemented

1. Sort by **RAM · Complex / Ascending**, check that candidate and complex RAM
   values exist, and select the desired molecules. Selection of the eight lowest
   values is currently manual.
2. Choose **Export SLURM batch · candidate + complex**.
3. Review process count, memory sizing, walltime, partition, account, and QOS.
4. Preview the exact inputs and calculated memory requests, then download the ZIP.
5. Transfer the ZIP and the separately built SIF to the cluster, extract the ZIP,
   and run `bash submit.sh /absolute/path/to/chemistry.sif` from the extracted
   `pfas-slurm-batch/` directory once the cluster runtime is available.

Eight selected molecules produce **16 separate jobs**: an isolated candidate and
a TFA complex for each. **Include one isolated TFA reference** adds a seventeenth
job. It is optional in the export; its energy is needed when calculating
adsorption energies from complex and isolated-system results.

The ZIP contains:

```text
pfas-slurm-batch/
  README.txt
  manifest.json
  submit.sh
  run.sbatch
  runs/
    <candidate>-candidate/
      input.in
      Pseudopotentials/
      Outputs/
    <candidate>-complex/
      ...
    tfa-tfa/                 # Only when requested
      ...
```

The image is deliberately not included in the ZIP. Each calculation uses its
own directory and copies of the referenced pseudopotentials. Export uses the
latest successful prepared inputs, including accepted edits to those files;
the external input-version selector is not part of bulk SLURM export.

Default scheduling values are **8 MPI processes**, **18 hours**, partition
`ice-cpu`, account `coc`, and QOS `coc-ice`. They are editable defaults, not a
confirmation that a particular PACE allocation currently accepts them. Blank
partition/account/QOS fields omit the corresponding submission flags.

Each job requests one node, one launcher task, and N CPUs with
`--ntasks=1 --cpus-per-task=N`. Inside the container, Open MPI launches N ranks
with one thread per rank. Generated launcher options select Open MPI 4's local
isolated launcher, disable its Slurm resource discovery, and disable MPI binding.
This is a single-node container MPI design, not a multi-node launcher.

`submit.sh` calls `sbatch` once per calculation and records accepted IDs in
`jobs.tsv`. It refuses to repeat submission if that file is nonempty. After a
partial submission failure, inspect those IDs and submit only the remaining
commands manually. Each run directory receives `pw.out`, `pw.err`, SLURM logs,
and QE scratch data under `Outputs/`.

### Automatic RAM sizing

Automatic sizing is the default, with **50% headroom**, rounding up to whole
GiB, and a minimum reservation of 1 GiB. Memory is requested with `--mem` for
each job independently.

Only successful RAM estimates matching the calculation's source-input and
pseudopotential hashes qualify. The newest usable estimate at the selected MPI
count is preferred over a one-process estimate:

- For matching MPI counts, use QE's reported total; if absent, multiply its
  maximum per-process estimate by that matching process count.
- Otherwise, use a one-process estimate directly as the **total job baseline**,
  preferring its total report and falling back to its per-process report.
  **Do not multiply this serial baseline by the requested MPI count.**
- Add the selected headroom, then round up. A 12 GiB baseline with the default
  50% headroom requests 18 GiB for either an 8-rank or a 16-rank job.

These are planning estimates, not measured peak usage or guaranteed memory
bounds. The browser's historical RAM display can remain populated even when
an estimate no longer matches the input being exported. Preview shows the
estimate source, sizing basis, and reservation. **Manual** sizing instead uses
one entered memory value for every exported job.

Changes to inputs, pseudopotentials, settings, or the selected estimate after
preview require another preview before download. Any unavailable calculation
blocks the complete ZIP; the UI lists reasons beside the download button.

During the initial export attempt, the optional TFA input existed but had no RAM
estimate. That blocked automatic export when TFA was included. Uncheck the TFA
reference and preview again to export only candidate/complex jobs, estimate TFA
from its entry, or explicitly select manual memory. This was a missing estimate,
not a permanent inability to run TFA.

### PACE runtime issue still open

The exported `run.sbatch` currently invokes **`apptainer` by name**. There is no
automatic Singularity fallback or module-load command in that script.
`PFAS_APPTAINER` configures the local dashboard adapter; it does not rewrite the
exported cluster launcher.

The last reported check on `login-ice-gnr-1` was:

```text
module spider apptainer
Lmod: Unable to find: "apptainer".
```

This establishes that the module lookup failed in that session. It does not
establish whether a container executable is already on PATH, is provided under
another module name, or is available on the allocated compute node. The next
deployment check is to inspect `command -v apptainer`, `command -v singularity`,
and the supported runtime/module setup on ICE, then validate it in a compute-node
allocation. Do not assume `module load apptainer` will work.

Actual PACE execution remains unverified in this development session. A ZIP
download or accepted `sbatch` submission does not establish successful QE
execution. Confirm runtime startup, MPI rank count, convergence, and `JOB DONE`
in a real job before scaling up.

## Persistence and configuration

| Setting | Default / purpose |
| --- | --- |
| `PFAS_ARTIFACTS` | `.dashboard/`; includes `tasks.sqlite`, task directories, and artifacts. |
| `PFAS_QE_INPUTS` | `qe_inputs/`; external input versions. |
| `PFAS_PSEUDOS` | `qespresso_pipeline/Pseudopotentials/`. |
| `PFAS_CHEM_PYTHON` | Override the chemistry subprocess interpreter. |
| `PFAS_PW`, `PFAS_MPI` | Local native QE and MPI executable overrides. |
| `PFAS_APPTAINER` | Local container executable override, default `apptainer`. |
| `PFAS_APPTAINER_IMAGE` | Local built SIF path; no default configured image. |
| `PFAS_PREPARE_TIMEOUT` | Preparation timeout in seconds, default 900. |

Queue settings, batches, and task history persist in SQLite. Waiting local work
can resume on restart; previously running attempts become interrupted and need
explicit retry. Back up the artifact directory together with its database.
Reading state for inspection should use a read-only SQLite connection rather
than constructing a task manager, whose initialization performs recovery.

## Source map

| Files | Responsibility |
| --- | --- |
| `__main__.py`, `server.py`, `config.py` | Launcher, API/static serving, and configuration. |
| `candidates.py` | CSV candidates and the neutral TFA reference. |
| `tasks.py`, `batches.py`, `persistence.py` | Task lifecycle, batch review/acceptance, and SQLite state. |
| `inputs.py`, `preparation.py`, `geometry.py` | Input discovery/validation, preparation, RAM parsing, and geometry extraction. |
| `runtimes.py`, `container_runner.py`, `processes.py`, `cpu.py` | Local launch commands, memory enforcement/accounting, cancellation, and CPU allocation. |
| `slurm.py` | Reviewed portable SLURM ZIP generation and memory sizing. |
| `diagram.py`, `export.py` | Diagram generation and static snapshot export. |
| `../web/frontend/src/` | React UI; `SlurmExportControls.tsx` implements export review/download. |
| `../web/tests/`, `../web/frontend/tests/` | Backend regressions and frontend tests. |

Important export endpoints are `POST /api/slurm/preview` and
`POST /api/slurm/export`. The latter downloads the ZIP; neither endpoint submits
to a scheduler. Local task targets labeled Slurm/Slurm array remain placeholders
that return HTTP 501.

## Validation and remaining work

Previous implementation-session checks passed: **32 SLURM export tests**, frontend
build, and Python lint. An earlier broader dashboard run passed 132 tests with
8 skipped, before the final memory-sizing changes. The container launcher also
started eight MPI ranks locally; that check was not a PACE QE production run.
These are historical results, not checks rerun for this documentation update.

Relevant checks from the repository root:

```bash
.venv/bin/python -m pytest web/tests/test_slurm_export.py -q
.venv/bin/python -m pytest web/tests -q
.venv/bin/ruff check dashboard web/tests/test_slurm_export.py
npm --prefix web/frontend run build
npm --prefix web/frontend test
```

In the restricted agent environment, FastAPI's in-process test client previously
required execution outside the sandbox to avoid hanging.

Still unimplemented or unverified:

- Resolve and verify the container runtime on PACE ICE.
- Run the exported launcher and QE successfully in a PACE allocation.
- Automatic SSH transfer/submission from the dashboard.
- Remote job monitoring, cancellation, and result import into dashboard history.
- SLURM array export and multi-node MPI.
- Automatic selection of the eight smallest calculations.
- Automatic adsorption-energy aggregation from candidate, complex, and TFA runs.
