# PFAS dashboard

From the repository root, with uv and Node/npm installed:

```bash
uv run --project web python -m dashboard
```

Open http://localhost:8000. The launcher installs the locked npm dependencies,
checks TypeScript, builds React, and serves the frontend and FastAPI from one
process. The web Python environment includes RDKit for diagrams and is separate from the
full chemistry environment needed for input preparation.
The dashboard displays clusters **500–667 inclusive**, in the existing order of
`shivani_ml_models/cluster_centers.csv` (168 candidates). CSV descriptors are
cluster averages. CID and SMILES identify each cluster's representative molecule;
all original fields are retained in its detail view.

Browsing never starts work. Use Generate diagram, Prepare inputs, or preview
and submit a QE run. Refresh updates the queue, elapsed time, and bounded log
tails. Only one task runs at a time. Preparation and QE are independent actions;
there is no bulk preparation, bulk QE, or adsorption aggregation.

## Chemistry configuration

Point the dashboard at a Python interpreter with RDKit, pymatgen, ASE, Open Babel
Python bindings, and the `obabel` and `cif2cell` commands. Its bin directory is
prepended to PATH for preparation and diagrams. For example, if the repository's
chemistry environment is already installed:

```bash
export PFAS_CHEM_PYTHON="$PWD/.venv/bin/python"
export PFAS_PSEUDOS="$PWD/qespresso_pipeline/Pseudopotentials"
uv run --project web python -m dashboard
```

Missing chemistry tools or element pseudopotentials reject preparation before
it enters the queue; browsing still works. The full preparation environment is not
installed by the web launcher. Diagrams work with the default web environment;
if `PFAS_CHEM_PYTHON` is set, that interpreter must also provide RDKit. The preparation child entry point is
`web/dashboard/preparation.py`; it reuses conversion, molecular-complex, and
patching helpers and never executes QE.

Inputs use the molecular `cluster` preset: PBE, D3, 60/600 Ry cutoffs, Gamma
sampling, mixing beta 0.15, 12 Å padding, 2.5 Å complex gap, spin 1, and relaxation.
The shared reference is **neutral TFA**, `O=C(O)C(F)(F)F`, CID 8442. Preparation
creates the reference if absent and otherwise reuses a compatible successful
reference, recording its source attempt. A candidate preparation creates
isolated-candidate and candidate–TFA inputs. Manifests record settings, source
CID/SMILES, and input/pseudopotential SHA-256 hashes. Representative geometries
and complex placement are starting models, not optimized adsorption structures.

`PFAS_PREPARE_TIMEOUT` sets the preparation timeout in seconds (default 900).
Diagrams run in isolated children with a 60-second timeout. Bulk diagrams skip
existing successful PNGs, deduplicate active attempts, and continue after failures.

## Local QE and MPI

`PFAS_PW` and `PFAS_MPI` override the `pw.x` and `mpirun` executable paths. Otherwise
both resolve through PATH. MPI is required only for more than one process.
Commands use argument lists directly, without a shell:

```text
pw.x -in input.in
mpirun -np N pw.x -in input.in
```

`OMP_NUM_THREADS=1` for all runs. Select a positive integer process count (default
one), preview the exact command and input, then submit. There are separate Run QE
controls for TFA, candidate, and complex. Slurm and Slurm array return HTTP 501
before creating a task. They do not submit anything.

Each run gets a new directory with an input captured when queued, immutable
pseudopotential copies with matching links, and its own `Outputs` directory.
Externally edited preparation inputs are accepted only with local
`pseudo_dir='./Pseudopotentials'` and `outdir='./Outputs'`. An input changed since
preview requires another preview. QE timeout is optional; by default it has no
application-imposed timeout. Stop terminates the entire process group, including
MPI children. Full stdout/stderr downloads become available when the attempt
finishes. The UI separately reports process exit, JOB DONE, SCF convergence,
relaxation completion, and provisional/confirmed last energies.

## Persistence and development

`PFAS_ARTIFACTS` selects persistent local storage (default `.dashboard/`). Keep the
SQLite database on local disk. Attempts and logs are preserved on retry. Graceful
shutdown stops active children; startup marks unfinished attempts interrupted.
Retry explicitly. Run only one dashboard server against a given artifact directory.

```bash
uv run --project web python -m dashboard --dev
```

Open the Vite URL (normally http://localhost:5173). Vite proxies `/api` to FastAPI;
both stop on interrupt. React uses built-in state, `fetch`, manual refresh, and
hash navigation. Components live under `web/frontend/src`. Plain CSS includes
narrow-screen layouts and a horizontally scrollable table.

The launcher accepts `--host` and `--port`; it binds to loopback by default.
Mutation routes require POST and reject cross-origin browser requests. Binding
to another interface **does not add authentication**. Keep access local or tunneled.

For a future PACE session, the proposed approach is to install the repository and
chemistry environment on a permitted compute node, run the server on loopback
inside an allocated job, and use SSH local forwarding through the PACE login
host. For example, adapt `ssh -L 8000:compute-node:8000 your-pace-login` to the
site's permitted tunnel topology. PACE connectivity and policy compatibility
remain **unverified**; Slurm execution and a shared live service are deferred.

## Read-only public snapshot

```bash
uv run --project web python -m dashboard export
```

Export does not trigger chemistry or QE. It builds the frontend, writes committed
public data under `web/snapshot/`, and assembles the static site under ignored
`web/public/` (use `--output` to change the assembled-site destination). Commit the
snapshot data to update the public dashboard. Only summary JSON, existing
successful diagrams, and selected result metadata are exported. Raw logs,
databases, machine paths, scratch files, credentials, and mutation controls are
excluded. The UI shows the export timestamp and local launch instructions.

The `dashboard.yml` workflow tests the backend, builds the frontend, copies
**committed** snapshot data, and deploys to GitHub Pages on main or manual dispatch.
Configure the repository Pages source as GitHub Actions. Publication depends on
that repository setting. Assets and snapshot requests are relative; candidate
links use hashes and work under `/pfas-environment-cleanup/` without server routing.

## Validation

```bash
uv run --project web --locked pytest -c web/pyproject.toml web/tests -m 'not slow'
uv run --project web --locked ty check --project web
uv run --project web --locked ruff check web/dashboard web/tests dashboard.py
npm run build --prefix web/frontend
npm run format:check --prefix web/frontend
```

Opt-in smoke checks use real tools, isolated temporary directories, a 90-second
preparation timeout and 45-second QE timeouts. The small H2 SCF fixture checks
native one- and two-process execution; it does not validate production scientific
accuracy or convergence of the 168 candidates:

```bash
PFAS_SMOKE=1 PFAS_CHEM_PYTHON="$PWD/.venv/bin/python" \
  uv run --project web --locked pytest -c web/pyproject.toml web/tests/test_smoke.py -q
```

Deferred: 3D viewing, bulk QE, automatic workflows, adsorption aggregation, Slurm,
authentication, containers, and a shared live service.
