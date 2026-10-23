# Run the dashboard on PACE

Run the backend and calculations in a Slurm compute allocation on PACE ICE,
then open the dashboard in your local computer's browser through SSH. The
browser uses your local computer; Python, preparation, QE, and saved artifacts
use PACE. The dashboard keeps diagrams and preparation in its own allocation,
and submits QE and RAM estimation to independent Slurm allocations.

## 1. Prepare a PACE checkout

For your account, connect with:

```bash
ssh jmccloskey30@login-ice.pace.gatech.edu
cd ~/pfas-environment-cleanup
```

The path above assumes the repository folder in your home directory is named
`pfas-environment-cleanup`; adjust it if needed. Use a checkout containing
`scripts/hpc/dashboard.sbatch`. If transferring from
your local computer, copy into a dedicated directory without deleting remote
files. For example, run locally after creating the destination on PACE:

```bash
rsync -av \
  --exclude=.git/ --exclude=.venv/ --exclude=web/.venv/ \
  --exclude=node_modules/ --exclude=dist/ --exclude=.dashboard/ \
  --exclude=__pycache__/ --exclude='*.sif' --exclude=.container-build/ \
  --exclude=data/ --exclude=dft_runs/ --exclude=dft_cases/ --exclude=compounds/ \
  ./ jmccloskey30@login-ice.pace.gatech.edu:~/pfas-environment-cleanup/
```

The default creates fresh dashboard history on PACE. Existing local history,
prepared inputs, and RAM estimates are not transferred by that command. Moving
an existing artifact directory requires stopping its dashboard first, copying
the complete directory including SQLite state, and checking stored paths and
pending jobs before restart. Startup can resume waiting work automatically.
Do not run two dashboard servers against one artifact directory.

## 2. Install and build once on PACE

Use Python **3.12**, `uv`, and Node/npm available on PACE. Do not copy a local
virtual environment to the cluster. Module names depend on the current PACE
software stack; inspect `module spider` rather than assuming a module exists.
Run setup where PACE permits software installation/builds and network access;
use an interactive compute allocation for resource intensive builds.

From the PACE repository root:

```bash
uv sync --locked --no-default-groups --group web
npm --prefix web/frontend ci --no-audit --no-fund
PFAS_DATA_MODE=live npm --prefix web/frontend run build
.venv/bin/python -c 'import fastapi, uvicorn, rdkit, psutil'
```

With `PFAS_APPTAINER_IMAGE` configured, the chemistry tools come from your SIF;
the dashboard needs only the `web` group above. For native input preparation,
use the following sync command instead. Open Babel may require compiler/build
dependencies on the cluster:

```bash
uv sync --locked --no-default-groups --group web --group preparation
```

Rebuild after frontend changes and resync after dependency changes. The job
script uses the prepared environment directly and performs no installations.
An alternative compatible environment can be selected with an **absolute**
`PFAS_DASHBOARD_PYTHON` path before submission.

## 3. Start the dashboard job

From the PACE repository root:

```bash
sbatch --partition=ice-cpu --account=coc --qos=coc-ice \
  scripts/hpc/dashboard.sbatch
```

These scheduling values match the repository's ICE examples; use the partition,
account, and QOS authorized for your allocation. The script requests one node,
4 CPUs, 8 GiB, and 4 hours. Override resources on `sbatch` as needed, for example
`--mem=64G --cpus-per-task=16 --time=08:00:00` before the script path. Diagrams and preparation share this small allocation. The UI defaults QE and
RAM estimation to separate allocations, whose resource requests are independent
of the dashboard's CPU and memory budgets.

Inspect your job and its log (replace `123456` with the submitted ID):

```bash
squeue -j 123456
tail -f pfas-dashboard-123456.log
```

The log appears after the job starts. It prints the compute hostname and the
exact tunnel command. Wait for Uvicorn's startup message. If startup fails,
inspect the log before opening the tunnel.

Optional configuration, exported **before** `sbatch`:

```bash
export PFAS_DASHBOARD_PORT=8000
export PFAS_ARTIFACTS="$PWD/.dashboard"
export PFAS_PACE_LOGIN=login-ice.pace.gatech.edu
export PFAS_APPTAINER_IMAGE="$HOME/chemistry.sif"  # Set the actual shared SIF path.
export PFAS_EXECUTION=auto
# For native QE, use paths valid on the compute node:
# export PFAS_PW=/absolute/path/to/pw.x
# export PFAS_MPI=/absolute/path/to/mpirun
```

`PFAS_EXECUTION=auto` enables Slurm submission when `SLURM_JOB_ID` and all four
commands (`sbatch`, `squeue`, `sacct`, `scancel`) are available. Use `slurm` to
explicitly require scheduler support, or `local` to preserve workstation
execution behavior. The current mode is shown in the UI. A PACE hostname alone
does not enable submission.

Within a Slurm allocation, Apptainer uses the allocation's resources and does
not require a D-Bus session or create its own memory cgroup. Container image
identity and required chemistry tools are still checked. Outside Slurm, the
existing verified per-task Apptainer memory isolation remains required.

If native QE needs modules, load the site-supported modules before starting
the dashboard and set absolute `PFAS_PW`/`PFAS_MPI` paths. Submission preserves
runtime/module paths but removes inherited allocation, MPI-session, and
`SBATCH_*` resource overrides. Verify your toolchain in a compute allocation.

## 4. Forward the port to your local computer

In a **second terminal on your local computer**, paste the command printed in
the job log. Its form is:

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -J jmccloskey30@login-ice.pace.gatech.edu \
  -L 127.0.0.1:8000:127.0.0.1:8000 jmccloskey30@COMPUTE_NODE
```

Replace `COMPUTE_NODE` with the hostname from the job log, and keep the terminal
open. This uses the login host as a jump host and terminates SSH on the allocated
compute node, where the dashboard listens on loopback. Compute-node SSH must be
permitted for your running allocation. Authenticate as required by PACE; use the
campus VPN if your access requires it.

Open **http://127.0.0.1:8000** locally. One tunnel carries both the React page
and API calls. If local port 8000 is occupied, change only the first port in
`-L` to `8001` and browse to `http://127.0.0.1:8001`.

PACE's [interactive SSH port-forwarding documentation](https://docs.pace.gatech.edu/interactiveJobs/pf_3rd_party_ssh/)
describes its cluster tunnel workflow. If your allocation rejects compute-node
SSH, confirm the supported tunnel method with PACE before changing the server's
bind address. The dashboard has execution controls and no login screen; keep
its loopback binding.

## 5. Run work

Select **Apptainer** when your chemistry tools are in the SIF. Diagrams and
preparation run sequentially inside the dashboard allocation. Their memory
values are queue reservations, not separate enforced limits. One available CPU
is reserved for the web server; saved budgets are capped to the allocation.

For production QE, choose **Separate Slurm allocation**, review the process
count, memory reservation, walltime, partition, account, and QOS, then preview
and queue. Each calculation gets its own single-node job. It runs the image's
MPI with one thread per rank, using the existing single-node launcher design.
For native QE, multiple MPI ranks use a single `srun` step; this requires a
host QE/MPI build compatible with PACE's Slurm MPI configuration. Container QE
uses only the image's MPI launcher. No host/container MPI mixing or multi-node
execution is introduced.

**Memory sizing → RAM estimate + headroom** (50% by default) uses matching input,
pseudopotential, runtime, image, and process-count estimates. Preview identifies
the sizing basis, including any one-process fallback. Manual sizing uses the
entered reservation instead.

For bulk **Estimate RAM**, the default is **Measure 16, 4, and 1 processes in
one 16-CPU allocation**. Enter that allocation's memory reservation explicitly;
it cannot yet be inferred from the results you are about to measure. The default
walltime is four hours, editable before submission. Each initialization has its
own timeout (default 120 seconds); selected inputs are checked sequentially at
16, 4, and 1 MPI processes. An individual failure does not discard successful
checks or prevent the next one. Retry reviews skip matching successes and queue
only missing or failed checks. Uncheck the grouped option to use a single
process count. Individual candidate run controls also accept a process count.

Use **RAM display** to select 16, 4, or 1 processes. In Slurm mode the browser
shows total estimated job RAM; sorting and filtering use the selected count.
A `≤` label identifies maximum per-process RAM multiplied by the process count
when QE did not report a total. History retains both reported values. These are
QE initialization estimates, not measured peak memory or guaranteed safe
reservations. Multi-process checks must report the requested MPI count.

## 6. Monitor, stop, and reconnect

The queue shows Slurm job IDs, state, pending reason, attempt logs, and the
shared scheduler log. Monitoring polls Slurm every 15 seconds, independently of
browser refreshes. A scheduler outage or accounting delay retains the last
known state. A successful scheduler exit does not by itself confirm QE
convergence or a valid RAM estimate.

**Stop job** and **Cancel batch** use `scancel`. RAM checks sharing one allocation
are canceled together; completed results remain available. **Pause queue** stops
new submissions, while already submitted jobs remain under Slurm control.

If submission fails or the scheduler reply is lost, the dashboard records an
unresolved submission and searches by its unique job name. It never retries
`sbatch` automatically, which could create duplicate calculations. The original
submission error remains visible. After independently confirming Slurm rejected
the job, use **I confirmed Slurm rejected this submission**, then preview a retry.
That action is rejected if a matching job is visible in Slurm.

Closing the SSH tunnel disconnects your browser. Stopping the dashboard or
letting its walltime expire interrupts its local tasks, but **independently
submitted QE and estimation jobs continue**. To stop only the dashboard on PACE:

```bash
scancel DASHBOARD_JOB_ID
```

Restart with the same `PFAS_ARTIFACTS` directory to reconnect to submitted jobs
and import their results. Compute workers write logs and atomic result files on
shared storage; they never write the dashboard SQLite database. Do not start
two dashboards against the same artifact directory. Use the new dashboard job's
hostname when recreating the tunnel.

## PACE acceptance check

After installing these changes, validate the workflow in your own allocation:

1. Confirm the banner says Slurm, and prepare one candidate using Apptainer.
   No D-Bus/cgroup memory-isolation probe should run.
2. Select that candidate for RAM estimation, enter an adequate manual memory
   reservation, and submit the grouped 16/4/1 checks. Verify one Slurm job ID,
   three attempt records, and the MPI counts in the logs.
3. Submit a small QE calculation in a separate allocation. Restart the dashboard
   while it runs and confirm monitoring reconnects and results are imported.
4. Cancel a test job and confirm Slurm cancellation appears in the queue.

Actual PACE startup, runtime compatibility, and QE execution still require this
cluster validation. Local automated tests simulate Slurm and exercise workers
with test executables; they do not establish successful PACE calculations.
