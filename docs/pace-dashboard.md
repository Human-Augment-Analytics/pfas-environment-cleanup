# Run the dashboard on PACE

Run the backend and calculations in a Slurm compute allocation on PACE ICE,
then open the dashboard in your local computer's browser through SSH. The
browser uses your local computer; Python, preparation, QE, and saved artifacts
use PACE. This workflow does not add remote Slurm submission to the UI.

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

For input preparation, use the following sync command instead. Open Babel may
require compiler/build dependencies on the cluster:

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
8 CPUs, 16 GiB, and 4 hours. Override resources on `sbatch` as needed, for example
`--mem=64G --cpus-per-task=16 --time=08:00:00` before the script path. Everything
run through the dashboard shares this allocation, including QE child processes.

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
# For native QE, use paths valid on the compute node:
# export PFAS_PW=/absolute/path/to/pw.x
# export PFAS_MPI=/absolute/path/to/mpirun
```

If QE needs modules, add the site-supported module setup to the job script
before its final `exec`. Verify QE and MPI within an allocation. Browsing only
needs the web environment. Apptainer is optional; the dashboard's container
adapter requires verified per-job cgroup limits, which may be unavailable in a
Slurm allocation. Native execution uses the allocation's overall Slurm limits.

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

## 5. Run work and stop

Use the UI's **Native** runtime for tools installed on the compute node. Its
local queue executes there inside the dashboard allocation. Leave capacity for
the web server: the dashboard reserves one available CPU and caps saved budgets
on startup. Set queue CPU and memory budgets to fit your Slurm request. Large
QE calculations need a correspondingly sized dashboard allocation or separately
exported Slurm jobs.

Closing the tunnel disconnects your browser while the Slurm job continues.
To stop the dashboard and its allocation, run on PACE:

```bash
scancel 123456
```

The server also stops when walltime expires. State remains in `PFAS_ARTIFACTS`;
running attempts can be interrupted and need review/retry. A new job can use a
different compute node: copy its new tunnel command. Stop the old job before
reusing the same artifact directory. Stop any other server using that directory;
the script's lock only coordinates launches through this script.

Cluster startup, compute-node SSH access, and QE execution must still be
validated in your PACE allocation.
