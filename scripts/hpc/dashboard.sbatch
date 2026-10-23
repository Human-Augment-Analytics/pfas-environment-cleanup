#!/usr/bin/env bash
#SBATCH --job-name=pfas-dashboard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=pfas-dashboard-%j.log

# Submit from the repository root; scheduling/account flags belong on sbatch.
set -euo pipefail

if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo "Submit from the repository root with: sbatch scripts/hpc/dashboard.sbatch" >&2
    exit 1
fi
cd "${SLURM_SUBMIT_DIR:?Missing Slurm submission directory}"
if [[ ! -f dashboard/server.py ]]; then
    echo "Submit this job from the PFAS repository root." >&2
    exit 1
fi

dashboard_python=${PFAS_DASHBOARD_PYTHON:-$PWD/.venv/bin/python}
dashboard_port=${PFAS_DASHBOARD_PORT:-8000}
if [[ ! $dashboard_port =~ ^[1-9][0-9]{3,4}$ ]] || (( dashboard_port < 1024 || dashboard_port > 65535 )); then
    echo "PFAS_DASHBOARD_PORT must be an integer from 1024 to 65535." >&2
    exit 1
fi
if [[ ! -x $dashboard_python || ! -f web/frontend/dist/index.html ]]; then
    echo "Python environment or frontend build missing. See docs/pace-dashboard.md." >&2
    exit 1
fi

export PFAS_ARTIFACTS=${PFAS_ARTIFACTS:-$PWD/.dashboard}
mkdir -p "$PFAS_ARTIFACTS"
# Prevent two launches of this script from recovering/writing the same queue.
exec 9>"$PFAS_ARTIFACTS/pace-dashboard.lock"
if ! flock -n 9; then
    echo "Another dashboard job is using $PFAS_ARTIFACTS." >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
dashboard_node=$(hostname -f)
dashboard_user=$(id -un)
dashboard_login=${PFAS_PACE_LOGIN:-login-ice.pace.gatech.edu}
echo "Dashboard job: $SLURM_JOB_ID on $dashboard_node"
echo "Artifacts: $PFAS_ARTIFACTS"
echo "Run in a terminal on your LOCAL computer (keep it open):"
printf 'ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -J %s@%s -L 127.0.0.1:8000:127.0.0.1:%s %s@%s\n' \
    "$dashboard_user" "$dashboard_login" "$dashboard_port" "$dashboard_user" "$dashboard_node"
echo "Then open http://127.0.0.1:8000 after Uvicorn reports startup complete."
echo "Stop with: scancel $SLURM_JOB_ID"

# Use one worker: each app instance owns a persistent task manager.
# Dependencies and the live frontend must be prepared before submission.
exec "$dashboard_python" -m uvicorn dashboard.server:create_app --factory \
    --host 127.0.0.1 --port "$dashboard_port" --workers 1
