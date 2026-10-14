#!/usr/bin/env bash
set -euo pipefail

echo "[job] starting at $(date)"
echo "[job] host: $(hostname)"
echo "[job] pwd: $(pwd)"

# Backend switch: cpu (default, byte-identical legacy behavior) or gpu
# (PACE-ICE ice-gpu partition via quantum-espresso/7.1-cuda in apptainer,
# proven by canary job 6239784; see plans/gpu_acceleration_plan.md).
PFAS_DFT_BACKEND="${PFAS_DFT_BACKEND:-cpu}"

if command -v module &> /dev/null; then
    if [[ "$PFAS_DFT_BACKEND" == "gpu" ]]; then
        # GPU track (canary-proven 2026-10-13): QE 7.1-cuda ships its own
        # toolchain (no gcc/openmpi prerequisites), runs inside the QE
        # apptainer image via srun, and needs apptainer (either in the
        # base image or from its module).
        module purge
        module load anaconda3
        module load quantum-espresso/7.1-cuda
        command -v apptainer &> /dev/null || module load apptainer 2>/dev/null || true
    else
        # PACE-ICE (RHEL9 lmod hierarchy), verified 2026-10-02: quantum-espresso/7.3
        # lives under the openmpi/4.1.5 + gcc/12.3.0 hierarchy and only becomes
        # loadable once both are active. The login default (mvapich2) cannot be
        # auto-swapped, and a bare `module load openmpi` fails under this hierarchy
        # (which would abort under set -e). Follow the sequence documented in the
        # QE 7.3 module file itself; anaconda3 must come BEFORE QE so QE's pw.x and
        # Open MPI stay ahead of anaconda's shadowing mpirun in PATH.
        module purge
        module load gcc/12.3.0
        module load openmpi/4.1.5
        module load anaconda3
        module load quantum-espresso/7.3
    fi
fi

if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
else
    echo "[error] conda command not found. Please ensure Anaconda/Miniconda is installed locally and in your PATH."
    exit 1
fi

# Locate the repo root. The script may live in scripts/ (repo checkout), at
# the repo root (older deployments), or be invoked through a symlink at the
# repo root, so resolve the physical script path and walk up instead of
# assuming a fixed relative position. Anchor the walk-up on the root-only
# qespresso_pipeline/ directory; it exists only at the repo root, so it
# reliably marks the project root.
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(dirname "$SCRIPT_PATH")"
while [[ "$PROJECT_ROOT" != "/" && ! -d "$PROJECT_ROOT/qespresso_pipeline" ]]; do
    PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
done
if [[ ! -f "$PROJECT_ROOT/qe_environment.yaml" ]]; then
    echo "[error] qe_environment.yaml not found at $SCRIPT_PATH or any parent directory."
    echo "[error] Deploy the full repository (see README) and run this script from within it."
    exit 1
fi
cd "$PROJECT_ROOT"

ENV_YAML="$PROJECT_ROOT/qe_environment.yaml"
ENV_PREFIX="${HOME}/.conda/envs/qe_pfas"

if [[ ! -f "$ENV_YAML" ]]; then
    echo "[error] Missing environment file: $ENV_YAML"
    exit 1
fi

# Serialize the env op across simultaneously starting jobs: two jobs that
# begin in the same second race on the shared prefix/pkgs cache and the
# loser's repodata collection crashes (dry-run job 6252238, 2026-10-13).
ENV_LOCK="$HOME/.conda/qe_pfas.env.lock"
mkdir -p "$HOME/.conda/envs"
if [[ ! -d "$ENV_PREFIX" ]]; then
    echo "[env] creating env at $ENV_PREFIX from $ENV_YAML"
    flock "$ENV_LOCK" conda env create -p "$ENV_PREFIX" -f "$ENV_YAML"
else
    echo "[env] updating env at $ENV_PREFIX from $ENV_YAML"
    flock "$ENV_LOCK" conda env update -p "$ENV_PREFIX" -f "$ENV_YAML" --prune
fi

echo "[env] python:"
conda run -p "$ENV_PREFIX" python --version
conda run -p "$ENV_PREFIX" python -c "import sys; print(sys.executable)"

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
MPI_TASKS="${SLURM_NTASKS:-1}"

CASE_NAME="${CASE_NAME:?CASE_NAME not set}"
ADSORBENT_NAME="${ADSORBENT_NAME:?ADSORBENT_NAME not set}"
PFAS_NAME="${PFAS_NAME:?PFAS_NAME not set}"

ADSORBENT_SOURCE="${ADSORBENT_SOURCE:-smiles}"
SYSTEM_TYPE="${SYSTEM_TYPE:-molecule}"
MODE="${MODE:-cluster}"

ADSORBENT_SMILES="${ADSORBENT_SMILES:-}"
ADSORBENT_CIF="${ADSORBENT_CIF:-}"
PFAS_SMILES="${PFAS_SMILES:-}"
PFAS_ENERGY_RY="${PFAS_ENERGY_RY:-}"

PSEUDO_DIR="${PSEUDO_DIR:-$PROJECT_ROOT/qespresso_pipeline/Pseudopotentials}"
if [[ "$PFAS_DFT_BACKEND" == "gpu" ]]; then
    # QE_7_1_SIF is exported by the quantum-espresso/7.1-cuda module; fail
    # fast here (not mid-leg inside pw.x) if the module chain ever changes.
    QE_7_1_SIF="${QE_7_1_SIF:?QE_7_1_SIF not set: quantum-espresso/7.1-cuda module did not export it (GPU backend)}"
    # One container pw.x per job, placed on the allocated GPU by srun;
    # run_adsorption_case.py appends "-in <file>" to this prefix. The inner
    # quotes survive shlex.split on the python side even if the SIF path
    # ever contains spaces. --bind /storage: the image cannot see the
    # campaign root on /storage otherwise.
    PW_COMMAND="${PW_COMMAND:-srun --mpi=pmi2 apptainer exec --nv --bind /storage \"$QE_7_1_SIF\" pw.x}"
else
    PW_COMMAND="${PW_COMMAND:-mpirun -np $MPI_TASKS pw.x}"
fi
COMPOUND_ROOT="${COMPOUND_ROOT:-$PROJECT_ROOT/compounds}"
WORKDIR="${WORKDIR:-$PROJECT_ROOT/dft_cases}"

ARGS=(
  --case-name "$CASE_NAME"
  --adsorbent-name "$ADSORBENT_NAME"
  --pfas-name "$PFAS_NAME"
  --adsorbent-source "$ADSORBENT_SOURCE"
  --system-type "$SYSTEM_TYPE"
  --mode "$MODE"
  --compound-root "$COMPOUND_ROOT"
  --workdir "$WORKDIR"
  --pseudo-dir "$PSEUDO_DIR"
  --pw-command "$PW_COMMAND"
)

if [[ -n "$ADSORBENT_SMILES" ]]; then
  ARGS+=(--adsorbent-smiles "$ADSORBENT_SMILES")
fi
if [[ -n "$ADSORBENT_CIF" ]]; then
  ARGS+=(--adsorbent-cif "$ADSORBENT_CIF")
fi
if [[ -n "$PFAS_SMILES" ]]; then
  ARGS+=(--pfas-smiles "$PFAS_SMILES")
fi
if [[ -n "$PFAS_ENERGY_RY" ]]; then
  ARGS+=(--pfas-energy-ry "$PFAS_ENERGY_RY")
fi

if [[ "${SKIP_ADS:-0}" == "1" ]]; then ARGS+=(--skip-ads); fi
if [[ "${SKIP_PFAS:-0}" == "1" ]]; then ARGS+=(--skip-pfas); fi
if [[ "${SKIP_COMPLEX:-0}" == "1" ]]; then ARGS+=(--skip-complex); fi

conda run -p "$ENV_PREFIX" python qespresso_pipeline/run_adsorption_case.py "${ARGS[@]}"

echo "[job] finished successfully at $(date)"
