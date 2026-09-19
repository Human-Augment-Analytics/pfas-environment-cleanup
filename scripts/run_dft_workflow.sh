#!/usr/bin/env bash
set -euo pipefail

echo "[job] starting at $(date)"
echo "[job] host: $(hostname)"
echo "[job] pwd: $(pwd)"

if command -v module &> /dev/null; then
    module load quantum-espresso/7.3 python/3.12.5
fi

# Locate the repo root. The script may live in scripts/ (repo checkout), at
# the repo root (older deployments), or be invoked through a symlink at the
# repo root, so resolve the physical script path and walk up instead of
# assuming a fixed relative position. Anchor the walk-up on the root-only
# qespresso_pipeline/ directory.
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(dirname "$SCRIPT_PATH")"
while [[ "$PROJECT_ROOT" != "/" && ! -d "$PROJECT_ROOT/qespresso_pipeline" ]]; do
    PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
done
if [[ ! -f "$PROJECT_ROOT/requirements-qe.txt" ]]; then
    echo "[error] requirements-qe.txt not found at $SCRIPT_PATH or any parent directory."
    echo "[error] Deploy the full repository (see README) and run this script from within it."
    exit 1
fi
cd "$PROJECT_ROOT"

ENV_DIR="$PROJECT_ROOT/.env-qe"
if [[ ! -x "$ENV_DIR/bin/python" || ! -f "$ENV_DIR/bin/activate" ]]; then
    echo "[error] Create .env-qe with uv before submitting jobs; see README's DFT setup."
    exit 1
fi
# Activation also puts obabel and cif2cell on PATH for Python subprocesses.
source "$ENV_DIR/bin/activate"

echo "[env] python: $(command -v python)"
python --version

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
PW_COMMAND="${PW_COMMAND:-mpirun -np $MPI_TASKS pw.x}"
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

python qespresso_pipeline/run_adsorption_case.py "${ARGS[@]}"

echo "[job] finished successfully at $(date)"
