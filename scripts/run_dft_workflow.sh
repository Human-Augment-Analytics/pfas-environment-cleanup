#!/usr/bin/env bash
set -euo pipefail

echo "[job] starting at $(date)"
echo "[job] host: $(hostname)"
echo "[job] pwd: $(pwd)"

module load anaconda3
eval "$(conda shell.bash hook)"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

ENV_YAML="$PROJECT_ROOT/qe_environment.yaml"
ENV_PREFIX="${HOME}/.conda/envs/qe_pfas"

if [[ ! -f "$ENV_YAML" ]]; then
    echo "[error] Missing environment file: $ENV_YAML"
    exit 1
fi

if [[ ! -d "$ENV_PREFIX" ]]; then
    echo "[env] creating env at $ENV_PREFIX from $ENV_YAML"
    conda env create -p "$ENV_PREFIX" -f "$ENV_YAML"
else
    echo "[env] updating env at $ENV_PREFIX from $ENV_YAML"
    conda env update -p "$ENV_PREFIX" -f "$ENV_YAML" --prune
fi

echo "[env] python:"
conda run -p "$ENV_PREFIX" python --version
conda run -p "$ENV_PREFIX" python -c "import sys; print(sys.executable)"

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

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
PW_COMMAND="${PW_COMMAND:-pw.x}"
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

echo "[job] finished at $(date)"