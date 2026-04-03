#!/usr/bin/env bash
set -euo pipefail

echo "[job] starting at $(date)"
echo "[job] host: $(hostname)"
echo "[job] pwd: $(pwd)"

module load anaconda3
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "[error] conda not found in PATH"
    exit 1
fi

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
    echo "[env] reusing existing env at $ENV_PREFIX"
fi

echo "[env] python:"
conda run -p "$ENV_PREFIX" python --version
conda run -p "$ENV_PREFIX" python -c "import sys; print(sys.executable)"

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "[job] SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unset}"
echo "[job] OMP_NUM_THREADS: $OMP_NUM_THREADS"

CASE_NAME="${CASE_NAME:-imidazolium_tfa}"
ADSORBENT_NAME="${ADSORBENT_NAME:-imidazolium}"
PFAS_NAME="${PFAS_NAME:-tfa}"

ADSORBENT_SMILES="${ADSORBENT_SMILES:-}"
PFAS_SMILES="${PFAS_SMILES:-}"
PFAS_ENERGY_RY="${PFAS_ENERGY_RY:-}"

PSEUDO_DIR="${PSEUDO_DIR:-$PROJECT_ROOT/qespresso_pipeline/Pseudopotentials}"
PW_COMMAND="${PW_COMMAND:-pw.x}"

COMPOUND_ROOT="${COMPOUND_ROOT:-$PROJECT_ROOT/compounds}"
WORKDIR="${WORKDIR:-$PROJECT_ROOT/dft_cases}"

SKIP_ARGS=()
if [[ "${SKIP_ADS:-0}" == "1" ]]; then
    SKIP_ARGS+=(--skip-ads)
fi
if [[ "${SKIP_PFAS:-0}" == "1" ]]; then
    SKIP_ARGS+=(--skip-pfas)
fi
if [[ "${SKIP_COMPLEX:-0}" == "1" ]]; then
    SKIP_ARGS+=(--skip-complex)
fi

EXTRA_ARGS=()
if [[ -n "$ADSORBENT_SMILES" ]]; then
    EXTRA_ARGS+=(--adsorbent-smiles "$ADSORBENT_SMILES")
fi
if [[ -n "$PFAS_SMILES" ]]; then
    EXTRA_ARGS+=(--pfas-smiles "$PFAS_SMILES")
fi
if [[ -n "$PFAS_ENERGY_RY" ]]; then
    EXTRA_ARGS+=(--pfas-energy-ry "$PFAS_ENERGY_RY")
fi

echo "[job] PW command: $PW_COMMAND"
echo "[job] CASE_NAME: $CASE_NAME"
echo "[job] ADSORBENT_NAME: $ADSORBENT_NAME"
echo "[job] PFAS_NAME: $PFAS_NAME"
echo "[job] SKIP_ADS=${SKIP_ADS:-0}"
echo "[job] SKIP_PFAS=${SKIP_PFAS:-0}"
echo "[job] SKIP_COMPLEX=${SKIP_COMPLEX:-0}"
echo "[job] PFAS_ENERGY_RY=${PFAS_ENERGY_RY:-unset}"
echo "[job] skip args: ${SKIP_ARGS[*]:-(none)}"

conda run -p "$ENV_PREFIX" python qespresso_pipeline/run_adsorption_case.py \
  --case-name "$CASE_NAME" \
  --adsorbent-name "$ADSORBENT_NAME" \
  --pfas-name "$PFAS_NAME" \
  --compound-root "$COMPOUND_ROOT" \
  --workdir "$WORKDIR" \
  --pseudo-dir "$PSEUDO_DIR" \
  --mode cluster \
  --pw-command "$PW_COMMAND" \
  "${SKIP_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"

echo "[job] finished at $(date)"