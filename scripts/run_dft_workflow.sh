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

# Project root = directory this script lives in
# PROJECT_ROOT="$PWD"
# cd "$PROJECT_ROOT"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

ENV_YAML="$PROJECT_ROOT/qe_environment.yaml"
ENV_PREFIX="${HOME}/.conda/envs/qe_pfas"

if [[ ! -f "$ENV_YAML" ]]; then
    echo "[error] Missing environment file: $ENV_YAML"
    exit 1
fi

# Create env once, reuse on later runs
if [[ ! -d "$ENV_PREFIX" ]]; then
    echo "[env] creating env at $ENV_PREFIX from $ENV_YAML"
    conda env create -p "$ENV_PREFIX" -f "$ENV_YAML"
else
    echo "[env] reusing existing env at $ENV_PREFIX"
fi

conda activate "$ENV_PREFIX"

echo "[env] python: $(which python)"
python --version

# Keep threading conservative on shared cluster
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Use SLURM CPU allocation if present, otherwise default to 4 ranks
MPI_RANKS="${SLURM_CPUS_PER_TASK:-4}"

echo "[job] MPI ranks: $MPI_RANKS"
echo "[job] OMP_NUM_THREADS: $OMP_NUM_THREADS"

# Adjust these paths/arguments as needed
CASE_NAME="imidazolium_tfa_demo"
ADSORBENT_SMILES='CCCCCCCCC=CCCCCCCCC1=NCC[N+]1(C)CCNC(=O)CCCCCCCC=CCCCCCCCC.COS(=O)(=O)[O-]'
PFAS_SMILES='FC(F)(F)C(=O)O'
PSEUDO_DIR="$PROJECT_ROOT/qespresso_pipeline/Pseudopotentials"

# Prefer srun on SLURM, fallback to mpirun, fallback to plain execution
# Threading setup: match the wrapper's --cpus-per-task allocation
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "[job] SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unset}"
echo "[job] OMP_NUM_THREADS: $OMP_NUM_THREADS"

PW_COMMAND="pw.x"
echo "[job] PW command: $PW_COMMAND"

echo "[job] PW command: $PW_COMMAND"

python qespresso_pipeline/run_adsorption_case.py \
  --case-name "$CASE_NAME" \
  --adsorbent-smiles "$ADSORBENT_SMILES" \
  --pfas-smiles "$PFAS_SMILES" \
  --pseudo-dir "$PSEUDO_DIR" \
  --mode cluster \
  --pw-command "$PW_COMMAND"

echo "[job] finished at $(date)"