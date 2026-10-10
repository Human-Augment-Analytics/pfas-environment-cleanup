#!/bin/bash
# Exp 002 step 1: relax isolated TFA with the pipeline's "cluster" preset
# (ecutwfc=60, ecutrho=600). Calls run_adsorption_case.py directly with the
# existing qe_pfas env instead of run_dft_workflow.sh, which runs
# `conda env update --prune` on the shared env other experiments use.
#SBATCH --job-name=exp02_tfa_relax
#SBATCH --partition=ice-cpu
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=test_runs/logs/relax-%j.out
#SBATCH --error=test_runs/logs/relax-%j.err
set -euo pipefail

ROOT="${HOME}/pfas-exp02"
cd "$ROOT"

module load anaconda3
module load quantum-espresso/7.3
module load openmpi
eval "$(conda shell.bash hook)"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

conda run --no-capture-output -p "${HOME}/.conda/envs/qe_pfas" \
  python qespresso_pipeline/run_adsorption_case.py \
    --case-name tfa_relax \
    --adsorbent-name none \
    --pfas-name tfa \
    --adsorbent-source smiles \
    --pfas-smiles 'FC(F)(F)C(=O)O' \
    --system-type molecule \
    --mode cluster \
    --compound-root test_runs/compounds \
    --workdir test_runs/dft_cases \
    --pseudo-dir qespresso_pipeline/Pseudopotentials \
    --pw-command "mpirun -np ${SLURM_NTASKS} pw.x" \
    --skip-ads --skip-complex

echo "[exp02] relax finished at $(date)"
