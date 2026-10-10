#!/bin/bash
# Exp 002 step 3: single-point SCF on relaxed TFA at one cutoff.
# Usage: sbatch --export=ALL,ECUT=40 experiments/exp002_ecutwfc/scf_tfa.sbatch
#SBATCH --job-name=exp02_tfa_scf
#SBATCH --partition=ice-cpu
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=test_runs/logs/scf-%j.out
#SBATCH --error=test_runs/logs/scf-%j.err
set -euo pipefail

: "${ECUT:?set ECUT via --export}"
cd "${HOME}/pfas-exp02/test_runs/scf/ecut${ECUT}"

module load quantum-espresso/7.3
module load openmpi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

mpirun -np "${SLURM_NTASKS}" pw.x -in tfa_scf.in > tfa_scf.out 2> tfa_scf.err
grep '^!' tfa_scf.out | tail -1
