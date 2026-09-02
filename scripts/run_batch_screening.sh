#!/bin/bash
#SBATCH --job-name=pfas_screen
#SBATCH --output=logs/screen_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=24:00:00
#SBATCH --array=2-101  # Adjust 51 to match the total number of lines in your CSV

CSV_FILE="molecular_adsorbents_smiles.csv"

ROW=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CSV_FILE")
ADS_NAME=$(echo "$ROW" | awk -F',' '{print $1}')
ADS_SMILES=$(echo "$ROW" | awk -F',' '{print $2}')

export CASE_NAME="${ADS_NAME}_TFA"
export ADSORBENT_NAME="$ADS_NAME"
export ADSORBENT_SMILES="$ADS_SMILES"
export PFAS_NAME="TFA"
export PFAS_SMILES="FC(F)(F)C(=O)O" 
export MODE="production"
export SYSTEM_TYPE="molecule"
export MPI_TASKS=$SLURM_NTASKS

bash run_dft_workflow.sh