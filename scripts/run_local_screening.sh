#!/bin/bash

CSV_FILE="molecular_adsorbents_smiles.csv"
PFAS_SMILES_TARGET="FC(F)(F)C(=O)O"

tail -n +2 "$CSV_FILE" | while IFS=',' read -r ADS_NAME ADS_SMILES; do
    
    export CASE_NAME="${ADS_NAME}_TFA"
    export ADSORBENT_NAME="$ADS_NAME"
    export ADSORBENT_SMILES="$ADS_SMILES"
    export PFAS_NAME="TFA"
    export PFAS_SMILES="$PFAS_SMILES_TARGET"
    export MODE="lowmem" 
    export SYSTEM_TYPE="molecule"
    
    bash run_dft_workflow.sh
    
    grep -H "Adsorption Energy" dft_cases/$CASE_NAME/results.json >> master_results.txt
done