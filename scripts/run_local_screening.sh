#!/bin/bash

# Resolve the repo root the same way scripts/run_dft_workflow.sh does: this
# script may be invoked from the repo root, from scripts/, or via
# "bash scripts/run_local_screening.sh", so walk up from this script until the
# repo root is found instead of assuming a fixed working directory. Anchor the
# walk-up on the root-only qespresso_pipeline/ directory: qe_environment.yaml
# is also tracked under scripts/ (audit finding B3) and must not anchor root
# detection.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
while [[ "$PROJECT_ROOT" != "/" && ! -d "$PROJECT_ROOT/qespresso_pipeline" ]]; do
    PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
done
if [[ ! -f "$PROJECT_ROOT/qe_environment.yaml" ]]; then
    echo "[error] qe_environment.yaml not found at $SCRIPT_DIR or any parent directory." >&2
    echo "[error] Deploy the full repository (see README) and run this script from within it." >&2
    exit 1
fi

CSV_FILE="$PROJECT_ROOT/scripts/molecular_adsorbents_smiles.csv"
WORKFLOW_SCRIPT="$PROJECT_ROOT/scripts/run_dft_workflow.sh"
MASTER_RESULTS="$PROJECT_ROOT/scripts/master_results.txt"
PFAS_SMILES_TARGET="FC(F)(F)C(=O)O"

if [[ ! -f "$CSV_FILE" ]]; then
    echo "[error] CSV file not found: $CSV_FILE" >&2
    exit 1
fi

# CSV header: ID,Name,SMILES,Category. Parse with python's csv module because
# several fields are quoted and contain commas (e.g. row 045's Name), which
# breaks naive IFS=',' read splitting (same approach as run_batch_screening.sh).
# Emit one "Name\tSMILES" pair per data row.
PARSED=$(python3 -c '
import csv, sys
with open(sys.argv[1], newline="") as f:
    rows = list(csv.reader(f))
out = []
for line_no, row in enumerate(rows, start=1):
    if line_no == 1 or not row:
        continue  # header line / blank line
    if len(row) < 3:
        sys.exit(f"no data columns at CSV line {line_no}")
    out.append(row[1] + "\t" + row[2])
if not out:
    sys.exit("no data rows found in CSV")
sys.stdout.write("\n".join(out))
' "$CSV_FILE") || exit 1

while IFS=$'\t' read -r ADS_NAME ADS_SMILES; do

    if [[ -z "$ADS_NAME" || -z "$ADS_SMILES" ]]; then
        echo "[error] could not extract Name/SMILES from a $CSV_FILE row: '$ADS_NAME'" >&2
        continue
    fi

    # Filesystem-safe adsorbent name: it becomes part of the case directory and
    # compounds/adsorbents/ path, and several Names contain spaces, slashes and
    # commas (e.g. "Crown Ether / 18-Crown-6").
    ADS_NAME=$(printf '%s' "$ADS_NAME" | tr -c 'A-Za-z0-9._-' '_' | sed -E 's/_+/_/g; s/^_+//; s/_+$//')

    export CASE_NAME="${ADS_NAME}_TFA"
    export ADSORBENT_NAME="$ADS_NAME"
    export ADSORBENT_SMILES="$ADS_SMILES"
    export PFAS_NAME="TFA"
    export PFAS_SMILES="$PFAS_SMILES_TARGET"
    export MODE="lowmem"
    export SYSTEM_TYPE="molecule"

    bash "$WORKFLOW_SCRIPT"

    # Harvest the adsorption energy from the file the pipeline actually writes
    # (results.json; the energy line the old grep targeted is stdout-only). The
    # writer skips results.json entirely when any total energy is zero/None,
    # so a missing file must not abort the sweep: warn and keep the Outputs
    # directories as debugging evidence instead of deleting them.
    RESULTS_JSON="$PROJECT_ROOT/dft_cases/$CASE_NAME/results.json"
    if python3 -c '
import json, sys
results_json, master_results, case_name = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    with open(results_json) as f:
        results = json.load(f)
except FileNotFoundError:
    sys.exit(f"results.json not found for case {case_name}: {results_json}")
adsorption_ev = (results.get("energies_ev") or {}).get("adsorption")
if adsorption_ev is None:
    sys.exit(f"no energies_ev.adsorption in {results_json}")
name = results.get("case_name") or case_name
with open(master_results, "a") as f:
    f.write(f"case={name} adsorption_energy_ev={adsorption_ev}\n")
' "$RESULTS_JSON" "$MASTER_RESULTS" "$CASE_NAME"; then
        rm -rf "$PROJECT_ROOT"/dft_cases/"$CASE_NAME"/*/Outputs/
    else
        echo "[warn] harvest failed for $CASE_NAME; keeping dft_cases/$CASE_NAME/*/Outputs/ for debugging" >&2
    fi
done <<< "$PARSED"
