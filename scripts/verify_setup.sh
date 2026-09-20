#!/bin/bash
# One-command setup verification for this repository. Run it after cloning
# (or whenever something behaves oddly): it checks the environment a new lab
# member otherwise has to reverse-engineer from the README, printing one
# PASS / SKIP / FAIL line per check and exiting non-zero if anything failed.
# It only inspects the environment and the checkout - it changes nothing.
#
# Usage:
#   bash scripts/verify_setup.sh

set -euo pipefail

PASSED=0
SKIPPED=0
FAILED=0

note() { echo "[verify] $*"; }
pass() { PASSED=$((PASSED + 1)); echo "PASS  $*"; }
skip() { SKIPPED=$((SKIPPED + 1)); echo "SKIP  $*"; }
fail() { FAILED=$((FAILED + 1)); echo "FAIL  $*"; }

# Locate the repository root the same way run_dft_workflow.sh does: walk up
# from this script until the qespresso_pipeline/ directory appears.
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(dirname "$SCRIPT_PATH")"
while [[ "$PROJECT_ROOT" != "/" && ! -d "$PROJECT_ROOT/qespresso_pipeline" ]]; do
  PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
done
if [[ ! -d "$PROJECT_ROOT/qespresso_pipeline" ]]; then
  note "cannot find the repository root (no qespresso_pipeline/ directory above $SCRIPT_PATH)"
  exit 1
fi
note "repository root: $PROJECT_ROOT"

# The conda prefix into which run_dft_workflow.sh builds the DFT toolchain
# (pw.x, obabel, cif2cell) from qe_environment.yaml.
QE_PREFIX="${HOME}/.conda/envs/qe_pfas"

echo
echo "conda and the 'pfas' environment"
echo "-------------------------------"
if command -v conda >/dev/null 2>&1; then
  pass "conda is available on PATH"
  if conda env list 2>/dev/null | grep -Eq "^pfas[[:space:]]"; then
    pass "conda environment 'pfas' exists (from environment.yaml)"
  else
    fail "conda environment 'pfas' is missing - create it with: conda env create -f environment.yaml"
  fi
  if [[ "${CONDA_DEFAULT_ENV:-}" == "pfas" ]]; then
    pass "'pfas' is the active environment"
  else
    note "active environment is '${CONDA_DEFAULT_ENV:-none}' - run repo scripts with: conda run -n pfas python <script> (or conda activate pfas)"
  fi
else
  fail "conda was not found on PATH - install Miniconda or Anaconda first"
fi

echo
echo "DFT toolchain (pw.x, obabel, cif2cell)"
echo "--------------------------------------"
for tool in pw.x obabel cif2cell; do
  where=""
  # type -P searches PATH for executable files only (it ignores shell
  # functions and aliases), unlike command -v.
  loc="$(type -P "$tool" 2>/dev/null || true)"
  if [[ -n "$loc" ]]; then
    where="on PATH ($loc)"
  elif [[ -x "$QE_PREFIX/bin/$tool" ]]; then
    where="in the qe environment ($QE_PREFIX/bin/$tool)"
  fi
  if [[ -n "$where" ]]; then
    pass "$tool is available $where"
  elif [[ "$tool" == "pw.x" ]] && [[ "$(type -t pw.x 2>/dev/null || true)" == "function" ]]; then
    skip "pw.x is defined only as a shell function (login-node module state) - it is usable inside SLURM jobs, not interactively; submit via scripts/dft_wrapper.py instead"
  elif [[ "$tool" == "pw.x" ]] && command -v module >/dev/null 2>&1; then
    skip "pw.x is not on PATH - on the PACE-ICE login nodes this is expected: the quantum-espresso module is restricted to compute-node jobs. Submit via scripts/dft_wrapper.py or sbatch instead of running pw.x by hand"
  else
    skip "$tool is not on PATH - it is provided by the qe environment built from qe_environment.yaml; scripts/run_dft_workflow.sh creates and updates that environment automatically (prefix: $QE_PREFIX)"
  fi
done

echo
echo "VESTA (optional, legacy manual path)"
echo "-----------------------------------"
if [[ -n "${VESTA_PATH:-}" ]]; then
  if [[ -x "$VESTA_PATH" ]]; then
    pass "VESTA_PATH points at an executable ($VESTA_PATH)"
  else
    fail "VESTA_PATH is set but '$VESTA_PATH' is not an executable file - point it at the VESTA binary or unset it (the automated pipeline does not need VESTA)"
  fi
elif command -v VESTA >/dev/null 2>&1; then
  pass "VESTA found on PATH"
else
  skip "VESTA not found - optional; only the legacy manual .mol to .cif step uses it (export VESTA_PATH=/path/to/VESTA if you need it)"
fi

# Read lines with plain bash rather than grep: some grep builds (notably
# Git-for-Windows) strip the carriage return from CRLF files before
# matching, which would hide the very thing this check looks for.
has_crlf() {
  local line
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" == *$'\r'* ]] && return 0
  done < "$1"
  return 1
}

echo
echo "Checkout line endings"
echo "--------------------"
crlf_files=""
for f in "$PROJECT_ROOT"/scripts/*.sh; do
  [[ -f "$f" ]] || continue
  if has_crlf "$f"; then
    crlf_files="${crlf_files}$(basename "$f") "
  fi
done
if [[ -z "$crlf_files" ]]; then
  pass "all scripts/*.sh use Unix (LF) line endings"
else
  fail "Windows (CRLF) line endings found in: $crlf_files - fix with 'dos2unix scripts/*.sh' or re-clone with core.autocrlf=input; these scripts fail on the cluster as CRLF"
fi

echo
note "summary: $PASSED passed, $SKIPPED skipped, $FAILED failed"
if [[ "$FAILED" -eq 0 ]]; then
  note "no blocking problems found"
  exit 0
fi
note "fix the FAIL lines above, then re-run"
exit 1
