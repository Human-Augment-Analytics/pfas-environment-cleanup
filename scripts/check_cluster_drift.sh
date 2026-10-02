#!/bin/bash
# Compare the copy of this repository on the cluster against this checkout.
# The SLURM jobs submitted by dft_wrapper.py run scripts/run_dft_workflow.sh
# from the cluster copy, so results are only reproducible when the two match.
# This script prints a diff of every file whose contents differ (ignoring
# line-ending differences) and exits non-zero on any drift, so it can gate
# a deploy or a batch of job submissions.
#
# It needs no counterpart installed on the cluster: it ships itself over
# SSH (ssh runs "bash -s" with this very file on standard input), so both
# manifests are produced by the same code.
#
# Usage:
#   bash scripts/check_cluster_drift.sh --user <gatech-user>
#
# Options (same flags and defaults as scripts/dft_wrapper.py):
#   --user          Gatech username for the SSH target (required)
#   --cluster       Login node          (default: login-ice.pace-gatech.edu)
#   --cluster-root  Path on the cluster (default: /storage/ice-shared/cs8903onl/mussmann-pfas)
#
# Exit codes:
#   0  the copies match
#   1  drift detected (diff printed above)
#   2  error (bad arguments, no repository root, or SSH failed)

set -euo pipefail

CLUSTER="login-ice.pace.gatech.edu"
CLUSTER_ROOT="/storage/ice-shared/cs8903onl/mussmann-pfas"
USER_NAME=""
PRINT_MANIFEST=0

note() { echo "[drift] $*"; }

usage_error() {
  echo "error: $1" >&2
  echo "usage: bash scripts/check_cluster_drift.sh --user <gatech-user> [--cluster HOST] [--cluster-root PATH]" >&2
  exit 2
}

# One "md5  relative-path" line per deployable file, sorted by path. Line
# endings are normalized (tr -d '\r') before hashing so a Windows checkout
# does not show up as drift, and the bulk data / run / cache directories
# that deploy_to_cluster.sh excludes are pruned here too, so the manifest
# covers exactly the set rsync deploys.
print_manifest() {
  find . \
    \( -name .git -o -name data -o -name dft_runs -o -name dft_cases -o -name compounds \
       -o -name logs -o -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name .venv \) -prune \
    -o -type f ! -name '*.pyc' ! -name master_results.txt -print \
  | while IFS= read -r f; do
      rel="${f#./}"
      hash="$(tr -d '\r' < "$f" | md5sum)"
      printf '%s  %s\n' "${hash%% *}" "$rel"
    done \
  | LC_ALL=C sort -k 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user)
      [[ $# -ge 2 ]] || usage_error "--user needs a value"
      USER_NAME="$2"
      shift 2
      ;;
    --cluster)
      [[ $# -ge 2 ]] || usage_error "--cluster needs a value"
      CLUSTER="$2"
      shift 2
      ;;
    --cluster-root)
      [[ $# -ge 2 ]] || usage_error "--cluster-root needs a value"
      CLUSTER_ROOT="$2"
      shift 2
      ;;
    --print-manifest)
      PRINT_MANIFEST=1
      shift
      ;;
    *)
      usage_error "unknown option: $1"
      ;;
  esac
done

# Remote mode: ssh ran "bash -s" with this script on stdin and exactly this
# flag. Produce the manifest of the current directory (the cluster root the
# caller cd'd into) and stop - nothing below applies on the remote side.
if [[ "$PRINT_MANIFEST" -eq 1 ]]; then
  if [[ ! -d qespresso_pipeline ]]; then
    echo "check_cluster_drift: $(pwd) has no qespresso_pipeline/ directory - not a repository copy" >&2
    exit 2
  fi
  print_manifest
  exit 0
fi

[[ -n "$USER_NAME" ]] || usage_error "--user is required"

# Locate the repository root the same way run_dft_workflow.sh and
# verify_setup.sh do: walk up from this script until the qespresso_pipeline/
# directory appears.
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(dirname "$SCRIPT_PATH")"
while [[ "$PROJECT_ROOT" != "/" && ! -d "$PROJECT_ROOT/qespresso_pipeline" ]]; do
  PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
done
if [[ ! -d "$PROJECT_ROOT/qespresso_pipeline" ]]; then
  echo "error: cannot find the repository root (no qespresso_pipeline/ directory above $SCRIPT_PATH)" >&2
  exit 2
fi
note "repository root: $PROJECT_ROOT"

SSH_TARGET="${USER_NAME}@${CLUSTER}"

TMP_LOCAL="$(mktemp)"
TMP_REMOTE="$(mktemp)"
trap 'rm -f "$TMP_LOCAL" "$TMP_REMOTE"' EXIT

note "building local manifest..."
( cd "$PROJECT_ROOT" && print_manifest ) > "$TMP_LOCAL"
note "  $(wc -l < "$TMP_LOCAL" | tr -d ' ') files"

note "building cluster manifest (${SSH_TARGET}:${CLUSTER_ROOT})..."
if ! ssh "$SSH_TARGET" "cd '${CLUSTER_ROOT}' && bash -s -- --print-manifest" < "$SCRIPT_PATH" > "$TMP_REMOTE"; then
  echo "error: could not build the cluster manifest (SSH failed, or ${CLUSTER_ROOT} is not a repository copy on ${SSH_TARGET})" >&2
  exit 2
fi
note "  $(wc -l < "$TMP_REMOTE" | tr -d ' ') files"

if diff -q "$TMP_LOCAL" "$TMP_REMOTE" >/dev/null; then
  note "OK: cluster copy matches this checkout ($(wc -l < "$TMP_LOCAL" | tr -d ' ') files)"
  exit 0
fi

note "DRIFT: the cluster copy differs from this checkout"
echo
echo "diff legend: '<' = this checkout, '>' = the cluster copy (${SSH_TARGET}:${CLUSTER_ROOT})"
echo
diff "$TMP_LOCAL" "$TMP_REMOTE" || true
echo
note "re-deploy with scripts/deploy_to_cluster.sh, or inspect the files above before submitting jobs"
exit 1
