#!/bin/bash
# Deploy this checkout to the cluster in one command, then verify the result.
# The SLURM jobs submitted by dft_wrapper.py run scripts/run_dft_workflow.sh
# from the cluster copy, so that copy must match this checkout for results to
# be reproducible. This script refuses to overwrite a cluster copy that has
# hand-edits (see the pre-flight below) unless you pass --force.
#
# What it does:
#   1. Refuses to run when any scripts/*.sh in this checkout has Windows
#      (CRLF) line endings - those fail on the cluster (bash scripts/verify_setup.sh
#      catches the same problem).
#   2. Runs scripts/check_cluster_drift.sh first: if the cluster copy differs
#      from this checkout, it refuses, because --delete would silently
#      overwrite whatever was changed over there. --force overrides this
#      (and redeploys this checkout as the source of truth).
#   3. rsyncs the repository to the cluster root, excluding the bulk data /
#      run / cache directories that live only on the cluster (same list
#      check_cluster_drift.sh uses, plus .git and tool caches).
#   4. Re-runs the drift check to confirm the deployed copy matches.
#
# Usage:
#   bash scripts/deploy_to_cluster.sh --user <gatech-user> [--force]
#
# Options (same flags and defaults as scripts/dft_wrapper.py):
#   --user          Gatech username for the SSH target (required)
#   --cluster       Login node          (default: login-ice.pace.gatech.edu)
#   --cluster-root  Path on the cluster (default: /storage/ice-shared/cs8903onl/mussmann-pfas)
#   --force         Redeploy even when the cluster copy has drifted
#
# Exit codes:
#   0  deployed and verified
#   1  refused (CRLF scripts, or cluster drift without --force)
#   2  error

set -euo pipefail

CLUSTER="login-ice.pace.gatech.edu"
CLUSTER_ROOT="/storage/ice-shared/cs8903onl/mussmann-pfas"
USER_NAME=""
FORCE=0

note() { echo "[deploy] $*"; }

usage_error() {
  echo "error: $1" >&2
  echo "usage: bash scripts/deploy_to_cluster.sh --user <gatech-user> [--cluster HOST] [--cluster-root PATH] [--force]" >&2
  exit 2
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
    --force)
      FORCE=1
      shift
      ;;
    *)
      usage_error "unknown option: $1"
      ;;
  esac
done

[[ -n "$USER_NAME" ]] || usage_error "--user is required"

command -v rsync >/dev/null 2>&1 || {
  echo "error: rsync is not on PATH - install it (on Windows: a Git-for-Windows/MSYS2 rsync or cwRsync) and retry" >&2
  exit 2
}

# Locate the repository root the same way run_dft_workflow.sh and
# verify_setup.sh do: walk up from this script until the qespresso_pipeline/
# directory appears.
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
PROJECT_ROOT="$SCRIPT_DIR"
while [[ "$PROJECT_ROOT" != "/" && ! -d "$PROJECT_ROOT/qespresso_pipeline" ]]; do
  PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
done
if [[ ! -d "$PROJECT_ROOT/qespresso_pipeline" ]]; then
  echo "error: cannot find the repository root (no qespresso_pipeline/ directory above $SCRIPT_PATH)" >&2
  exit 2
fi
note "repository root: $PROJECT_ROOT"

DRIFT_SCRIPT="$SCRIPT_DIR/check_cluster_drift.sh"
if [[ ! -f "$DRIFT_SCRIPT" ]]; then
  echo "error: $DRIFT_SCRIPT not found - deploy and drift-check belong together" >&2
  exit 2
fi

SSH_TARGET="${USER_NAME}@${CLUSTER}"

# Pre-flight 1: refuse to ship CRLF shell scripts - they fail on the cluster.
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

crlf_files=""
for f in "$SCRIPT_DIR"/*.sh; do
  [[ -f "$f" ]] || continue
  if has_crlf "$f"; then
    crlf_files="${crlf_files}$(basename "$f") "
  fi
done
if [[ -n "$crlf_files" ]]; then
  echo "error: Windows (CRLF) line endings found in: $crlf_files" >&2
  echo "  these scripts fail on the cluster as CRLF. Fix with 'dos2unix scripts/*.sh' or" >&2
  echo "  re-clone with core.autocrlf=input; bash scripts/verify_setup.sh checks for this too." >&2
  exit 1
fi
note "pre-flight 1/2: scripts/*.sh all use Unix (LF) line endings"

# Pre-flight 2: if the cluster copy has drifted from this checkout, refuse
# unless --force - rsync --delete would silently overwrite those changes.
set +e
bash "$DRIFT_SCRIPT" --user "$USER_NAME" --cluster "$CLUSTER" --cluster-root "$CLUSTER_ROOT"
DRIFT_RC=$?
set -e

case "$DRIFT_RC" in
  0)
    note "pre-flight 2/2: cluster copy already matches this checkout"
    ;;
  1)
    if [[ "$FORCE" -eq 1 ]]; then
      note "--force given: redeploying this checkout over the drifted cluster copy (its hand-edits, if any, are lost)"
    else
      echo "error: the cluster copy differs from this checkout (diff above) - rsync --delete would overwrite that." >&2
      echo "  Inspect the diff, reconcile by hand, or rerun with --force to make this checkout the source of truth." >&2
      exit 1
    fi
    ;;
  *)
    echo "error: the drift pre-flight itself failed (exit $DRIFT_RC) - fix that before deploying" >&2
    exit 2
    ;;
esac

note "deploying ${SSH_TARGET}:${CLUSTER_ROOT} ..."
# Excludes mirror check_cluster_drift.sh, plus .git and tool caches: the
# drift manifest prunes the same names so the post-deploy check converges.
if ! ( cd "$PROJECT_ROOT" && rsync -av --delete \
    --exclude .git/ --exclude data/ --exclude dft_runs/ --exclude dft_cases/ --exclude compounds/ \
    --exclude logs/ --exclude __pycache__/ --exclude .pytest_cache/ --exclude .ruff_cache/ --exclude .venv/ \
    --exclude '*.pyc' --exclude master_results.txt \
    ./ "${SSH_TARGET}:${CLUSTER_ROOT}/" ); then
  echo "error: rsync failed - the cluster copy may be partially updated; re-run the deploy" >&2
  exit 2
fi

note "re-checking the deployed copy..."
set +e
bash "$DRIFT_SCRIPT" --user "$USER_NAME" --cluster "$CLUSTER" --cluster-root "$CLUSTER_ROOT"
VERIFY_RC=$?
set -e

if [[ "$VERIFY_RC" -ne 0 ]]; then
  echo "error: deployed, but the verification drift check still fails (exit $VERIFY_RC) - inspect the diff above" >&2
  exit 2
fi

note "done: cluster copy deployed and verified"
exit 0
