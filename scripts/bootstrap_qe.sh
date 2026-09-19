#!/usr/bin/env bash
# Source this file from a DFT workflow; do not execute it as a standalone script.
# It creates/updates .env-qe from ../requirements-qe.txt and makes QE 7.3
# deterministic. Set QE_INSTALL_LOCAL=1 to build the matching local QE copy.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "[error] Source this helper: source scripts/bootstrap_qe.sh" >&2
    exit 2
fi

QE_VERSION="7.3"
PYTHON_VERSION="3.12.5"
BOOTSTRAP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$BOOTSTRAP_DIR/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.env-qe"
REQUIREMENTS="$PROJECT_ROOT/requirements-qe.txt"

bootstrap_error() {
    echo "[error] $*" >&2
    return 1
}

is_pace() {
    [[ "${PACE_CLUSTER:-}" == "1" || \
       "${SLURM_CLUSTER_NAME:-}" == *ice* || \
       "${HOSTNAME:-}" == *.pace.gatech.edu || \
       -d /storage/ice-shared ]]
}

qe_version_is_expected() {
    local banner
    banner="$("$QE_PW_X" --version 2>&1)" || {
        echo "$banner" >&2
        return 1
    }
    [[ "$banner" == *"v.${QE_VERSION}"* ]]
}

install_local_qe() {
    local install_root="${QE_LOCAL_PREFIX:-$HOME/opt/qe-$QE_VERSION}"
    local archive source_dir

    command -v curl >/dev/null || bootstrap_error "curl is required to install local QE" || return 1
    command -v make >/dev/null || bootstrap_error "make is required to install local QE" || return 1
    command -v gfortran >/dev/null || bootstrap_error "gfortran is required to install local QE" || return 1
    command -v mpirun >/dev/null || bootstrap_error "an MPI implementation is required to install local QE" || return 1
    [[ ! -e "$install_root" ]] || bootstrap_error "local QE target already exists: $install_root" || return 1

    archive="$(mktemp --suffix=.tar.gz)"
    source_dir="${archive%.tar.gz}/q-e-qe-$QE_VERSION"
    mkdir -p "$(dirname -- "$install_root")"
    echo "[qe] downloading and building QE $QE_VERSION at $install_root"
    curl --fail --location --output "$archive" \
        "https://gitlab.com/QEF/q-e/-/archive/qe-$QE_VERSION/q-e-qe-$QE_VERSION.tar.gz" || return 1
    mkdir -p "${archive%.tar.gz}"
    tar -xzf "$archive" -C "${archive%.tar.gz}" || return 1
    (
        cd "$source_dir" || exit 1
        ./configure && make -j"${QE_BUILD_JOBS:-2}" all
    ) || return 1
    mv "$source_dir" "$install_root"
    rm -rf "${archive%.tar.gz}" "$archive"
}

if is_pace; then
    type module >/dev/null 2>&1 || bootstrap_error "PACE was detected but the module command is unavailable" || return 1
    module load "quantum-espresso/$QE_VERSION" "python/$PYTHON_VERSION" "uv/0.9.17" || return 1
    QE_PW_X="$(command -v pw.x)"
else
    QE_LOCAL_PREFIX="${QE_LOCAL_PREFIX:-$HOME/opt/qe-$QE_VERSION}"
    if [[ ! -x "$QE_LOCAL_PREFIX/bin/pw.x" && "${QE_INSTALL_LOCAL:-0}" == "1" ]]; then
        install_local_qe || return 1
    fi
    [[ -x "$QE_LOCAL_PREFIX/bin/pw.x" ]] || bootstrap_error \
        "QE $QE_VERSION was not found at $QE_LOCAL_PREFIX/bin/pw.x; set QE_INSTALL_LOCAL=1 to build it" || return 1
    export PATH="$QE_LOCAL_PREFIX/bin:$PATH"
    QE_PW_X="$QE_LOCAL_PREFIX/bin/pw.x"
    command -v uv >/dev/null || bootstrap_error "uv is required outside PACE" || return 1
fi

qe_version_is_expected || bootstrap_error "expected QE $QE_VERSION, got: $QE_PW_X" || return 1
command -v mpirun >/dev/null || bootstrap_error "no MPI launcher is available for QE $QE_VERSION" || return 1

if [[ ! -x "$VENV_DIR/bin/python" ]] || ! "$VENV_DIR/bin/python" -c \
    'import sys; raise SystemExit(sys.version_info[:3] != (3, 12, 5))'; then
    echo "[env] creating $VENV_DIR with Python $PYTHON_VERSION"
    uv venv --clear --python "$PYTHON_VERSION" "$VENV_DIR" || return 1
fi
echo "[env] synchronizing $VENV_DIR"
uv pip sync --python "$VENV_DIR/bin/python" "$REQUIREMENTS" || return 1
"$VENV_DIR/bin/python" -c 'import ase, cif2cell, numpy, openbabel, pymatgen'

export QE_PW_X
export QE_PYTHON="$VENV_DIR/bin/python"
