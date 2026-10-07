"""Preparation-only child entry point. Never imports an execution wrapper."""

import hashlib
import json
import shutil
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pseudo_names(text):
    import re

    names = re.findall(r"(?m)^\s*\w+\s+[\d.]+\s+([^\s!]+\.UPF)\s*", text)
    if not names or any(Path(n).name != n for n in names):
        raise ValueError("Missing or unsafe ATOMIC_SPECIES pseudopotentials")
    return sorted(set(names))


def prepare(request, directory):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qespresso_pipeline"))
    from run_adsorption_case import (
        build_molecular_complex_cif,
        get_mode_settings,
        patch_qe_input,
        prepare_from_smiles,
    )
    from smiles_to_qe import run_cif2cell

    directory = Path(directory)
    settings = get_mode_settings("cluster", "molecule")
    if request.get("shared_tfa"):
        shared = Path(request["shared_tfa"])
        for suffix in ("mol", "cif", "in"):
            shutil.copyfile(shared / ("tfa." + suffix), directory / ("tfa." + suffix))
        tfa_mol = directory / "tfa.mol"
    else:
        tfa_mol, _, _ = prepare_from_smiles(
            "O=C(O)C(F)(F)F", directory / "tfa", settings
        )
    if request["candidate"] != "tfa":
        mol, _, _ = prepare_from_smiles(
            request["smiles"], directory / "candidate", settings
        )
        build_molecular_complex_cif(
            mol, tfa_mol, directory / "complex.cif", padding=12.0, vdw_gap=2.5
        )
        patch_qe_input(
            run_cif2cell(directory / "complex.cif", str(directory / "complex")),
            settings,
            1,
            0.0,
        )
    inputs = list(directory.glob("*.in"))
    hashes = {}
    for path in inputs:
        for name in pseudo_names(path.read_text()):
            hashes[name] = digest(Path(request["pseudos"]) / name)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "settings": settings,
                "geometry": {"padding": 12, "vdw_gap": 2.5},
                "source": request,
                "inputs": {p.name: digest(p) for p in inputs},
                "pseudopotentials": hashes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    prepare(json.loads(Path(sys.argv[1]).read_text()), sys.argv[2])
