"""Preparation-only child entry point. Never imports an execution wrapper."""

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

from dashboard.core import config


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


SPECIES_END_CARDS = (
    "ATOMIC_POSITIONS",
    "K_POINTS",
    "CELL_PARAMETERS",
    "ATOMIC_FORCES",
    "CONSTRAINTS",
    "OCCUPATIONS",
    "HUBBARD",
    "SOLVENTS",
    "ADDITIONAL_K_POINTS",
)
MOLECULE_PADDING = 12.0
COMPLEX_GAP = 2.5


def species_rows(text: str) -> list[str]:
    lines = [line.split("!")[0].strip() for line in text.splitlines()]
    starts = [
        index for index, line in enumerate(lines) if line.upper() == "ATOMIC_SPECIES"
    ]
    if len(starts) != 1:
        raise ValueError("Missing or duplicate ATOMIC_SPECIES")
    rows = []
    for line in lines[starts[0] + 1 :]:
        if not line:
            continue
        if line.split()[0].upper() in SPECIES_END_CARDS:
            break
        rows.append(line)
    return rows


def species_name(line: str) -> str:
    values = line.split()
    if (
        len(values) != 3
        or not re.fullmatch(r"[A-Za-z]\w*", values[0])
        or not re.fullmatch(
            r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?", values[1]
        )
    ):
        raise ValueError("Invalid ATOMIC_SPECIES row")
    name = values[2]
    if Path(name).name != name or "\\" in name or not name.lower().endswith(".upf"):
        raise ValueError("Missing or unsafe ATOMIC_SPECIES pseudopotentials")
    return name


def pseudo_names(text: str) -> list[str]:
    names = [species_name(line) for line in species_rows(text)]
    if not names:
        raise ValueError("Missing or unsafe ATOMIC_SPECIES pseudopotentials")
    count = re.search(r"\bntyp\s*=\s*(\d+)", text, re.IGNORECASE)
    if count and len(names) != int(count[1]):
        raise ValueError("ATOMIC_SPECIES count does not match ntyp")
    return sorted(set(names))


def prepare_reference(request: dict, directory: Path, settings: dict) -> Path:
    from run_adsorption_case import prepare_from_smiles

    from dashboard.core import candidates

    if request.get("shared_tfa"):
        shared = Path(request["shared_tfa"])
        for suffix in ("mol", "cif", "in"):
            shutil.copyfile(shared / ("tfa." + suffix), directory / ("tfa." + suffix))
        return directory / "tfa.mol"
    molecule, _, _ = prepare_from_smiles(
        candidates.TFA["smiles"], directory / "tfa", settings
    )
    return molecule


def prepare_candidate(
    request: dict, directory: Path, settings: dict, reference: Path
) -> None:
    from run_adsorption_case import (
        build_molecular_complex_cif,
        patch_qe_input,
        prepare_from_smiles,
    )
    from smiles_to_qe import run_cif2cell

    molecule, _, _ = prepare_from_smiles(
        request["smiles"], directory / "candidate", settings
    )
    build_molecular_complex_cif(
        molecule,
        reference,
        directory / "complex.cif",
        padding=MOLECULE_PADDING,
        vdw_gap=COMPLEX_GAP,
    )
    converted = run_cif2cell(directory / "complex.cif", str(directory / "complex"))
    patch_qe_input(converted, settings, 1, 0.0)


def write_manifest(request: dict, directory: Path, settings: dict) -> None:
    inputs = list(directory.glob("*.in"))
    hashes = {
        name: digest(Path(request["pseudos"]) / name)
        for path in inputs
        for name in pseudo_names(path.read_text())
    }
    manifest = {
        "settings": settings,
        "geometry": {"padding": MOLECULE_PADDING, "vdw_gap": COMPLEX_GAP},
        "source": request,
        "inputs": {path.name: digest(path) for path in inputs},
        "pseudopotentials": hashes,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))


def prepare(request: dict, directory: Path | str) -> None:
    # Resolve the existing pipeline in the isolated chemistry interpreter.
    sys.path.insert(0, str(config.ROOT / "qespresso_pipeline"))
    from run_adsorption_case import get_mode_settings

    directory = Path(directory)
    settings = get_mode_settings("cluster", "molecule")
    reference = prepare_reference(request, directory, settings)
    if request["candidate"] != "tfa":
        prepare_candidate(request, directory, settings, reference)
    write_manifest(request, directory, settings)


def main(*, request: Path, directory: Path) -> None:
    prepare(json.loads(request.read_text()), directory)
