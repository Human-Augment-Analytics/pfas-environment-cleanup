"""Read prepared QE coordinates without running chemistry or QE."""

import hashlib
import math
import re
from collections.abc import Callable, Mapping
from pathlib import Path

BOHR = 0.529177210903
MAX_ATOMS = 100000
CELL_DIMENSIONS = 3


def number(value: str) -> float:
    result = float(value.replace("D", "e").replace("d", "e"))
    if not math.isfinite(result):
        raise ValueError("Non-finite coordinate")
    return result


def parameter(text: str, name: str) -> float | None:
    match = re.search(rf"\b{re.escape(name)}\s*=\s*([-+\d.eEdD]+)", text, re.IGNORECASE)
    return number(match[1]) if match else None


def lattice_parameter(text: str) -> float | None:
    angstroms = parameter(text, "A")
    if angstroms is not None:
        return angstroms
    bohr = parameter(text, "celldm(1)")
    return bohr * BOHR if bohr is not None else None


def card(lines: list[str], name: str) -> tuple[int, str]:
    for index, line in enumerate(lines):
        if line.upper().startswith(name):
            return index + 1, line[len(name) :].strip(" {}()").lower() or "alat"
    raise ValueError(f"Missing {name}")


def scale(unit: str, alat: float | None) -> float:
    if unit == "angstrom":
        return 1.0
    if unit == "bohr":
        return BOHR
    if unit == "alat" and alat is not None:
        return alat
    raise ValueError(f"Unsupported or unspecified length unit: {unit}")


def cell_vectors(text: str, lines: list[str], alat: float | None) -> list[list[float]]:
    if parameter(text, "ibrav") != 0:
        raise ValueError("Fractional coordinates require an explicit cell (ibrav=0)")
    start, unit = card(lines, "CELL_PARAMETERS")
    factor = scale(unit, alat)
    cell = [
        [number(value) * factor for value in line.split()[:CELL_DIMENSIONS]]
        for line in lines[start : start + CELL_DIMENSIONS]
    ]
    if len(cell) != CELL_DIMENSIONS or any(len(row) != CELL_DIMENSIONS for row in cell):
        raise ValueError("Invalid cell")
    return cell


def atom(
    line: str, unit: str, alat: float | None, cell: list[list[float]] | None
) -> dict:
    values = line.split()
    if len(values) < CELL_DIMENSIONS + 1:
        raise ValueError("Incomplete atomic coordinates")
    element = re.match(r"[A-Z][a-z]?", values[0])
    if not element:
        raise ValueError("Invalid atom label")
    # Fractional coordinates use the cell; other units use a scalar conversion.
    position = [number(value) for value in values[1 : CELL_DIMENSIONS + 1]]
    if cell:
        position = [
            sum(position[j] * cell[j][i] for j in range(CELL_DIMENSIONS))
            for i in range(CELL_DIMENSIONS)
        ]
    else:
        position = [value * scale(unit, alat) for value in position]
    return {"element": element[0], "position": position}


def geometry(text: str) -> dict:
    lines = [line.split("!")[0].strip() for line in text.splitlines()]
    alat = lattice_parameter(text)
    start, unit = card(lines, "ATOMIC_POSITIONS")
    count = parameter(text, "nat")
    if count is None or count != int(count) or not 0 < count <= MAX_ATOMS:
        raise ValueError("Invalid atom count")
    # Validate the entire card before returning coordinates to the browser.
    cell = cell_vectors(text, lines, alat) if unit == "crystal" else None
    atoms = [atom(line, unit, alat, cell) for line in lines[start : start + int(count)]]
    if len(atoms) != int(count):
        raise ValueError("Incomplete atomic coordinates")
    return {
        "atoms": atoms,
        "units": "angstrom",
        "input_hash": hashlib.sha256(text.encode()).hexdigest(),
    }


def read_geometry(path: Path) -> dict:
    try:
        return geometry(path.read_text())
    except (ValueError, IndexError) as error:
        return {"error": str(error)}
    except OSError:
        return {"error": "Prepared input cannot be read"}


def prepared_geometries(
    task: Mapping[str, object], input_path: Callable[[str], Path | None]
) -> dict:
    if task["kind"] != "prepare" or task["status"] != "succeeded":
        return {}
    results = {}
    for system in ("candidate", "complex", "tfa"):
        path = input_path(system + ".in")
        if path and path.is_file():
            results[system] = read_geometry(path)
    return results


def preview_geometry(text: str) -> dict:
    try:
        return geometry(text)
    except (ValueError, IndexError) as error:
        return {"error": str(error)}
