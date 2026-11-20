"""Inspect saved QE relaxations and reuse complete, fixed-cell coordinate cards."""

import math
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypedDict

from dashboard.core import execution
from dashboard.core.chemistry import geometry, inputs

CONTINUATION_PREFIX = "continuation:"
POSITIONS_CARD = "ATOMIC_POSITIONS"
NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
ENERGY_PATTERN = rf"!\s+total energy\s*=\s*({NUMBER_PATTERN})\s+Ry"
FORCE_PATTERN = rf"Total force\s*=\s*({NUMBER_PATTERN})"
SCF_PATTERN = r"convergence has been achieved in\s+(\d+)\s+iterations"
FAILED_SCF_MARKER = "convergence NOT achieved"
FIXED_CELL_CALCULATION = "relax"
RESTART_MODE_KEY = "restart_mode"
FRESH_RESTART_MODE = "from_scratch"
EMPTY_ELECTRONS_BLOCK = "\n&ELECTRONS\n/\n"
SYSTEM_NAMELIST = "system"
ELECTRONS_NAMELIST = "electrons"
FRESH_ELECTRONIC_SETTINGS = {"startingpot": "atomic", "startingwfc": "atomic+random"}
JOB_DONE_MARKER = "JOB DONE."
RELAXATION_MARKERS = ("End of BFGS Geometry Optimization", "bfgs converged")
ERROR_MARKER = "Error in routine"
COORDINATE_FIELDS = geometry.CELL_DIMENSIONS + 1


class Step(TypedDict):
    energy_ry: float
    total_force_ry_bohr: float | None
    scf_iterations: int | None


class Output(TypedDict):
    steps: list[Step]
    positions: list[str]
    failed_scf_cycles: int
    job_done: bool
    relaxation_completed: bool
    qe_error: bool
    partial_positions: bool


class Report(TypedDict):
    steps: list[Step]
    failed_scf_cycles: int
    calculation: str
    force_threshold_ry_bohr: float | None
    energy_threshold_ry: float | None
    initial_geometry: dict
    latest_geometry: dict | None
    continuation_error: str


def calculation(text: str) -> str:
    match = re.search(
        r"\bcalculation\s*=\s*['\"]([^'\"]+)",
        inputs.mask(text, strings=False),
        re.IGNORECASE,
    )
    return match[1].lower() if match else execution.QE_DEFAULT_CALCULATION


def position_rows(lines: Iterator[str], count: int) -> list[str]:
    rows = []
    for _ in range(count):
        line = next(lines, "").strip()
        values = line.split()
        if len(values) < geometry.CELL_DIMENSIONS + 1:
            return []
        try:
            for value in values[1 : geometry.CELL_DIMENSIONS + 1]:
                if not math.isfinite(geometry.number(value)):
                    return []
        except ValueError:
            return []
        rows.append(line)
    return rows


def record_scf(line: str, output: Output, iterations: int | None) -> int | None:
    # A converged SCF energy starts a sample; its force follows it.
    if match := re.search(SCF_PATTERN, line):
        iterations = int(match[1])
    if match := re.search(ENERGY_PATTERN, line):
        output["steps"].append(
            Step(
                energy_ry=geometry.number(match[1]),
                total_force_ry_bohr=None,
                scf_iterations=iterations,
            )
        )
        iterations = None
    if (match := re.search(FORCE_PATTERN, line)) and output["steps"]:
        output["steps"][-1]["total_force_ry_bohr"] = geometry.number(match[1])
    output["failed_scf_cycles"] += FAILED_SCF_MARKER in line
    return iterations


def read_output(lines: Iterable[str], count: int) -> Output:
    """Stream large logs, retaining SCF summaries and only the latest full card."""
    output = empty_output()
    iterations = None
    stream = iter(lines)
    for line in stream:
        iterations = record_scf(line, output, iterations)
        record_completion(line, output)
        if count and line.strip().upper().startswith(POSITIONS_CARD):
            record_positions(line, stream, count, output)
    return output


def empty_output() -> Output:
    return {
        "steps": [],
        "positions": [],
        "failed_scf_cycles": 0,
        "job_done": False,
        "relaxation_completed": False,
        "qe_error": False,
        "partial_positions": False,
    }


def record_completion(line: str, output: Output) -> None:
    output["job_done"] |= JOB_DONE_MARKER in line
    output["qe_error"] |= ERROR_MARKER in line
    output["relaxation_completed"] |= any(
        marker.lower() in line.lower() for marker in RELAXATION_MARKERS
    )


def record_positions(
    line: str, stream: Iterator[str], count: int, output: Output
) -> None:
    # Manual preview may reuse an older full card; automation rejects a partial tail.
    rows = position_rows(stream, count)
    output["partial_positions"] = not bool(rows)
    if rows:
        output["positions"] = [line.strip(), *rows]


def original_positions(text: str) -> tuple[list[str], int, int]:
    lines = text.splitlines()
    count = geometry.parameter(text, "nat")
    if count is None or count != int(count) or not 0 < count <= geometry.MAX_ATOMS:
        raise ValueError("Invalid atom count")
    start, _ = geometry.card([line.strip() for line in lines], POSITIONS_CARD)
    if len(position_rows(iter(lines[start:]), int(count))) != int(count):
        raise ValueError("Incomplete original atomic coordinates")
    return lines, start, int(count)


def constrained_positions(originals: list[str], updates: list[str]) -> list[str]:
    # Output flags are not authoritative: retain the submitted constraints.
    rows = []
    for original, updated in zip(originals, updates, strict=True):
        before, after = original.split("!")[0].split(), updated.split()
        if before[0] != after[0]:
            raise ValueError("Output atom order differs from the original input")
        rows.append(" ".join([*after[:COORDINATE_FIELDS], *before[COORDINATE_FIELDS:]]))
    return rows


def replace_positions(text: str, positions: list[str]) -> str:
    if calculation(text) != FIXED_CELL_CALCULATION:
        raise ValueError(
            "Coordinate continuation currently requires a fixed-cell relax calculation"
        )
    if not positions:
        raise ValueError("No complete ATOMIC_POSITIONS card was printed by QE")
    lines, start, count = original_positions(text)
    if len(positions) != count + 1:
        raise ValueError("Output atom count differs from the original input")
    rows = constrained_positions(lines[start : start + count], positions[1:])
    lines[start - 1 : start + count] = [positions[0], *rows]
    result = "\n".join(lines) + "\n"
    geometry.geometry(result)
    return fresh_start(result)


def fresh_start(text: str) -> str:
    start, end, _ = inputs.control_span(text)
    body = fresh_assignment(text[start:end], RESTART_MODE_KEY, FRESH_RESTART_MODE)
    return fresh_electrons(text[:start] + body + text[end:])


def fresh_electrons(text: str) -> str:
    """Explicitly avoid file-based density/wavefunctions in fresh directories."""
    clean = inputs.mask(text)
    blocks = list(re.finditer(r"&electrons\b", clean, re.IGNORECASE))
    if not blocks:
        # QE reads SYSTEM before ELECTRONS; insert after SYSTEM when omitted.
        _, system_end = namelist_span(text, SYSTEM_NAMELIST)
        insertion = system_end + 1
        return fresh_electrons(
            text[:insertion] + EMPTY_ELECTRONS_BLOCK + text[insertion:]
        )
    start, end = namelist_span(text, ELECTRONS_NAMELIST)
    body = text[start:end]
    for name, value in FRESH_ELECTRONIC_SETTINGS.items():
        body = fresh_assignment(body, name, value)
    return text[:start] + body + text[end:]


def namelist_span(text: str, name: str) -> tuple[int, int]:
    clean = inputs.mask(text)
    blocks = list(re.finditer(rf"&{name}\b", clean, re.IGNORECASE))
    if len(blocks) != 1:
        raise ValueError(f"Continuation requires exactly one &{name.upper()} block")
    start = blocks[0].end()
    end = clean.find("/", start)
    if end < 0 or "&" in clean[start:end]:
        raise ValueError(f"Malformed &{name.upper()} block")
    return start, end


def fresh_assignment(body: str, name: str, value: str) -> str:
    matches = list(re.finditer(rf"\b{name}\s*=", inputs.mask(body), re.IGNORECASE))
    assignment = f"{name}='{value}'"
    if not matches:
        return f"\n {assignment},\n" + body
    if len(matches) != 1:
        raise ValueError(f"Multiple {name} assignments")
    match = matches[0]
    quoted = re.match(r"\s*(['\"])[^'\"]*\1", body[match.end() :])
    if not quoted:
        raise ValueError(f"Malformed {name} assignment")
    return body[: match.start()] + assignment + body[match.end() + quoted.end() :]


def inspect(directory: Path) -> tuple[str, Output]:
    text = (directory / execution.INPUT_FILE).read_text()
    count = geometry.parameter(text, "nat")
    atom_count = (
        int(count)
        if count and count == int(count) and 0 < count <= geometry.MAX_ATOMS
        else 0
    )
    with (directory / execution.STDOUT_LOG_FILE).open(errors="replace") as stream:
        return text, read_output(stream, atom_count)


def continuation(directory: Path) -> str:
    text, output = inspect(directory)
    return replace_positions(text, output["positions"])


def report(directory: Path) -> Report:
    text, output = inspect(directory)
    problem, latest = "", None
    try:
        latest = geometry.geometry(replace_positions(text, output["positions"]))
    except ValueError as error:
        problem = str(error)
    return {
        "steps": output["steps"],
        "failed_scf_cycles": output["failed_scf_cycles"],
        "calculation": calculation(text),
        "force_threshold_ry_bohr": geometry.parameter(text, "forc_conv_thr"),
        "energy_threshold_ry": geometry.parameter(text, "etot_conv_thr"),
        "initial_geometry": geometry.preview_geometry(text),
        "latest_geometry": latest,
        "continuation_error": problem,
    }
