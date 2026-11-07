"""Inspect saved QE relaxations and reuse complete, fixed-cell coordinate cards."""

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
RESTART_PATTERN = r"\brestart_mode\s*=\s*(['\"])[^'\"]*\1"
FRESH_RESTART_ASSIGNMENT = "restart_mode='from_scratch'"
COORDINATE_FIELDS = geometry.CELL_DIMENSIONS + 1


class Step(TypedDict):
    energy_ry: float
    total_force_ry_bohr: float | None
    scf_iterations: int | None


class Output(TypedDict):
    steps: list[Step]
    positions: list[str]
    failed_scf_cycles: int


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
                geometry.number(value)
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
    output: Output = {"steps": [], "positions": [], "failed_scf_cycles": 0}
    iterations = None
    stream = iter(lines)
    for line in stream:
        iterations = record_scf(line, output, iterations)
        if (
            count
            and line.strip().upper().startswith(POSITIONS_CARD)
            and (rows := position_rows(stream, count))
        ):
            output["positions"] = [line.strip(), *rows]
    return output


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
    body = text[start:end]
    # Mask comments so a commented restart assignment cannot hide the real one.
    matches = list(
        re.finditer(RESTART_PATTERN, inputs.mask(body, strings=False), re.IGNORECASE)
    )
    if len(matches) > 1:
        raise ValueError("Multiple restart_mode assignments")
    if matches:
        match = matches[0]
        body = body[: match.start()] + FRESH_RESTART_ASSIGNMENT + body[match.end() :]
    else:
        body = f"\n {FRESH_RESTART_ASSIGNMENT},\n" + body
    return text[:start] + body + text[end:]


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
