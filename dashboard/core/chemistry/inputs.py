"""External input discovery, conservative namelist edits, and QE RAM reports."""

import hashlib
import re
from collections.abc import Iterable, Mapping
from decimal import Decimal
from pathlib import Path

from dashboard.core import models


def discover(
    root: Path, candidates: Mapping[str, object]
) -> dict[str, tuple[str, str, Path]]:
    root = root.resolve()
    result = {}
    if not root.is_dir():
        return result
    for owner in candidates:
        directory = root / owner
        if not directory.resolve().is_relative_to(root) or not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.in")):
            match = re.fullmatch(r"(candidate|complex|tfa)\.([\w.-]+)\.in", path.name)
            if not match or (owner == "tfa") != (match[1] == "tfa"):
                continue
            if not path.resolve().is_relative_to(root) or not path.is_file():
                continue
            identifier = hashlib.sha256(
                str(path.relative_to(root)).encode()
            ).hexdigest()
            result[identifier] = (owner, match[1], path)
    return result


def quoted_end(text: str, start: int) -> int:
    index, quote = start + 1, text[start]
    while index < len(text):
        if text[index] == quote:
            if index + 1 < len(text) and text[index + 1] == quote:
                index += 2
                continue
            return index + 1
        index += 1
    raise ValueError("Unterminated input string")


def mask(text: str, strings: bool = True) -> str:
    """Blank comments/strings without moving Fortran namelist offsets."""
    output, index = list(text), 0
    while index < len(text):
        if text[index] == "!":
            end = text.find("\n", index)
            end = len(text) if end < 0 else end
            output[index:end] = " " * (end - index)
            index = end
        elif text[index] in "'\"":
            end = quoted_end(text, index)
            if strings:
                output[index:end] = [
                    "\n" if char == "\n" else " " for char in text[index:end]
                ]
            index = end
        else:
            index += 1
    return "".join(output)


def control_span(text: str) -> tuple[int, int, str]:
    clean = mask(text)
    starts = list(re.finditer(r"&control\b", clean, re.IGNORECASE))
    if len(starts) != 1:
        raise ValueError("Input requires exactly one &CONTROL block")
    start = starts[0].end()
    end = clean.find("/", start)
    if end < 0 or "&" in clean[start:end]:
        raise ValueError("Malformed &CONTROL block")
    return start, end, clean


def dry_input(text: str) -> str:
    start, end, clean = control_span(text)
    assignments = list(re.finditer(r"\bnstep\b", clean[start:end], re.IGNORECASE))
    edits = []
    for name in assignments:
        offset = start + name.start()
        value = re.match(
            r"nstep\s*=\s*[+-]?\d+(?=\s*(?:,|$|[a-z_]\w*\s*=))",
            clean[offset:end],
            re.IGNORECASE,
        )
        if not value:
            raise ValueError("Malformed CONTROL nstep assignment")
        edits.append((offset, offset + value.end(), "nstep=0"))
    if not edits:
        edits = [(start, start, "\n nstep=0,\n")]
    for a, b, replacement in reversed(edits):
        text = text[:a] + replacement + text[b:]
    return text


def validate_references(text: str) -> None:
    start, end, _ = control_span(text)
    body = text[start:end]
    clean = mask(body)
    requirements = {"pseudo_dir": "./Pseudopotentials", "outdir": "./Outputs"}
    for key in (*requirements, "wfcdir", "prefix"):
        names = list(re.finditer(rf"\b{key}\b\s*=", clean, re.IGNORECASE))
        if len(names) > 1 or (key in requirements and len(names) != 1):
            raise ValueError(f"Input requires one {key} assignment")
        if not names:
            continue
        value = re.match(
            r"\s*(['\"])(.*?)\1(?=\s|,|$)",
            mask(body, strings=False)[names[0].end() :],
            re.DOTALL,
        )
        if not value:
            raise ValueError(f"Invalid {key} reference")
        reference = value[2]
        expected = requirements.get(key, "./Outputs" if key == "wfcdir" else None)
        if expected is not None and reference != expected:
            raise ValueError(f"Input must use {key}='{expected}'")
        if key == "prefix" and (
            "/" in reference or "\\" in reference or reference in ("", ".", "..")
        ):
            raise ValueError("Unsafe scratch reference")


RAM_PATTERN = r"Estimated\s+(max\s+dynamical\s+RAM\s+per\s+process|total\s+dynamical\s+RAM)\s*>?\s*[:=]?\s*(\d+(?:\.\d*)?(?:[EeDd][+-]?\d+)?)\s*(MB|GB)\b"
QE_ERROR_PATTERN = r"Error in routine|%{5,}|MPI_ABORT"


def ram_value(match: re.Match[str]) -> models.RAMValue | None:
    value, unit = match[2], match[3]
    amount = Decimal(value.replace("D", "E").replace("d", "e"))
    if not amount.is_finite() or amount < 0:
        return None
    return models.RAMValue(
        value=value,
        unit=unit,
        bytes=int(amount * (1024 ** (2 if unit.upper() == "MB" else 3))),
    )


def ram_reports(lines: Iterable[str]) -> tuple[models.RAMReport, bool]:
    result: models.RAMReport = {"per_process": None, "total": None}
    error = False
    for line in lines:
        error |= bool(re.search(QE_ERROR_PATTERN, line, re.IGNORECASE))
        match = re.search(RAM_PATTERN, line, re.IGNORECASE)
        if match and (value := ram_value(match)) is not None:
            result["per_process" if match[1].lower().startswith("max") else "total"] = (
                value
            )
    return result, error
