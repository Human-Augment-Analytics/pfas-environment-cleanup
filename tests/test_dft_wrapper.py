"""Fast smoke test for the dependency-free DFT submission CLI."""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_dft_wrapper_help() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/dft_wrapper.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "DFT wrapper: submit-if-missing and fetch results." in result.stdout
