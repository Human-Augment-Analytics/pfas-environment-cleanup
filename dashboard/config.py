import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def preparation_python():
    configured = os.getenv("PFAS_CHEM_PYTHON")
    if configured:
        return configured
    return sys.executable


@dataclass
class Config:
    artifacts: Path = field(
        default_factory=lambda: Path(
            os.getenv("PFAS_ARTIFACTS", ROOT / ".dashboard")
        ).resolve()
    )
    python: str = field(
        default_factory=lambda: os.getenv("PFAS_CHEM_PYTHON", sys.executable)
    )
    prepare_python: str = field(default_factory=preparation_python)
    pseudos: Path = field(
        default_factory=lambda: Path(
            os.getenv("PFAS_PSEUDOS", ROOT / "qespresso_pipeline/Pseudopotentials")
        ).resolve()
    )
    pw: str = field(default_factory=lambda: os.getenv("PFAS_PW", "pw.x"))
    mpi: str = field(default_factory=lambda: os.getenv("PFAS_MPI", "mpirun"))
    prepare_timeout: float = field(
        default_factory=lambda: float(os.getenv("PFAS_PREPARE_TIMEOUT", "900"))
    )
