import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


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
