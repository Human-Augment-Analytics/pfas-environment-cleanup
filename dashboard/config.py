import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .execution import (
    EXECUTION_ENV,
    SCHEDULER_COMMANDS,
    SLURM_JOB_ID_ENV,
    ExecutionMode,
)

ROOT = Path(__file__).parents[1]
CHEM_PYTHON_ENV = "PFAS_CHEM_PYTHON"
ARTIFACTS_ENV = "PFAS_ARTIFACTS"
APPTAINER_ENV = "PFAS_APPTAINER"
APPTAINER_IMAGE_ENV = "PFAS_APPTAINER_IMAGE"
PSEUDOS_ENV = "PFAS_PSEUDOS"
QE_INPUTS_ENV = "PFAS_QE_INPUTS"
PW_ENV = "PFAS_PW"
MPI_ENV = "PFAS_MPI"
PREPARE_TIMEOUT_ENV = "PFAS_PREPARE_TIMEOUT"


def preparation_python() -> str:
    configured = os.getenv(CHEM_PYTHON_ENV)
    if configured:
        return configured
    return sys.executable


@dataclass
class Config:
    execution: str = field(
        default_factory=lambda: os.getenv(EXECUTION_ENV, ExecutionMode.AUTO)
    )

    @property
    def allocation(self) -> bool:
        return self.execution != ExecutionMode.LOCAL and bool(
            os.getenv(SLURM_JOB_ID_ENV)
        )

    @property
    def scheduler_enabled(self) -> bool:
        import shutil

        if self.execution not in ExecutionMode:
            raise ValueError("PFAS_EXECUTION must be auto, local, or slurm")
        found = all(shutil.which(c) for c in SCHEDULER_COMMANDS)
        if self.execution == ExecutionMode.SLURM and not found:
            raise ValueError(
                "Slurm execution requires sbatch, squeue, sacct, and scancel"
            )
        return found and (self.execution == ExecutionMode.SLURM or self.allocation)

    artifacts: Path = field(
        default_factory=lambda: Path(
            os.getenv(ARTIFACTS_ENV, ROOT / ".dashboard")
        ).resolve()
    )
    apptainer: str = field(
        default_factory=lambda: os.getenv(APPTAINER_ENV, "apptainer")
    )
    image: Path | None = field(
        default_factory=lambda: (
            Path(os.environ[APPTAINER_IMAGE_ENV]).resolve()
            if os.getenv(APPTAINER_IMAGE_ENV)
            else None
        )
    )
    python: str = field(
        default_factory=lambda: os.getenv(CHEM_PYTHON_ENV, sys.executable)
    )
    prepare_python: str = field(default_factory=preparation_python)
    pseudos: Path = field(
        default_factory=lambda: Path(
            os.getenv(PSEUDOS_ENV, ROOT / "qespresso_pipeline/Pseudopotentials")
        ).resolve()
    )
    qe_inputs: Path = field(
        default_factory=lambda: Path(
            os.getenv(QE_INPUTS_ENV, ROOT / "qe_inputs")
        ).resolve()
    )
    pw: str = field(default_factory=lambda: os.getenv(PW_ENV, "pw.x"))
    mpi: str = field(default_factory=lambda: os.getenv(MPI_ENV, "mpirun"))
    prepare_timeout: float = field(
        default_factory=lambda: float(os.getenv(PREPARE_TIMEOUT_ENV, "900"))
    )
