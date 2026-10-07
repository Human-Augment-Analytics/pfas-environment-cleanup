import os
import re
import signal
import subprocess
import time


def terminate(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass
        except ProcessLookupError:
            return
        # The leader can exit while MPI children remain.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def execute(
    command, directory, cancel, timeout=None, on_start=lambda p: None, env=None
):
    start = time.monotonic()
    with (
        (directory / "stdout.log").open("wb") as out,
        (directory / "stderr.log").open("wb") as err,
    ):
        process = subprocess.Popen(
            command,
            cwd=directory,
            stdout=out,
            stderr=err,
            start_new_session=True,
            env={**(env or os.environ), "OMP_NUM_THREADS": "1"},
        )
        on_start(process)
        while process.poll() is None:
            if cancel.is_set() or (timeout and time.monotonic() - start > timeout):
                terminate(process)
                return (
                    process.returncode,
                    "canceled" if cancel.is_set() else "failed",
                    "Stopped" if cancel.is_set() else "Timed out",
                )
            time.sleep(0.05)
        return (
            process.returncode,
            "succeeded" if process.returncode == 0 else "failed",
            "",
        )


def evidence(text, calculation="scf"):
    energies = re.findall(r"!\s+total energy\s*=\s*([-+\d.EeDd]+)\s+Ry", text)
    completed = "JOB DONE." in text
    scf = "convergence has been achieved" in text
    relax = (
        "End of BFGS Geometry Optimization" in text or "bfgs converged" in text.lower()
    )
    failed = "convergence NOT achieved" in text or "Error in routine" in text
    confirmed = (
        completed
        and scf
        and not failed
        and (calculation not in ("relax", "vc-relax") or relax)
    )
    return {
        "job_done": completed,
        "scf_converged": scf,
        "relaxation_completed": relax,
        "confirmed": confirmed,
        "energy_ry": float(energies[-1].replace("D", "E").replace("d", "e"))
        if energies
        else None,
        "provisional": not confirmed,
    }


def tail(path, limit=16000):
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - limit))
        return stream.read(limit).decode(errors="replace")
