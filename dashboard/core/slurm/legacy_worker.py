"""Compatibility entry point for jobs submitted before direct Slurm scripts.

New submissions do not invoke this module. Keep it until old queued jobs drain.
"""

import json
import subprocess
from pathlib import Path

from dashboard.core.config import Config
from dashboard.core.runtimes.local import Runtimes
from dashboard.core.slurm.export import write_direct_job


def main(*, manifest: Path) -> None:
    manifest_path = manifest
    stored_manifest = json.loads(manifest_path.read_text())
    stored = stored_manifest["config"]
    config = Config(
        artifacts=Path(stored["artifacts"]),
        pseudos=Path(stored["pseudos"]),
        apptainer=stored["apptainer"],
        image=Path(stored["image"]) if stored["image"] else None,
    )
    script = write_direct_job(
        stored_manifest["tasks"], config, Runtimes(config), manifest_path.parent
    )
    subprocess.run(["bash", str(script)], check=True)
