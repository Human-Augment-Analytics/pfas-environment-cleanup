"""Keep CLI parsing in app and explicit, keyword-only workflows in core."""

import importlib
import inspect
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("app_name", "core_name", "arguments"),
    [
        ("cli", "dashboard.core.main", ["serve", "--port", "8123"]),
        ("diagram", "dashboard.core.chemistry.diagram", ["O", "diagram.png"]),
        (
            "preparation",
            "dashboard.core.chemistry.preparation",
            ["request.json", "output"],
        ),
        ("cpu", "dashboard.core.runtimes.cpu", ["1,2", "echo", "example"]),
        ("container", "dashboard.core.runtimes.container", ["--probe", "128"]),
        ("legacy_worker", "dashboard.core.slurm.legacy_worker", ["manifest.json"]),
    ],
)
def test_app_passes_parsed_keywords_to_core(
    app_name: str, core_name: str, arguments: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = importlib.import_module("dashboard.app." + app_name)
    core = importlib.import_module(core_name)
    assert all(
        parameter.kind == inspect.Parameter.KEYWORD_ONLY
        for parameter in inspect.signature(core.main).parameters.values()
    )
    expected = vars(app.get_parser().parse_args(arguments))
    captured: dict[str, object] = {}

    def capture(**keywords: object) -> int:
        captured.update(keywords)
        return 0

    monkeypatch.setattr(core, "main", capture)
    monkeypatch.setattr(sys, "argv", [app_name, *arguments])
    if app_name == "container":
        with pytest.raises(SystemExit) as result:
            app.main()
        assert result.value.code == 0
    else:
        app.main()
    assert captured == expected


def test_importing_diagram_does_not_generate_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    importlib.reload(importlib.import_module("dashboard.core.chemistry.diagram"))
    assert list(tmp_path.iterdir()) == []


def test_pace_launcher_factory_is_importable() -> None:
    import re

    from dashboard.core import config

    script = (config.ROOT / "scripts/hpc/dashboard.sbatch").read_text()
    factory = re.search(r"-m uvicorn ([\w.]+):(\w+)", script)
    assert factory is not None
    module = importlib.import_module(factory[1])
    assert callable(getattr(module, factory[2]))
