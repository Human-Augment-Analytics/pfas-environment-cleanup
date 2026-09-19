"""Fast local quality checks, run with ``./scripts/dev <task>``."""

from pathlib import Path
from shlex import quote

from invoke import task


ROOT = Path(__file__).parent
TOOL_BIN = ROOT / ".env-pfas-ci" / "bin"


def _run(context, label: str, command: str) -> None:
    print(f"\n==> {label}\n$ {command}")
    context.run(f"cd {quote(str(ROOT))} && {command}")


@task
def ruff(context) -> None:
    """Run Ruff's repository-wide correctness checks."""
    _run(context, "Ruff", f"CLICOLOR_FORCE=1 {quote(str(TOOL_BIN / 'ruff'))} check .")


@task
def mypy(context) -> None:
    """Run the configured mypy checks."""
    _run(context, "mypy", quote(str(TOOL_BIN / "mypy")))


@task(pre=[ruff, mypy])
def quality(context) -> None:
    """Run all static quality checks."""
    pass


@task
def fast_test(context) -> None:
    """Run the fast unit-test suite, excluding explicitly slow tests."""
    _run(context, "fast pytest suite", f"{quote(str(TOOL_BIN / 'pytest'))} -q -m 'not slow'")


@task
def slow_test(context) -> None:
    """Run explicitly slow tests in the full data/ML Conda environment."""
    _run(
        context,
        "slow pytest suite in the pfas Conda environment",
        "conda run -n pfas python -m pytest -q -m slow --timeout=0 --session-timeout=0",
    )


@task
def update(context) -> None:
    """Reinstall the pinned developer requirements."""
    _run(
        context,
        "developer requirement update",
        f"uv pip install --reinstall --python {quote(str(TOOL_BIN / 'python'))} "
        "-r requirements-dev.txt",
    )


@task(pre=[quality, fast_test])
def ci(context) -> None:
    """Run the complete CI-equivalent quality suite."""
    pass
