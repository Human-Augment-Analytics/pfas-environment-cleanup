"""Fast local quality checks, run with ``./scripts/dev <task>``."""

from pathlib import Path
from shlex import quote

from invoke import task


ROOT = Path(__file__).parent
TOOL_BIN = ROOT / ".venv" / "bin"


def _run(context, label: str, command: str) -> None:
    print(f"\n==> {label}\n$ {command}")
    context.run(f"cd {quote(str(ROOT))} && {command}")


@task
def ruff(context) -> None:
    """Run Ruff's repository-wide correctness checks."""
    _run(context, "Ruff", f"{quote(str(TOOL_BIN / 'ruff'))} check .")


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


@task(pre=[quality, fast_test])
def ci(context) -> None:
    """Run the complete CI-equivalent quality suite."""
    pass
