"""Enforce and report the fast-test performance budget."""

from __future__ import annotations

from typing import Any

import pytest


SLOW_TEST_WARNING_SECONDS = 1.0


def pytest_configure(config: pytest.Config) -> None:
    config._slow_fast_test_reports: list[tuple[str, float]] = []  # type: ignore[attr-defined]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Keep explicitly slow tests out of the fast-test performance budget."""
    for item in items:
        if item.get_closest_marker("slow"):
            item.add_marker(pytest.mark.timeout(0))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[Any]):
    outcome = yield
    report = outcome.get_result()
    if (
        report.when == "call"
        and not item.get_closest_marker("slow")
        and report.duration > SLOW_TEST_WARNING_SECONDS
    ):
        item.config._slow_fast_test_reports.append((report.nodeid, report.duration))  # type: ignore[attr-defined]


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter, exitstatus: int, config: pytest.Config
) -> None:
    slow_reports = config._slow_fast_test_reports  # type: ignore[attr-defined]
    if not slow_reports:
        return

    terminalreporter.write_sep("=", "fast-test budget warnings")
    for nodeid, duration in slow_reports:
        terminalreporter.write_line(
            f"WARNING: {nodeid} took {duration:.2f}s; the fast-test warning budget is 1.00s."
        )
