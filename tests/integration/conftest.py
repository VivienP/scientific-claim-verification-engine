"""Integration-test gating.

Per `.claude/rules/offline-tests.md`, integration tests make real network
calls and are NOT run by the pre-commit hook. They are run manually with:

    pytest tests/integration/ -v --run-integration

Without the flag, every test in `tests/integration/` is skipped at collection
time. This keeps the default `pytest tests/unit/` invocation fast and
reproducible.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that make real network calls.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-integration"):
        return
    skip_marker = pytest.mark.skip(reason="Integration test — pass --run-integration to enable.")
    for item in items:
        item.add_marker(skip_marker)
