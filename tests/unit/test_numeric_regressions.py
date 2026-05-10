"""Regression tests for _find_or_ci_triple wired to the captured JSONL entries.

Loads eval/regressions/2026-05-10/elicit_numeric_workflow/regression.jsonl
and asserts the expected behavior for each entry.

Each JSONL row has:
  - extracted_assertions: list of dicts matching NumericAssertion fields
  - expected_behavior: "no_check_applies" | "consistent"

TDD: these tests fail on the old engine, pass after the Bug A/B fix.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from src.numeric.checks import NumericAssertion
from src.numeric.engine import _find_or_ci_triple

_REGRESSION_JSONL = pathlib.Path(
    "eval/regressions/2026-05-10/elicit_numeric_workflow/regression.jsonl"
)


def _load_regression_rows() -> list[tuple[str, list[NumericAssertion], str]]:
    """Return (regression_id, assertions, expected_behavior) for each row."""
    rows: list[tuple[str, list[NumericAssertion], str]] = []
    with _REGRESSION_JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            assertions = [
                NumericAssertion(
                    raw_text=a["raw_text"],
                    value=a["value"],
                    unit=a.get("unit"),
                    role=a["role"],
                    context=a["context"],
                )
                for a in entry["extracted_assertions"]
            ]
            rows.append((entry["regression_id"], assertions, entry["expected_behavior"]))
    return rows


_REGRESSION_ROWS = _load_regression_rows()


@pytest.mark.parametrize(
    "regression_id,assertions,expected_behavior",
    _REGRESSION_ROWS,
    ids=[r[0] for r in _REGRESSION_ROWS],
)
def test_regression_entry(
    regression_id: str,
    assertions: list[NumericAssertion],
    expected_behavior: str,
) -> None:
    """For each regression entry, verify _find_or_ci_triple behaves as expected."""
    result = _find_or_ci_triple(assertions)

    if expected_behavior == "no_check_applies":
        assert result is None, (
            f"[{regression_id}] Expected None (no check applies) "
            f"but got {result!r} — Bug A/B regression not fixed."
        )
    elif expected_behavior == "consistent":
        from src.numeric.checks import check_or_ci_consistency

        assert result is not None, (
            f"[{regression_id}] Expected a triple but got None — happy path broke."
        )
        ratio, ci_low, ci_high = result
        check_result = check_or_ci_consistency(ratio, ci_low, ci_high, extracted=assertions)
        assert check_result.consistent is True, (
            f"[{regression_id}] Expected consistent=True but got {check_result!r}"
        )
    else:
        pytest.fail(f"[{regression_id}] Unknown expected_behavior: {expected_behavior!r}")
