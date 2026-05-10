"""Auto-discovers regression cases from eval/regressions/ and parametrizes them.

Each new regression added by /skillify-failure becomes a permanent test that
remains as a `pytest.skip` (with context) until the user wires the actual
pipeline call for that failure_category.

CI visibility caveat: when at least one regression case exists, the
parametrized test renders as one SKIPPED line per case (visible in `-v`
output). When zero cases exist (initial state), pytest's empty-parametrize
behavior emits a single placeholder SKIPPED entry — also visible. Neither
state silently passes; both fail loud if you replace `pytest.skip` with
a bad assertion.

When wiring a category, replace the skip with a real assertion. The recommended
pattern: import the relevant pipeline function, run it on `claim_text`, and
assert the verdict matches `expected_verdict`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

REGRESSIONS_DIR = Path(__file__).resolve().parents[2] / "eval" / "regressions"


def _collect_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    if not REGRESSIONS_DIR.is_dir():
        return cases
    for jsonl_path in sorted(REGRESSIONS_DIR.rglob("regression.jsonl")):
        for raw_line in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            cases.append(json.loads(line))
    return cases


_CASES = _collect_cases()


@pytest.mark.parametrize(
    "case",
    _CASES,
    ids=[str(c.get("regression_id", f"unknown-{i}")) for i, c in enumerate(_CASES)],
)
def test_regression_resolved(case: dict[str, Any]) -> None:
    """Placeholder until wired per failure_category. Skips are visible — they don't false-pass."""
    expected = case.get("expected_verdict")
    rid = case.get("regression_id", "unknown")
    category = case.get("failure_category", "unknown")
    if expected == "TBD":
        pytest.skip(f"expected_verdict is TBD for {rid} — fill it before this test is meaningful")
    pytest.skip(
        f"regression {rid} ({category}) awaits manual wiring — see eval/regressions/README.md"
    )


def test_collector_finds_jsonl_when_present() -> None:
    """Sanity check: the collector globs the right directory and parses jsonl correctly.

    Always passes, regardless of whether any regressions exist yet.

    Schema note: the original `failure_category` field was generalised to also
    accept `bug_class` for the richer numeric-workflow regression entries
    (eval/regressions/2026-05-10/elicit_numeric_workflow/). Both serve the same
    purpose — categorising the failure mode — so we require at least one.
    """
    cases = _collect_cases()
    assert isinstance(cases, list)
    for case in cases:
        assert "regression_id" in case, f"missing regression_id in case: {case}"
        has_category = "failure_category" in case or "bug_class" in case
        assert has_category, (
            f"missing categorisation field (failure_category or bug_class) in case: {case}"
        )
        assert "claim_text" in case, f"missing claim_text in case: {case}"
