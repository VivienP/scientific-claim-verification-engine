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


def test_goodwin_nejm_2022_abstract_only_returns_unverifiable() -> None:
    """Regression pin for the Goodwin NEJM 2022 / 20% sustained response case.

    The verifier previously emitted status='unsupported', confidence=0.75 while
    internally recording evidence_quality='abstract_only' and
    fulltext_available=False. After A2, the emission gate must downgrade this
    to status='unverifiable', confidence=None.

    This test loads the regression entry from eval/regressions/ and asserts
    that the new behavior matches the expected_behavior described there.
    """
    from unittest.mock import MagicMock, patch

    from anthropic.types import TextBlock

    from src.models import Claim, ResolvedSource

    # Load the regression entry to verify it exists and has the right shape.
    goodwin_entry: dict[str, object] | None = None
    for case in _CASES:
        if case.get("regression_id") == "elicit_psilocybin__ae1ff864":
            goodwin_entry = case
            break

    assert goodwin_entry is not None, (
        "Regression entry 'elicit_psilocybin__ae1ff864' not found in "
        "eval/regressions/2026-05-11/abstract_only_unsupported/regression.jsonl"
    )
    assert goodwin_entry["claim_text"] == (
        "Sustained response rates at 12 weeks were only 20% in the largest randomized trial"
    )

    claim = Claim(
        claim_id="elicit_psilocybin__ae1ff864",
        claim_text=str(goodwin_entry["claim_text"]),
        cited_authors=["Goodwin"],
        cited_year=2022,
        claim_type="factual_numeric",
    )
    source = ResolvedSource(
        found=True,
        doi=str(goodwin_entry["source_doi"]),
        title=str(goodwin_entry["source_title"]),
        # Representative NEJM abstract (does not mention the 20% figure explicitly)
        abstract=(
            "Background: Psilocybin therapy has shown promise as a treatment for "
            "treatment-resistant major depressive disorder. Methods: We conducted a "
            "phase 2 double-blind randomized controlled trial comparing single doses of "
            "psilocybin (25 mg, 10 mg, 1 mg control) in 233 patients. Primary outcome "
            "was change in MADRS score at 3 weeks. Secondary outcomes included response "
            "and remission rates at 3 weeks, and sustained response at 12 weeks. "
            "Results: MADRS score decreased significantly in the 25-mg group. "
            "The incidences of response and remission at 3 weeks, but not sustained "
            "response at 12 weeks, were generally supportive of the primary results. "
            "Adverse events occurred in 179 of 233 participants (77%)."
        ),
        similarity_score=0.95,
    )

    # Mock the LLM to return the original buggy response (unsupported + 0.75).
    # After A2, the emission gate must downgrade this to (unverifiable, None).
    def _text_block(text: str) -> TextBlock:
        return TextBlock(type="text", text=text)

    with patch("src.verify.anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [
            _text_block(
                '{"status": "unsupported", "confidence": 0.75, "explanation": '
                '"The abstract does not report a specific 20% sustained response rate '
                "at 12 weeks. It states that sustained response at 12 weeks was NOT "
                'generally supportive of the primary results in the 25-mg group."}'
            )
        ]
        mock_response.usage.input_tokens = 200
        mock_response.usage.output_tokens = 60
        mock_response.usage.cache_read_input_tokens = 200
        mock_response.usage.cache_creation_input_tokens = 0
        mock_client.messages.create.return_value = mock_response

        from src.verify import verify_claim

        result, step = verify_claim(claim, source)

    # Assert: the Goodwin case must now return unverifiable, NOT unsupported.
    assert result.status == "unverifiable", (
        f"Expected 'unverifiable' but got {result.status!r}. "
        "The A2 emission gate must downgrade unsupported+abstract_only -> unverifiable."
    )
    assert result.confidence is None
    assert result.evidence_quality == "abstract_only"
    # C3 (2026-05-23): verify.py now emits "insufficient_evidence_depth" (post-MIRROR
    # 3.4, qualitative claims downgrade too so the reason is no longer claim-type-specific).
    # The "numeric_claim_abstract_only" literal is kept in UnverifiableReason for
    # backward-compat with historical provenance.jsonl files but is no longer emitted
    # at new call sites in verify.py.
    assert result.unverifiable_reason == "insufficient_evidence_depth", (
        f"Expected reason 'insufficient_evidence_depth' but got "
        f"{result.unverifiable_reason!r}. verify.py must pass "
        "unverifiable_reason='insufficient_evidence_depth' to the helper (C3)."
    )
    assert step.unverifiable_reason == "insufficient_evidence_depth", (
        "ProvenanceStep.unverifiable_reason must mirror result.unverifiable_reason."
    )
    # F1: the explanation is rewritten by safe_verification_result so the
    # verdict and explanation stay consistent. The original LLM "unsupported"
    # narrative is preserved as a truncated suffix.
    assert "Pipeline could not verify" in result.explanation, (
        f"Expected pipeline-limit framing in explanation, got: {result.explanation!r}"
    )
    assert "abstract_only" in result.explanation, (
        "Explanation must surface the evidence_quality that triggered the downgrade."
    )
    # The original LLM explanation should be preserved (truncated) as a suffix:
    assert (
        "20% sustained response rate" in result.explanation
        or "20% sustained response" in result.explanation
    ), "Original LLM explanation should be preserved (truncated) as a suffix."


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
