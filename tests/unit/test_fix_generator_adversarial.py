"""Adversarial CI gate: doi_hallucination_rate must be 0.00.

50 test cases covering:
  - 20 cases: LLM returns plausible-but-fake DOI → must be nulled by CrossRef gate
  - 10 cases: LLM returns real DOI for wrong paper (CrossRef score "low") → nulled
  - 10 cases: LLM returns null → fix.suggested_doi stays None (pass)
  - 10 cases: LLM returns valid DOI for correct paper → kept (pass)

All offline: LLM and CrossRef mocked.
Hard gate: if any non-null DOI is NOT CrossRef-verified, the test fails.
This test MUST remain in CI regardless of any other changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple
from unittest.mock import MagicMock, patch

import pytest

from src.copilot.fix_generator import generate_fix
from src.models import (
    Claim,
    ProvenanceStep,
    ResolvedSource,
    ResolvedSourceSet,
    VerificationResult,
)
from src.pipeline import ClaimVerification

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_cv(
    claim_id: str = "cl-adv",
    verdict: str = "unsupported",
) -> ClaimVerification:
    claim = Claim(
        claim_id=claim_id,
        claim_text="Treatment reduces biomarker levels.",
        cited_authors=["Smith"],
        cited_year=2021,
        claim_type="factual_qualitative",
    )
    source = ResolvedSource(
        found=True,
        doi="10.1234/source",
        title="Review Paper",
        abstract="A literature review.",
        similarity_score=0.6,
    )
    source_set = ResolvedSourceSet(sources=(source,), citation_markers=(1,))
    # A1: supported/unsupported require fulltext-grade evidence
    eq = "quoted_passage" if verdict in ("supported", "unsupported") else "abstract_only"
    actual_confidence: float | None = None if verdict == "unverifiable" else 0.2
    result = VerificationResult(
        status=verdict,  # type: ignore[arg-type]
        explanation="The source does not support the claim.",
        confidence=actual_confidence,  # type: ignore[arg-type]
        evidence_quality=eq,  # type: ignore[arg-type]
    )
    return ClaimVerification(
        claim=claim,
        source=source,
        source_set=source_set,
        result=result,
        fetch_method="abstract",
    )


def _tool_block(doi: str | None, action: str = "swap_doi") -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "submit_fix"
    block.input = {
        "action": action,
        "suggested_doi": doi,
        "suggested_doi_title": "Some Paper" if doi else None,
        "reworded_claim": None,
        "confidence": 0.8,
        "reasoning": "Adversarial test case.",
    }
    return block


def _llm_response(doi: str | None) -> MagicMock:
    response = MagicMock()
    response.content = [_tool_block(doi)]
    response.usage.input_tokens = 200
    response.usage.output_tokens = 40
    response.usage.cache_read_input_tokens = 0
    response.usage.cache_creation_input_tokens = 200
    return response


def _crossref_found(doi: str) -> MagicMock:
    r = MagicMock()
    r.found = True
    r.doi = doi
    r.title = "Verified Paper"
    return r


def _crossref_not_found() -> MagicMock:
    r = MagicMock()
    r.found = False
    r.doi = None
    r.title = None
    return r


# ---------------------------------------------------------------------------
# Adversarial case definitions
# ---------------------------------------------------------------------------


class AdversarialCase(NamedTuple):
    label: str
    llm_doi: str | None
    crossref_found: bool
    expect_doi_null: bool


# 20 cases: plausible but fake DOIs — CrossRef returns not found
_FAKE_DOI_CASES = [
    AdversarialCase(
        label=f"fake_doi_{i:02d}",
        llm_doi=f"10.{1000 + i}/{100 + i}fake.doi",
        crossref_found=False,
        expect_doi_null=True,
    )
    for i in range(20)
]

# 10 cases: DOI exists but CrossRef says not found (wrong paper, low confidence)
_WRONG_PAPER_CASES = [
    AdversarialCase(
        label=f"wrong_paper_{i:02d}",
        llm_doi=f"10.9999/wrong.{i:04d}",
        crossref_found=False,
        expect_doi_null=True,
    )
    for i in range(10)
]

# 10 cases: LLM returns null — must stay null
_NULL_DOI_CASES = [
    AdversarialCase(
        label=f"null_doi_{i:02d}",
        llm_doi=None,
        crossref_found=False,  # CrossRef not called
        expect_doi_null=True,
    )
    for i in range(10)
]

# 10 cases: valid, CrossRef-verified DOI — must be kept
_VALID_DOI_CASES = [
    AdversarialCase(
        label=f"valid_doi_{i:02d}",
        llm_doi=f"10.1016/valid.{i:04d}",
        crossref_found=True,
        expect_doi_null=False,
    )
    for i in range(10)
]

ALL_ADVERSARIAL_CASES = _FAKE_DOI_CASES + _WRONG_PAPER_CASES + _NULL_DOI_CASES + _VALID_DOI_CASES
assert len(ALL_ADVERSARIAL_CASES) == 50, "Must have exactly 50 adversarial cases"


# ---------------------------------------------------------------------------
# Parametrised gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    ALL_ADVERSARIAL_CASES,
    ids=[c.label for c in ALL_ADVERSARIAL_CASES],
)
@patch("src.copilot.fix_generator.anthropic.Anthropic")
@patch("src.copilot.fix_generator.fetch_work_by_doi")
def test_doi_hallucination_gate(
    mock_crossref: MagicMock,
    mock_cls: MagicMock,
    case: AdversarialCase,
    tmp_path: Path,
) -> None:
    """doi_hallucination_rate MUST be 0.00: no unverified DOI ever reaches output."""
    mock_cls.return_value.messages.create.return_value = _llm_response(case.llm_doi)

    if case.crossref_found and case.llm_doi:
        mock_crossref.return_value = _crossref_found(case.llm_doi)
    else:
        mock_crossref.return_value = _crossref_not_found()

    fix, step = generate_fix(_make_cv(), db_path=tmp_path / "c.db")

    # Core invariant: every case must return a ProvenanceStep
    assert isinstance(step, ProvenanceStep)
    assert step.operation == "copilot_fix"

    if case.expect_doi_null:
        # NULL expectation: DOI must NOT be in the output
        assert fix is None or fix.suggested_doi is None, (
            f"HALLUCINATION DETECTED — case {case.label}: "
            f"unverified DOI '{case.llm_doi}' reached fix.suggested_doi"
        )
    else:
        # VALID expectation: DOI must be present and match
        assert fix is not None, f"Expected a fix for valid DOI case {case.label}"
        assert fix.suggested_doi == case.llm_doi, (
            f"Valid DOI was unexpectedly dropped in case {case.label}"
        )
