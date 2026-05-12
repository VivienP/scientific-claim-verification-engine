"""Exhaustive policy-table tests for src/policy/evidence_sufficiency.

The policy is a pure function with three orthogonal gates (resolution status,
depth, access). Tests cover the cross-product of input shapes the pipeline
actually produces:

  * Sufficient — qualitative claim on abstract; any claim on fulltext.
  * Insufficient(numeric_claim_abstract_only) — numeric claim on abstract / title / none.
  * Insufficient(resolution_source_disagreement) — disputed verdict regardless of depth.
  * Insufficient(resolution_low_confidence) — low_confidence verdict + numeric claim.
  * Insufficient(fulltext_unavailable) — blocked/unresolved access + no text.

Each row is a single ``Sufficient`` or ``Insufficient(reason)`` assertion so a
breaking change to the policy fails the most specific case rather than a
broad behavioral test.
"""

from __future__ import annotations

from src.models import Claim, EvidenceBundle, ResolvedSource
from src.policy import (
    Insufficient,
    Sufficient,
    assess_evidence_sufficiency,
)


def _numeric_claim() -> Claim:
    return Claim(
        claim_id="num-1",
        claim_text="The HR for MACE was 0.74 (95% CI 0.58-0.95) at week 12.",
        cited_authors=["Smith"],
        cited_year=2022,
        claim_type="factual_numeric",
    )


def _qualitative_claim() -> Claim:
    return Claim(
        claim_id="qual-1",
        claim_text="Psilocybin shows promise for treatment-resistant depression.",
        cited_authors=["Goodwin"],
        cited_year=2022,
        claim_type="factual_qualitative",
    )


def _source() -> ResolvedSource:
    return ResolvedSource(
        found=True,
        doi="10.1/x",
        title="Sample title",
        abstract="Sample abstract.",
        similarity_score=1.0,
    )


def _bundle(
    *,
    depth: str = "abstract",
    access: str = "available",
    resolution: str = "single_source_only",
    text: str | None = "abstract text",
) -> EvidenceBundle:
    return EvidenceBundle(
        text=text,
        depth=depth,  # type: ignore[arg-type]
        access_status=access,  # type: ignore[arg-type]
        source_resolution_status=resolution,  # type: ignore[arg-type]
    )


class TestSufficient:
    """Inputs that should NOT trip the policy gate."""

    def test_qualitative_claim_on_abstract_is_sufficient(self) -> None:
        decision = assess_evidence_sufficiency(
            _qualitative_claim(), _source(), _bundle(depth="abstract")
        )
        assert isinstance(decision, Sufficient)

    def test_qualitative_claim_on_fulltext_is_sufficient(self) -> None:
        decision = assess_evidence_sufficiency(
            _qualitative_claim(),
            _source(),
            _bundle(depth="fulltext", text="full body"),
        )
        assert isinstance(decision, Sufficient)

    def test_numeric_claim_on_fulltext_is_sufficient(self) -> None:
        decision = assess_evidence_sufficiency(
            _numeric_claim(),
            _source(),
            _bundle(depth="fulltext", text="full body"),
        )
        assert isinstance(decision, Sufficient)

    def test_qualitative_claim_on_title_is_sufficient(self) -> None:
        decision = assess_evidence_sufficiency(
            _qualitative_claim(),
            _source(),
            _bundle(depth="title", text=None),
        )
        assert isinstance(decision, Sufficient)

    def test_corroborated_resolution_is_sufficient(self) -> None:
        decision = assess_evidence_sufficiency(
            _qualitative_claim(),
            _source(),
            _bundle(resolution="corroborated"),
        )
        assert isinstance(decision, Sufficient)


class TestDepthGate:
    """Numeric claim + abstract/title/none -> Insufficient(numeric_claim_abstract_only)."""

    def test_numeric_claim_on_abstract_is_insufficient(self) -> None:
        decision = assess_evidence_sufficiency(
            _numeric_claim(), _source(), _bundle(depth="abstract")
        )
        assert isinstance(decision, Insufficient)
        assert decision.reason == "numeric_claim_abstract_only"

    def test_numeric_claim_on_title_is_insufficient(self) -> None:
        decision = assess_evidence_sufficiency(
            _numeric_claim(),
            _source(),
            _bundle(depth="title", text=None),
        )
        assert isinstance(decision, Insufficient)
        assert decision.reason == "numeric_claim_abstract_only"

    def test_numeric_claim_on_none_depth_is_insufficient(self) -> None:
        decision = assess_evidence_sufficiency(
            _numeric_claim(),
            _source(),
            _bundle(depth="none", access="unavailable", text=None),
        )
        assert isinstance(decision, Insufficient)
        assert decision.reason == "numeric_claim_abstract_only"


class TestResolutionGate:
    """Resolution-verdict gates fire ahead of depth gates."""

    def test_disputed_resolution_is_insufficient_even_on_fulltext(self) -> None:
        # The disputed gate has highest priority: a wrong paper means no
        # verification, regardless of how deep the wrong-paper evidence is.
        decision = assess_evidence_sufficiency(
            _qualitative_claim(),
            _source(),
            _bundle(depth="fulltext", text="body", resolution="disputed"),
        )
        assert isinstance(decision, Insufficient)
        assert decision.reason == "resolution_source_disagreement"

    def test_low_confidence_resolution_blocks_numeric_claim_only(self) -> None:
        decision = assess_evidence_sufficiency(
            _numeric_claim(),
            _source(),
            _bundle(depth="fulltext", text="body", resolution="low_confidence"),
        )
        assert isinstance(decision, Insufficient)
        assert decision.reason == "resolution_low_confidence"

    def test_low_confidence_resolution_does_not_block_qualitative_claim(self) -> None:
        decision = assess_evidence_sufficiency(
            _qualitative_claim(),
            _source(),
            _bundle(depth="fulltext", text="body", resolution="low_confidence"),
        )
        # low_confidence affects only specific-numeric claims; qualitative
        # claims can still be verified against a weakly-resolved source.
        assert isinstance(decision, Sufficient)


class TestAccessGate:
    """Access-status gates fire when no text is available at all."""

    def test_unresolved_access_with_no_text_is_insufficient(self) -> None:
        decision = assess_evidence_sufficiency(
            _qualitative_claim(),
            _source(),
            _bundle(depth="none", access="unresolved", text=None),
        )
        assert isinstance(decision, Insufficient)
        assert decision.reason == "fulltext_unavailable"

    def test_blocked_access_with_no_text_is_insufficient(self) -> None:
        decision = assess_evidence_sufficiency(
            _qualitative_claim(),
            _source(),
            _bundle(depth="none", access="blocked", text=None),
        )
        assert isinstance(decision, Insufficient)
        assert decision.reason == "fulltext_unavailable"

    def test_unavailable_access_with_no_text_is_sufficient_for_qualitative(self) -> None:
        # "unavailable" by itself is not a fail-stop — depth gates already
        # cover the no-text case for numeric claims, and qualitative claims
        # can survive a fetch failure as long as some text is available.
        # When depth=none + access=unavailable + qualitative, the policy
        # currently lets it through; the verifier's short-circuit (no source
        # text) handles the final emission.
        decision = assess_evidence_sufficiency(
            _qualitative_claim(),
            _source(),
            _bundle(depth="none", access="unavailable", text=None),
        )
        assert isinstance(decision, Sufficient)


class TestPolicyOrdering:
    """When multiple gates would fire, the higher-priority one wins."""

    def test_disputed_takes_priority_over_depth(self) -> None:
        # Both disputed (highest) and numeric+abstract (lower) apply.
        # Expected: disputed wins.
        decision = assess_evidence_sufficiency(
            _numeric_claim(),
            _source(),
            _bundle(depth="abstract", resolution="disputed"),
        )
        assert isinstance(decision, Insufficient)
        assert decision.reason == "resolution_source_disagreement"

    def test_low_confidence_numeric_takes_priority_over_depth(self) -> None:
        decision = assess_evidence_sufficiency(
            _numeric_claim(),
            _source(),
            _bundle(depth="abstract", resolution="low_confidence"),
        )
        assert isinstance(decision, Insufficient)
        # Resolution gate fires before depth gate.
        assert decision.reason == "resolution_low_confidence"

    def test_depth_takes_priority_over_access(self) -> None:
        # numeric + abstract + blocked: depth gate fires first.
        decision = assess_evidence_sufficiency(
            _numeric_claim(),
            _source(),
            _bundle(depth="abstract", access="blocked"),
        )
        assert isinstance(decision, Insufficient)
        assert decision.reason == "numeric_claim_abstract_only"


class TestPolicyPurity:
    """The policy must not touch ``source`` fields beyond the bundle."""

    def test_decision_is_independent_of_source_doi(self) -> None:
        bundle = _bundle()
        a = assess_evidence_sufficiency(_qualitative_claim(), _source(), bundle)
        b = assess_evidence_sufficiency(
            _qualitative_claim(),
            ResolvedSource(
                found=True,
                doi="10.99/other",
                title="other",
                abstract="other abs",
                similarity_score=0.1,
            ),
            bundle,
        )
        # Same claim text, same bundle -> same decision regardless of source fields.
        assert type(a) is type(b)
