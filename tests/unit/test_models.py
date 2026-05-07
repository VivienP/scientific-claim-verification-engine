"""Unit tests for src/models.py — frozen dataclasses and type aliases."""

from __future__ import annotations

import dataclasses

import pytest

from src.models import Claim, ProvenanceStep, ResolvedSource, VerificationResult
from src.numeric.checks import NumericAssertion, NumericCheckResult


class TestClaim:
    def test_claim_fields(self) -> None:
        claim = Claim(
            claim_id="abc-123",
            claim_text="X causes Y",
            cited_authors=["Smith", "Jones"],
            cited_year=2020,
            claim_type="causal",
        )
        assert claim.claim_id == "abc-123"
        assert claim.claim_text == "X causes Y"
        assert claim.cited_authors == ["Smith", "Jones"]
        assert claim.cited_year == 2020
        assert claim.claim_type == "causal"
        assert claim.citation_markers == []

    def test_claim_frozen(self) -> None:
        claim = Claim(
            claim_id="abc-123",
            claim_text="X causes Y",
            cited_authors=[],
            cited_year=None,
            claim_type="factual_qualitative",
        )
        with pytest.raises((AttributeError, TypeError)):
            claim.claim_text = "modified"  # type: ignore[misc]


class TestResolvedSource:
    def test_resolved_source_found(self) -> None:
        source = ResolvedSource(
            found=True,
            doi="10.1000/test",
            title="Test Paper",
            abstract="An abstract.",
            similarity_score=0.95,
        )
        assert source.found is True
        assert source.doi == "10.1000/test"
        assert source.similarity_score == 0.95
        assert source.title_match_score is None
        assert source.resolution_low_confidence is False


class TestResolvedSourceSet:
    """S2-P4: ResolvedSourceSet wraps multi-citation resolution results."""

    @staticmethod
    def _src(
        *,
        found: bool = True,
        doi: str | None = "10.1/x",
        title_match: float | None = 1.0,
    ) -> ResolvedSource:
        return ResolvedSource(
            found=found,
            doi=doi,
            title="t" if found else None,
            abstract="a" if found else None,
            similarity_score=1.0 if found else None,
            title_match_score=title_match,
        )

    def test_primary_picks_highest_title_match(self) -> None:
        from src.models import ResolvedSourceSet

        weak = self._src(doi="10.1/weak", title_match=0.3)
        strong = self._src(doi="10.1/strong", title_match=0.95)
        rs_set = ResolvedSourceSet(sources=(weak, strong), citation_markers=(81, 82))
        assert rs_set.primary().doi == "10.1/strong"

    def test_primary_prefers_found_over_unfound(self) -> None:
        from src.models import ResolvedSourceSet

        unfound = self._src(found=False, doi=None, title_match=None)
        found = self._src(doi="10.1/y", title_match=0.5)
        rs_set = ResolvedSourceSet(sources=(unfound, found), citation_markers=(1, 2))
        assert rs_set.primary().doi == "10.1/y"

    def test_primary_returns_not_found_on_empty(self) -> None:
        from src.models import ResolvedSourceSet

        rs_set = ResolvedSourceSet(sources=(), citation_markers=())
        primary = rs_set.primary()
        assert primary.found is False
        assert primary.doi is None

    def test_found_sources_filters_unresolved(self) -> None:
        from src.models import ResolvedSourceSet

        a = self._src(doi="10.1/a")
        b = self._src(found=False, doi=None)
        c = self._src(doi="10.1/c")
        rs_set = ResolvedSourceSet(sources=(a, b, c), citation_markers=(1, 2, 3))
        found = rs_set.found_sources()
        assert len(found) == 2
        assert {s.doi for s in found} == {"10.1/a", "10.1/c"}


class TestVerificationResult:
    def test_verification_result_fields(self) -> None:
        result = VerificationResult(
            status="supported",
            explanation="The abstract supports this.",
            confidence=0.9,
        )
        assert result.status == "supported"
        assert result.confidence == 0.9
        assert result.retrieval_status == "fulltext_unavailable"
        assert result.evidence_quality == "abstract_only"


class TestProvenanceStep:
    def test_provenance_step_fields(self) -> None:
        step = ProvenanceStep(
            step_id="step-001",
            claim_id="claim-001",
            operation="verify",
            input_hash="abc123",
            output_hash="def456",
            model_id="claude-sonnet-4-6",
            timestamp=1234567890.0,
            tokens_in=100,
            tokens_out=50,
            cache_hit=True,
            confidence=0.85,
        )
        assert step.step_id == "step-001"
        assert step.operation == "verify"
        assert step.model_id == "claude-sonnet-4-6"
        assert step.tokens_in == 100
        assert step.cache_hit is True


class TestVerificationResultNumericCheck:
    def test_default_numeric_check_is_none(self) -> None:
        v = VerificationResult(status="supported", explanation="ok", confidence=0.9)
        assert v.numeric_check is None

    def test_numeric_check_round_trip_through_asdict(self) -> None:
        nc = NumericCheckResult(
            check_type="or_ci_consistency",
            consistent=True,
            extracted=[
                NumericAssertion(
                    raw_text="OR 40.53",
                    value=40.53,
                    unit=None,
                    role="primary",
                    context="odds ratio",
                ),
            ],
            explanation="OR/CI internally consistent.",
        )
        v = VerificationResult(
            status="supported",
            explanation="ok",
            confidence=0.9,
            numeric_check=nc,
        )
        d = dataclasses.asdict(v)
        assert d["numeric_check"]["check_type"] == "or_ci_consistency"
        assert d["numeric_check"]["consistent"] is True
        assert d["numeric_check"]["extracted"][0]["value"] == 40.53
