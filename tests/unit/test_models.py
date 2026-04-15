"""Unit tests for src/models.py — frozen dataclasses and type aliases."""

from __future__ import annotations

import pytest

from src.models import Claim, ProvenanceStep, ResolvedSource, VerificationResult


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

    def test_claim_cited_year_none(self) -> None:
        claim = Claim(
            claim_id="abc",
            claim_text="Some claim",
            cited_authors=[],
            cited_year=None,
            claim_type="methodological",
        )
        assert claim.cited_year is None

    def test_claim_all_types(self) -> None:
        for claim_type in ("factual_numeric", "factual_qualitative", "methodological", "causal"):
            claim = Claim(
                claim_id="x",
                claim_text="t",
                cited_authors=[],
                cited_year=None,
                claim_type=claim_type,  # type: ignore[arg-type]
            )
            assert claim.claim_type == claim_type


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

    def test_resolved_source_not_found(self) -> None:
        source = ResolvedSource(
            found=False,
            doi=None,
            title=None,
            abstract=None,
            similarity_score=None,
        )
        assert source.found is False
        assert source.similarity_score is None

    def test_resolved_source_frozen(self) -> None:
        source = ResolvedSource(
            found=False, doi=None, title=None, abstract=None, similarity_score=None
        )
        with pytest.raises((AttributeError, TypeError)):
            source.found = True  # type: ignore[misc]


class TestVerificationResult:
    def test_verification_result_fields(self) -> None:
        result = VerificationResult(
            status="supported",
            explanation="The abstract supports this.",
            confidence=0.9,
        )
        assert result.status == "supported"
        assert result.confidence == 0.9

    def test_verification_result_all_statuses(self) -> None:
        for status in ("supported", "unsupported", "not_addressed", "partially_supported"):
            result = VerificationResult(
                status=status,  # type: ignore[arg-type]
                explanation="Explanation.",
                confidence=0.5,
            )
            assert result.status == status

    def test_verification_result_frozen(self) -> None:
        result = VerificationResult(status="supported", explanation="ok", confidence=1.0)
        with pytest.raises((AttributeError, TypeError)):
            result.status = "unsupported"  # type: ignore[misc]


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

    def test_provenance_step_none_fields(self) -> None:
        step = ProvenanceStep(
            step_id="step-002",
            claim_id="claim-002",
            operation="resolve",
            input_hash="aaa",
            output_hash="bbb",
            model_id=None,
            timestamp=1234567890.0,
            tokens_in=None,
            tokens_out=None,
            cache_hit=None,
            confidence=None,
        )
        assert step.model_id is None
        assert step.tokens_in is None
        assert step.cache_hit is None

    def test_provenance_step_frozen(self) -> None:
        step = ProvenanceStep(
            step_id="s",
            claim_id="c",
            operation="extract",
            input_hash="h1",
            output_hash="h2",
            model_id=None,
            timestamp=0.0,
            tokens_in=None,
            tokens_out=None,
            cache_hit=None,
            confidence=None,
        )
        with pytest.raises((AttributeError, TypeError)):
            step.operation = "resolve"  # type: ignore[misc]

    def test_provenance_step_all_operations(self) -> None:
        for op in ("extract", "resolve", "verify", "aggregate"):
            step = ProvenanceStep(
                step_id="s",
                claim_id="c",
                operation=op,  # type: ignore[arg-type]
                input_hash="h1",
                output_hash="h2",
                model_id=None,
                timestamp=0.0,
                tokens_in=None,
                tokens_out=None,
                cache_hit=None,
                confidence=None,
            )
            assert step.operation == op
