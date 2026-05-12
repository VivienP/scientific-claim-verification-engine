"""Unit tests for A2 changes in src/verify_multi.py."""

from __future__ import annotations

from src.models import (
    Claim,
    ResolvedSource,
    ResolvedSourceSet,
    VerificationResult,
)


def _make_claim(claim_id: str = "claim-m1") -> Claim:
    return Claim(
        claim_id=claim_id,
        claim_text="Protein folding rates increase with temperature.",
        cited_authors=["Smith"],
        cited_year=2020,
        claim_type="factual_qualitative",
    )


def _make_result(
    status: str,
    confidence: float | None,
    *,
    verification_depth: str = "abstract",
    evidence_quality: str = "abstract_only",
) -> VerificationResult:
    return VerificationResult(
        status=status,  # type: ignore[arg-type]
        explanation="test",
        confidence=confidence,
        verification_depth=verification_depth,  # type: ignore[arg-type]
        evidence_quality=evidence_quality,  # type: ignore[arg-type]
    )


class TestAggregateMultiSourceUnverifiable:
    """A2: _aggregate_multi_source_verdicts handles the new 'unverifiable' status."""

    def test_all_unverifiable_returns_unverifiable(self) -> None:
        from src.verify_multi import _aggregate_multi_source_verdicts

        results = [
            _make_result("unverifiable", None),
            _make_result("unverifiable", None),
        ]
        assert _aggregate_multi_source_verdicts(results) == "unverifiable"

    def test_unverifiable_with_supported_returns_partially_supported(self) -> None:
        """unverifiable contributes no evidence signal, like not_addressed."""
        from src.verify_multi import _aggregate_multi_source_verdicts

        results = [
            _make_result("unverifiable", None),
            _make_result(
                "supported",
                0.9,
                verification_depth="fulltext",
                evidence_quality="quoted_passage",
            ),
        ]
        assert _aggregate_multi_source_verdicts(results) == "partially_supported"

    def test_mixed_unverifiable_and_not_addressed_returns_not_addressed(self) -> None:
        from src.verify_multi import _aggregate_multi_source_verdicts

        results = [
            _make_result("unverifiable", None),
            _make_result("not_addressed", 0.5),
        ]
        assert _aggregate_multi_source_verdicts(results) == "not_addressed"


class TestVerifyMultiDepthAggregation:
    """A2: verify_claim_multi_source aggregates depth/evidence from best source."""

    def test_aggregated_depth_inherits_from_best_source(self) -> None:
        """With two sources, one with fulltext evidence and one abstract-only,
        the aggregate verification_depth should come from the best source (fulltext).
        Uses patch on src.verify to intercept the deferred imports inside
        verify_claim_multi_source."""
        from unittest.mock import patch

        fulltext_result = VerificationResult(
            status="supported",
            explanation="quoted from passage",
            confidence=0.9,
            verification_depth="fulltext",
            evidence_quality="quoted_passage",
            fulltext_available=True,
            retrieval_status="passage_found",
        )
        abstract_result = VerificationResult(
            status="unverifiable",
            explanation="abstract only",
            confidence=None,
            verification_depth="abstract",
            evidence_quality="abstract_only",
        )

        s1 = ResolvedSource(
            found=True, doi="10.1/a", title="A", abstract="abs", similarity_score=1.0
        )
        s2 = ResolvedSource(
            found=True, doi="10.1/b", title="B", abstract="abs", similarity_score=1.0
        )
        rs_set = ResolvedSourceSet(sources=(s1, s2), citation_markers=(1, 2))

        # verify_claim_multi_source uses deferred imports from src.verify
        with patch("src.verify.verify_claim") as mock_vc:
            from src.models import ProvenanceStep

            def _make_step(claim_id: str) -> ProvenanceStep:
                return ProvenanceStep(
                    step_id="s",
                    claim_id=claim_id,
                    operation="verify",
                    input_hash="i",
                    output_hash="o",
                    model_id="m",
                    timestamp=0.0,
                    tokens_in=10,
                    tokens_out=5,
                    cache_hit=False,
                    confidence=None,
                )

            # Both sources go through verify_claim (no passages provided).
            # First call returns fulltext-grade result, second returns abstract.
            mock_vc.side_effect = [
                (fulltext_result, _make_step("claim-m1")),
                (abstract_result, _make_step("claim-m1")),
            ]

            from src.verify_multi import verify_claim_multi_source

            result, steps = verify_claim_multi_source(_make_claim(), rs_set)
            # Best source is fulltext -> aggregated depth must be fulltext
            assert result.verification_depth == "fulltext"
            assert result.evidence_quality == "quoted_passage"
            assert len(steps) == 3  # 2 per-source steps + 1 aggregate step


class TestEmptySourceSetGuard:
    """A1 hardening (2026-05-12): empty ResolvedSourceSet must not crash with
    a schema-invariant ValueError. The pipeline guards this at the call site,
    but the function-level contract must also produce a valid VerificationResult
    so any direct caller gets a sensible response instead of a runtime crash.
    """

    def test_empty_source_set_returns_not_addressed_without_crashing(self) -> None:
        from src.verify_multi import verify_claim_multi_source

        empty_set = ResolvedSourceSet(sources=(), citation_markers=())
        result, steps = verify_claim_multi_source(_make_claim(), empty_set)

        assert result.status == "not_addressed"
        # confidence must be non-None (schema invariant 1: only unverifiable -> None)
        assert result.confidence is not None
        assert result.evidence_quality == "no_evidence"
        # No per-source LLM steps are emitted when there are no sources.
        assert steps == []
