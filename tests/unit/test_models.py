"""Unit tests for src/models.py — frozen dataclasses and type aliases."""

from __future__ import annotations

import dataclasses

import pytest

from src.models import Claim, ResolvedSource, VerificationResult, safe_verification_result
from src.numeric.checks import NumericAssertion, NumericCheckResult


class TestClaim:
    def test_claim_frozen(self) -> None:
        """Runtime check: dataclasses.frozen=True actually raises on assignment.

        Trivial field-roundtrip tests for frozen dataclasses are mypy-enforced
        and tautological; only the runtime frozen behaviour needs locking here.
        """
        claim = Claim(
            claim_id="abc-123",
            claim_text="X causes Y",
            cited_authors=[],
            cited_year=None,
            claim_type="factual_qualitative",
        )
        with pytest.raises((AttributeError, TypeError)):
            claim.claim_text = "modified"  # type: ignore[misc]


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

    def test_primary_returns_first_found_in_marker_order(self) -> None:
        """When the user writes `[7, 9]` they are stating that ref [7] is the
        primary citation by textual intent, regardless of which retrieved
        source happens to score higher on title-match. `primary()` must
        honor that order so the report's headline DOI matches the
        author's first-listed marker.

        Bug A (2026-05-08, Valsci validation run): for `[7, 9]` (Kinney+Lo), both
        sources resolved correctly, but `primary()` used to pick Lo because
        its title_match_score was higher — even though the user wrote
        Kinney first. The new contract: marker order is primary; scoring
        is only a tiebreaker when no source is found in marker order.
        """
        from src.models import ResolvedSourceSet

        # marker [7] → kinney (lower title_match), marker [9] → lo (higher)
        kinney = self._src(doi="10.1/kinney", title_match=0.4)
        lo = self._src(doi="10.1/lo", title_match=1.0)
        rs_set = ResolvedSourceSet(sources=(kinney, lo), citation_markers=(7, 9))
        assert rs_set.primary().doi == "10.1/kinney", (
            "First-marker source must be the primary regardless of score"
        )

    def test_primary_skips_unfound_to_first_found_in_marker_order(self) -> None:
        from src.models import ResolvedSourceSet

        unfound = self._src(found=False, doi=None, title_match=None)
        found_a = self._src(doi="10.1/a", title_match=0.3)
        found_b = self._src(doi="10.1/b", title_match=0.95)
        # Marker [1] failed; among found sources [2] is first → wins
        rs_set = ResolvedSourceSet(sources=(unfound, found_a, found_b), citation_markers=(1, 2, 3))
        assert rs_set.primary().doi == "10.1/a"

    def test_primary_prefers_found_over_unfound(self) -> None:
        from src.models import ResolvedSourceSet

        unfound = self._src(found=False, doi=None, title_match=None)
        found = self._src(doi="10.1/y", title_match=0.5)
        rs_set = ResolvedSourceSet(sources=(unfound, found), citation_markers=(1, 2))
        assert rs_set.primary().doi == "10.1/y"

    def test_primary_returns_first_unfound_when_all_failed(self) -> None:
        """When no source resolved, return the first attempt (still unfound-shaped)
        rather than picking arbitrarily. This preserves the marker order
        invariant even on full-failure runs.
        """
        from src.models import ResolvedSourceSet

        u1 = self._src(found=False, doi=None, title_match=None)
        u2 = self._src(found=False, doi=None, title_match=None)
        rs_set = ResolvedSourceSet(sources=(u1, u2), citation_markers=(7, 9))
        primary = rs_set.primary()
        assert primary.found is False

    def test_primary_returns_not_found_on_empty(self) -> None:
        from src.models import ResolvedSourceSet

        rs_set = ResolvedSourceSet(sources=(), citation_markers=())
        primary = rs_set.primary()
        assert primary.found is False
        assert primary.doi is None

    def test_primary_single_source_returns_it(self) -> None:
        """Single-source claims (the common case) are unaffected by this change."""
        from src.models import ResolvedSourceSet

        only = self._src(doi="10.1/only", title_match=0.7)
        rs_set = ResolvedSourceSet(sources=(only,), citation_markers=(42,))
        assert rs_set.primary().doi == "10.1/only"

    def test_found_sources_filters_unresolved(self) -> None:
        from src.models import ResolvedSourceSet

        a = self._src(doi="10.1/a")
        b = self._src(found=False, doi=None)
        c = self._src(doi="10.1/c")
        rs_set = ResolvedSourceSet(sources=(a, b, c), citation_markers=(1, 2, 3))
        found = rs_set.found_sources()
        assert len(found) == 2
        assert {s.doi for s in found} == {"10.1/a", "10.1/c"}


class TestVerificationResultNumericCheck:
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
        # evidence_quality="quoted_passage" satisfies Invariant 2 for status="supported".
        v = VerificationResult(
            status="supported",
            explanation="ok",
            confidence=0.9,
            numeric_check=nc,
            evidence_quality="quoted_passage",
        )
        d = dataclasses.asdict(v)
        assert d["numeric_check"]["check_type"] == "or_ci_consistency"
        assert d["numeric_check"]["consistent"] is True
        assert d["numeric_check"]["extracted"][0]["value"] == 40.53


class TestVerificationResultInvariant:
    """A1: __post_init__ validator enforces confidence-evidence coupling invariant."""

    def test_post_init_rejects_unverifiable_with_confidence(self) -> None:
        with pytest.raises(ValueError, match="unverifiable status requires confidence=None"):
            VerificationResult(
                status="unverifiable",
                explanation="cannot determine",
                confidence=0.75,
                evidence_quality="abstract_only",
            )

    def test_post_init_rejects_non_unverifiable_with_none_confidence(self) -> None:
        with pytest.raises(ValueError, match="requires non-null confidence"):
            VerificationResult(
                status="supported",
                explanation="ok",
                confidence=None,
                evidence_quality="quoted_passage",
            )

    def test_direct_construction_supported_on_abstract_only_is_legal(self) -> None:
        """Decision log 2026-05-11: Invariant 2 is DROPPED from __post_init__.
        Direct VerificationResult construction with (supported, abstract_only) is legal.
        The downgrade enforcement lives in safe_verification_result(), not the schema.
        """
        result = VerificationResult(
            status="supported",
            explanation="ok",
            confidence=0.9,
            evidence_quality="abstract_only",
        )
        assert result.status == "supported"
        assert result.confidence == 0.9

    def test_direct_construction_supported_on_title_only_is_legal(self) -> None:
        """Invariant 2 dropped: direct construction with (supported, title_only) is legal."""
        result = VerificationResult(
            status="supported",
            explanation="ok",
            confidence=0.9,
            evidence_quality="title_only",
        )
        assert result.status == "supported"

    def test_direct_construction_unsupported_on_citing_paper_context_is_legal(self) -> None:
        """Invariant 2 dropped: (unsupported, citing_paper_context) direct construction is legal."""
        result = VerificationResult(
            status="unsupported",
            explanation="contradicted",
            confidence=0.8,
            evidence_quality="citing_paper_context",
        )
        assert result.status == "unsupported"

    def test_post_init_accepts_supported_with_fulltext_quoted_passage(self) -> None:
        result = VerificationResult(
            status="supported",
            explanation="ok",
            confidence=0.9,
            evidence_quality="quoted_passage",
            verification_depth="fulltext",
            fulltext_available=True,
            retrieval_status="passage_found",
        )
        assert result.status == "supported"
        assert result.confidence == 0.9

    @pytest.mark.parametrize(
        "eq",
        [
            "abstract_only",
            "title_only",
            "citing_paper_context",
            "no_evidence",
            "quoted_passage",
            "passages_searched_no_quote",
        ],
    )
    def test_post_init_accepts_unverifiable_with_none_confidence_and_any_evidence_quality(
        self, eq: str
    ) -> None:
        result = VerificationResult(
            status="unverifiable",
            explanation="cannot determine",
            confidence=None,
            evidence_quality=eq,  # type: ignore[arg-type]  # parametrize passes str, not Literal
        )
        assert result.status == "unverifiable"
        assert result.confidence is None

    def test_dataclasses_replace_preserves_invariant_1_only(self) -> None:
        """dataclasses.replace() calls __init__ -> __post_init__. Verify it
        catches Invariant 1 violations (confidence/unverifiable coupling), but NOT
        the dropped Invariant 2 (evidence-quality coupling).
        """
        valid = VerificationResult(
            status="not_addressed",
            explanation="ok",
            confidence=0.5,
            evidence_quality="abstract_only",
        )
        # Replacing status to "supported" while evidence_quality stays "abstract_only"
        # is now LEGAL (Invariant 2 dropped). Direct construction is permitted.
        result = dataclasses.replace(valid, status="supported", confidence=0.9)
        assert result.status == "supported"
        # Invariant 1 still holds: replacing to unverifiable must have confidence=None.
        with pytest.raises(ValueError, match="unverifiable status requires confidence=None"):
            dataclasses.replace(valid, status="unverifiable", confidence=0.5)

    def test_partially_supported_allowed_on_abstract_only(self) -> None:
        """partially_supported is a hedge, not a confident verdict. Allowed on any evidence."""
        result = VerificationResult(
            status="partially_supported",
            explanation="some elements match",
            confidence=0.6,
            evidence_quality="abstract_only",
        )
        assert result.status == "partially_supported"

    def test_not_addressed_allowed_on_abstract_only(self) -> None:
        """not_addressed = the source doesn't discuss the claim. Valid from abstract."""
        result = VerificationResult(
            status="not_addressed",
            explanation="source does not discuss this",
            confidence=0.9,
            evidence_quality="abstract_only",
        )
        assert result.status == "not_addressed"

    def test_safe_verification_result_downgrades_supported_on_abstract_only(self) -> None:
        result = safe_verification_result(
            status="supported",
            confidence=0.9,
            explanation="abstract says so",
            evidence_quality="abstract_only",
        )
        assert result.status == "unverifiable"
        assert result.confidence is None
        assert result.evidence_quality == "abstract_only"

    def test_safe_verification_result_downgrades_unsupported_on_abstract_only(self) -> None:
        result = safe_verification_result(
            status="unsupported",
            confidence=0.75,
            explanation="abstract does not support",
            evidence_quality="abstract_only",
        )
        assert result.status == "unverifiable"
        assert result.confidence is None

    def test_safe_verification_result_passes_through_valid_fulltext(self) -> None:
        result = safe_verification_result(
            status="supported",
            confidence=0.9,
            explanation="quoted from passage",
            evidence_quality="quoted_passage",
        )
        assert result.status == "supported"
        assert result.confidence == 0.9

    def test_safe_verification_result_passes_through_not_addressed(self) -> None:
        result = safe_verification_result(
            status="not_addressed",
            confidence=0.9,
            explanation="source silent",
            evidence_quality="abstract_only",
        )
        assert result.status == "not_addressed"
        assert result.confidence == 0.9

    def test_helper_passes_qualitative_supported_on_abstract_through(self) -> None:
        """Decision log 2026-05-11: qualitative claims with confident verdict on abstract
        are NOT downgraded. The abstract is sufficient for 'X reduces Y'-style verdicts.
        """
        result = safe_verification_result(
            status="supported",
            confidence=0.9,
            evidence_quality="abstract_only",
            claim_text="psilocybin reduces depression symptoms",  # no numeric pattern
            explanation="abstract directly addresses the qualitative claim",
        )
        assert result.status == "supported"
        assert result.confidence == 0.9
        assert result.evidence_quality == "abstract_only"

    def test_helper_downgrades_numeric_claim_with_claim_text(self) -> None:
        """Numeric claim (20% response rate) on abstract-only evidence is downgraded."""
        result = safe_verification_result(
            status="unsupported",
            confidence=0.75,
            evidence_quality="abstract_only",
            claim_text="Sustained response rates at 12 weeks were only 20% in the largest trial",
            explanation="abstract does not mention the 20% figure",
        )
        assert result.status == "unverifiable"
        assert result.confidence is None

    def test_helper_legacy_none_claim_text_downgrades(self) -> None:
        """Legacy callers (claim_text=None) are still downgraded as fail-safe."""
        result = safe_verification_result(
            status="supported",
            confidence=0.9,
            evidence_quality="abstract_only",
            claim_text=None,
            explanation="abstract says so",
        )
        assert result.status == "unverifiable"
        assert result.confidence is None
