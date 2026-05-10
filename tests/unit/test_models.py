"""Unit tests for src/models.py — frozen dataclasses and type aliases."""

from __future__ import annotations

import dataclasses

import pytest

from src.models import Claim, ResolvedSource, VerificationResult
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
