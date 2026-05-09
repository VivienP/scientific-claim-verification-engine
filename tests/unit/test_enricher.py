"""Unit tests for src/copilot/enricher.py — all sub-components mocked, offline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.copilot.enricher import CopilotConfig, CopilotEnricher
from src.copilot.models import CopilotMode, EnrichedVerification, RecommendedFix
from src.copilot.primary_source import SourceClassification
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


def _make_cv(verdict: str = "unsupported") -> ClaimVerification:
    claim = Claim(
        claim_id="cl-enrich-01",
        claim_text="Drug X significantly reduces biomarker Y.",
        cited_authors=["Jones"],
        cited_year=2022,
        claim_type="factual_qualitative",
    )
    source = ResolvedSource(
        found=True,
        doi="10.1234/source",
        title="Review Paper",
        abstract="A systematic review of Drug X studies.",
        similarity_score=0.7,
    )
    source_set = ResolvedSourceSet(sources=(source,), citation_markers=(1,))
    result = VerificationResult(
        status=verdict,  # type: ignore[arg-type]
        explanation="Source does not support the magnitude claimed.",
        confidence=0.3,
    )
    return ClaimVerification(
        claim=claim,
        source=source,
        source_set=source_set,
        result=result,
        fetch_method="abstract",
    )


def _stub_rationale(cv: ClaimVerification, **_: object) -> tuple[str, ProvenanceStep]:
    step = ProvenanceStep(
        step_id="r-step",
        claim_id=cv.claim.claim_id,
        operation="copilot_rationale",
        input_hash="a" * 64,
        output_hash="b" * 64,
        model_id="claude-sonnet-4-6",
        timestamp=0.0,
        tokens_in=100,
        tokens_out=10,
        cache_hit=False,
        confidence=None,
    )
    return "The source is a review article and does not report primary data.", step


def _stub_classify(
    cv: ClaimVerification,
    **_: object,
) -> tuple[SourceClassification, ProvenanceStep]:
    clf = SourceClassification(
        is_primary_source=False,
        study_design="systematic_review",
        risk_of_bias="unknown",
    )
    step = ProvenanceStep(
        step_id="c-step",
        claim_id=cv.claim.claim_id,
        operation="copilot_primary_source",
        input_hash="c" * 64,
        output_hash="d" * 64,
        model_id=None,
        timestamp=0.0,
        tokens_in=None,
        tokens_out=None,
        cache_hit=None,
        confidence=None,
    )
    return clf, step


def _stub_lookup(doi: str | None, **_: object) -> tuple[str | None, str | None, ProvenanceStep]:
    step = ProvenanceStep(
        step_id="l-step",
        claim_id="",
        operation="copilot_primary_lookup",
        input_hash="e" * 64,
        output_hash="f" * 64,
        model_id=None,
        timestamp=0.0,
        tokens_in=None,
        tokens_out=None,
        cache_hit=None,
        confidence=None,
    )
    return "10.9999/primary", "Primary RCT Paper", step


def _stub_fix(cv: ClaimVerification, **_: object) -> tuple[RecommendedFix | None, ProvenanceStep]:
    fix = RecommendedFix(
        action="swap_doi",
        regulatory_risk_level="high",
        suggested_doi="10.9999/primary",
        suggested_doi_title="Primary RCT Paper",
        reworded_claim=None,
        confidence=0.85,
        provenance_step_id="fix-step",
    )
    step = ProvenanceStep(
        step_id="fix-step",
        claim_id=cv.claim.claim_id,
        operation="copilot_fix",
        input_hash="g" * 64,
        output_hash="h" * 64,
        model_id="claude-sonnet-4-6",
        timestamp=0.0,
        tokens_in=300,
        tokens_out=60,
        cache_hit=False,
        confidence=0.85,
    )
    return fix, step


# ---------------------------------------------------------------------------
# enrich_one — happy path (pharma mode)
# ---------------------------------------------------------------------------


class TestEnrichOnePharma:
    @patch("src.copilot.enricher.extract_rationale", side_effect=_stub_rationale)
    @patch("src.copilot.enricher.classify_source", side_effect=_stub_classify)
    @patch("src.copilot.enricher.find_primary_source_doi", side_effect=_stub_lookup)
    @patch("src.copilot.enricher.generate_fix", side_effect=_stub_fix)
    def test_returns_enriched_verification(
        self,
        mock_fix: MagicMock,
        mock_lookup: MagicMock,
        mock_classify: MagicMock,
        mock_rationale: MagicMock,
        tmp_path: Path,
    ) -> None:
        enricher = CopilotEnricher(CopilotConfig(db_path=tmp_path / "c.db"))
        ev = enricher.enrich_one(_make_cv())
        assert isinstance(ev, EnrichedVerification)

    @patch("src.copilot.enricher.extract_rationale", side_effect=_stub_rationale)
    @patch("src.copilot.enricher.classify_source", side_effect=_stub_classify)
    @patch("src.copilot.enricher.find_primary_source_doi", side_effect=_stub_lookup)
    @patch("src.copilot.enricher.generate_fix", side_effect=_stub_fix)
    def test_base_is_unchanged(
        self,
        mock_fix: MagicMock,
        mock_lookup: MagicMock,
        mock_classify: MagicMock,
        mock_rationale: MagicMock,
        tmp_path: Path,
    ) -> None:
        cv = _make_cv()
        enricher = CopilotEnricher(CopilotConfig(db_path=tmp_path / "c.db"))
        ev = enricher.enrich_one(cv)
        assert ev.base is cv

    @patch("src.copilot.enricher.extract_rationale", side_effect=_stub_rationale)
    @patch("src.copilot.enricher.classify_source", side_effect=_stub_classify)
    @patch("src.copilot.enricher.find_primary_source_doi", side_effect=_stub_lookup)
    @patch("src.copilot.enricher.generate_fix", side_effect=_stub_fix)
    def test_copilot_fields_populated(
        self,
        mock_fix: MagicMock,
        mock_lookup: MagicMock,
        mock_classify: MagicMock,
        mock_rationale: MagicMock,
        tmp_path: Path,
    ) -> None:
        enricher = CopilotEnricher(CopilotConfig(db_path=tmp_path / "c.db"))
        ev = enricher.enrich_one(_make_cv())
        assert isinstance(ev.copilot.verdict_rationale, str)
        assert len(ev.copilot.verdict_rationale) > 0

    @patch("src.copilot.enricher.extract_rationale", side_effect=_stub_rationale)
    @patch("src.copilot.enricher.classify_source", side_effect=_stub_classify)
    @patch("src.copilot.enricher.find_primary_source_doi", side_effect=_stub_lookup)
    @patch("src.copilot.enricher.generate_fix", side_effect=_stub_fix)
    def test_copilot_steps_include_all_steps(
        self,
        mock_fix: MagicMock,
        mock_lookup: MagicMock,
        mock_classify: MagicMock,
        mock_rationale: MagicMock,
        tmp_path: Path,
    ) -> None:
        enricher = CopilotEnricher(CopilotConfig(db_path=tmp_path / "c.db"))
        ev = enricher.enrich_one(_make_cv())
        ops = {s.operation for s in ev.copilot_steps}
        assert "copilot_rationale" in ops
        assert "copilot_primary_source" in ops
        assert "copilot_primary_lookup" in ops
        assert "copilot_fix" in ops

    @patch("src.copilot.enricher.extract_rationale", side_effect=_stub_rationale)
    @patch("src.copilot.enricher.classify_source", side_effect=_stub_classify)
    @patch("src.copilot.enricher.find_primary_source_doi", side_effect=_stub_lookup)
    @patch("src.copilot.enricher.generate_fix", side_effect=_stub_fix)
    def test_mode_is_pharma(
        self,
        mock_fix: MagicMock,
        mock_lookup: MagicMock,
        mock_classify: MagicMock,
        mock_rationale: MagicMock,
        tmp_path: Path,
    ) -> None:
        enricher = CopilotEnricher(CopilotConfig(db_path=tmp_path / "c.db"))
        ev = enricher.enrich_one(_make_cv())
        assert ev.mode == CopilotMode.PHARMA

    @patch("src.copilot.enricher.extract_rationale", side_effect=_stub_rationale)
    @patch("src.copilot.enricher.classify_source", side_effect=_stub_classify)
    @patch("src.copilot.enricher.find_primary_source_doi", side_effect=_stub_lookup)
    @patch("src.copilot.enricher.generate_fix", side_effect=_stub_fix)
    def test_pharma_fields_populated(
        self,
        mock_fix: MagicMock,
        mock_lookup: MagicMock,
        mock_classify: MagicMock,
        mock_rationale: MagicMock,
        tmp_path: Path,
    ) -> None:
        enricher = CopilotEnricher(CopilotConfig(db_path=tmp_path / "c.db"))
        ev = enricher.enrich_one(_make_cv())
        assert ev.copilot.is_primary_source is False
        assert ev.copilot.study_design == "systematic_review"
        assert ev.copilot.primary_source_doi == "10.9999/primary"


# ---------------------------------------------------------------------------
# enrich_one — general mode (pharma fields disabled)
# ---------------------------------------------------------------------------


class TestEnrichOneGeneralMode:
    @patch("src.copilot.enricher.extract_rationale", side_effect=_stub_rationale)
    @patch("src.copilot.enricher.generate_fix", side_effect=_stub_fix)
    def test_general_mode_pharma_fields_are_none(
        self, mock_fix: MagicMock, mock_rationale: MagicMock, tmp_path: Path
    ) -> None:
        config = CopilotConfig(mode=CopilotMode.GENERAL, db_path=tmp_path / "c.db")
        enricher = CopilotEnricher(config)
        ev = enricher.enrich_one(_make_cv())
        assert ev.copilot.is_primary_source is None
        assert ev.copilot.study_design is None
        assert ev.copilot.risk_of_bias is None
        assert ev.copilot.primary_source_doi is None


# ---------------------------------------------------------------------------
# enrich_one — supported verdict skips fix
# ---------------------------------------------------------------------------


class TestEnrichOneSupportedSkipsFix:
    @patch("src.copilot.enricher.extract_rationale", side_effect=_stub_rationale)
    @patch("src.copilot.enricher.classify_source", side_effect=_stub_classify)
    @patch("src.copilot.enricher.find_primary_source_doi", side_effect=_stub_lookup)
    @patch("src.copilot.enricher.generate_fix")
    def test_fix_not_called_for_supported(
        self,
        mock_fix: MagicMock,
        mock_lookup: MagicMock,
        mock_classify: MagicMock,
        mock_rationale: MagicMock,
        tmp_path: Path,
    ) -> None:
        enricher = CopilotEnricher(CopilotConfig(db_path=tmp_path / "c.db"))
        enricher.enrich_one(_make_cv(verdict="supported"))
        mock_fix.assert_not_called()


# ---------------------------------------------------------------------------
# enrich_all
# ---------------------------------------------------------------------------


class TestEnrichAll:
    @patch("src.copilot.enricher.extract_rationale", side_effect=_stub_rationale)
    @patch("src.copilot.enricher.classify_source", side_effect=_stub_classify)
    @patch("src.copilot.enricher.find_primary_source_doi", side_effect=_stub_lookup)
    @patch("src.copilot.enricher.generate_fix", side_effect=_stub_fix)
    def test_returns_list_of_same_length(
        self,
        mock_fix: MagicMock,
        mock_lookup: MagicMock,
        mock_classify: MagicMock,
        mock_rationale: MagicMock,
        tmp_path: Path,
    ) -> None:
        cvs = [_make_cv(), _make_cv("partially_supported"), _make_cv("supported")]
        enricher = CopilotEnricher(CopilotConfig(db_path=tmp_path / "c.db"))
        results = enricher.enrich_all(cvs)
        assert len(results) == 3
        assert all(isinstance(r, EnrichedVerification) for r in results)
