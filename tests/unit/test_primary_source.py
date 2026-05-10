"""Unit tests for src/copilot/primary_source.py — offline, zero network, zero LLM."""

from __future__ import annotations

from src.copilot.primary_source import (
    _assess_risk_of_bias,
    _classify_study_design,
    _extract_max_n,
    classify_source,
)
from src.models import (
    Claim,
    ProvenanceStep,
    ResolvedSource,
    ResolvedSourceSet,
    VerificationResult,
)
from src.pipeline import ClaimVerification

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cv(
    abstract: str | None = None,
    doi: str | None = "10.1234/test",
) -> ClaimVerification:
    claim = Claim(
        claim_id="cl-test",
        claim_text="Treatment X reduces marker Y.",
        cited_authors=["Jones"],
        cited_year=2021,
        claim_type="factual_qualitative",
    )
    source = ResolvedSource(
        found=abstract is not None,
        doi=doi,
        title="Test Paper",
        abstract=abstract,
        similarity_score=0.85,
    )
    source_set = ResolvedSourceSet(sources=(source,), citation_markers=(1,))
    result = VerificationResult(
        status="supported",
        explanation="Evidence found.",
        confidence=0.8,
    )
    return ClaimVerification(
        claim=claim,
        source=source,
        source_set=source_set,
        result=result,
        fetch_method="abstract",
    )


# ---------------------------------------------------------------------------
# _extract_max_n
# ---------------------------------------------------------------------------


class TestExtractMaxN:
    def test_n_equals_pattern(self) -> None:
        assert _extract_max_n("We enrolled n = 150 patients.") == 150

    def test_capital_n_pattern(self) -> None:
        assert _extract_max_n("N=300 subjects were included.") == 300

    def test_patients_pattern(self) -> None:
        assert _extract_max_n("240 patients completed the study.") == 240

    def test_participants_pattern(self) -> None:
        assert _extract_max_n("Fifty 50 participants were randomised.") == 50

    def test_returns_max_when_multiple(self) -> None:
        assert _extract_max_n("Group A (n=45) and group B (n=55).") == 55

    def test_returns_none_when_no_match(self) -> None:
        assert _extract_max_n("This is a review article.") is None

    def test_empty_string(self) -> None:
        assert _extract_max_n("") is None


# ---------------------------------------------------------------------------
# _classify_study_design — secondary sources
# ---------------------------------------------------------------------------


class TestClassifyStudyDesignSecondary:
    def test_meta_analysis(self) -> None:
        abstract = "We conducted a meta-analysis of 24 randomized trials on statin therapy."
        design, is_primary = _classify_study_design(abstract)
        assert design == "meta_analysis"
        assert is_primary is False

    def test_systematic_review(self) -> None:
        abstract = (
            "Systematic review of the literature following PRISMA guidelines "
            "to assess outcomes of immunotherapy."
        )
        design, is_primary = _classify_study_design(abstract)
        assert design == "systematic_review"
        assert is_primary is False

    def test_narrative_review(self) -> None:
        abstract = "This narrative review summarises current evidence on GLP-1 agonists."
        design, is_primary = _classify_study_design(abstract)
        assert design == "narrative_review"
        assert is_primary is False

    def test_literature_review_variant(self) -> None:
        abstract = "A comprehensive literature review of lactate measurement methods."
        design, is_primary = _classify_study_design(abstract)
        assert design == "narrative_review"
        assert is_primary is False

    def test_guidelines(self) -> None:
        abstract = "These clinical practice guidelines provide recommendations for T2D management."
        design, is_primary = _classify_study_design(abstract)
        assert design == "guidelines"
        assert is_primary is False


# ---------------------------------------------------------------------------
# _classify_study_design — primary sources
# ---------------------------------------------------------------------------


class TestClassifyStudyDesignPrimary:
    def test_rct(self) -> None:
        abstract = (
            "A randomized controlled trial of 320 patients with type 2 diabetes. "
            "Patients were double-blind placebo-controlled over 52 weeks."
        )
        design, is_primary = _classify_study_design(abstract)
        assert design == "rct"
        assert is_primary is True

    def test_rct_abbreviation(self) -> None:
        abstract = "This RCT enrolled 200 patients. The primary endpoint was HbA1c."
        design, is_primary = _classify_study_design(abstract)
        assert design == "rct"
        assert is_primary is True

    def test_cohort_study(self) -> None:
        abstract = "A prospective cohort study of 1200 adults followed for 10 years."
        design, is_primary = _classify_study_design(abstract)
        assert design == "observational"
        assert is_primary is True

    def test_case_control(self) -> None:
        abstract = "In this matched case-control study, 500 cases and 500 controls were recruited."
        design, is_primary = _classify_study_design(abstract)
        assert design == "case_control"
        assert is_primary is True

    def test_animal_model(self) -> None:
        abstract = "Using a mouse model of Alzheimer's disease, we demonstrated amyloid clearance."
        design, is_primary = _classify_study_design(abstract)
        assert design == "animal_model"
        assert is_primary is True


# ---------------------------------------------------------------------------
# Edge cases — no abstract
# ---------------------------------------------------------------------------


class TestEdgeCasesNoAbstract:
    def test_empty_string_returns_unknown(self) -> None:
        design, is_primary = _classify_study_design("")
        assert design == "unknown"
        assert is_primary is False

    def test_classify_source_no_abstract_source(self) -> None:
        cv = _make_cv(abstract=None)
        clf, _step = classify_source(cv)
        assert clf.is_primary_source is False
        assert clf.study_design == "unknown"
        assert clf.risk_of_bias == "unknown"

    def test_classify_source_whitespace_only_abstract(self) -> None:
        cv = _make_cv(abstract="   ")
        clf, _step = classify_source(cv)
        assert clf.is_primary_source is False
        assert clf.study_design == "unknown"


# ---------------------------------------------------------------------------
# _assess_risk_of_bias
# ---------------------------------------------------------------------------


class TestRiskOfBias:
    def test_high_quality_rct_large_n(self) -> None:
        abstract = (
            "Double-blind, randomized placebo-controlled trial with n = 250 patients. "
            "Allocation concealment was ensured."
        )
        rob = _assess_risk_of_bias(abstract, "rct", is_primary=True)
        assert rob == "low"

    def test_pilot_study_is_high(self) -> None:
        rob = _assess_risk_of_bias(
            "A pilot study of 8 patients showed promising results.", "rct", is_primary=True
        )
        assert rob == "high"

    def test_case_report_is_high(self) -> None:
        rob = _assess_risk_of_bias(
            "We report a case report of a 34-year-old patient with rare ALS variant.",
            "rct",
            is_primary=True,
        )
        assert rob == "high"

    def test_very_small_n_is_high(self) -> None:
        rob = _assess_risk_of_bias(
            "n = 5 patients were enrolled in this exploratory study.", "rct", is_primary=True
        )
        assert rob == "high"

    def test_secondary_source_is_unknown(self) -> None:
        rob = _assess_risk_of_bias(
            "This meta-analysis included 15 trials.", "meta_analysis", is_primary=False
        )
        assert rob == "unknown"

    def test_no_abstract_is_unknown(self) -> None:
        rob = _assess_risk_of_bias("", "rct", is_primary=True)
        assert rob == "unknown"


# ---------------------------------------------------------------------------
# classify_source — stage 1 (crossref_work_type)
# ---------------------------------------------------------------------------


class TestClassifySourceStage1:
    def test_always_secondary_type_book(self) -> None:
        cv = _make_cv(abstract="An RCT showed reduction in HbA1c levels.")
        clf, _step = classify_source(cv, crossref_work_type="book")
        assert clf.is_primary_source is False
        assert clf.study_design == "narrative_review"

    def test_always_secondary_type_reference_entry(self) -> None:
        cv = _make_cv(abstract="Study data...")
        clf, _ = classify_source(cv, crossref_work_type="reference-entry")
        assert clf.is_primary_source is False

    def test_likely_secondary_book_chapter(self) -> None:
        cv = _make_cv(abstract="A prospective cohort study...")
        clf, _ = classify_source(cv, crossref_work_type="book-chapter")
        assert clf.is_primary_source is False
        assert clf.study_design == "unknown"

    def test_preprint_is_primary(self) -> None:
        cv = _make_cv(abstract="Randomized controlled trial of 120 participants.")
        clf, _ = classify_source(cv, crossref_work_type="posted-content")
        assert clf.is_primary_source is True
        assert clf.study_design == "rct"

    def test_none_crossref_type_falls_through_to_stage2(self) -> None:
        rct_abstract = "Randomized double-blind trial with n = 200 subjects."
        cv = _make_cv(abstract=rct_abstract)
        clf, _ = classify_source(cv)  # crossref_work_type=None
        assert clf.is_primary_source is True
        assert clf.study_design == "rct"


# ---------------------------------------------------------------------------
# classify_source — ProvenanceStep
# ---------------------------------------------------------------------------


class TestClassifySourceProvenance:
    def test_step_shape_and_determinism(self) -> None:
        """One consolidated check on the deterministic-step contract.

        Classify_source is pure-Python (no LLM); the step must carry the
        right operation, the claim_id of the input, no model/token data,
        valid SHA-256 hashes, and produce identical hashes for identical
        inputs but different hashes for different abstracts.
        """
        cv1 = _make_cv(abstract="A systematic review of 10 RCTs.")
        cv2 = _make_cv(abstract="A randomized controlled trial of 200 patients.")
        _, step1a = classify_source(cv1)
        _, step1b = classify_source(cv1)
        _, step2 = classify_source(cv2)

        assert isinstance(step1a, ProvenanceStep)
        assert step1a.operation == "copilot_primary_source"
        assert step1a.model_id is None
        assert step1a.tokens_in is None and step1a.tokens_out is None
        assert step1a.claim_id == cv1.claim.claim_id
        assert len(step1a.input_hash) == 64 and len(step1a.output_hash) == 64
        assert step1a.output_hash == step1b.output_hash  # determinism
        assert step1a.output_hash != step2.output_hash  # input-sensitivity
