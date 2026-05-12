"""Unit tests for the E2E recall measurement script.

Covers schema validation, claim alignment, DOI normalization, and metric
computation. The pipeline orchestration (`_run_pipeline`) is integration-only
and not tested here — it depends on Anthropic + OpenAlex calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.e2e.measurement import (
    align_claims,
    author_overlap,
    compute_metrics,
    normalize_doi,
    normalize_surname,
    score_pair,
    year_match,
)
from eval.e2e.schema import (
    SCHEMA_VERSION,
    GroundTruthClaim,
    load_reference_paper,
)
from src.models import Claim, ResolvedSource, VerificationResult


def _make_extracted(
    claim_id: str,
    text: str,
    *,
    authors: list[str] | None = None,
    year: int | None = None,
) -> Claim:
    return Claim(
        claim_id=claim_id,
        claim_text=text,
        cited_authors=authors or [],
        cited_year=year,
        claim_type="factual_qualitative",
    )


def _make_gt(
    gt_id: str,
    text: str,
    *,
    authors: list[str] | None = None,
    year: int | None = None,
    doi: str | None = None,
    origin: str = "secondary",
) -> GroundTruthClaim:
    return GroundTruthClaim(
        gt_claim_id=gt_id,
        claim_text=text,
        section="results",
        claim_type="factual_qualitative",
        claim_origin=origin,  # type: ignore[arg-type]
        cited_authors=authors or [],
        cited_year=year,
        ground_truth_doi=doi,
        ground_truth_title=None,
    )


def _make_source(*, found: bool, doi: str | None = None) -> ResolvedSource:
    return ResolvedSource(
        found=found,
        doi=doi,
        title=None,
        abstract="abstract" if found else None,
        similarity_score=0.9 if found else None,
    )


def _make_verification(status: str) -> VerificationResult:
    # A1: supported/unsupported require fulltext-grade evidence
    eq = "quoted_passage" if status in ("supported", "unsupported") else "abstract_only"
    actual_confidence: float | None = None if status == "unverifiable" else 0.5
    return VerificationResult(
        status=status,  # type: ignore[arg-type]
        explanation="",
        confidence=actual_confidence,  # type: ignore[arg-type]
        evidence_quality=eq,  # type: ignore[arg-type]
    )


def test_run_pipeline_passes_parsed_bibliography_to_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.measure_e2e_recall as measure
    from src.models import ResolvedSourceSet

    claim = _make_extracted(
        "e1",
        "X is supported [1].",
        authors=["Smith"],
        year=None,
    )
    source = _make_source(found=True, doi=None)
    captured: dict[str, object] = {}

    def fake_resolve_multi(
        claims: list[Claim], **kwargs: object
    ) -> tuple[dict[str, ResolvedSourceSet], list[object]]:
        captured["claims"] = claims
        captured["bibliography"] = kwargs.get("bibliography")
        rs_set = ResolvedSourceSet(sources=(source,), citation_markers=())
        return {"e1": rs_set}, []

    def fake_verify_one_claim(
        claim_arg: Claim, source_set_arg: object, *, citing_paper_text: str | None, config: object
    ) -> object:
        from src.pipeline import ClaimVerification

        return ClaimVerification(
            claim=claim_arg,
            source=source,
            source_set=source_set_arg,  # type: ignore[arg-type]
            result=_make_verification("supported"),
            fetch_method="abstract_fallback",
        )

    monkeypatch.setattr(measure, "extract_claims", lambda text: ([claim], object()))
    monkeypatch.setattr(measure, "resolve_citations_multi", fake_resolve_multi)
    monkeypatch.setattr(measure, "verify_one_claim", fake_verify_one_claim)
    monkeypatch.setattr(measure, "_compute_cost", lambda steps: 0.0)

    text = "Body claim [1].\n\nReferences\n[1]\nSmith A. 'Reference title'. In: J (2020).\n"
    measure._run_pipeline(text, max_cost_usd=1.0)

    bibliography = captured["bibliography"]
    assert isinstance(bibliography, dict)
    assert 1 in bibliography


# ---------------------------------------------------------------------------
# Surname / author / year helpers
# ---------------------------------------------------------------------------


def test_normalize_surname_handles_variants() -> None:
    assert normalize_surname("Smith") == "smith"
    assert normalize_surname("Smith, J.") == "smith"
    assert normalize_surname("John Smith") == "smith"
    assert normalize_surname("Smith et al.") == "smith"
    assert normalize_surname("  Wang  ") == "wang"


def test_author_overlap_intersects_surnames() -> None:
    assert author_overlap(["Smith"], ["Smith, J."]) is True
    assert author_overlap(["John Smith"], ["Smith"]) is True
    assert author_overlap(["Wang"], ["Smith"]) is False


def test_author_overlap_empty_gt_is_neutral() -> None:
    # Primary claims have empty cited_authors; matching should not penalize.
    assert author_overlap(["Smith"], []) is True


def test_year_match_tolerance() -> None:
    assert year_match(2020, 2020) is True
    assert year_match(2020, 2021) is True
    assert year_match(2020, 2019) is True
    assert year_match(2020, 2022) is False


def test_year_match_neutral_when_missing() -> None:
    assert year_match(None, 2020) is True
    assert year_match(2020, None) is True
    assert year_match(None, None) is True


# ---------------------------------------------------------------------------
# Pair scoring
# ---------------------------------------------------------------------------


def test_score_pair_strong_lexical_alone_matches() -> None:
    extracted = _make_extracted("e1", "Wang et al. demonstrated a microneedle patch")
    gt = _make_gt("c1", "Wang et al. demonstrated a microneedle patch")
    assert score_pair(extracted, gt) >= 0.85


def test_score_pair_below_floor_returns_zero() -> None:
    extracted = _make_extracted("e1", "completely unrelated text about cats")
    gt = _make_gt("c1", "microneedle interstitial fluid extraction protocol")
    assert score_pair(extracted, gt) == 0.0


def test_score_pair_requires_author_or_year_when_lexical_weak() -> None:
    # Lexical is in the floor..strong band (0.5-0.85). Author mismatch + year mismatch.
    # In that band we require author OR year to agree, so score must be 0.
    extracted = _make_extracted(
        "e1", "kinase activity in cell membrane", authors=["Wong"], year=2018
    )
    gt = _make_gt("c1", "kinase pathway controls cell signalling", authors=["Smith"], year=2022)
    score = score_pair(extracted, gt)
    assert score == 0.0  # author mismatch + year mismatch + mid lexical = no match


def test_score_pair_accepts_mid_lexical_with_authoryear_match() -> None:
    extracted = _make_extracted(
        "e1", "Smith showed protein folding kinetics", authors=["Smith"], year=2020
    )
    gt = _make_gt(
        "c1",
        "Smith demonstrated that protein folding follows specific kinetics",
        authors=["Smith"],
        year=2020,
    )
    score = score_pair(extracted, gt)
    assert score > 0.0


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def test_align_claims_one_to_one() -> None:
    extracted = [
        _make_extracted("e1", "Wang demonstrated a microneedle patch", authors=["Wang"], year=2024),
        _make_extracted(
            "e2", "Smith showed protein folding kinetics", authors=["Smith"], year=2020
        ),
    ]
    gt = [
        _make_gt("c1", "Wang demonstrated a microneedle patch", authors=["Wang"], year=2024),
        _make_gt("c2", "Smith showed protein folding kinetics", authors=["Smith"], year=2020),
    ]
    matches = align_claims(gt, extracted)
    assigned = {m.gt_claim_id: m.extracted_claim_id for m in matches}
    assert assigned == {"c1": "e1", "c2": "e2"}


def test_align_claims_unmatched_gt_returns_none() -> None:
    extracted = [_make_extracted("e1", "totally different topic about astronomy")]
    gt = [_make_gt("c1", "microneedle interstitial fluid extraction")]
    matches = align_claims(gt, extracted)
    assert matches[0].extracted_claim_id is None


def test_align_claims_no_double_assignment() -> None:
    # Two GT claims with similar text — only one extracted should match the best
    extracted = [
        _make_extracted("e1", "Wang demonstrated a microneedle patch", authors=["Wang"], year=2024),
    ]
    gt = [
        _make_gt("c1", "Wang demonstrated a microneedle patch", authors=["Wang"], year=2024),
        _make_gt("c2", "Wang demonstrated a microneedle patch design", authors=["Wang"], year=2024),
    ]
    matches = align_claims(gt, extracted)
    assigned_extracted = [m.extracted_claim_id for m in matches if m.extracted_claim_id]
    assert len(assigned_extracted) == len(set(assigned_extracted))  # no duplicates


# ---------------------------------------------------------------------------
# DOI normalization
# ---------------------------------------------------------------------------


def test_normalize_doi_strips_prefixes() -> None:
    assert normalize_doi("10.1038/nature12345") == "10.1038/nature12345"
    assert normalize_doi("https://doi.org/10.1038/nature12345") == "10.1038/nature12345"
    assert normalize_doi("doi:10.1038/nature12345") == "10.1038/nature12345"
    assert normalize_doi("  10.1038/Nature12345  ") == "10.1038/nature12345"
    assert normalize_doi(None) is None


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def test_compute_metrics_perfect_pipeline() -> None:
    extracted = [
        _make_extracted("e1", "claim A", authors=["Wang"], year=2024),
    ]
    gt = [
        _make_gt("c1", "claim A", authors=["Wang"], year=2024, doi="10.1038/x"),
    ]
    sources = {"e1": _make_source(found=True, doi="10.1038/x")}
    verifications = {"e1": _make_verification("supported")}
    matches = align_claims(gt, extracted)

    m = compute_metrics(gt, extracted, sources, verifications, matches)
    assert m.extraction_recall == 1.0
    assert m.extraction_precision == 1.0
    assert m.resolution_accuracy == 1.0
    assert m.e2e_coverage_useful == 1.0
    assert m.not_addressed_unknown_cause == 0.0


def test_compute_metrics_silent_wrong_doi_caught() -> None:
    extracted = [
        _make_extracted("e1", "claim A", authors=["Wang"], year=2024),
    ]
    gt = [
        _make_gt("c1", "claim A", authors=["Wang"], year=2024, doi="10.1038/correct"),
    ]
    sources = {"e1": _make_source(found=True, doi="10.1038/wrong")}  # silent wrong DOI
    verifications = {"e1": _make_verification("supported")}
    matches = align_claims(gt, extracted)

    m = compute_metrics(gt, extracted, sources, verifications, matches)
    assert m.extraction_recall == 1.0
    assert m.resolution_accuracy == 0.0  # caught
    assert m.e2e_coverage_useful == 0.0


def test_compute_metrics_extractor_misses_half() -> None:
    extracted = [_make_extracted("e1", "claim A", authors=["Wang"], year=2024)]
    gt = [
        _make_gt("c1", "claim A", authors=["Wang"], year=2024, doi="10.1038/a"),
        _make_gt(
            "c2",
            "claim B about something else",
            authors=["Smith"],
            year=2020,
            doi="10.1038/b",
        ),
    ]
    sources = {"e1": _make_source(found=True, doi="10.1038/a")}
    verifications = {"e1": _make_verification("supported")}
    matches = align_claims(gt, extracted)

    m = compute_metrics(gt, extracted, sources, verifications, matches)
    assert m.extraction_recall == 0.5
    assert m.extraction_precision == 1.0
    assert m.e2e_coverage_useful == 0.5


def test_compute_metrics_primary_claims_excluded_from_resolution() -> None:
    extracted = [_make_extracted("e1", "primary finding of this paper")]
    gt = [
        _make_gt(
            "c1",
            "primary finding of this paper",
            origin="primary",
        ),
    ]
    sources = {"e1": _make_source(found=False)}
    verifications = {"e1": _make_verification("not_addressed")}
    matches = align_claims(gt, extracted)

    m = compute_metrics(gt, extracted, sources, verifications, matches)
    # Primary claim is matched → extraction recall = 1.0
    assert m.extraction_recall == 1.0
    # No secondary GT claims with DOI → resolution_accuracy and e2e_coverage are 0 (no denominator)
    assert m.counts["n_secondary_with_doi_gt"] == 0
    assert m.counts["n_resolution_attempts_for_matched"] == 0


def test_compute_metrics_not_addressed_marked_unknown_cause() -> None:
    extracted = [_make_extracted("e1", "claim A")]
    gt = [_make_gt("c1", "claim A")]
    sources = {"e1": _make_source(found=False)}
    verifications = {"e1": _make_verification("not_addressed")}
    matches = align_claims(gt, extracted)

    m = compute_metrics(gt, extracted, sources, verifications, matches)
    assert m.counts["n_not_addressed"] == 1
    assert m.not_addressed_unknown_cause == 1.0  # 100% unknown — Phase D not done


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def _write_paper_json(tmp_path: Path, claims: list[dict[str, object]]) -> Path:
    data = {
        "schema_version": SCHEMA_VERSION,
        "paper": {
            "title": "Test paper",
            "authors": ["Author A"],
            "year": 2023,
            "source_text_path": "tests/fixtures/dummy.txt",
        },
        "claims": claims,
    }
    path = tmp_path / "paper.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _valid_claim(gt_id: str = "c001") -> dict[str, object]:
    return {
        "gt_claim_id": gt_id,
        "claim_text": "test claim",
        "section": "introduction",
        "claim_type": "factual_qualitative",
        "claim_origin": "secondary",
        "cited_authors": ["Smith"],
        "cited_year": 2020,
        "ground_truth_doi": "10.1038/x",
        "ground_truth_title": "Test title",
    }


def test_load_reference_paper_valid(tmp_path: Path) -> None:
    path = _write_paper_json(tmp_path, [_valid_claim()])
    paper = load_reference_paper(path)
    assert len(paper.claims) == 1
    assert paper.claims[0].gt_claim_id == "c001"


def test_load_reference_paper_rejects_invalid_section(tmp_path: Path) -> None:
    bad = _valid_claim()
    bad["section"] = "abstract"
    path = _write_paper_json(tmp_path, [bad])
    with pytest.raises(ValueError, match="section"):
        load_reference_paper(path)


def test_load_reference_paper_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = _write_paper_json(tmp_path, [_valid_claim("c001"), _valid_claim("c001")])
    with pytest.raises(ValueError, match="Duplicate"):
        load_reference_paper(path)


def test_load_reference_paper_rejects_missing_field(tmp_path: Path) -> None:
    bad = _valid_claim()
    del bad["ground_truth_doi"]
    path = _write_paper_json(tmp_path, [bad])
    with pytest.raises(ValueError, match="missing"):
        load_reference_paper(path)
