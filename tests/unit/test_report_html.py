"""Unit tests for src/copilot/report_html.py — fully offline, no I/O outside tmp_path."""

from __future__ import annotations

from pathlib import Path

from src.copilot.models import (
    CopilotFields,
    CopilotMode,
    EnrichedVerification,
    RecommendedFix,
)
from src.copilot.report_html import build_copilot_report
from src.models import (
    Claim,
    ProvenanceStep,
    ResolvedSource,
    ResolvedSourceSet,
    VerificationResult,
)
from src.pipeline import ClaimVerification

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_cv(
    claim_id: str = "cl-01",
    verdict: str = "unsupported",
    claim_text: str = "Drug X reduces biomarker Y by 50%.",
    doi: str | None = "10.1234/source",
    retrieval: str = "abstract",
) -> ClaimVerification:
    claim = Claim(
        claim_id=claim_id,
        claim_text=claim_text,
        cited_authors=["Jones"],
        cited_year=2022,
        claim_type="factual_qualitative",
    )
    source = ResolvedSource(
        found=doi is not None,
        doi=doi,
        title="A Review of Drug X" if doi else None,
        abstract="Review article abstract.",
        similarity_score=0.7,
    )
    source_set = ResolvedSourceSet(sources=(source,), citation_markers=(1,))
    result = VerificationResult(
        status=verdict,  # type: ignore[arg-type]
        explanation="Source does not support the magnitude claimed.",
        confidence=0.3,
        retrieval_status=retrieval,  # type: ignore[arg-type]
    )
    return ClaimVerification(
        claim=claim,
        source=source,
        source_set=source_set,
        result=result,
        fetch_method="abstract",
    )


def _make_step(claim_id: str, op: str = "copilot_rationale") -> ProvenanceStep:
    return ProvenanceStep(
        step_id="step-" + op,
        claim_id=claim_id,
        operation=op,
        input_hash="a" * 64,
        output_hash="b" * 64,
        model_id="claude-sonnet-4-6",
        timestamp=0.0,
        tokens_in=100,
        tokens_out=10,
        cache_hit=False,
        confidence=None,
    )


def _make_fix(action: str = "swap_doi", doi: str | None = "10.9999/primary") -> RecommendedFix:
    return RecommendedFix(
        action=action,  # type: ignore[arg-type]
        regulatory_risk_level="high",
        suggested_doi=doi,
        suggested_doi_title="Primary RCT Paper",
        reworded_claim=None,
        confidence=0.85,
        provenance_step_id="fix-step-id",
    )


def _make_enriched(
    cv: ClaimVerification | None = None,
    *,
    rationale: str = "The cited source is a review and does not report primary data.",
    is_primary_source: bool | None = False,
    study_design: str | None = "systematic_review",
    risk_of_bias: str | None = "high",
    primary_source_doi: str | None = "10.9999/primary",
    fix: RecommendedFix | None = None,
    mode: CopilotMode = CopilotMode.PHARMA,
) -> EnrichedVerification:
    if cv is None:
        cv = _make_cv()
    if fix is None and cv.result.status in {"unsupported", "partially_supported"}:
        fix = _make_fix()
    copilot = CopilotFields(
        verdict_rationale=rationale,
        recommended_fix=fix,
        is_primary_source=is_primary_source,
        study_design=study_design,  # type: ignore[arg-type]
        risk_of_bias=risk_of_bias,  # type: ignore[arg-type]
        conflicting_evidence_flag=None,
        primary_source_doi=primary_source_doi,
        novelty_claim=None,
    )
    return EnrichedVerification(
        base=cv,
        copilot=copilot,
        copilot_steps=(_make_step(cv.claim.claim_id),),
        mode=mode,
    )


# ---------------------------------------------------------------------------
# build_copilot_report — basic mechanics
# ---------------------------------------------------------------------------


class TestReportFileWriting:
    def test_creates_html_file(self, tmp_path: Path) -> None:
        path = build_copilot_report(tmp_path, [_make_enriched()])
        assert path.exists()
        assert path.name == "copilot_report.html"
        assert path.parent == tmp_path

    def test_returns_written_path(self, tmp_path: Path) -> None:
        path = build_copilot_report(tmp_path, [_make_enriched()])
        assert path == tmp_path / "copilot_report.html"

    def test_creates_run_dir_if_missing(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "new_run"
        path = build_copilot_report(new_dir, [_make_enriched()])
        assert path.exists()
        assert new_dir.is_dir()

    def test_empty_enriched_list_renders(self, tmp_path: Path) -> None:
        path = build_copilot_report(tmp_path, [])
        html = path.read_text(encoding="utf-8")
        assert "<html" in html
        assert "</html>" in html


# ---------------------------------------------------------------------------
# Content assertions
# ---------------------------------------------------------------------------


class TestReportContent:
    def test_includes_run_id(self, tmp_path: Path) -> None:
        path = build_copilot_report(tmp_path, [_make_enriched()], run_id="run-xyz")
        html = path.read_text(encoding="utf-8")
        assert "run-xyz" in html

    def test_includes_claim_text(self, tmp_path: Path) -> None:
        cv = _make_cv(claim_text="Treatment doubles survival rate.")
        path = build_copilot_report(tmp_path, [_make_enriched(cv)])
        html = path.read_text(encoding="utf-8")
        assert "Treatment doubles survival rate." in html

    def test_includes_verdict_badge(self, tmp_path: Path) -> None:
        path = build_copilot_report(tmp_path, [_make_enriched()])
        html = path.read_text(encoding="utf-8")
        assert "badge-unsupported" in html
        assert "Unsupported" in html

    def test_includes_rationale(self, tmp_path: Path) -> None:
        ev = _make_enriched(rationale="Custom rationale text here.")
        path = build_copilot_report(tmp_path, [ev])
        html = path.read_text(encoding="utf-8")
        assert "Custom rationale text here." in html

    def test_includes_source_doi(self, tmp_path: Path) -> None:
        path = build_copilot_report(tmp_path, [_make_enriched()])
        html = path.read_text(encoding="utf-8")
        assert "10.1234/source" in html
        assert 'href="https://doi.org/10.1234/source"' in html

    def test_includes_pharma_evidence_fields(self, tmp_path: Path) -> None:
        path = build_copilot_report(tmp_path, [_make_enriched()])
        html = path.read_text(encoding="utf-8")
        assert "Secondary source" in html
        assert "Systematic Review" in html
        assert "High" in html  # risk of bias

    def test_includes_recommended_fix_block(self, tmp_path: Path) -> None:
        path = build_copilot_report(tmp_path, [_make_enriched()])
        html = path.read_text(encoding="utf-8")
        assert "Recommended Fix" in html
        assert "10.9999/primary" in html
        assert "swap doi" in html.lower()

    def test_includes_cost_and_runtime(self, tmp_path: Path) -> None:
        path = build_copilot_report(
            tmp_path,
            [_make_enriched()],
            total_cost_usd=0.456,
            runtime_seconds=42.7,
        )
        html = path.read_text(encoding="utf-8")
        assert "$0.456" in html
        assert "42s" in html

    def test_includes_provenance_hashes_collapsed(self, tmp_path: Path) -> None:
        path = build_copilot_report(tmp_path, [_make_enriched()])
        html = path.read_text(encoding="utf-8")
        assert "<details" in html
        assert "Provenance" in html
        assert "copilot_rationale" in html


# ---------------------------------------------------------------------------
# Conditional rendering
# ---------------------------------------------------------------------------


class TestConditionalRendering:
    def test_no_fix_block_when_supported(self, tmp_path: Path) -> None:
        cv = _make_cv(verdict="supported")
        ev = _make_enriched(cv=cv, fix=None)
        path = build_copilot_report(tmp_path, [ev])
        html = path.read_text(encoding="utf-8")
        assert "Recommended Fix" not in html
        assert "badge-supported" in html

    def test_general_mode_omits_pharma_fields(self, tmp_path: Path) -> None:
        ev = _make_enriched(
            mode=CopilotMode.GENERAL,
            is_primary_source=None,
            study_design=None,
            risk_of_bias=None,
            primary_source_doi=None,
        )
        path = build_copilot_report(tmp_path, [ev])
        html = path.read_text(encoding="utf-8")
        # The "Evidence" row should not appear at all in GENERAL mode.
        assert "Evidence</span>" not in html
        assert "Source type" not in html
        assert "Risk of bias" not in html

    def test_no_source_resolved_message(self, tmp_path: Path) -> None:
        cv = _make_cv(doi=None)
        ev = _make_enriched(cv=cv)
        path = build_copilot_report(tmp_path, [ev])
        html = path.read_text(encoding="utf-8")
        assert "No source resolved" in html


# ---------------------------------------------------------------------------
# Verdict distribution + summary stats
# ---------------------------------------------------------------------------


class TestSummaryStats:
    def test_counts_per_verdict(self, tmp_path: Path) -> None:
        evs = [
            _make_enriched(_make_cv("c1", "supported"), fix=None),
            _make_enriched(_make_cv("c2", "supported"), fix=None),
            _make_enriched(_make_cv("c3", "unsupported")),
            _make_enriched(_make_cv("c4", "partially_supported")),
            _make_enriched(_make_cv("c5", "not_addressed"), fix=None),
        ]
        path = build_copilot_report(tmp_path, evs)
        html = path.read_text(encoding="utf-8")
        # Total = 5, supported = 2, partial = 1, unsupported = 1, not_addressed = 1
        assert '>5</span>\n      <span class="lbl">Claims</span>' in html or ">5<" in html
        assert "Supported" in html
        assert "Partial" in html

    def test_n_with_fix_count(self, tmp_path: Path) -> None:
        evs = [
            _make_enriched(_make_cv("c1", "supported"), fix=None),
            _make_enriched(_make_cv("c2", "unsupported")),  # has fix
            _make_enriched(_make_cv("c3", "partially_supported")),  # has fix
        ]
        path = build_copilot_report(tmp_path, evs)
        html = path.read_text(encoding="utf-8")
        # Two fixes presented to the reviewer
        assert "TOTAL_FIXES = 2" in html


# ---------------------------------------------------------------------------
# XSS / autoescape safety
# ---------------------------------------------------------------------------


class TestAutoescapeSafety:
    def test_html_in_claim_text_is_escaped(self, tmp_path: Path) -> None:
        cv = _make_cv(claim_text="<script>alert('xss')</script>")
        path = build_copilot_report(tmp_path, [_make_enriched(cv=cv)])
        html = path.read_text(encoding="utf-8")
        # Raw script tag must NOT appear; entities must.
        assert "<script>alert('xss')</script>" not in html
        assert "&lt;script&gt;" in html

    def test_html_in_rationale_is_escaped(self, tmp_path: Path) -> None:
        ev = _make_enriched(rationale="<img src=x onerror=alert(1)>")
        path = build_copilot_report(tmp_path, [ev])
        html = path.read_text(encoding="utf-8")
        assert "<img src=x onerror=alert(1)>" not in html
        assert "&lt;img" in html


# ---------------------------------------------------------------------------
# HITL JS export
# ---------------------------------------------------------------------------


class TestHITLExportButton:
    def test_export_button_present(self, tmp_path: Path) -> None:
        path = build_copilot_report(tmp_path, [_make_enriched()])
        html = path.read_text(encoding="utf-8")
        assert "Export Review Session" in html
        assert "exportSession" in html

    def test_accept_reject_buttons_per_fix(self, tmp_path: Path) -> None:
        ev = _make_enriched()
        path = build_copilot_report(tmp_path, [ev])
        html = path.read_text(encoding="utf-8")
        assert "Accept" in html
        assert "Reject" in html
        # The recordDecision JS function must be present.
        assert "function recordDecision" in html

    def test_session_key_includes_run_id(self, tmp_path: Path) -> None:
        path = build_copilot_report(tmp_path, [_make_enriched()], run_id="trial-2026")
        html = path.read_text(encoding="utf-8")
        assert "copilot_session_trial-2026" in html
