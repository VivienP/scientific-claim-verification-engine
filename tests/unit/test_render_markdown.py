"""Unit tests for src/render_markdown.py — offline, deterministic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.render_markdown import render_markdown, render_markdown_from_file


def _make_report(
    *,
    report_id: str = "run-001",
    claims: list[dict[str, Any]] | None = None,
    summary_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total_claims": 0,
        "supported": 0,
        "unsupported": 0,
        "not_addressed": 0,
        "partially_supported": 0,
        "unverifiable": 0,
        "unverifiable_by_reason": {},
        "citation_found_rate": 0.0,
        "verifiability_status": "no_citations_found",
        "fulltext_verified": 0,
        "no_passage_found": 0,
        "fulltext_unavailable": 0,
        "resolution_low_confidence": 0,
        "retracted_sources": 0,
        "numeric_checks_run": 0,
        "numeric_inconsistencies_flagged": 0,
        "abstract_only_verdicts": 0,
        "fulltext_success_rate": 0.0,
        "not_addressed_breakdown": {
            "no_source": 0,
            "paywall": 0,
            "no_passage": 0,
            "claim_absent": 0,
        },
        "total_cost_usd": 0.0123,
        "usage_by_stage": {},
        "cross_modal_disagreements": 0,
    }
    if summary_overrides:
        summary.update(summary_overrides)
    return {
        "report_id": report_id,
        "timestamp": 1715500000.0,
        "input_text": "Some input text.",
        "summary": summary,
        "claims": claims or [],
    }


def _make_claim(
    *,
    claim_id: str = "c1",
    claim_text: str = "Drug X reduces biomarker Y by 30%.",
    status: str = "supported",
    confidence: float | None = 0.85,
    cited_authors: list[str] | None = None,
    cited_year: int | None = 2022,
    doi: str | None = "10.1234/abc",
    title: str | None = "A landmark study",
    found: bool = True,
    passages: list[str] | None = None,
    evidence_quality: str = "quoted_passage",
    retrieval_status: str = "passage_found",
    explanation: str = "The paper directly reports a 30% reduction.",
    unverifiable_reason: str | None = None,
    numeric_check: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "claim_text": claim_text,
        "claim_type": "factual_numeric",
        "cited_authors": cited_authors or ["Smith"],
        "cited_year": cited_year,
        "source": {
            "found": found,
            "doi": doi,
            "title": title,
            "abstract": "abstract text" if found else None,
            "similarity_score": 0.9 if found else None,
            "title_match_score": 0.95 if found else None,
            "resolution_low_confidence": False,
            "oa_url": None,
            "pmcid": None,
            "retraction_status": False,
        },
        "verification": {
            "status": status,
            "explanation": explanation,
            "confidence": confidence,
            "source_passages": passages or [],
            "source_section": "results",
            "fulltext_available": True,
            "verification_depth": "fulltext",
            "retrieval_status": retrieval_status,
            "evidence_quality": evidence_quality,
            "retraction_status": False,
            "numeric_check": numeric_check,
            "unverifiable_reason": unverifiable_reason,
        },
    }


class TestHeader:
    def test_includes_run_id(self) -> None:
        md = render_markdown(_make_report(report_id="run-xyz"))
        assert "run-xyz" in md

    def test_includes_total_cost(self) -> None:
        md = render_markdown(_make_report(summary_overrides={"total_cost_usd": 0.0456}))
        assert "$0.0456" in md or "0.0456" in md

    def test_starts_with_h1(self) -> None:
        md = render_markdown(_make_report())
        assert md.lstrip().startswith("# ")


class TestSummary:
    def test_counts_each_verdict(self) -> None:
        md = render_markdown(
            _make_report(
                summary_overrides={
                    "total_claims": 5,
                    "supported": 2,
                    "unsupported": 1,
                    "not_addressed": 1,
                    "unverifiable": 1,
                },
                claims=[_make_claim(claim_id=f"c{i}") for i in range(5)],
            )
        )
        assert "Supported" in md
        assert "Unsupported" in md
        assert "Not addressed" in md
        assert "Unverifiable" in md

    def test_citation_resolution_rate_shown(self) -> None:
        md = render_markdown(
            _make_report(
                summary_overrides={"total_claims": 4, "citation_found_rate": 0.75},
                claims=[_make_claim(claim_id=f"c{i}") for i in range(4)],
            )
        )
        assert "75" in md  # rendered as percentage
        assert "3/4" in md  # resolved/total

    def test_empty_claims_renders_valid_markdown(self) -> None:
        md = render_markdown(_make_report())
        assert "# " in md  # at least one header
        assert "Verification" in md


class TestClaims:
    def test_renders_one_section_per_claim(self) -> None:
        claims = [
            _make_claim(claim_id="c1", claim_text="First claim."),
            _make_claim(claim_id="c2", claim_text="Second claim."),
        ]
        md = render_markdown(_make_report(claims=claims))
        assert "First claim." in md
        assert "Second claim." in md

    def test_verdict_label_uppercase_in_heading(self) -> None:
        md = render_markdown(_make_report(claims=[_make_claim(status="supported")]))
        assert "SUPPORTED" in md

    def test_unverifiable_verdict_rendered(self) -> None:
        md = render_markdown(
            _make_report(
                claims=[
                    _make_claim(
                        status="unverifiable",
                        confidence=None,
                        evidence_quality="abstract_only",
                        retrieval_status="fulltext_unavailable",
                        unverifiable_reason="numeric_claim_abstract_only",
                    )
                ]
            )
        )
        assert "UNVERIFIABLE" in md
        assert "numeric_claim_abstract_only" in md

    def test_doi_rendered_as_link(self) -> None:
        md = render_markdown(_make_report(claims=[_make_claim(doi="10.1038/nature12373")]))
        assert "10.1038/nature12373" in md
        assert "https://doi.org/10.1038/nature12373" in md

    def test_no_source_resolved_message(self) -> None:
        md = render_markdown(_make_report(claims=[_make_claim(found=False, doi=None, title=None)]))
        assert "not resolved" in md.lower() or "no source" in md.lower()

    def test_passages_rendered_as_blockquotes(self) -> None:
        md = render_markdown(
            _make_report(
                claims=[_make_claim(passages=["The study reports a 30% reduction in biomarker Y."])]
            )
        )
        assert "> The study reports a 30% reduction in biomarker Y." in md

    def test_confidence_shown(self) -> None:
        md = render_markdown(_make_report(claims=[_make_claim(confidence=0.87)]))
        assert "0.87" in md

    def test_numeric_check_inconsistency_flagged(self) -> None:
        md = render_markdown(
            _make_report(
                claims=[
                    _make_claim(
                        numeric_check={
                            "consistent": False,
                            "check_type": "relative_diff",
                            "claim_value": 30.0,
                            "source_value": 20.0,
                            "explanation": "Claimed 30%, source reports 20%.",
                        }
                    )
                ]
            )
        )
        assert "INCONSISTENT" in md or "inconsistent" in md


class TestEscaping:
    def test_special_markdown_chars_in_claim_text_preserved_or_escaped(self) -> None:
        md = render_markdown(
            _make_report(
                claims=[
                    _make_claim(claim_text="Claim with *asterisks* and [brackets] and `backticks`.")
                ]
            )
        )
        # Either escaped or rendered verbatim — must not break the markdown structure.
        assert "asterisks" in md
        assert "brackets" in md
        assert "backticks" in md

    def test_newlines_in_explanation_handled(self) -> None:
        md = render_markdown(
            _make_report(claims=[_make_claim(explanation="Line one.\nLine two.\nLine three.")])
        )
        assert "Line one." in md
        assert "Line three." in md


class TestFromFile:
    def test_writes_md_next_to_json(self, tmp_path: Path) -> None:
        report_path = tmp_path / "report.json"
        report_path.write_text(json.dumps(_make_report()), encoding="utf-8")
        md_path = render_markdown_from_file(report_path)
        assert md_path == tmp_path / "report.md"
        assert md_path.exists()
        assert md_path.read_text(encoding="utf-8").lstrip().startswith("# ")

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            render_markdown_from_file(tmp_path / "nonexistent.json")


class TestForwardCompat:
    def test_missing_optional_summary_fields_does_not_crash(self) -> None:
        minimal = {
            "report_id": "old-run",
            "timestamp": 0.0,
            "input_text": "",
            "summary": {
                "total_claims": 0,
                "supported": 0,
                "unsupported": 0,
                "not_addressed": 0,
                "partially_supported": 0,
                "total_cost_usd": 0.0,
            },
            "claims": [],
        }
        md = render_markdown(minimal)
        assert "old-run" in md

    def test_missing_unverifiable_reason_renders(self) -> None:
        claim = _make_claim(
            status="unverifiable",
            confidence=None,
            evidence_quality="abstract_only",
            unverifiable_reason=None,
        )
        md = render_markdown(_make_report(claims=[claim]))
        assert "UNVERIFIABLE" in md


class TestAutoWiredFromBuildReport:
    """When build_report writes report.json, it should also write report.md."""

    def test_build_report_emits_markdown(self, tmp_path: Path) -> None:
        from src.models import Claim, ProvenanceStep, ResolvedSource, VerificationResult
        from src.report import build_report

        claim = Claim(
            claim_id="c1",
            claim_text="X causes Y.",
            cited_authors=["Smith"],
            cited_year=2020,
            claim_type="causal",
        )
        source = ResolvedSource(
            found=True,
            doi="10.1234/x",
            title="A Paper",
            abstract="abstract",
            similarity_score=0.9,
        )
        result = VerificationResult(
            status="supported",
            explanation="Yes.",
            confidence=0.9,
            evidence_quality="quoted_passage",
            source_passages=["X causes Y, p<0.01."],
        )
        step = ProvenanceStep(
            step_id="s1",
            claim_id="c1",
            operation="verify",
            input_hash="a",
            output_hash="b",
            model_id="claude-sonnet-4-6",
            timestamp=0.0,
            tokens_in=100,
            tokens_out=10,
            cache_hit=True,
            confidence=0.9,
        )

        run_dir = build_report(
            "run-md-test",
            "Input.",
            [claim],
            {"c1": source},
            {"c1": result},
            [step],
            output_dir=tmp_path,
        )

        md_path = run_dir / "report.md"
        assert md_path.exists()
        md = md_path.read_text(encoding="utf-8")
        assert "run-md-test" in md
        assert "X causes Y." in md
        assert "SUPPORTED" in md
