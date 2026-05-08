"""Unit tests for src/report.py — report writing and provenance."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.models import Claim, ProvenanceStep, ResolvedSource, VerificationResult


def _make_claim(claim_id: str = "c1") -> Claim:
    return Claim(
        claim_id=claim_id,
        claim_text="X causes Y.",
        cited_authors=["Smith"],
        cited_year=2020,
        claim_type="causal",
    )


def _make_source(found: bool = True) -> ResolvedSource:
    return ResolvedSource(
        found=found,
        doi=None,
        title="Test Paper" if found else None,
        abstract="Abstract text." if found else None,
        similarity_score=0.9 if found else None,
    )


def _make_result(status: str = "supported") -> VerificationResult:
    return VerificationResult(status=status, explanation="Ok.", confidence=0.9)  # type: ignore[arg-type]


def _make_step(
    step_id: str = "step-1",
    claim_id: str = "c1",
    operation: str = "verify",
    tokens_in: int | None = 100,
    tokens_out: int | None = 50,
    cache_hit: bool | None = True,
) -> ProvenanceStep:
    return ProvenanceStep(
        step_id=step_id,
        claim_id=claim_id,
        operation=operation,  # type: ignore[arg-type]
        input_hash="aaa",
        output_hash="bbb",
        model_id="claude-sonnet-4-6" if tokens_in else None,
        timestamp=time.time(),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cache_hit=cache_hit,
        confidence=0.9,
    )


class TestBuildReport:
    def test_report_json_structure(self, tmp_path: Path) -> None:
        from src.report import build_report

        claims = [_make_claim("c1"), _make_claim("c2")]
        sources = {"c1": _make_source(), "c2": _make_source(found=False)}
        results = {"c1": _make_result("supported"), "c2": _make_result("not_addressed")}
        steps = [_make_step("s1", "c1"), _make_step("s2", "c2", tokens_in=None)]

        run_dir = build_report(
            "report-002",
            "Text.",
            claims,
            sources,
            results,
            steps,
            output_dir=tmp_path,
        )

        with open(run_dir / "report.json") as f:
            report = json.load(f)

        assert report["report_id"] == "report-002"
        assert "timestamp" in report
        assert report["input_text"] == "Text."
        assert "summary" in report
        assert report["summary"]["total_claims"] == 2
        assert report["summary"]["supported"] == 1
        assert report["summary"]["not_addressed"] == 1
        assert "claims" in report
        assert len(report["claims"]) == 2

    def test_summary_stats(self, tmp_path: Path) -> None:
        from src.report import build_report

        claims = [_make_claim(f"c{i}") for i in range(4)]
        sources = {
            "c0": _make_source(found=True),
            "c1": _make_source(found=True),
            "c2": _make_source(found=False),
            "c3": _make_source(found=True),
        }
        results = {
            "c0": _make_result("supported"),
            "c1": _make_result("unsupported"),
            "c2": _make_result("not_addressed"),
            "c3": _make_result("partially_supported"),
        }
        steps = [_make_step(f"s{i}", f"c{i}") for i in range(4)]

        run_dir = build_report(
            "report-003", "Text.", claims, sources, results, steps, output_dir=tmp_path
        )

        with open(run_dir / "report.json") as f:
            report = json.load(f)

        summary = report["summary"]
        assert summary["total_claims"] == 4
        assert summary["supported"] == 1
        assert summary["unsupported"] == 1
        assert summary["not_addressed"] == 1
        assert summary["partially_supported"] == 1
        assert summary["citation_found_rate"] == pytest.approx(3 / 4)

    def test_ec5_empty_claims_valid_report(self, tmp_path: Path) -> None:
        """EC-5: Empty claims list produces valid report with total_claims=0."""
        from src.report import build_report

        run_dir = build_report("report-004", "Text.", [], {}, {}, [], output_dir=tmp_path)

        with open(run_dir / "report.json") as f:
            report = json.load(f)

        assert report["summary"]["total_claims"] == 0
        assert report["claims"] == []

    def test_ec7_missing_directory_created(self, tmp_path: Path) -> None:
        """EC-7: output_dir pointing to non-existent nested path is created."""
        from src.report import build_report

        nested = tmp_path / "deep" / "nested" / "path"
        # nested does not exist yet

        run_dir = build_report("report-005", "Text.", [], {}, {}, [], output_dir=nested)

        assert (run_dir / "report.json").exists()
        assert (run_dir / "provenance.jsonl").exists()

    def test_provenance_jsonl_one_line_per_step_plus_aggregate(self, tmp_path: Path) -> None:
        """provenance.jsonl has one line per input step + 1 aggregate step."""
        from src.report import build_report

        steps = [_make_step(f"s{i}", "c1") for i in range(3)]
        run_dir = build_report(
            "report-006",
            "T.",
            [_make_claim()],
            {"c1": _make_source()},
            {"c1": _make_result()},
            steps,
            output_dir=tmp_path,
        )

        lines = (run_dir / "provenance.jsonl").read_text().strip().split("\n")
        assert len(lines) == 4  # 3 input steps + 1 aggregate

    def test_cost_calculation_in_summary(self, tmp_path: Path) -> None:
        from src.report import build_report

        # 1000 tokens_in (uncached), 100 tokens_out → cost = 1000*3/1e6 + 100*15/1e6 = 0.003 + 0.0015 = 0.0045
        step = _make_step("s1", "c1", tokens_in=1000, tokens_out=100, cache_hit=False)
        run_dir = build_report(
            "report-010",
            "T.",
            [_make_claim()],
            {"c1": _make_source()},
            {"c1": _make_result()},
            [step],
            output_dir=tmp_path,
        )

        with open(run_dir / "report.json") as f:
            report = json.load(f)

        assert report["summary"]["total_cost_usd"] == pytest.approx(0.0045, rel=1e-3)

    def test_cost_calculation_cached_tokens(self, tmp_path: Path) -> None:
        from src.report import build_report

        # 1000 cache-hit tokens_in, 100 tokens_out → cost = 1000*0.30/1e6 + 100*15/1e6 = 0.0003 + 0.0015 = 0.0018
        step = _make_step("s1", "c1", tokens_in=1000, tokens_out=100, cache_hit=True)
        run_dir = build_report(
            "report-011",
            "T.",
            [_make_claim()],
            {"c1": _make_source()},
            {"c1": _make_result()},
            [step],
            output_dir=tmp_path,
        )

        with open(run_dir / "report.json") as f:
            report = json.load(f)

        assert report["summary"]["total_cost_usd"] == pytest.approx(0.0018, rel=1e-3)


class TestPhase1ReportFields:
    def test_fulltext_verified_count_in_summary(self, tmp_path: Path) -> None:
        from src.report import build_report

        claims = [_make_claim(f"c{i}") for i in range(3)]
        sources = {f"c{i}": _make_source() for i in range(3)}
        results = {
            "c0": VerificationResult(
                status="supported",
                explanation="ok",
                confidence=0.9,
                verification_depth="fulltext",
                fulltext_available=True,
                retrieval_status="passage_found",
            ),
            "c1": VerificationResult(
                status="unsupported",
                explanation="ok",
                confidence=0.9,
                verification_depth="fulltext",
                fulltext_available=True,
                retrieval_status="passage_found",
            ),
            "c2": VerificationResult(  # abstract path
                status="supported",
                explanation="ok",
                confidence=0.9,
            ),
        }
        steps = [_make_step(f"s{i}", f"c{i}") for i in range(3)]

        run_dir = build_report(
            "report-fulltext", "Text.", claims, sources, results, steps, output_dir=tmp_path
        )
        with open(run_dir / "report.json") as f:
            report = json.load(f)

        assert report["summary"]["fulltext_verified"] == 2

    def test_source_passages_serialized(self, tmp_path: Path) -> None:
        from src.report import build_report

        claims = [_make_claim("c1")]
        sources = {"c1": _make_source()}
        results = {
            "c1": VerificationResult(
                status="supported",
                explanation="ok",
                confidence=0.9,
                source_passages=["First quote.", "Second quote."],
                source_section="results",
                fulltext_available=True,
                verification_depth="fulltext",
                retrieval_status="passage_found",
                evidence_quality="quoted_passage",
            )
        }
        steps = [_make_step("s1", "c1")]

        run_dir = build_report(
            "report-passages", "Text.", claims, sources, results, steps, output_dir=tmp_path
        )
        with open(run_dir / "report.json") as f:
            report = json.load(f)

        verification = report["claims"][0]["verification"]
        assert verification["source_passages"] == ["First quote.", "Second quote."]
        assert verification["source_section"] == "results"
        assert verification["verification_depth"] == "fulltext"

    def test_numeric_checks_summary_counts(self, tmp_path: Path) -> None:
        from src.numeric.checks import NumericCheckResult
        from src.report import build_report

        nc_consistent = NumericCheckResult(
            check_type="or_ci_consistency",
            consistent=True,
            extracted=[],
            explanation="OK",
        )
        nc_flagged = NumericCheckResult(
            check_type="or_ci_consistency",
            consistent=False,
            extracted=[],
            explanation="bad",
        )

        claims = [_make_claim(f"c{i}") for i in range(3)]
        sources = {f"c{i}": _make_source() for i in range(3)}
        results = {
            "c0": VerificationResult(
                status="supported",
                explanation="ok",
                confidence=0.9,
                numeric_check=nc_consistent,
            ),
            "c1": VerificationResult(
                status="unsupported",
                explanation="ok",
                confidence=0.9,
                numeric_check=nc_flagged,
            ),
            "c2": VerificationResult(
                status="not_addressed",
                explanation="ok",
                confidence=0.9,
                numeric_check=None,
            ),
        }
        steps = [_make_step(f"s{i}", f"c{i}") for i in range(3)]

        run_dir = build_report(
            "report-numeric",
            "Text.",
            claims,
            sources,
            results,
            steps,
            output_dir=tmp_path,
        )
        with open(run_dir / "report.json") as f:
            report = json.load(f)

        assert report["summary"]["numeric_checks_run"] == 2
        assert report["summary"]["numeric_inconsistencies_flagged"] == 1

    def test_retracted_sources_count_in_summary(self, tmp_path: Path) -> None:
        from src.report import build_report

        claims = [_make_claim("c1"), _make_claim("c2")]
        sources = {
            "c1": ResolvedSource(
                found=True,
                doi="10.1/r",
                title="T",
                abstract="a",
                similarity_score=1.0,
                retraction_status=True,
            ),
            "c2": _make_source(),
        }
        results = {"c1": _make_result(), "c2": _make_result()}
        steps = [_make_step(f"s{i}", f"c{i + 1}") for i in range(2)]

        run_dir = build_report(
            "report-retr", "Text.", claims, sources, results, steps, output_dir=tmp_path
        )
        with open(run_dir / "report.json") as f:
            report = json.load(f)

        assert report["summary"]["retracted_sources"] == 1

    def test_evidence_diagnostic_summary_counts(self, tmp_path: Path) -> None:
        from src.report import build_report

        claims = [_make_claim(f"c{i}") for i in range(4)]
        sources = {
            "c0": _make_source(),
            "c1": _make_source(),
            "c2": ResolvedSource(
                found=True,
                doi="10.1/weak",
                title="Weak Match",
                abstract="a",
                similarity_score=1.0,
                resolution_low_confidence=True,
            ),
            "c3": _make_source(found=False),
        }
        results = {
            "c0": VerificationResult(
                status="supported",
                explanation="ok",
                confidence=0.9,
                verification_depth="fulltext",
                fulltext_available=True,
                retrieval_status="passage_found",
            ),
            "c1": VerificationResult(
                status="not_addressed",
                explanation="No relevant passage.",
                confidence=0.9,
                verification_depth="abstract",
                fulltext_available=True,
                retrieval_status="no_passage_found",
            ),
            "c2": _make_result("not_addressed"),
            "c3": _make_result("not_addressed"),
        }
        steps = [_make_step(f"s{i}", f"c{i}") for i in range(4)]

        run_dir = build_report(
            "report-diagnostics",
            "Text.",
            claims,
            sources,
            results,
            steps,
            output_dir=tmp_path,
        )
        with open(run_dir / "report.json") as f:
            report = json.load(f)

        assert report["summary"]["fulltext_verified"] == 1
        assert report["summary"]["no_passage_found"] == 1
        assert report["summary"]["fulltext_unavailable"] == 2
        assert report["summary"]["resolution_low_confidence"] == 1


class TestVerifiabilityStatus:
    """Tests for verifiability_status field in report summary."""

    def test_verifiable_when_majority_citations_found(self, tmp_path: Path) -> None:
        """citation_found_rate > 0.5 → verifiable."""
        from src.report import build_report

        claims = [_make_claim(f"c{i}") for i in range(4)]
        sources = {f"c{i}": _make_source(found=(i < 3)) for i in range(4)}  # 3/4 found
        results = {f"c{i}": _make_result() for i in range(4)}
        steps = [_make_step(f"s{i}", f"c{i}") for i in range(4)]

        run_dir = build_report("vs-001", "T.", claims, sources, results, steps, output_dir=tmp_path)

        with open(run_dir / "report.json") as f:
            report = json.load(f)

        assert report["summary"]["verifiability_status"] == "verifiable"

    def test_no_citations_found_when_rate_is_zero(self, tmp_path: Path) -> None:
        """citation_found_rate == 0.0 → no_citations_found."""
        from src.report import build_report

        claims = [_make_claim("c1"), _make_claim("c2")]
        sources = {"c1": _make_source(found=False), "c2": _make_source(found=False)}
        results = {"c1": _make_result("not_addressed"), "c2": _make_result("not_addressed")}
        steps = [_make_step("s1", "c1"), _make_step("s2", "c2")]

        run_dir = build_report("vs-002", "T.", claims, sources, results, steps, output_dir=tmp_path)

        with open(run_dir / "report.json") as f:
            report = json.load(f)

        assert report["summary"]["verifiability_status"] == "no_citations_found"

    def test_low_citation_density_when_rate_between_zero_and_half(self, tmp_path: Path) -> None:
        """0 < citation_found_rate <= 0.5 → low_citation_density."""
        from src.report import build_report

        claims = [_make_claim(f"c{i}") for i in range(4)]
        sources = {f"c{i}": _make_source(found=(i == 0)) for i in range(4)}  # 1/4 found
        results = {f"c{i}": _make_result() for i in range(4)}
        steps = [_make_step(f"s{i}", f"c{i}") for i in range(4)]

        run_dir = build_report("vs-003", "T.", claims, sources, results, steps, output_dir=tmp_path)

        with open(run_dir / "report.json") as f:
            report = json.load(f)

        assert report["summary"]["verifiability_status"] == "low_citation_density"


class TestUsageByStage:
    """S4b-6: per-stage cost / token bucketing in report.summary.usage_by_stage."""

    def test_buckets_steps_by_operation(self, tmp_path: Path) -> None:
        from src.report import build_report

        claims = [_make_claim("c1")]
        sources = {"c1": _make_source()}
        results = {"c1": _make_result()}
        steps = [
            _make_step("s1", "c1", operation="extract", tokens_in=200, tokens_out=80),
            _make_step("s2", "c1", operation="resolve", tokens_in=None, tokens_out=None),
            _make_step("s3", "c1", operation="verify", tokens_in=300, tokens_out=120),
            _make_step("s4", "c1", operation="verify", tokens_in=250, tokens_out=100),
        ]

        run_dir = build_report("ub-001", "T.", claims, sources, results, steps, output_dir=tmp_path)
        with open(run_dir / "report.json") as f:
            report = json.load(f)

        usage = report["summary"]["usage_by_stage"]
        # The auto-emitted aggregate step is meta (it covers the report
        # itself) and is intentionally excluded from per-stage usage so
        # the operator sees the audited operations only.
        assert set(usage.keys()) == {"extract", "resolve", "verify"}
        # Verify is summed across both steps.
        assert usage["verify"]["tokens_in"] == 550
        assert usage["verify"]["tokens_out"] == 220
        assert usage["verify"]["n_steps"] == 2
        # Extract has its own bucket.
        assert usage["extract"]["tokens_in"] == 200
        assert usage["extract"]["n_steps"] == 1
        # Resolve had no token data — bucket exists with zeros for visibility.
        assert usage["resolve"]["tokens_in"] == 0
        assert usage["resolve"]["n_steps"] == 1

    def test_cache_hit_count_per_stage(self, tmp_path: Path) -> None:
        from src.report import build_report

        claims = [_make_claim("c1")]
        sources = {"c1": _make_source()}
        results = {"c1": _make_result()}
        steps = [
            _make_step("s1", "c1", operation="verify", cache_hit=True),
            _make_step("s2", "c1", operation="verify", cache_hit=False),
            _make_step("s3", "c1", operation="verify", cache_hit=True),
        ]

        run_dir = build_report("ub-002", "T.", claims, sources, results, steps, output_dir=tmp_path)
        with open(run_dir / "report.json") as f:
            report = json.load(f)

        assert report["summary"]["usage_by_stage"]["verify"]["n_cache_hits"] == 2

    def test_per_stage_cost_sums_to_total(self, tmp_path: Path) -> None:
        from src.report import build_report

        claims = [_make_claim("c1")]
        sources = {"c1": _make_source()}
        results = {"c1": _make_result()}
        steps = [
            _make_step("s1", "c1", operation="extract", tokens_in=200, tokens_out=80),
            _make_step("s2", "c1", operation="verify", tokens_in=300, tokens_out=120),
        ]

        run_dir = build_report("ub-003", "T.", claims, sources, results, steps, output_dir=tmp_path)
        with open(run_dir / "report.json") as f:
            report = json.load(f)

        usage = report["summary"]["usage_by_stage"]
        per_stage_total = sum(float(b["cost_usd"]) for b in usage.values())
        # The auto-emitted aggregate step contributes 0 cost (no tokens).
        assert per_stage_total == report["summary"]["total_cost_usd"]
