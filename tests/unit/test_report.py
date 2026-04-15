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
        title="Test Paper",
        abstract="Abstract text.",
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
    def test_happy_path_creates_files(self, tmp_path: Path) -> None:
        from src.report import build_report

        claims = [_make_claim("c1")]
        sources = {"c1": _make_source()}
        results = {"c1": _make_result()}
        steps = [_make_step("s1", "c1")]

        run_dir = build_report(
            "report-001",
            "Some scientific text.",
            claims,
            sources,
            results,
            steps,
            output_dir=tmp_path,
        )

        report_file = run_dir / "report.json"
        provenance_file = run_dir / "provenance.jsonl"
        assert report_file.exists()
        assert provenance_file.exists()

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

    def test_provenance_jsonl_valid_json(self, tmp_path: Path) -> None:
        from src.report import build_report

        steps = [_make_step("s1", "c1")]
        run_dir = build_report(
            "report-007",
            "T.",
            [_make_claim()],
            {"c1": _make_source()},
            {"c1": _make_result()},
            steps,
            output_dir=tmp_path,
        )

        for line in (run_dir / "provenance.jsonl").read_text().strip().split("\n"):
            obj = json.loads(line)
            assert "step_id" in obj
            assert "operation" in obj

    def test_aggregate_provenance_step_written(self, tmp_path: Path) -> None:
        from src.report import build_report

        run_dir = build_report("report-008", "T.", [], {}, {}, [], output_dir=tmp_path)

        lines = (run_dir / "provenance.jsonl").read_text().strip().split("\n")
        last = json.loads(lines[-1])
        assert last["operation"] == "aggregate"

    def test_returns_path_to_run_dir(self, tmp_path: Path) -> None:
        from src.report import build_report

        run_dir = build_report("report-009", "T.", [], {}, {}, [], output_dir=tmp_path)
        assert isinstance(run_dir, Path)
        assert run_dir.name == "report-009"

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
