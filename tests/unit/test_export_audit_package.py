"""Unit tests for the external-review audit package exporter."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import cast

from scripts.export_audit_package import export_audit_package, failure_type_for_claim


def _claim(
    claim_id: str,
    *,
    status: str,
    retrieval_status: str = "passage_found",
    evidence_quality: str = "quoted_passage",
    source_found: bool = True,
    doi: str | None = "10.1000/example",
    resolution_low_confidence: bool = False,
    unverifiable_reason: str | None = None,
    numeric_check: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "claim_text": f"Claim {claim_id}",
        "claim_type": "factual_numeric",
        "cited_authors": ["Smith"],
        "cited_year": 2024,
        "source": {
            "found": source_found,
            "doi": doi,
            "title": "Example source" if source_found else None,
            "resolution_low_confidence": resolution_low_confidence,
        },
        "verification": {
            "status": status,
            "explanation": "Verifier rationale.",
            "confidence": None if status == "unverifiable" else 0.82,
            "source_passages": ["quoted passage"] if evidence_quality != "no_evidence" else [],
            "verification_depth": "fulltext" if retrieval_status == "passage_found" else "abstract",
            "retrieval_status": retrieval_status,
            "evidence_quality": evidence_quality,
            "unverifiable_reason": unverifiable_reason,
            "numeric_check": numeric_check,
        },
    }


def _report(path: Path, claims: list[dict[str, object]]) -> Path:
    def status_of(claim: dict[str, object]) -> str:
        return cast(str, cast(dict[str, object], claim["verification"])["status"])

    def is_resolution_low_confidence(claim: dict[str, object]) -> bool:
        return cast(bool, cast(dict[str, object], claim["source"])["resolution_low_confidence"])

    summary = {
        "total_claims": len(claims),
        "supported": sum(1 for c in claims if status_of(c) == "supported"),
        "partially_supported": sum(1 for c in claims if status_of(c) == "partially_supported"),
        "unsupported": sum(1 for c in claims if status_of(c) == "unsupported"),
        "not_addressed": sum(1 for c in claims if status_of(c) == "not_addressed"),
        "unverifiable": sum(1 for c in claims if status_of(c) == "unverifiable"),
        "citation_found_rate": 0.8,
        "fulltext_success_rate": 0.5,
        "resolution_low_confidence": sum(1 for c in claims if is_resolution_low_confidence(c)),
        "not_addressed_breakdown": {
            "no_source": 1,
            "paywall": 1,
            "no_passage": 0,
            "claim_absent": 0,
        },
        "unverifiable_by_reason": {"numeric_claim_abstract_only": 1},
        "numeric_checks_run": 1,
        "numeric_inconsistencies_flagged": 1,
        "total_cost_usd": 0.42,
    }
    report = {
        "report_id": "pilot-report",
        "timestamp": 1778603609.0,
        "input_text": "input",
        "summary": summary,
        "claims": claims,
    }
    report_path = path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path


def test_failure_type_for_claim_uses_actionable_audit_categories() -> None:
    assert (
        failure_type_for_claim(
            _claim("c1", status="unverifiable", unverifiable_reason="numeric_claim_abstract_only")
        )
        == "fulltext_required_numeric_claim"
    )
    assert failure_type_for_claim(_claim("c2", status="unsupported")) == "source_contradicts_claim"
    assert (
        failure_type_for_claim(_claim("c3", status="not_addressed", source_found=False, doi=None))
        == "no_source_anchor"
    )
    assert (
        failure_type_for_claim(_claim("c4", status="supported", resolution_low_confidence=True))
        == "wrong_source_risk"
    )
    assert (
        failure_type_for_claim(
            _claim(
                "c5",
                status="supported",
                numeric_check={"consistent": False, "ambiguous": False},
            )
        )
        == "numeric_inconsistency"
    )


def test_export_audit_package_writes_claim_csv_with_review_columns(tmp_path: Path) -> None:
    report_path = _report(
        tmp_path,
        [
            _claim("c1", status="supported"),
            _claim("c2", status="unsupported"),
            _claim("c3", status="not_addressed", source_found=False, doi=None),
            _claim("c4", status="unverifiable", unverifiable_reason="numeric_claim_abstract_only"),
        ],
    )

    output_dir = export_audit_package(report_path, tmp_path / "package")

    rows = list(csv.DictReader((output_dir / "claims.csv").open(encoding="utf-8")))
    assert [row["claim_id"] for row in rows] == ["c1", "c2", "c3", "c4"]
    assert rows[1]["failure_type"] == "source_contradicts_claim"
    assert rows[2]["failure_type"] == "no_source_anchor"
    assert rows[3]["failure_type"] == "fulltext_required_numeric_claim"
    assert "human_verdict" in rows[0]
    assert "human_note" in rows[0]
    assert "customer_action" in rows[0]


def test_export_audit_package_writes_summary_limitations_and_manifest(tmp_path: Path) -> None:
    report_path = _report(
        tmp_path,
        [
            _claim("c1", status="supported"),
            _claim("c2", status="unsupported"),
            _claim("c3", status="not_addressed", source_found=False, doi=None),
            _claim("c4", status="unverifiable", unverifiable_reason="numeric_claim_abstract_only"),
        ],
    )

    output_dir = export_audit_package(report_path, tmp_path / "package")

    audit_summary = (output_dir / "audit_summary.md").read_text(encoding="utf-8")
    limitations = (output_dir / "limitations.md").read_text(encoding="utf-8")
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert "pilot-report" in audit_summary
    assert "source_contradicts_claim" in audit_summary
    assert "fulltext_required_numeric_claim" in audit_summary
    assert "human-review queue" in limitations
    assert "LLM-reported confidence is not a decision signal" in limitations
    assert manifest["report_id"] == "pilot-report"
    assert manifest["outputs"] == [
        "claims.csv",
        "audit_summary.md",
        "limitations.md",
        "manifest.json",
    ]
