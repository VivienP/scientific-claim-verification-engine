"""Export an external-review audit package from an existing report.json.

The exporter is intentionally post-processing only: it does not call an LLM,
does not rerun resolution, and does not change verdict semantics. Its job is
to turn the canonical report into files a buyer or reviewer can inspect.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import structlog

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_OUTPUTS = ["claims.csv", "audit_summary.md", "limitations.md", "manifest.json"]


def _mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value)
    return {}


def _string(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _bool_string(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return ""


def _passage_count(value: object) -> int:
    if isinstance(value, list):
        return len(value)
    return 0


def _numeric_check_consistent(value: object) -> str:
    check = _mapping(value)
    if not check:
        return ""
    consistent = check.get("consistent")
    if isinstance(consistent, bool):
        return "true" if consistent else "false"
    return ""


def failure_type_for_claim(claim: Mapping[str, Any]) -> str:
    """Return the review category that should drive triage for one claim."""
    source = _mapping(claim.get("source"))
    verification = _mapping(claim.get("verification"))
    numeric_check = _mapping(verification.get("numeric_check"))

    if numeric_check.get("consistent") is False:
        if numeric_check.get("ambiguous") is True:
            return "numeric_pairing_uncertain"
        return "numeric_inconsistency"

    if source.get("found") is False or not source.get("doi"):
        return "no_source_anchor"

    if source.get("resolution_low_confidence") is True:
        return "wrong_source_risk"

    status = _string(verification.get("status"))
    retrieval_status = _string(verification.get("retrieval_status"))
    reason = _string(verification.get("unverifiable_reason"))

    if status == "unverifiable":
        if reason == "numeric_claim_abstract_only":
            return "fulltext_required_numeric_claim"
        if reason == "fulltext_unavailable" or retrieval_status == "fulltext_unavailable":
            return "fulltext_unavailable"
        if reason == "insufficient_evidence_depth":
            return "insufficient_evidence_depth"
        return "unverifiable"
    if status == "unsupported":
        return "source_contradicts_claim"
    if status == "not_addressed":
        if retrieval_status == "fulltext_unavailable":
            return "source_silent_or_paywalled"
        if retrieval_status == "no_passage_found":
            return "no_relevant_passage_found"
        return "source_silent"
    if status == "partially_supported":
        return "claim_needs_qualification"
    if status == "supported":
        return "source_supports_claim"
    return "unknown"


def _claim_row(claim: Mapping[str, Any]) -> dict[str, str]:
    source = _mapping(claim.get("source"))
    verification = _mapping(claim.get("verification"))
    return {
        "claim_id": _string(claim.get("claim_id")),
        "status": _string(verification.get("status")),
        "failure_type": failure_type_for_claim(claim),
        "claim_text": _string(claim.get("claim_text")),
        "doi": _string(source.get("doi")),
        "source_title": _string(source.get("title")),
        "evidence_quality": _string(verification.get("evidence_quality")),
        "retrieval_status": _string(verification.get("retrieval_status")),
        "verification_depth": _string(verification.get("verification_depth")),
        "confidence": _string(verification.get("confidence")),
        "resolution_low_confidence": _bool_string(source.get("resolution_low_confidence")),
        "unverifiable_reason": _string(verification.get("unverifiable_reason")),
        "numeric_check_consistent": _numeric_check_consistent(verification.get("numeric_check")),
        "source_passages_count": str(_passage_count(verification.get("source_passages"))),
        "human_verdict": "",
        "human_note": "",
        "customer_action": "",
    }


def _write_claims_csv(claims: list[Mapping[str, Any]], output_dir: Path) -> Counter[str]:
    rows = [_claim_row(claim) for claim in claims]
    fieldnames = list(rows[0].keys()) if rows else list(_claim_row({}).keys())
    with (output_dir / "claims.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return Counter(row["failure_type"] for row in rows)


def _summary_lines(summary: Mapping[str, Any]) -> list[str]:
    total = int(summary.get("total_claims") or 0)
    cost = float(summary.get("total_cost_usd") or 0.0)
    citation_rate = float(summary.get("citation_found_rate") or 0.0)
    fulltext_rate = float(summary.get("fulltext_success_rate") or 0.0)
    return [
        f"- Total claims: {total}",
        f"- Citation found rate: {citation_rate:.1%}",
        f"- Fulltext success rate: {fulltext_rate:.1%}",
        f"- Unsupported: {int(summary.get('unsupported') or 0)}",
        f"- Not addressed: {int(summary.get('not_addressed') or 0)}",
        f"- Unverifiable: {int(summary.get('unverifiable') or 0)}",
        "- Numeric inconsistencies flagged: "
        f"{int(summary.get('numeric_inconsistencies_flagged') or 0)}",
        f"- Estimated model cost: ${cost:.2f}",
    ]


def _write_audit_summary(
    report: Mapping[str, Any], failure_counts: Counter[str], output_dir: Path
) -> None:
    report_id = _string(report.get("report_id")) or "unknown"
    summary = _mapping(report.get("summary"))
    queue = [
        item
        for item in [
            "source_contradicts_claim",
            "numeric_inconsistency",
            "wrong_source_risk",
            "fulltext_required_numeric_claim",
            "no_source_anchor",
            "claim_needs_qualification",
        ]
        if failure_counts.get(item, 0) > 0
    ]
    content = [
        f"# Audit package: {report_id}",
        "",
        "## Run summary",
        *_summary_lines(summary),
        "",
        "## Failure taxonomy",
    ]
    content.extend(f"- {key}: {failure_counts[key]}" for key in sorted(failure_counts))
    content.extend(
        [
            "",
            "## Human-review queue",
            *(f"- {item}: {failure_counts[item]}" for item in queue),
            "",
            "Use `claims.csv` as the working file for reviewer adjudication.",
        ]
    )
    (output_dir / "audit_summary.md").write_text("\n".join(content) + "\n", encoding="utf-8")


def _write_limitations(summary: Mapping[str, Any], output_dir: Path) -> None:
    not_addressed = _mapping(summary.get("not_addressed_breakdown"))
    unverifiable = _mapping(summary.get("unverifiable_by_reason"))
    content = [
        "# Limitations and review protocol",
        "",
        "This package is generated from an existing `report.json`; it does not rerun the pipeline.",
        "",
        "## Known limitations",
        "- LLM-reported confidence is not a decision signal; use `evidence_quality`, "
        "`retrieval_status`, and source passages.",
        "- Open-access coverage limits verdict strength when publisher full text is unavailable.",
        "- Low-confidence or disputed resolution should be reviewed before treating a "
        "verdict as source-grounded.",
        "- Numeric flags require human review when multiple metrics appear in one sentence.",
        "",
        "## Access and resolution diagnostics",
        f"- Not addressed / no source: {int(not_addressed.get('no_source') or 0)}",
        f"- Not addressed / paywall: {int(not_addressed.get('paywall') or 0)}",
        f"- Unverifiable by reason: {dict(unverifiable)}",
        "",
        "## human-review queue",
        "Prioritize `source_contradicts_claim`, `numeric_inconsistency`, `wrong_source_risk`, "
        "`fulltext_required_numeric_claim`, and `no_source_anchor` rows in `claims.csv`.",
    ]
    (output_dir / "limitations.md").write_text("\n".join(content) + "\n", encoding="utf-8")


def _write_manifest(report_path: Path, report: Mapping[str, Any], output_dir: Path) -> None:
    manifest = {
        "report_id": _string(report.get("report_id")),
        "source_report": str(report_path),
        "generated_at": time.time(),
        "outputs": _OUTPUTS,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def export_audit_package(report_path: Path, output_dir: Path) -> Path:
    """Write a buyer/reviewer-facing audit package and return the output directory."""
    report = cast(Mapping[str, Any], json.loads(report_path.read_text(encoding="utf-8")))
    claims = [cast(Mapping[str, Any], claim) for claim in report.get("claims", [])]
    summary = _mapping(report.get("summary"))

    output_dir.mkdir(parents=True, exist_ok=True)
    failure_counts = _write_claims_csv(claims, output_dir)
    _write_audit_summary(report, failure_counts, output_dir)
    _write_limitations(summary, output_dir)
    _write_manifest(report_path, report, output_dir)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Path to report.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write the audit package. Defaults to <report-dir>/audit_package.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.report.parent / "audit_package"
    export_audit_package(args.report, output_dir)
    logger.info("audit_package_exported", output_dir=str(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
