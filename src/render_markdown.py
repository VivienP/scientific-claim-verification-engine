"""Render a report.json dict as a human-readable markdown document.

Pure read-side module: zero LLM calls, zero network I/O, deterministic.
Consumes the same dict shape that ``src.report.build_report`` writes to
``reports/runs/{report_id}/report.json``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

logger: structlog.BoundLogger = structlog.get_logger(__name__)


_VERDICT_LABEL: dict[str, str] = {
    "supported": "SUPPORTED",
    "unsupported": "UNSUPPORTED",
    "partially_supported": "PARTIALLY SUPPORTED",
    "not_addressed": "NOT ADDRESSED",
    "unverifiable": "UNVERIFIABLE",
}

# Display order for the summary verdict table. Conservative verdicts last
# so the reader sees positive findings first when scanning.
_VERDICT_ORDER: tuple[str, ...] = (
    "supported",
    "partially_supported",
    "unsupported",
    "not_addressed",
    "unverifiable",
)


def render_markdown(report: dict[str, Any]) -> str:
    """Render report.json dict as a markdown document.

    Pure function — no I/O, no mutation of the input. Missing optional
    fields render as best-effort placeholders so older reports remain
    legible.
    """
    sections: list[str] = []
    sections.append(_render_header(report))
    sections.append(_render_summary(report.get("summary", {})))
    claims = report.get("claims", [])
    if claims:
        sections.append(_render_claims(claims))
    return "\n\n".join(sections).rstrip() + "\n"


def render_markdown_from_file(report_path: Path) -> Path:
    """Read report.json from disk and write report.md alongside it.

    Returns the path to the written .md file.
    """
    if not report_path.exists():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    md = render_markdown(report)
    md_path = report_path.with_suffix(".md")
    md_path.write_text(md, encoding="utf-8")
    return md_path


def _render_header(report: dict[str, Any]) -> str:
    report_id = report.get("report_id", "<unknown>")
    ts_raw = report.get("timestamp", 0.0)
    summary = report.get("summary", {})
    cost = summary.get("total_cost_usd", 0.0)
    try:
        dt = datetime.fromtimestamp(float(ts_raw), tz=UTC)
        ts_iso = dt.isoformat(timespec="seconds")
    except (TypeError, ValueError, OSError):
        ts_iso = "<unknown>"
    lines = [
        "# Verification report",
        "",
        f"**Run:** `{report_id}`  ",
        f"**Generated:** {ts_iso}  ",
        f"**Total cost:** ${float(cost):.4f}",
    ]
    return "\n".join(lines)


def _render_summary(summary: dict[str, Any]) -> str:
    total = int(summary.get("total_claims", 0))
    table_rows = ["| Verdict | Count |", "|---|---|"]
    for key in _VERDICT_ORDER:
        # Title-case from the upper-case label; "Not addressed" / "Partially supported".
        label = _VERDICT_LABEL[key].capitalize()
        count = int(summary.get(key, 0))
        table_rows.append(f"| {label} | {count} |")
    table_rows.append(f"| **Total** | **{total}** |")

    lines: list[str] = ["## Summary", "", *table_rows]

    citation_rate = summary.get("citation_found_rate")
    if citation_rate is not None and total > 0:
        pct = round(float(citation_rate) * 100)
        resolved = round(float(citation_rate) * total)
        lines.append("")
        lines.append(f"**Citation resolution:** {resolved}/{total} found ({pct}%)")

    fulltext_verified = summary.get("fulltext_verified")
    no_passage = summary.get("no_passage_found")
    unavailable = summary.get("fulltext_unavailable")
    if any(v is not None for v in (fulltext_verified, no_passage, unavailable)):
        lines.append(
            f"**Fulltext access:** verified={int(fulltext_verified or 0)}, "
            f"no_passage={int(no_passage or 0)}, unavailable={int(unavailable or 0)}"
        )

    numeric_run = int(summary.get("numeric_checks_run", 0) or 0)
    numeric_bad = int(summary.get("numeric_inconsistencies_flagged", 0) or 0)
    if numeric_run > 0:
        lines.append(f"**Numeric checks:** {numeric_run} run, {numeric_bad} inconsistencies")

    warnings = _render_summary_warnings(summary)
    if warnings:
        lines.append("")
        lines.extend(warnings)

    return "\n".join(lines)


def _render_summary_warnings(summary: dict[str, Any]) -> list[str]:
    """Surface counts that matter for reviewer trust.

    Retractions, low-confidence resolution, and cross-modal disagreements
    each get a single blockquote line when their count is non-zero.
    """
    warnings: list[str] = []
    retracted = int(summary.get("retracted_sources", 0) or 0)
    if retracted > 0:
        warnings.append(f"> **Warning:** {retracted} cited source(s) retracted.")
    low_conf = int(summary.get("resolution_low_confidence", 0) or 0)
    if low_conf > 0:
        warnings.append(
            f"> **Warning:** {low_conf} citation(s) resolved with low confidence — verify manually."
        )
    cross_modal = int(summary.get("cross_modal_disagreements", 0) or 0)
    if cross_modal > 0:
        warnings.append(
            f"> **Note:** {cross_modal} cross-modal disagreement(s) — "
            "primary verdict confidence downgraded."
        )
    return warnings


def _render_claims(claims: list[dict[str, Any]]) -> str:
    parts: list[str] = ["## Claims"]
    for i, claim in enumerate(claims, start=1):
        parts.append("")
        parts.append(_render_one_claim(i, claim))
    return "\n".join(parts)


def _render_one_claim(index: int, claim: dict[str, Any]) -> str:
    verification = claim.get("verification", {})
    source = claim.get("source", {})
    status = str(verification.get("status", "not_addressed"))
    label = _VERDICT_LABEL.get(status, status.upper())
    confidence = verification.get("confidence")
    conf_str = f" — confidence {float(confidence):.2f}" if confidence is not None else ""

    citation = _format_citation(claim.get("cited_authors") or [], claim.get("cited_year"))
    heading = f"### {index}. [{label}] {citation}{conf_str}"

    lines: list[str] = [heading, ""]

    claim_text = str(claim.get("claim_text", ""))
    lines.append(f"> {_inline_safe(claim_text)}")
    lines.append("")

    lines.append(_format_source_line(source))
    lines.append(_format_evidence_line(verification))

    numeric_line = _format_numeric_line(verification.get("numeric_check"))
    if numeric_line:
        lines.append(numeric_line)

    if status == "unverifiable":
        reason = verification.get("unverifiable_reason") or "unspecified"
        lines.append(f"- **Unverifiable reason:** `{reason}`")

    passages = verification.get("source_passages") or []
    if passages:
        lines.append("")
        lines.append("**Passages:**")
        lines.append("")
        for passage in passages:
            lines.append(f"> {_inline_safe(str(passage))}")

    explanation = str(verification.get("explanation") or "").strip()
    if explanation:
        lines.append("")
        lines.append(f"**Why:** {_inline_safe(explanation)}")

    return "\n".join(lines)


def _format_citation(authors: list[Any], year: int | None) -> str:
    author_str = ", ".join(str(a) for a in authors) if authors else "—"
    return f"{author_str} ({year})" if year is not None else author_str


def _format_source_line(source: dict[str, Any]) -> str:
    if not source.get("found"):
        return "- **Source:** not resolved"
    doi = source.get("doi")
    title = source.get("title")
    link = f"[{doi}](https://doi.org/{doi})" if doi else "—"
    title_part = f" — *{_inline_safe(str(title))}*" if title else ""
    low_conf = source.get("resolution_low_confidence")
    flag = " **(LOW CONFIDENCE)**" if low_conf else ""
    return f"- **Source:** {link}{title_part}{flag}"


def _format_evidence_line(verification: dict[str, Any]) -> str:
    retrieval = verification.get("retrieval_status", "unknown")
    quality = verification.get("evidence_quality", "unknown")
    depth = verification.get("verification_depth", "unknown")
    return f"- **Evidence:** depth={depth}, retrieval={retrieval}, quality={quality}"


def _format_numeric_line(numeric: dict[str, Any] | None) -> str | None:
    if not numeric:
        return None
    consistent = bool(numeric.get("consistent"))
    label = "consistent" if consistent else "INCONSISTENT"
    check_type = numeric.get("check_type", "unknown")
    detail = numeric.get("explanation") or ""
    suffix = f" — {_inline_safe(str(detail))}" if detail else ""
    return f"- **Numeric check:** {label} ({check_type}){suffix}"


def _inline_safe(text: str) -> str:
    """Collapse whitespace so passages and claim text fit on one blockquote line.

    No HTML-escape needed: markdown renders raw text in blockquotes safely.
    Newlines are the only character that breaks the blockquote structure;
    collapsing them avoids visual fragmentation across multi-line strings.
    """
    return " ".join(text.split())


def main() -> None:
    """CLI entry: render the latest run, or a given report.json, to markdown."""
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if len(sys.argv) > 1:
        report_path = Path(sys.argv[1])
    else:
        runs_dir = Path("reports/runs")
        candidates = sorted(
            (r / "report.json" for r in runs_dir.iterdir() if (r / "report.json").exists()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print("No report.json found under reports/runs/", file=sys.stderr)
            sys.exit(1)
        report_path = candidates[0]

    md_path = render_markdown_from_file(report_path)
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
