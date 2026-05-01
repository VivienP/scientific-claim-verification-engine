#!/usr/bin/env python
"""Pretty-print a verification report.json for terminal screenshots.

Usage:
    python scripts/show_report.py                              # latest run
    python scripts/show_report.py reports/runs/<id>/report.json
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# ANSI colour codes
# ---------------------------------------------------------------------------
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"

_VERDICT_COLOUR: dict[str, str] = {
    "supported": _GREEN,
    "unsupported": _RED,
    "not_addressed": _YELLOW,
    "partially_supported": _BLUE,
}
_VERDICT_LABEL: dict[str, str] = {
    "supported": "SUPPORTED",
    "unsupported": "UNSUPPORTED",
    "not_addressed": "NOT ADDRESSED",
    "partially_supported": "PARTIALLY SUPPORTED",
}

_WIDTH = 76

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_latest_report() -> Path:
    runs_dir = Path("reports/runs")
    if not runs_dir.exists():
        raise FileNotFoundError(f"Directory not found: {runs_dir}")
    candidates = sorted(
        (r / "report.json" for r in runs_dir.iterdir() if (r / "report.json").exists()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No report.json found in reports/runs/")
    return candidates[0]


def _wrap(text: str, indent: int) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=_WIDTH, initial_indent=prefix, subsequent_indent=prefix)


def _badge(status: str) -> str:
    colour = _VERDICT_COLOUR.get(status, "")
    label = _VERDICT_LABEL.get(status, status.upper())
    return f"{_BOLD}{colour}[{label}]{_RESET}"


def _truncate(text: str, max_len: int) -> str:
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def _fmt_float(value: float | int | str | None) -> str:
    return "n/a" if value is None else f"{float(value):.2f}"


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def show(report_path: Path) -> None:
    report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    summary: dict[str, Any] = report["summary"]
    claims: list[dict[str, Any]] = report["claims"]

    border = "═" * _WIDTH
    thin = "─" * _WIDTH

    print(f"\n{_BOLD}{border}{_RESET}")
    print(
        f" {_BOLD}VERIFICATION REPORT{_RESET}"
        f"  •  {summary['total_claims']} claims"
        f"  •  ${summary['total_cost_usd']:.4f}"
    )
    print(f" {_DIM}Report : {report['report_id']}{_RESET}")
    print(f"{_BOLD}{border}{_RESET}\n")

    for i, claim in enumerate(claims, 1):
        status: str = claim["verification"]["status"]
        authors: list[str] = claim["cited_authors"]
        year: int | None = claim["cited_year"]
        citation = f"{', '.join(authors)} ({year})" if authors else "—"

        print(f" {_BOLD}{i}{_RESET}  {_badge(status)}  {_DIM}{citation}{_RESET}")

        claim_text = _truncate(claim["claim_text"], 220)
        print(_wrap(f'Claim : "{claim_text}"', indent=5))

        source: dict[str, Any] = claim["source"]
        if source["found"] and source["title"]:
            title = _truncate(str(source["title"]), 72)
            print(_wrap(f"Source: {title}", indent=5))
            match = _fmt_float(source.get("title_match_score"))
            weak = "LOW CONFIDENCE" if source.get("resolution_low_confidence") else "ok"
            print(
                _wrap(
                    f"Match : year={_fmt_float(source.get('similarity_score'))} "
                    f"title={match} ({weak})",
                    indent=5,
                )
            )
        else:
            print(f"     {_YELLOW}Source: ✗ Not found in OpenAlex/CrossRef{_RESET}")

        verification: dict[str, Any] = claim["verification"]
        retrieval = str(verification.get("retrieval_status", "unknown"))
        evidence = str(verification.get("evidence_quality", "unknown"))
        conf: float = verification["confidence"]
        evidence_line = f"Evidence: retrieval={retrieval} quality={evidence} confidence={conf:.2f}"
        print(_wrap(evidence_line, indent=5))

        passages = verification.get("source_passages") or []
        for passage in passages[:2]:
            print(_wrap(f"Quote : {passage!s}", indent=5))

        numeric = verification.get("numeric_check")
        if numeric:
            numeric_status = "consistent" if numeric.get("consistent") else "INCONSISTENT"
            print(_wrap(f"Numeric: {numeric_status} ({numeric.get('check_type')})", indent=5))

        explanation = _truncate(str(claim["verification"]["explanation"]), 300)
        print(_wrap(f"Why   : {explanation}", indent=5))
        print()

    print(f"{_DIM}{thin}{_RESET}")
    status_summary = (
        f"  supported={summary['supported']}"
        f"  unsupported={summary['unsupported']}"
        f"  not_addressed={summary['not_addressed']}"
        f"  partially={summary['partially_supported']}"
        f"  no_passage={summary.get('no_passage_found', 0)}"
        f"  weak_resolution={summary.get('resolution_low_confidence', 0)}"
    )
    print(status_summary)
    print(f"{_BOLD}{border}{_RESET}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if len(sys.argv) > 1:
        report_path = Path(sys.argv[1])
    else:
        try:
            report_path = _find_latest_report()
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

    if not report_path.exists():
        print(f"Error: {report_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    show(report_path)


if __name__ == "__main__":
    main()
