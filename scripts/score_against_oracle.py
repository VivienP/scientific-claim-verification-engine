#!/usr/bin/env python
"""Score a pipeline run's resolution correctness against an oracle file.

The standard `report.json.summary.citation_found_rate` reports whether the resolver
returned ANY DOI per claim — not whether that DOI is the *correct* one.
A benchmark can show high citation-found rates while hiding a much lower correct-source
rate, with claims resolving to the citing paper itself or to unrelated work.

This script reads `report.json` and `oracle.json` from a real-paper validation
directory and emits a `correct_source_rate` plus per-claim diagnostics.

Usage:

    python scripts/score_against_oracle.py benchmarks/real_papers/<run>/

Outputs Markdown by default; pass `--json` to emit machine-readable.
Exits non-zero when `correct_source_rate < --fail-below`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClaimAudit:
    """One claim's resolution-correctness audit, derivable from report + oracle."""

    claim_id: str
    cited_first_author: str | None
    cited_year: int | None
    expected_doi: str | None
    resolved_doi: str | None
    is_correct: bool
    is_self_cite: bool  # claim has no external citation (e.g. self-result)
    notes: str = ""


@dataclass(frozen=True)
class OracleScorecard:
    n_claims_total: int
    n_self_cite: int
    n_external: int
    n_external_resolved_any: int
    n_external_resolved_correct: int
    citation_found_rate: float
    correct_source_rate: float
    claim_audits: tuple[ClaimAudit, ...]


def _doi_eq(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    return a.strip().lower() == b.strip().lower()


def _find_oracle_for_claim(
    cited_first_author: str | None,
    cited_year: int | None,
    oracle_claims: list[dict[str, object]],
) -> dict[str, object] | None:
    """Match a report claim to an oracle entry by (first_author, year).

    Tolerates claims that flatten multi-citations into one author list:
    the first cited author is the most reliable signature.
    """
    if not cited_first_author:
        return None
    for entry in oracle_claims:
        oracle_author = entry.get("cited_authors_first")
        oracle_year = entry.get("cited_year")
        if (
            isinstance(oracle_author, str)
            and oracle_author.lower() == cited_first_author.lower()
            and oracle_year == cited_year
        ):
            return entry
    return None


def score_run(report: dict[str, object], oracle: dict[str, object]) -> OracleScorecard:
    """Compute per-claim audit + summary metrics from a report + oracle pair."""
    claims = report.get("claims", [])
    oracle_claims = oracle.get("claims", [])
    if not isinstance(claims, list) or not isinstance(oracle_claims, list):
        msg = "report.claims and oracle.claims must both be lists"
        raise TypeError(msg)

    audits: list[ClaimAudit] = []
    for c in claims:
        if not isinstance(c, dict):
            continue
        cited_authors = c.get("cited_authors") or []
        cited_first = (
            cited_authors[0] if isinstance(cited_authors, list) and cited_authors else None
        )
        cited_year = c.get("cited_year")
        if not isinstance(cited_year, int):
            cited_year = None
        source = c.get("source") or {}
        resolved_doi = source.get("doi") if isinstance(source, dict) else None

        oracle_entry = _find_oracle_for_claim(cited_first, cited_year, oracle_claims)
        if oracle_entry is None:
            # No oracle entry → either a self-cite claim or an unmapped
            # claim that the oracle author missed. Treat as self-cite when
            # cited_first is None (no external citation).
            is_self_cite = cited_first is None
            audits.append(
                ClaimAudit(
                    claim_id=str(c.get("claim_id", "")),
                    cited_first_author=cited_first,
                    cited_year=cited_year,
                    expected_doi=None,
                    resolved_doi=resolved_doi if isinstance(resolved_doi, str) else None,
                    is_correct=False,
                    is_self_cite=is_self_cite,
                    notes="no oracle entry" if not is_self_cite else "self-result claim",
                )
            )
            continue

        expected_doi = oracle_entry.get("expected_doi")
        expected_doi_str = expected_doi if isinstance(expected_doi, str) else None
        resolved_doi_str = resolved_doi if isinstance(resolved_doi, str) else None
        is_correct = _doi_eq(resolved_doi_str, expected_doi_str)
        audits.append(
            ClaimAudit(
                claim_id=str(c.get("claim_id", "")),
                cited_first_author=cited_first,
                cited_year=cited_year,
                expected_doi=expected_doi_str,
                resolved_doi=resolved_doi_str,
                is_correct=is_correct,
                is_self_cite=False,
                notes=str(oracle_entry.get("_note", "")),
            )
        )

    n_total = len(audits)
    n_self_cite = sum(1 for a in audits if a.is_self_cite)
    n_external = n_total - n_self_cite
    n_external_resolved_any = sum(
        1 for a in audits if not a.is_self_cite and a.resolved_doi is not None
    )
    n_external_resolved_correct = sum(1 for a in audits if a.is_correct)
    citation_found_rate = n_external_resolved_any / n_external if n_external else 0.0
    correct_source_rate = n_external_resolved_correct / n_external if n_external else 0.0

    return OracleScorecard(
        n_claims_total=n_total,
        n_self_cite=n_self_cite,
        n_external=n_external,
        n_external_resolved_any=n_external_resolved_any,
        n_external_resolved_correct=n_external_resolved_correct,
        citation_found_rate=citation_found_rate,
        correct_source_rate=correct_source_rate,
        claim_audits=tuple(audits),
    )


def render_markdown(scorecard: OracleScorecard) -> str:
    """Format an OracleScorecard as a Markdown table for terminal or file output."""
    lines = [
        "## Resolution-correctness scorecard",
        "",
        f"- Total claims: **{scorecard.n_claims_total}**",
        f"- Self-result claims (no external citation): **{scorecard.n_self_cite}**",
        f"- Externally-cited claims: **{scorecard.n_external}**",
        (
            f"- External claims resolved to *any* DOI: "
            f"**{scorecard.n_external_resolved_any}/{scorecard.n_external}** "
            f"({scorecard.citation_found_rate:.1%})"
        ),
        (
            f"- External claims resolved to *correct* DOI: "
            f"**{scorecard.n_external_resolved_correct}/{scorecard.n_external}** "
            f"({scorecard.correct_source_rate:.1%})"
        ),
        "",
        "### Per-claim audit",
        "",
        "| First author | Year | Expected DOI | Resolved DOI | Correct |",
        "|---|---|---|---|---|",
    ]
    for a in scorecard.claim_audits:
        if a.is_self_cite:
            continue
        ok = "yes" if a.is_correct else "no"
        lines.append(
            f"| {a.cited_first_author or ''} "
            f"| {a.cited_year or ''} "
            f"| `{a.expected_doi or '(none)'}` "
            f"| `{a.resolved_doi or '(none)'}` "
            f"| {ok} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path containing report.json and oracle.json",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of markdown")
    parser.add_argument(
        "--fail-below",
        type=float,
        default=None,
        metavar="THRESHOLD",
        help="Exit with code 1 if correct_source_rate falls below this fraction",
    )
    args = parser.parse_args()

    report_path = args.run_dir / "report.json"
    oracle_path = args.run_dir / "oracle.json"
    if not report_path.exists():
        print(f"missing {report_path}", file=sys.stderr)
        return 2
    if not oracle_path.exists():
        print(f"missing {oracle_path}", file=sys.stderr)
        return 2

    report = json.loads(report_path.read_text(encoding="utf-8"))
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    scorecard = score_run(report, oracle)

    if args.json:
        payload = {
            **{k: v for k, v in asdict(scorecard).items() if k != "claim_audits"},
            "claim_audits": [asdict(a) for a in scorecard.claim_audits],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(render_markdown(scorecard))

    if args.fail_below is not None and scorecard.correct_source_rate < args.fail_below:
        print(
            f"\nFAIL: correct_source_rate={scorecard.correct_source_rate:.2%} "
            f"< threshold={args.fail_below:.2%}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
