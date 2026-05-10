#!/usr/bin/env python
"""Diagnose why CTran fails on a run, per-claim.

CTran is *transparent* iff ``source_passages`` is non-empty OR
``evidence_quality`` is in the transparent allow-list. The allow-list,
mirroring :func:`src.aar._claim_is_transparent`, is:

    {abstract_only, quoted_passage, title_only, passages_searched_no_quote}

The ``passages_searched_no_quote`` value was introduced in Phase A.2 to
distinguish "fulltext was retrieved and BM25 selected passages, but the
LLM didn't quote any" from "no passages were ever shown to the LLM"
(``no_evidence``). The auditor still sees what was searched in the former
case, so it counts as transparent.

A claim is a *failure* when ``source_passages`` is empty AND
``evidence_quality`` is outside that allow-list (typically
``no_evidence`` or ``citing_paper_context``).

This diagnoser categorises each failing claim into a fix-mode bucket so we
can target Phase A.2 work at the dominant failure rather than guessing.

Usage:

    # Single run
    python scripts/diagnose_ctran.py benchmarks/real_papers/valsci_brice_2025

    # Multiple runs (rolled up)
    python scripts/diagnose_ctran.py reports/runs/lactate_isf reports/runs/api-abc12345

    # Markdown output to a file
    python scripts/diagnose_ctran.py --all-benchmarks \
        --output reports/phase_a2/ctran_failure_matrix.md

Categories (descending leverage for typical bio-med inputs):

    A1_doi_unresolved          — ``source.found`` is False; CrossRef returned
                                 no match. Fix: resolver / extractor
                                 (out of A.2 scope).
    A2a_retrieval_failed       — ``source.found`` True, ``retrieval_status`` is
                                 ``fulltext_unavailable``, ``evidence_quality``
                                 is ``no_evidence``. The pipeline could not
                                 retrieve any text for the resolved DOI.
                                 Fix: extend ``fetch_fulltext.py`` chain
                                 (preprints, OpenAlex OA URL).
    A2b_verifier_did_not_quote — ``source.found`` True, ``retrieval_status``
                                 is ``passage_found`` (BM25 selected passages
                                 from a real fulltext), but ``source_passages``
                                 ended up empty AND ``evidence_quality`` is
                                 ``no_evidence``. The verifier saw passages
                                 but didn't quote any. Fix: verifier should
                                 emit the BM25-selected passages even when
                                 the verdict is unsupported.
    A3_citing_context_only     — ``evidence_quality=citing_paper_context``.
                                 Source DOI exists but neither abstract nor
                                 fulltext came back; verifier fell back to
                                 the citing paper's own context. Fix: Europe
                                 PMC abstract chain; fulltext via preprints /
                                 OpenAlex OA URL.
    A4_other_failure           — Unexpected combination not matched above.
    PASS_*                     — Transparent claims, broken down by depth.

The diagnoser is read-only — it never mutates a run directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger: structlog.BoundLogger = structlog.get_logger(__name__)


# Mirrors src.aar._claim_is_transparent (kept in sync — adding a value here
# without updating aar.py would silently desynchronise this diagnoser from
# the actual CTran metric).
_TRANSPARENT_QUALITIES = frozenset(
    {
        "abstract_only",
        "quoted_passage",
        "title_only",
        "passages_searched_no_quote",  # Phase A.2 — fulltext + BM25, LLM didn't quote.
    }
)
_NON_TRANSPARENT_KNOWN = frozenset({"no_evidence", "citing_paper_context"})


@dataclass(frozen=True)
class ClaimDiagnosis:
    """Per-claim categorisation."""

    claim_id: str
    claim_text: str
    transparent: bool
    category: str
    evidence_quality: str | None
    retrieval_status: str | None
    source_doi: str | None
    source_found: bool
    source_passages_count: int

    @property
    def truncated_text(self) -> str:
        return (self.claim_text[:100] + "...") if len(self.claim_text) > 100 else self.claim_text


def _categorise(claim: dict[str, Any]) -> tuple[bool, str]:
    """Return (transparent, category_key) for one claim from report.json.

    The transparent boolean mirrors :func:`src.aar._claim_is_transparent`
    so this script's totals always match the AAR scorecard. Failure
    categories sub-split A2 by ``retrieval_status`` — this distinction
    drives which fix lands first in Phase A.2.
    """
    verification = claim.get("verification") or {}
    source = claim.get("source") or {}

    passages = verification.get("source_passages") or []
    quality = verification.get("evidence_quality")
    retrieval_status = verification.get("retrieval_status")
    source_found = bool(source.get("found"))

    transparent = bool(passages) or quality in _TRANSPARENT_QUALITIES
    if transparent:
        if passages:
            return True, "PASS_quoted_passage"
        return True, f"PASS_{quality}"

    if not source_found:
        return False, "A1_doi_unresolved"
    if quality == "no_evidence":
        # Sub-split: did retrieval actually fail, or did the verifier
        # have passages but choose not to quote them?
        if retrieval_status == "passage_found":
            return False, "A2b_verifier_did_not_quote"
        return False, "A2a_retrieval_failed"
    if quality == "citing_paper_context":
        return False, "A3_citing_context_only"
    return False, "A4_other_failure"


def diagnose_run(run_dir: Path) -> list[ClaimDiagnosis]:
    """Read ``run_dir/report.json`` and return one diagnosis per claim."""
    report_path = run_dir / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"report.json not found in {run_dir}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    out: list[ClaimDiagnosis] = []
    for claim in report.get("claims", []):
        verification = claim.get("verification") or {}
        source = claim.get("source") or {}
        transparent, category = _categorise(claim)
        out.append(
            ClaimDiagnosis(
                claim_id=str(claim.get("claim_id", "")),
                claim_text=str(claim.get("claim_text", "")),
                transparent=transparent,
                category=category,
                evidence_quality=verification.get("evidence_quality"),
                retrieval_status=verification.get("retrieval_status"),
                source_doi=source.get("doi"),
                source_found=bool(source.get("found")),
                source_passages_count=len(verification.get("source_passages") or []),
            )
        )
    return out


def render_markdown(
    diagnoses_by_run: dict[str, list[ClaimDiagnosis]],
) -> str:
    """Render a markdown report rolling up across runs.

    Includes:
      - per-run summary (CTran %, category counts)
      - rolled-up category counts across all runs
      - per-claim table for failing claims, grouped by category
      - dominant-failure-mode call-out
    """
    lines: list[str] = []
    lines.append("# CTran failure diagnostic — Phase A.2 baseline\n")
    lines.append(
        "Generated by `scripts/diagnose_ctran.py`. Each row in the per-claim tables "
        "below is a CTran failure. Categories are defined in the script docstring.\n"
    )

    # ---- Per-run summary ----
    lines.append("## Per-run summary\n")
    lines.append("| Run | Claims | Transparent | CTran |")
    lines.append("|---|---|---|---|")
    for run_name, diagnoses in diagnoses_by_run.items():
        n = len(diagnoses)
        n_pass = sum(1 for d in diagnoses if d.transparent)
        ctran = (n_pass / n) if n else 0.0
        lines.append(f"| `{run_name}` | {n} | {n_pass} | {ctran:.2%} |")
    lines.append("")

    # ---- Rolled-up category counts ----
    rollup: Counter[str] = Counter()
    for diagnoses in diagnoses_by_run.values():
        for d in diagnoses:
            rollup[d.category] += 1

    failing_total = sum(c for k, c in rollup.items() if not k.startswith("PASS"))
    passing_total = sum(c for k, c in rollup.items() if k.startswith("PASS"))
    grand_total = failing_total + passing_total

    lines.append("## Rolled-up category counts\n")
    lines.append(
        f"Total claims across runs: **{grand_total}** "
        f"(transparent: **{passing_total}**, failing: **{failing_total}**)\n"
    )
    lines.append("| Category | Count | % of total | % of failures |")
    lines.append("|---|---|---|---|")
    for cat, count in sorted(rollup.items(), key=lambda kv: (-kv[1], kv[0])):
        pct_total = count / grand_total if grand_total else 0
        pct_fail = (count / failing_total) if failing_total and not cat.startswith("PASS") else None
        pct_fail_str = f"{pct_fail:.1%}" if pct_fail is not None else "—"
        lines.append(f"| `{cat}` | {count} | {pct_total:.1%} | {pct_fail_str} |")
    lines.append("")

    # ---- Dominant failure mode call-out ----
    failing_only = {k: v for k, v in rollup.items() if not k.startswith("PASS")}
    if failing_only:
        dominant_cat, dominant_count = max(failing_only.items(), key=lambda kv: kv[1])
        dominant_pct = dominant_count / failing_total if failing_total else 0
        lines.append("## Dominant failure mode\n")
        lines.append(
            f"**`{dominant_cat}`** — {dominant_count} of {failing_total} failures "
            f"({dominant_pct:.1%}). This is the bucket Step 2 should target first."
        )
        lines.append("")
        lines.append(_recommendation_for(dominant_cat))
        lines.append("")

    # ---- Per-claim failure tables ----
    lines.append("## Per-claim failure detail\n")
    for run_name, diagnoses in diagnoses_by_run.items():
        failures = [d for d in diagnoses if not d.transparent]
        if not failures:
            continue
        lines.append(f"### `{run_name}` ({len(failures)} failures)\n")
        lines.append(
            "| claim_id | category | evidence_quality | retrieval_status | doi | claim_text |"
        )
        lines.append("|---|---|---|---|---|---|")
        # Group by category for readability.
        for d in sorted(failures, key=lambda x: (x.category, x.claim_id)):
            doi = d.source_doi or "—"
            lines.append(
                f"| `{d.claim_id[:12]}` | `{d.category}` | "
                f"`{d.evidence_quality or '—'}` | `{d.retrieval_status or '—'}` | "
                f"`{doi[:40]}` | {d.truncated_text} |"
            )
        lines.append("")

    return "\n".join(lines)


def _recommendation_for(dominant_category: str) -> str:
    """Plain-English recommendation for the dominant failure bucket."""
    table = {
        "A1_doi_unresolved": (
            "→ Out of Phase A.2 scope. Resolver / extractor work (Phase 1). "
            "If this dominates, Phase A.2 cannot lift CTran much without first "
            "fixing the bibliography parser + CrossRef matcher."
        ),
        "A2a_retrieval_failed": (
            "→ **Fix: extend `src/fetch_fulltext.py`.** Resolved DOI but no "
            "text retrieved at all. Add a preprints client "
            "(`src/clients/preprints.py`, bioRxiv/medRxiv) and wire "
            "`OpenAlex.best_oa_location.pdf_url` between unpaywall and the "
            "abstract fallback rung."
        ),
        "A2b_verifier_did_not_quote": (
            "→ **Fix: verifier behaviour, not retrieval.** BM25 selected "
            "passages from real fulltext (retrieval_status=passage_found), "
            "but the verifier emitted `evidence_quality=no_evidence` AND "
            "empty `source_passages` — the auditor sees nothing despite the "
            "pipeline having seen everything. Require the verifier to emit "
            "the passages it considered (even when they don't support the "
            "claim) so the auditor can see what was searched. Touch points: "
            "`src/verify_prompts.py` (output schema) and `src/verify.py` "
            "(passage projection into `VerificationResult.source_passages`). "
            "Cheaper than building new clients; same lift."
        ),
        "A3_citing_context_only": (
            "→ **Fix: extend the abstract / fulltext chain.** Source DOI is "
            "valid but neither abstract nor fulltext came back; the verifier "
            "fell back to the citing paper's own surrounding context. Build "
            "`src/clients/preprints.py` (bioRxiv/medRxiv DOI cross-ref) and "
            "wire `OpenAlex.best_oa_location.pdf_url` as a new rung in "
            "`fetch_fulltext.py`."
        ),
        "A4_other_failure": (
            "→ Inspect the failing rows below — an unrecognised "
            "evidence_quality / retrieval_status combination suggests a "
            "schema or pipeline bug rather than a retrieval ceiling."
        ),
    }
    return table.get(dominant_category, "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dirs",
        nargs="*",
        type=Path,
        help="One or more run directories containing report.json.",
    )
    parser.add_argument(
        "--all-benchmarks",
        action="store_true",
        help="Diagnose every run dir under benchmarks/real_papers/, "
        "benchmarks/real_outputs/, and benchmarks/canary/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the markdown report to this path (default: stdout).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of markdown.",
    )
    args = parser.parse_args()

    targets: list[Path] = list(args.run_dirs)
    if args.all_benchmarks:
        for root in [
            PROJECT_ROOT / "benchmarks" / "real_papers",
            PROJECT_ROOT / "benchmarks" / "real_outputs",
            PROJECT_ROOT / "benchmarks" / "canary",
        ]:
            if not root.exists():
                continue
            for child in sorted(root.iterdir()):
                if (child / "report.json").exists():
                    targets.append(child)

    if not targets:
        print("error: no run dirs supplied. Use --all-benchmarks or pass paths.", file=sys.stderr)
        return 2

    diagnoses_by_run: dict[str, list[ClaimDiagnosis]] = {}
    for run_dir in targets:
        try:
            diagnoses_by_run[run_dir.name] = diagnose_run(run_dir)
        except FileNotFoundError as exc:
            logger.warning("diagnose_run_skipped", run_dir=str(run_dir), error=str(exc))
            print(f"warning: {exc}", file=sys.stderr)

    if not diagnoses_by_run:
        print("error: no runs produced diagnoses.", file=sys.stderr)
        return 2

    if args.json:
        payload: dict[str, list[dict[str, Any]]] = {
            run: [
                {
                    "claim_id": d.claim_id,
                    "transparent": d.transparent,
                    "category": d.category,
                    "evidence_quality": d.evidence_quality,
                    "retrieval_status": d.retrieval_status,
                    "source_doi": d.source_doi,
                    "source_found": d.source_found,
                    "source_passages_count": d.source_passages_count,
                }
                for d in ds
            ]
            for run, ds in diagnoses_by_run.items()
        }
        out_text = json.dumps(payload, indent=2)
    else:
        out_text = render_markdown(diagnoses_by_run)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out_text, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(out_text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
