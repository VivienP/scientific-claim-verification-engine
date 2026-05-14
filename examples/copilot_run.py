"""End-to-end Copilot example: V1 pipeline → enricher → HTML report.

Demonstrates:

  text → run_pipeline()             → list[ClaimVerification]
       → CopilotEnricher.enrich_all → list[EnrichedVerification]
       → build_copilot_report()     → copilot_report.html

Usage:
    # End-to-end live run (requires ANTHROPIC_API_KEY):
    python -m examples.copilot_run --input path/to/text.txt --run-dir reports/runs/copilot-demo

    # Replay an existing pipeline output (offline):
    python -m examples.copilot_run --pipeline-output reports/runs/X/report.json \
        --run-dir reports/runs/copilot-replay

This module also exports ``serialize_enriched`` and ``deserialize_enriched``
helpers used by ``scripts/run_copilot_eval.py`` to score persisted runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import structlog

from src.copilot.enricher import CopilotConfig, CopilotEnricher
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

logger: structlog.BoundLogger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Public helpers — Serialisation
# ---------------------------------------------------------------------------


def serialize_enriched(enriched: list[EnrichedVerification]) -> list[dict[str, Any]]:
    """Convert a list of ``EnrichedVerification`` into a JSON-friendly list of dicts.

    Round-trips with :func:`deserialize_enriched`. Tuples become lists, but the
    deserialiser restores them.
    """
    return [_ev_to_dict(ev) for ev in enriched]


def deserialize_enriched(raw: list[dict[str, Any]]) -> list[EnrichedVerification]:
    """Inverse of :func:`serialize_enriched`."""
    return [_ev_from_dict(d) for d in raw]


def write_enriched(path: Path, enriched: list[EnrichedVerification]) -> None:
    """Write the enriched run to ``path`` as JSON. Creates parents if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = serialize_enriched(enriched)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Run V1 + Copilot end-to-end.")
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to a UTF-8 .txt file containing the source text to verify.",
    )
    parser.add_argument(
        "--pipeline-output",
        type=Path,
        help="Path to an existing report.json (replay enrichment without re-running V1).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory to write enriched.json and copilot_report.html into.",
    )
    parser.add_argument(
        "--mode",
        choices=["pharma", "academic", "general"],
        default="pharma",
        help="Copilot mode — controls which evidence-quality fields are populated.",
    )
    parser.add_argument(
        "--no-primary-lookup",
        action="store_true",
        help="Disable Semantic Scholar primary-source lookup.",
    )
    parser.add_argument(
        "--no-fix",
        action="store_true",
        help="Disable LLM recommended_fix generation.",
    )
    args = parser.parse_args()

    if not args.input and not args.pipeline_output:
        print("error: must provide either --input or --pipeline-output", file=sys.stderr)
        return 2

    config = CopilotConfig(
        mode=CopilotMode(args.mode),
        enable_primary_lookup=not args.no_primary_lookup,
        enable_recommended_fix=not args.no_fix,
        db_path=args.run_dir / "_cache.db",
    )

    args.run_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()

    if args.input is not None:
        cvs, _pipeline_steps = _run_v1_pipeline(args.input)
    else:
        cvs = _load_pipeline_output(args.pipeline_output)

    print(f"Loaded {len(cvs)} ClaimVerification objects.", file=sys.stderr)

    enricher = CopilotEnricher(config)
    enriched = enricher.enrich_all(cvs)
    print(f"Enriched {len(enriched)} claims.", file=sys.stderr)

    runtime = time.time() - start

    # Persist enriched JSON for downstream scoring.
    enriched_path = args.run_dir / "enriched.json"
    write_enriched(enriched_path, enriched)
    print(f"Wrote {enriched_path}", file=sys.stderr)

    # Render HTML report.
    html_path = build_copilot_report(
        args.run_dir,
        enriched,
        run_id=args.run_dir.name,
        runtime_seconds=runtime,
        total_cost_usd=_estimate_cost(enriched),
    )
    print(f"Wrote {html_path}", file=sys.stderr)

    return 0


# ---------------------------------------------------------------------------
# V1 pipeline integration (lazy — only imported on real runs to keep the
# offline test surface minimal).
# ---------------------------------------------------------------------------


def _run_v1_pipeline(input_path: Path) -> tuple[list[ClaimVerification], list[ProvenanceStep]]:
    from src.pipeline import PipelineConfig, run_pipeline

    text = input_path.read_text(encoding="utf-8")
    config = PipelineConfig()
    return run_pipeline(text, config=config)


def _load_pipeline_output(path: Path) -> list[ClaimVerification]:
    """Reconstruct ClaimVerification list from an existing report.json.

    Best-effort — the V1 report.json schema is the authoritative source. We
    rebuild the dataclasses needed for the enricher, ignoring fields the
    enricher does not consume.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "claims" in raw:
        items = raw["claims"]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(f"Cannot interpret pipeline output at {path}")

    cvs: list[ClaimVerification] = []
    for item in items:
        cvs.append(_cv_from_dict(item))
    return cvs


def _estimate_cost(enriched: list[EnrichedVerification]) -> float:
    """Sum copilot_steps token usage and convert to USD using Sonnet-4 pricing.

    Sonnet-4 (2026-05): input $3.00 / MTok, output $15.00 / MTok.
    Cache reads at ~10% of input. Conservative estimate that ignores cache
    discounts (worst case) so the displayed cost is an upper bound.
    """
    total_in = 0
    total_out = 0
    for ev in enriched:
        for step in ev.copilot_steps:
            if step.tokens_in:
                total_in += step.tokens_in
            if step.tokens_out:
                total_out += step.tokens_out
    return round((total_in * 3.0 + total_out * 15.0) / 1_000_000, 4)


# ---------------------------------------------------------------------------
# Internal: dataclass <-> dict converters
# ---------------------------------------------------------------------------


def _ev_to_dict(ev: EnrichedVerification) -> dict[str, Any]:
    return {
        "base": _cv_to_dict(ev.base),
        "copilot": _copilot_fields_to_dict(ev.copilot),
        "copilot_steps": [asdict(s) for s in ev.copilot_steps],
        "mode": ev.mode.value,
    }


def _ev_from_dict(d: dict[str, Any]) -> EnrichedVerification:
    return EnrichedVerification(
        base=_cv_from_dict(d["base"]),
        copilot=_copilot_fields_from_dict(d["copilot"]),
        copilot_steps=tuple(ProvenanceStep(**s) for s in d.get("copilot_steps", [])),
        mode=CopilotMode(d["mode"]),
    )


def _cv_to_dict(cv: ClaimVerification) -> dict[str, Any]:
    return {
        "claim": asdict(cv.claim),
        "source": asdict(cv.source),
        "source_set": {
            "sources": [asdict(s) for s in cv.source_set.sources],
            "citation_markers": list(cv.source_set.citation_markers),
        },
        "result": asdict(cv.result),
        "fetch_method": cv.fetch_method,
    }


def _cv_from_dict(d: dict[str, Any]) -> ClaimVerification:
    claim = Claim(**d["claim"])
    source = ResolvedSource(**d["source"])
    ss_raw = d.get("source_set", {"sources": [asdict(source)], "citation_markers": []})
    source_set = ResolvedSourceSet(
        sources=tuple(ResolvedSource(**s) for s in ss_raw.get("sources", [])),
        citation_markers=tuple(ss_raw.get("citation_markers", [])),
    )
    result_raw = dict(d["result"])
    # Drop unknown fields rather than crash on schema drift.
    valid_fields = set(VerificationResult.__dataclass_fields__.keys())
    result = VerificationResult(**{k: v for k, v in result_raw.items() if k in valid_fields})
    return ClaimVerification(
        claim=claim,
        source=source,
        source_set=source_set,
        result=result,
        fetch_method=d.get("fetch_method", "abstract"),
    )


def _copilot_fields_to_dict(c: CopilotFields) -> dict[str, Any]:
    return {
        "verdict_rationale": c.verdict_rationale,
        "recommended_fix": asdict(c.recommended_fix) if c.recommended_fix else None,
        "is_primary_source": c.is_primary_source,
        "study_design": c.study_design,
        "risk_of_bias": c.risk_of_bias,
        "conflicting_evidence_flag": c.conflicting_evidence_flag,
        "primary_source_doi": c.primary_source_doi,
        "novelty_claim": c.novelty_claim,
    }


def _copilot_fields_from_dict(d: dict[str, Any]) -> CopilotFields:
    fix = None
    if d.get("recommended_fix"):
        fix = RecommendedFix(**d["recommended_fix"])
    return CopilotFields(
        verdict_rationale=d["verdict_rationale"],
        recommended_fix=fix,
        is_primary_source=d.get("is_primary_source"),
        study_design=d.get("study_design"),
        risk_of_bias=d.get("risk_of_bias"),
        conflicting_evidence_flag=d.get("conflicting_evidence_flag"),
        primary_source_doi=d.get("primary_source_doi"),
        novelty_claim=d.get("novelty_claim"),
    )


if __name__ == "__main__":
    sys.exit(main())
