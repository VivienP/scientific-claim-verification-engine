"""Report generation — aggregates pipeline results and writes report.json + provenance.jsonl."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
import uuid
from pathlib import Path

import structlog

from src.models import (
    Claim,
    ProvenanceStep,
    ResolvedSource,
    VerifiabilityStatus,
    VerificationResult,
)

logger: structlog.BoundLogger = structlog.get_logger(__name__)

# Cost rates for claude-sonnet-4-6 (USD per token)
_COST_INPUT_PER_TOKEN = 3.0 / 1_000_000
_COST_INPUT_CACHED_PER_TOKEN = 0.30 / 1_000_000
_COST_OUTPUT_PER_TOKEN = 15.0 / 1_000_000


def _hash(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def _step_cost(step: ProvenanceStep) -> float:
    """Cost of a single step in USD. Zero when the step has no token data."""
    cost = 0.0
    if step.tokens_in is not None:
        # cache_hit=None (short-circuit steps) treated as uncached — conservative estimate
        rate = _COST_INPUT_CACHED_PER_TOKEN if step.cache_hit else _COST_INPUT_PER_TOKEN
        cost += step.tokens_in * rate
    if step.tokens_out is not None:
        cost += step.tokens_out * _COST_OUTPUT_PER_TOKEN
    return cost


def _compute_cost(steps: list[ProvenanceStep]) -> float:
    return sum(_step_cost(s) for s in steps)


def _compute_usage_by_stage(steps: list[ProvenanceStep]) -> dict[str, dict[str, int | float]]:
    """Bucket cumulative token + cost spend by ``operation`` field.

    Returns one entry per distinct operation seen in the step list, each
    carrying ``tokens_in``, ``tokens_out``, ``cost_usd``, ``n_steps``,
    and ``n_cache_hits``. Stages with no token data (deterministic
    operations like ``fetch_fulltext`` or ``aggregate``) appear with
    zero token totals and zero cost — explicit visibility is preferable
    to silent omission.

    The Valsci 2025 paper formalised this as a key audit-tool deliverable:
    operators want to know "which phase is dominating cost?" without
    having to re-derive it from raw provenance lines.
    """
    by_stage: dict[str, dict[str, int | float]] = {}
    for step in steps:
        bucket = by_stage.setdefault(
            step.operation,
            {
                "tokens_in": 0,
                "tokens_out": 0,
                "cost_usd": 0.0,
                "n_steps": 0,
                "n_cache_hits": 0,
            },
        )
        bucket["n_steps"] = int(bucket["n_steps"]) + 1
        if step.tokens_in is not None:
            bucket["tokens_in"] = int(bucket["tokens_in"]) + step.tokens_in
        if step.tokens_out is not None:
            bucket["tokens_out"] = int(bucket["tokens_out"]) + step.tokens_out
        if step.cache_hit:
            bucket["n_cache_hits"] = int(bucket["n_cache_hits"]) + 1
        bucket["cost_usd"] = float(bucket["cost_usd"]) + _step_cost(step)
    return by_stage


def _verifiability_status(citation_found_rate: float) -> VerifiabilityStatus:
    if citation_found_rate == 0.0:
        return "no_citations_found"
    if citation_found_rate <= 0.5:
        return "low_citation_density"
    return "verifiable"


def _compute_summary_stats(
    claims: list[Claim],
    results: dict[str, VerificationResult],
    sources: dict[str, ResolvedSource],
) -> dict[str, int | float | str]:
    """Pure helper — compute summary statistics. No I/O."""
    total = len(claims)

    def result_for(claim: Claim) -> VerificationResult:
        return results.get(
            claim.claim_id,
            VerificationResult(status="not_addressed", explanation="", confidence=0.0),
        )

    def source_for(claim: Claim) -> ResolvedSource:
        return sources.get(
            claim.claim_id,
            ResolvedSource(found=False, doi=None, title=None, abstract=None, similarity_score=None),
        )

    supported = sum(1 for c in claims if result_for(c).status == "supported")
    unsupported = sum(1 for c in claims if result_for(c).status == "unsupported")
    not_addressed = sum(1 for c in claims if result_for(c).status == "not_addressed")
    partially_supported = sum(1 for c in claims if result_for(c).status == "partially_supported")

    found_count = sum(1 for c in claims if source_for(c).abstract is not None)
    citation_found_rate = found_count / total if total > 0 else 0.0

    fulltext_verified = sum(1 for c in claims if result_for(c).retrieval_status == "passage_found")
    no_passage_found = sum(
        1 for c in claims if result_for(c).retrieval_status == "no_passage_found"
    )
    fulltext_unavailable = sum(
        1 for c in claims if result_for(c).retrieval_status == "fulltext_unavailable"
    )
    retracted_sources = sum(1 for c in claims if source_for(c).retraction_status)
    resolution_low_confidence = sum(1 for c in claims if source_for(c).resolution_low_confidence)
    numeric_checks_run = sum(1 for c in claims if result_for(c).numeric_check is not None)
    numeric_inconsistencies_flagged = sum(
        1 for c in claims if ((nc := result_for(c).numeric_check) is not None and not nc.consistent)
    )

    return {
        "total_claims": total,
        "supported": supported,
        "unsupported": unsupported,
        "not_addressed": not_addressed,
        "partially_supported": partially_supported,
        "citation_found_rate": citation_found_rate,
        "verifiability_status": _verifiability_status(citation_found_rate),
        "fulltext_verified": fulltext_verified,
        "no_passage_found": no_passage_found,
        "fulltext_unavailable": fulltext_unavailable,
        "resolution_low_confidence": resolution_low_confidence,
        "retracted_sources": retracted_sources,
        "numeric_checks_run": numeric_checks_run,
        "numeric_inconsistencies_flagged": numeric_inconsistencies_flagged,
    }


def build_report(
    report_id: str,
    input_text: str,
    claims: list[Claim],
    sources: dict[str, ResolvedSource],
    results: dict[str, VerificationResult],
    provenance_steps: list[ProvenanceStep],
    *,
    output_dir: Path | None = None,
) -> Path:
    """Aggregate pipeline results and write report.json + provenance.jsonl.

    report_id must be generated by the caller (pipeline entry point), not here.
    Creates reports/runs/{report_id}/ with mkdir(parents=True, exist_ok=True).
    output_dir defaults to project_root/reports/.
    Empty claims list produces valid report with total_claims=0.
    Returns path to the run directory (reports/runs/{report_id}/).
    Side effects: creates two files. No other side effects.
    """
    resolved_output_dir = output_dir if output_dir is not None else _default_output_dir()
    run_dir = resolved_output_dir / "runs" / report_id
    run_dir.mkdir(parents=True, exist_ok=True)

    stats = _compute_summary_stats(claims, results, sources)
    total_cost = _compute_cost(provenance_steps)
    stats["total_cost_usd"] = total_cost
    stats["usage_by_stage"] = _compute_usage_by_stage(provenance_steps)  # type: ignore[assignment]
    stats["cross_modal_disagreements"] = sum(
        1 for s in provenance_steps if s.operation == "verify_cross_modal" and s.confidence is None
    )

    claim_records = []
    for claim in claims:
        source = sources.get(
            claim.claim_id,
            ResolvedSource(found=False, doi=None, title=None, abstract=None, similarity_score=None),
        )
        result = results.get(
            claim.claim_id,
            VerificationResult(status="not_addressed", explanation="No result.", confidence=0.0),
        )
        claim_records.append(
            {
                "claim_id": claim.claim_id,
                "claim_text": claim.claim_text,
                "claim_type": claim.claim_type,
                "cited_authors": claim.cited_authors,
                "cited_year": claim.cited_year,
                "source": dataclasses.asdict(source),
                "verification": dataclasses.asdict(result),
            }
        )

    report = {
        "report_id": report_id,
        "timestamp": time.time(),
        "input_text": input_text,
        "summary": stats,
        "claims": claim_records,
    }

    with open(run_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Build aggregate provenance step.
    # claim_id=report_id by convention: aggregate step belongs to the run, not a single claim.
    # input_hash: full claim_records payload so any change in verdicts, sources, or explanations
    # is detectable. output_hash: stats + claim_records captures both summary and per-claim output.
    aggregate_step = ProvenanceStep(
        step_id=str(uuid.uuid4()),
        claim_id=report_id,
        operation="aggregate",
        input_hash=_hash(repr(claim_records)),
        output_hash=_hash(repr((stats, claim_records))),
        model_id=None,
        timestamp=time.time(),
        tokens_in=None,
        tokens_out=None,
        cache_hit=None,
        confidence=None,
    )

    all_steps = [*provenance_steps, aggregate_step]

    with open(run_dir / "provenance.jsonl", "w", encoding="utf-8") as f:
        for step in all_steps:
            f.write(json.dumps(dataclasses.asdict(step)) + "\n")

    logger.info(
        "report_written",
        report_id=report_id,
        total_claims=stats["total_claims"],
        total_cost_usd=total_cost,
        run_dir=str(run_dir),
    )

    return run_dir


def _default_output_dir() -> Path:
    """Return default output directory: <project_root>/reports/."""
    return Path(__file__).parent.parent / "reports"
