"""Multi-source aggregation layer for the verification pipeline.

Extracted from src/verify.py. Contains the aggregation policy
(_aggregate_multi_source_verdicts) and the multi-source orchestrator
(verify_claim_multi_source) that fan out to single-source verifiers
and reduce per-source verdicts to a single aggregated VerificationResult.

These two functions share no internal state with the single-source LLM
verifiers in src/verify.py and are imported here via deferred (function-level)
imports to avoid a circular dependency at module initialisation time.
"""

from __future__ import annotations

import hashlib
import time
import uuid

import structlog

from src.models import (
    Claim,
    PaperChunk,
    ProvenanceStep,
    ResolvedSourceSet,
    UnverifiableReason,
    VerificationResult,
    VerificationStatus,
    safe_verification_result,
)
from src.verify_prompts import MODEL_ID

logger: structlog.BoundLogger = structlog.get_logger(__name__)


def _aggregate_multi_source_verdicts(
    per_source: list[VerificationResult],
) -> VerificationStatus:
    """Aggregate per-source verdicts for a multi-citation claim.

    S2-P4 aggregation rule (Codex's, biased toward partially_supported on
    mixed evidence — matches annotator behavior on the four multi-source
    lactate-ISF claims, all of which are expected `partially_supported`):

      * any source `supported` AND all others in {supported, partially}
        -> supported
      * any source `supported` AND any in {unsupported, not_addressed}
        -> partially_supported (mixed)
      * all sources `unsupported` -> unsupported
      * all sources `not_addressed` (e.g., empty set / all unfound)
        -> not_addressed
      * everything else -> partially_supported
    """
    if not per_source:
        return "not_addressed"
    statuses = [r.status for r in per_source]
    # Treat "unverifiable" like "not_addressed" for aggregation — it
    # contributes no evidence signal. When every per-source result is
    # unverifiable, the aggregate is also unverifiable.
    if all(s == "unverifiable" for s in statuses):
        return "unverifiable"
    has_supported = any(s == "supported" for s in statuses)
    has_partial = any(s == "partially_supported" for s in statuses)
    has_unsupported = any(s == "unsupported" for s in statuses)
    has_not_addressed = any(s in ("not_addressed", "unverifiable") for s in statuses)

    if has_supported and not has_unsupported and not has_not_addressed:
        return "supported"
    if has_supported:
        return "partially_supported"
    if all(s == "unsupported" for s in statuses):
        return "unsupported"
    if all(s in ("not_addressed", "unverifiable") for s in statuses):
        return "not_addressed"
    if has_partial or has_unsupported:
        return "partially_supported"
    return "partially_supported"


def verify_claim_multi_source(
    claim: Claim,
    source_set: ResolvedSourceSet,
    *,
    passages_per_source: dict[str, list[PaperChunk]] | None = None,
    model_id: str = MODEL_ID,
    api_key: str | None = None,
) -> tuple[VerificationResult, list[ProvenanceStep]]:
    """Verify a claim against every source in `source_set`, aggregate, and return.

    For each source in the set:
      - If `passages_per_source[source.doi]` is non-empty, run
        `verify_claim_fulltext` on those passages.
      - Else run `verify_claim` against `source.abstract` (via the existing
        single-source path; that path itself routes to title-only mode when
        abstract is None and title is informative).

    Per-source verdicts are then aggregated via `_aggregate_multi_source_verdicts`.
    The returned VerificationResult records the aggregate status, a synthetic
    explanation listing per-source verdicts, and `confidence` set to the mean
    of per-source confidences.

    The returned ProvenanceStep list contains one step per source plus any
    nested fulltext+numeric steps for sources that took the fulltext path.
    """
    # Deferred imports to avoid a circular dependency:
    # verify_multi -> verify (verify_claim, verify_claim_fulltext)
    # verify -> verify_multi (re-export of these two symbols)
    from src.verify import verify_claim, verify_claim_fulltext

    passages_per_source = passages_per_source or {}
    per_source_results: list[VerificationResult] = []
    all_steps: list[ProvenanceStep] = []
    explanations: list[str] = []

    for source in source_set:
        passages = passages_per_source.get(source.doi or "", []) if source.doi else []
        if passages:
            result, step = verify_claim_fulltext(
                claim, source, passages, model_id=model_id, api_key=api_key
            )
            all_steps.append(step)
        else:
            result, step = verify_claim(claim, source, model_id=model_id, api_key=api_key)
            all_steps.append(step)
        per_source_results.append(result)
        marker_label = source.doi or source.title or "(unresolved)"
        explanations.append(f"[{marker_label}] {result.status}: {result.explanation}")

    # Empty-source-set guard. Without this, aggregation produces
    # confidence=None on a non-unverifiable status, which __post_init__
    # rejects with a runtime ValueError. The pipeline guards this at the
    # call site, but direct callers shouldn't trip the invariant either.
    if not per_source_results:
        return (
            VerificationResult(
                status="not_addressed",
                confidence=0.0,
                explanation="Empty source set — no resolved sources to verify against.",
                evidence_quality="no_evidence",
            ),
            all_steps,
        )

    aggregated_status = _aggregate_multi_source_verdicts(per_source_results)
    # Exclude confidence=None (unverifiable) and confidence=0.0 (parse errors)
    # so they don't drag the aggregate confidence down.
    confidences = [
        r.confidence for r in per_source_results if r.confidence is not None and r.confidence > 0
    ]
    aggregated_confidence: float | None = (
        sum(confidences) / len(confidences) if confidences else None
    )

    # Derive verification_depth and evidence_quality from the best available
    # per-source result rather than hardcoding to "abstract". Priority:
    # fulltext > citing_paper_context > abstract > title_only — the aggregation
    # uses the most informative evidence available.
    depth_priority = {
        "fulltext": 0,
        "citing_paper_context": 1,
        "abstract": 2,
        "title_only": 3,
    }
    primary_result = (
        min(
            per_source_results,
            key=lambda r: depth_priority.get(r.verification_depth, 99),
        )
        if per_source_results
        else None
    )
    agg_depth = primary_result.verification_depth if primary_result else "abstract"
    agg_evidence = primary_result.evidence_quality if primary_result else "no_evidence"

    # Route through the helper: low-evidence confident verdicts are downgraded to unverifiable.
    aggregated = safe_verification_result(
        status=aggregated_status,
        confidence=aggregated_confidence,
        explanation=" || ".join(explanations) if explanations else "Empty source set.",
        verification_depth=agg_depth,
        evidence_quality=agg_evidence,
        retraction_status=any(s.retraction_status for s in source_set),
        claim_text=claim.claim_text,
        # If downgraded: proximate cause is insufficient evidence depth across sources.
        unverifiable_reason="insufficient_evidence_depth",
    )

    agg_unverifiable_reason: UnverifiableReason | None = (
        "insufficient_evidence_depth" if aggregated.status == "unverifiable" else None
    )
    all_steps.append(
        ProvenanceStep(
            step_id=str(uuid.uuid4()),
            claim_id=claim.claim_id,
            operation="aggregate",
            input_hash=hashlib.sha256(repr(per_source_results).encode()).hexdigest(),
            output_hash=hashlib.sha256(repr(aggregated).encode()).hexdigest(),
            model_id=None,
            timestamp=time.time(),
            tokens_in=None,
            tokens_out=None,
            cache_hit=None,
            confidence=aggregated.confidence,
            unverifiable_reason=agg_unverifiable_reason,
        )
    )

    return aggregated, all_steps
