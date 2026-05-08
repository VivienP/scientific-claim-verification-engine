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
    VerificationResult,
    VerificationStatus,
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
    has_supported = any(s == "supported" for s in statuses)
    has_partial = any(s == "partially_supported" for s in statuses)
    has_unsupported = any(s == "unsupported" for s in statuses)
    has_not_addressed = any(s == "not_addressed" for s in statuses)

    if has_supported and not has_unsupported and not has_not_addressed:
        return "supported"
    if has_supported:
        return "partially_supported"
    if all(s == "unsupported" for s in statuses):
        return "unsupported"
    if all(s == "not_addressed" for s in statuses):
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

    aggregated_status = _aggregate_multi_source_verdicts(per_source_results)
    # Exclude confidence=0.0: these are parse-error results, not meaningful low-confidence verdicts.
    confidences = [r.confidence for r in per_source_results if r.confidence > 0]
    aggregated_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    aggregated = VerificationResult(
        status=aggregated_status,
        explanation=" || ".join(explanations) if explanations else "Empty source set.",
        confidence=aggregated_confidence,
        verification_depth="abstract",
        evidence_quality="abstract_only" if per_source_results else "no_evidence",
        retraction_status=any(s.retraction_status for s in source_set),
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
            confidence=aggregated_confidence,
        )
    )

    return aggregated, all_steps
