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
    EvidenceBundle,
    PaperChunk,
    ProvenanceStep,
    ResolvedSource,
    ResolvedSourceSet,
    UnverifiableReason,
    VerificationResult,
    VerificationStatus,
    safe_verification_result,
)
from src.policy import Insufficient, assess_evidence_sufficiency
from src.verify_prompts import MODEL_ID

# Reused by ``verify_claim_multi_source`` when the per-source policy gate
# emits an ``Insufficient`` verdict. Kept in sync with the pipeline-side
# table in ``src/pipeline.py``; duplicated to avoid an inverse import.
_UNVERIFIABLE_REASON_TEXT: dict[UnverifiableReason, str] = {
    "insufficient_evidence_depth": ("evidence depth was insufficient for this claim type"),
    "fulltext_unavailable": ("full text was not retrievable by the current fetch chain"),
    "numeric_claim_abstract_only": (
        "this claim contains specific numeric Results-section assertions "
        "that cannot be confirmed from abstract-only evidence"
    ),
    "parse_error": "the verifier response could not be parsed",
    "resolution_low_confidence": (
        "the resolver flagged the source resolution as low-confidence and "
        "the claim contains specific numerics that would silently contaminate"
    ),
    "resolution_source_disagreement": (
        "multiple resolver clients disagreed on the source identity"
    ),
}


def _policy_gated_result(
    *,
    reason: UnverifiableReason,
    evidence: EvidenceBundle,
    source: ResolvedSource,
) -> VerificationResult:
    """Deterministic per-source ``unverifiable`` verdict for the multi-source path.

    Matches the metadata the LLM verifier would have emitted at the same depth
    so the aggregation logic downstream sees a comparable shape.
    """
    if evidence.depth == "fulltext":
        verification_depth: str = "fulltext"
        evidence_quality = "passages_searched_no_quote"
        retrieval_status = "passage_found"
        fulltext_available = True
    elif evidence.depth == "abstract":
        verification_depth = "abstract"
        evidence_quality = "abstract_only"
        retrieval_status = "fulltext_unavailable"
        fulltext_available = False
    elif evidence.depth == "title":
        verification_depth = "title_only"
        evidence_quality = "title_only"
        retrieval_status = "fulltext_unavailable"
        fulltext_available = False
    else:
        verification_depth = "abstract"
        evidence_quality = "no_evidence"
        retrieval_status = "fulltext_unavailable"
        fulltext_available = False
    explanation = (
        "Pipeline declined to invoke the verifier for this source: "
        f"{_UNVERIFIABLE_REASON_TEXT[reason]}. "
        "No LLM call was made — the verdict is deterministic."
    )
    return VerificationResult(
        status="unverifiable",
        confidence=None,
        explanation=explanation,
        evidence_quality=evidence_quality,  # type: ignore[arg-type]
        verification_depth=verification_depth,  # type: ignore[arg-type]
        retrieval_status=retrieval_status,  # type: ignore[arg-type]
        fulltext_available=fulltext_available,
        retraction_status=source.retraction_status,
        unverifiable_reason=reason,
    )


def _policy_step(
    claim: Claim, source: ResolvedSource, reason: UnverifiableReason
) -> ProvenanceStep:
    """Provenance step for a per-source policy-gated unverifiable verdict."""
    return ProvenanceStep(
        step_id=str(uuid.uuid4()),
        claim_id=claim.claim_id,
        operation="verify",
        input_hash=hashlib.sha256(repr((claim, source)).encode()).hexdigest(),
        output_hash=hashlib.sha256(f"unverifiable|{reason}".encode()).hexdigest(),
        model_id=None,
        timestamp=time.time(),
        tokens_in=None,
        tokens_out=None,
        cache_hit=None,
        confidence=None,
        unverifiable_reason=reason,
    )


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
    evidence_per_source: dict[str, EvidenceBundle] | None = None,
    model_id: str = MODEL_ID,
    api_key: str | None = None,
) -> tuple[VerificationResult, list[ProvenanceStep]]:
    """Verify a claim against every source in `source_set`, aggregate, and return.

    For each source in the set:
      - If ``evidence_per_source[source.doi]`` is supplied and the policy
        gate returns ``Insufficient``, emit a deterministic per-source
        ``unverifiable`` verdict and skip the LLM. This applies the same
        evidence-sufficiency contract enforced by the single-source path
        in ``src/pipeline.py::verify_one_claim``.
      - Else, if ``passages_per_source[source.doi]`` is non-empty, run
        ``verify_claim_fulltext`` on those passages.
      - Otherwise run ``verify_claim`` against ``source.abstract`` (the
        single-source path routes to title-only mode when abstract is None
        and the title is informative).

    Per-source verdicts are then aggregated via ``_aggregate_multi_source_verdicts``.
    The returned VerificationResult records the aggregate status, a synthetic
    explanation listing per-source verdicts, and ``confidence`` set to the mean
    of per-source confidences.

    The returned ProvenanceStep list contains one step per source plus any
    nested fulltext+numeric steps for sources that took the fulltext path,
    and the aggregate step at the end.
    """
    # Deferred imports to avoid a circular dependency:
    # verify_multi -> verify (verify_claim, verify_claim_fulltext)
    # verify -> verify_multi (re-export of these two symbols)
    from src.verify import verify_claim, verify_claim_fulltext

    passages_per_source = passages_per_source or {}
    evidence_per_source = evidence_per_source or {}
    per_source_results: list[VerificationResult] = []
    all_steps: list[ProvenanceStep] = []
    explanations: list[str] = []

    for source in source_set:
        passages = passages_per_source.get(source.doi or "", []) if source.doi else []
        evidence = evidence_per_source.get(source.doi or "") if source.doi else None
        if source.found and evidence is not None:
            decision = assess_evidence_sufficiency(claim, source, evidence)
            if isinstance(decision, Insufficient):
                result = _policy_gated_result(
                    reason=decision.reason, evidence=evidence, source=source
                )
                step = _policy_step(claim, source, decision.reason)
                all_steps.append(step)
                per_source_results.append(result)
                marker_label = source.doi or source.title or "(unresolved)"
                explanations.append(
                    f"[{marker_label}] {result.status} ({decision.reason}): {result.explanation}"
                )
                continue
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
