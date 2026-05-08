"""LLM-based single-claim verification against source abstract."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
import uuid
from typing import Any

import anthropic
import structlog
from anthropic.types import TextBlock

from src.models import (
    Claim,
    PaperChunk,
    ProvenanceStep,
    ResolvedSource,
    ResolvedSourceSet,
    VerificationResult,
    VerificationStatus,
)
from src.verify_prompts import (
    _CITING_CONTEXT_MAX_CONFIDENCE,
    _CITING_CONTEXT_SYSTEM_PROMPT,
    _CITING_CONTEXT_WINDOW_CHARS,
    _FULLTEXT_SYSTEM_PROMPT,
    _PARSE_ERROR_RESULT,
    _SHORT_CIRCUIT_RESULT,
    _SYSTEM_PROMPT,
    _TITLE_ONLY_MAX_CONFIDENCE,
    _TITLE_ONLY_MIN_TITLE_LENGTH,
    _TITLE_ONLY_SYSTEM_PROMPT,
    _VALID_STATUSES,
    MODEL_ID,
    _build_passages_block,
    _extract_citing_context_window,
    _hash,
    _make_short_circuit_step,
    _parse_cache_hit,
    _strip_fences,
)

# Re-export all symbols that tests import from this module path.
__all__ = [
    "MODEL_ID",
    "_CITING_CONTEXT_MAX_CONFIDENCE",
    "_CITING_CONTEXT_SYSTEM_PROMPT",
    "_CITING_CONTEXT_WINDOW_CHARS",
    "_FULLTEXT_SYSTEM_PROMPT",
    "_PARSE_ERROR_RESULT",
    "_SHORT_CIRCUIT_RESULT",
    "_SYSTEM_PROMPT",
    "_TITLE_ONLY_MAX_CONFIDENCE",
    "_TITLE_ONLY_MIN_TITLE_LENGTH",
    "_TITLE_ONLY_SYSTEM_PROMPT",
    "_VALID_STATUSES",
    "_aggregate_multi_source_verdicts",
    "_build_passages_block",
    "_extract_citing_context_window",
    "_hash",
    "_make_short_circuit_step",
    "_parse_cache_hit",
    "_strip_fences",
    "verify_claim",
    "verify_claim_citing_context",
    "verify_claim_fulltext",
    "verify_claim_fulltext_with_numeric",
    "verify_claim_multi_source",
    "verify_claim_title_only",
]

logger: structlog.BoundLogger = structlog.get_logger(__name__)


def verify_claim_fulltext(
    claim: Claim,
    source: ResolvedSource,
    passages: list[PaperChunk],
    *,
    model_id: str = MODEL_ID,
    api_key: str | None = None,
) -> tuple[VerificationResult, ProvenanceStep]:
    """Verify a claim against top-k full-text passages via Claude API.

    Falls back to verify_claim() (abstract-only) if passages is empty.
    Uses _FULLTEXT_SYSTEM_PROMPT (>1024 tokens, prompt-cached).
    Returns VerificationResult with verification_depth="fulltext",
    fulltext_available=True, source_passages and source_section populated,
    retraction_status mirrored from source.retraction_status.
    Never raises. Falls back to a parse-error result on malformed responses.
    """
    if not passages:
        abstract_result, step = verify_claim(claim, source, model_id=model_id, api_key=api_key)
        result = dataclasses.replace(
            abstract_result,
            fulltext_available=True,
            verification_depth="abstract",
            retrieval_status="no_passage_found",
            retraction_status=source.retraction_status,
        )
        return (
            result,
            dataclasses.replace(
                step,
                input_hash=_hash(repr((claim, source, passages))),
                output_hash=_hash(repr(result)),
                confidence=result.confidence,
            ),
        )

    ts = time.time()
    effective_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=effective_key)

    user_message = f"<claim>{claim.claim_text}</claim>\n" + _build_passages_block(passages)

    response = client.messages.create(
        model=model_id,
        # max_tokens=2048: full-text verifier may quote up to 3 source_passages
        # plus a multi-sentence explanation. Observed parse errors on claim 003
        # were caused by truncated JSON when 1024 was insufficient.
        max_tokens=2048,
        system=[
            {
                "type": "text",
                "text": _FULLTEXT_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    )

    tokens_in: int = response.usage.input_tokens
    tokens_out: int = response.usage.output_tokens
    cache_hit = _parse_cache_hit(response.usage)

    logger.info(
        "verify_fulltext_llm_call",
        model_id=model_id,
        claim_id=claim.claim_id,
        passage_count=len(passages),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cache_hit=cache_hit,
    )

    first_block = response.content[0] if response.content else None
    response_text = first_block.text if isinstance(first_block, TextBlock) else ""

    try:
        parsed: dict[str, Any] = json.loads(_strip_fences(response_text))
        status_raw = str(parsed["status"])
        if status_raw not in _VALID_STATUSES:
            raise ValueError(f"Invalid status: {status_raw}")
        status: VerificationStatus = status_raw  # type: ignore[assignment]
        raw_passages = parsed.get("source_passages", [])
        source_passages = [str(p) for p in raw_passages] if isinstance(raw_passages, list) else []
        source_section_raw = parsed.get("source_section")
        source_section = str(source_section_raw) if source_section_raw else None
        result = VerificationResult(
            status=status,
            explanation=str(parsed["explanation"]),
            confidence=float(parsed["confidence"]),
            source_passages=source_passages,
            source_section=source_section,
            fulltext_available=True,
            verification_depth="fulltext",
            retrieval_status="passage_found",
            evidence_quality="quoted_passage" if source_passages else "no_evidence",
            retraction_status=source.retraction_status,
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.error(
            "verify_fulltext_parse_error",
            claim_id=claim.claim_id,
            raw_response=response_text[:200],
            error=str(exc),
        )
        result = VerificationResult(
            status="not_addressed",
            explanation="Parse error.",
            confidence=0.0,
            source_passages=[],
            source_section=None,
            fulltext_available=True,
            verification_depth="fulltext",
            retrieval_status="passage_found",
            evidence_quality="no_evidence",
            retraction_status=source.retraction_status,
        )

    step = ProvenanceStep(
        step_id=str(uuid.uuid4()),
        claim_id=claim.claim_id,
        operation="verify",
        input_hash=_hash(repr((claim, source, passages))),
        output_hash=_hash(repr(result)),
        model_id=model_id,
        timestamp=ts,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cache_hit=cache_hit,
        confidence=result.confidence,
    )

    return result, step


def verify_claim_fulltext_with_numeric(
    claim: Claim,
    source: ResolvedSource,
    passages: list[PaperChunk],
    *,
    model_id: str = MODEL_ID,
    api_key: str | None = None,
) -> tuple[VerificationResult, list[ProvenanceStep]]:
    """Run full-text LLM verification AND the deterministic numeric engine.

    Returns the VerificationResult with `numeric_check` populated when the engine
    runs successfully, plus the full list of provenance steps (verify + extract +
    optional check).

    Numeric engine is invoked only when the claim contains numeric assertions that
    yield an OR/CI triple. When the engine returns None (no triple found), the
    VerificationResult is returned unchanged with `numeric_check=None`.

    Never raises.
    """
    from src.numeric.engine import run_numeric_check

    result, verify_step = verify_claim_fulltext(
        claim, source, passages, model_id=model_id, api_key=api_key
    )

    numeric_result, numeric_steps = run_numeric_check(
        claim.claim_text, claim_id=claim.claim_id, model_id=model_id, api_key=api_key
    )

    if numeric_result is not None:
        result = dataclasses.replace(result, numeric_check=numeric_result)

    return result, [verify_step, *numeric_steps]


def verify_claim_citing_context(
    claim: Claim,
    source: ResolvedSource,
    citing_paper_text: str,
    *,
    model_id: str = MODEL_ID,
    api_key: str | None = None,
) -> tuple[VerificationResult, ProvenanceStep]:
    """Verify a claim against the citing paper's own internal context.

    S3-P1 last-resort verifier: when the cited source cannot be retrieved
    (Layers 1-4 failed — no abstract, no full text, no informative title),
    check whether the citing paper's own surrounding text is consistent with
    the claim being attributed to the citation. This is internal-consistency
    evidence, NOT third-party verification — and is capped at
    `partially_supported` accordingly.

    The prompt explicitly tells the LLM that supported is forbidden; a
    deterministic post-LLM guard re-applies the cap so prompt non-compliance
    cannot leak `supported` verdicts.

    Returns a `VerificationResult` with:
        verification_depth = "citing_paper_context"
        evidence_quality   = "citing_paper_context"
        confidence         <= _CITING_CONTEXT_MAX_CONFIDENCE
    The explanation is prefixed with "[Internal-consistency only]" so the
    Medical Writer audit consumer sees the contract distinction at a glance.
    """
    ts = time.time()
    effective_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=effective_key)

    citation_label = ", ".join(claim.cited_authors) or "(unattributed)"
    if claim.cited_year is not None:
        citation_label = f"{citation_label} ({claim.cited_year})"
    context = _extract_citing_context_window(citing_paper_text, claim.claim_text)
    user_message = (
        f"<claim>{claim.claim_text}</claim>\n"
        f"<cited_reference>{citation_label}</cited_reference>\n"
        f"<citing_paper_context>{context}</citing_paper_context>"
    )

    response = client.messages.create(
        model=model_id,
        max_tokens=512,
        system=[
            {
                "type": "text",
                "text": _CITING_CONTEXT_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    )

    tokens_in: int = response.usage.input_tokens
    tokens_out: int = response.usage.output_tokens
    cache_hit = _parse_cache_hit(response.usage)

    logger.info(
        "verify_citing_context_llm_call",
        model_id=model_id,
        claim_id=claim.claim_id,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cache_hit=cache_hit,
    )

    first_block = response.content[0] if response.content else None
    response_text = first_block.text if isinstance(first_block, TextBlock) else ""

    try:
        parsed: dict[str, Any] = json.loads(_strip_fences(response_text))
        status_raw = str(parsed["status"])
        if status_raw not in _VALID_STATUSES:
            raise ValueError(f"Invalid status: {status_raw}")
        confidence = float(parsed["confidence"])
        # Hard cap: internal consistency cannot establish supported.
        if status_raw == "supported":
            status_raw = "partially_supported"
        confidence = min(confidence, _CITING_CONTEXT_MAX_CONFIDENCE)
        status: VerificationStatus = status_raw  # type: ignore[assignment]
        raw_explanation = str(parsed["explanation"])
        explanation = (
            raw_explanation
            if "internal-consistency" in raw_explanation.lower()
            else f"[Internal-consistency only] {raw_explanation}"
        )
        result = VerificationResult(
            status=status,
            explanation=explanation,
            confidence=confidence,
            verification_depth="citing_paper_context",
            evidence_quality="citing_paper_context",
            retraction_status=source.retraction_status,
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.error(
            "verify_citing_context_parse_error",
            claim_id=claim.claim_id,
            raw_response=response_text[:200],
            error=str(exc),
        )
        result = VerificationResult(
            status="not_addressed",
            explanation="Parse error.",
            confidence=0.0,
            verification_depth="citing_paper_context",
            evidence_quality="no_evidence",
            retraction_status=source.retraction_status,
        )

    step = ProvenanceStep(
        step_id=str(uuid.uuid4()),
        claim_id=claim.claim_id,
        operation="verify",
        input_hash=_hash(repr((claim, source, len(citing_paper_text)))),
        output_hash=_hash(repr(result)),
        model_id=model_id,
        timestamp=ts,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cache_hit=cache_hit,
        confidence=result.confidence,
    )

    return result, step


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


def verify_claim_title_only(
    claim: Claim,
    source: ResolvedSource,
    *,
    model_id: str = MODEL_ID,
    api_key: str | None = None,
) -> tuple[VerificationResult, ProvenanceStep]:
    """Verify a claim against the source title (and optionally journal) only.

    Bug B fix (S1-P1-B): when the resolver finds the right paper but neither
    CrossRef nor PubMed exposes an abstract (common for IEEE proceedings,
    Elsevier paywalls, and older journals), the previous `verify_claim`
    short-circuited to `not_addressed`. For claims whose target title is
    near-verbatim with the claim text (e.g. "Porosity control of polylactic
    acid porous microneedles using microfluidic technology"), the title is
    informative enough to warrant `partially_supported` rather than
    `not_addressed`.

    Hard guarantees enforced post-LLM (defensive against prompt non-compliance):
        - status `supported` is downgraded to `partially_supported`
        - confidence is clamped to <= _TITLE_ONLY_MAX_CONFIDENCE (0.7)
        - evidence_quality is always `title_only`
        - verification_depth is always `title_only`
    """
    ts = time.time()
    effective_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=effective_key)

    title = source.title or ""
    user_message = f"<claim>{claim.claim_text}</claim>\n<title>{title}</title>"

    response = client.messages.create(
        model=model_id,
        max_tokens=512,
        system=[
            {
                "type": "text",
                "text": _TITLE_ONLY_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    )

    tokens_in: int = response.usage.input_tokens
    tokens_out: int = response.usage.output_tokens
    cache_hit = _parse_cache_hit(response.usage)

    logger.info(
        "verify_title_only_llm_call",
        model_id=model_id,
        claim_id=claim.claim_id,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cache_hit=cache_hit,
    )

    first_block = response.content[0] if response.content else None
    response_text = first_block.text if isinstance(first_block, TextBlock) else ""

    try:
        parsed: dict[str, Any] = json.loads(_strip_fences(response_text))
        status_raw = str(parsed["status"])
        if status_raw not in _VALID_STATUSES:
            raise ValueError(f"Invalid status: {status_raw}")
        confidence = float(parsed["confidence"])
        # Hard cap: title-only evidence cannot establish supported.
        if status_raw == "supported":
            status_raw = "partially_supported"
        confidence = min(confidence, _TITLE_ONLY_MAX_CONFIDENCE)
        status: VerificationStatus = status_raw  # type: ignore[assignment]
        result = VerificationResult(
            status=status,
            explanation=str(parsed["explanation"]),
            confidence=confidence,
            verification_depth="title_only",
            evidence_quality="title_only",
            retraction_status=source.retraction_status,
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.error(
            "verify_title_only_parse_error",
            claim_id=claim.claim_id,
            raw_response=response_text[:200],
            error=str(exc),
        )
        result = VerificationResult(
            status="not_addressed",
            explanation="Parse error.",
            confidence=0.0,
            verification_depth="title_only",
            evidence_quality="no_evidence",
            retraction_status=source.retraction_status,
        )

    step = ProvenanceStep(
        step_id=str(uuid.uuid4()),
        claim_id=claim.claim_id,
        operation="verify",
        input_hash=_hash(repr((claim, source))),
        output_hash=_hash(repr(result)),
        model_id=model_id,
        timestamp=ts,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cache_hit=cache_hit,
        confidence=result.confidence,
    )

    return result, step


def verify_claim(
    claim: Claim,
    source: ResolvedSource,
    *,
    model_id: str = MODEL_ID,
    api_key: str | None = None,
) -> tuple[VerificationResult, ProvenanceStep]:
    """Verify a single claim against its resolved source abstract via Claude API.

    Short-circuits (no LLM call) when source.found=False, or when both
    source.abstract is None and source.title is too short to verify against.
    When abstract is None but the title is informative (>= _TITLE_ONLY_MIN_TITLE_LENGTH
    chars), routes to `verify_claim_title_only` (Bug B fix S1-P1-B).
    System prompt >1024 tokens → cache_control={"type": "ephemeral"}.
    """
    if not source.found:
        return _SHORT_CIRCUIT_RESULT, _make_short_circuit_step(claim, source)
    if source.abstract is None:
        if source.title and len(source.title) >= _TITLE_ONLY_MIN_TITLE_LENGTH:
            return verify_claim_title_only(claim, source, model_id=model_id, api_key=api_key)
        return _SHORT_CIRCUIT_RESULT, _make_short_circuit_step(claim, source)

    ts = time.time()
    effective_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=effective_key)

    user_message = f"<claim>{claim.claim_text}</claim>\n<source>{source.abstract}</source>"

    response = client.messages.create(
        model=model_id,
        max_tokens=512,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    )

    tokens_in: int = response.usage.input_tokens
    tokens_out: int = response.usage.output_tokens
    cache_hit = _parse_cache_hit(response.usage)

    logger.info(
        "verify_llm_call",
        model_id=model_id,
        claim_id=claim.claim_id,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cache_hit=cache_hit,
    )

    first_block = response.content[0] if response.content else None
    response_text = first_block.text if isinstance(first_block, TextBlock) else ""
    result: VerificationResult

    try:
        parsed: dict[str, Any] = json.loads(_strip_fences(response_text))
        status_raw = str(parsed["status"])
        if status_raw not in _VALID_STATUSES:
            raise ValueError(f"Invalid status: {status_raw}")
        status: VerificationStatus = status_raw  # type: ignore[assignment]
        result = VerificationResult(
            status=status,
            explanation=str(parsed["explanation"]),
            confidence=float(parsed["confidence"]),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.error(
            "verify_parse_error",
            claim_id=claim.claim_id,
            raw_response=response_text[:200],
            error=str(exc),
        )
        result = _PARSE_ERROR_RESULT

    step = ProvenanceStep(
        step_id=str(uuid.uuid4()),
        claim_id=claim.claim_id,
        operation="verify",
        input_hash=_hash(repr((claim, source))),
        output_hash=_hash(repr(result)),
        model_id=model_id,
        timestamp=ts,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cache_hit=cache_hit,
        confidence=result.confidence,
    )

    return result, step
