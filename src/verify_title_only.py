"""Title-only verifier: fallback path when no abstract is available."""

from __future__ import annotations

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
    ProvenanceStep,
    ResolvedSource,
    UnverifiableReason,
    VerificationResult,
    safe_verification_result,
)
from src.verify_prompts import (
    _TITLE_ONLY_MAX_CONFIDENCE,
    _TITLE_ONLY_SYSTEM_PROMPT,
    _VALID_STATUSES,
    MODEL_ID,
    _hash,
    _parse_cache_hit,
    _strip_fences,
)

logger: structlog.BoundLogger = structlog.get_logger(__name__)


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
        # Route through the helper: `unsupported` + `title_only` is downgraded to unverifiable.
        # `supported` is already capped to `partially_supported` above.
        # Off-topic titles surface as `unverifiable` (not `unsupported`) — acceptable for Phase 1.
        result = safe_verification_result(
            status=status_raw,
            confidence=confidence,
            explanation=str(parsed["explanation"]),
            verification_depth="title_only",
            evidence_quality="title_only",
            retraction_status=source.retraction_status,
            claim_text=claim.claim_text,
            # Title-only evidence is structurally insufficient for any specific claim.
            unverifiable_reason="insufficient_evidence_depth",
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

    unverifiable_reason: UnverifiableReason | None = (
        "insufficient_evidence_depth" if result.status == "unverifiable" else None
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
        unverifiable_reason=unverifiable_reason,
    )

    return result, step
