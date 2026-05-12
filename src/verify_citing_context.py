"""Citing-context verifier: last-resort internal-consistency path."""

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
    _CITING_CONTEXT_MAX_CONFIDENCE,
    _CITING_CONTEXT_SYSTEM_PROMPT,
    _VALID_STATUSES,
    MODEL_ID,
    _extract_citing_context_window,
    _hash,
    _parse_cache_hit,
    _strip_fences,
)

logger: structlog.BoundLogger = structlog.get_logger(__name__)


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
        raw_explanation = str(parsed["explanation"])
        explanation = (
            raw_explanation
            if "internal-consistency" in raw_explanation.lower()
            else f"[Internal-consistency only] {raw_explanation}"
        )
        # Route through the helper: `unsupported` + `citing_paper_context` is downgraded.
        # `supported` is already capped to `partially_supported` above.
        result = safe_verification_result(
            status=status_raw,
            confidence=confidence,
            explanation=explanation,
            verification_depth="citing_paper_context",
            evidence_quality="citing_paper_context",
            retraction_status=source.retraction_status,
            claim_text=claim.claim_text,
            # Citing-paper context is internal consistency only, not
            # source-of-truth evidence; insufficient depth for any specific claim.
            unverifiable_reason="insufficient_evidence_depth",
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

    unverifiable_reason: UnverifiableReason | None = (
        "insufficient_evidence_depth" if result.status == "unverifiable" else None
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
        unverifiable_reason=unverifiable_reason,
    )

    return result, step
