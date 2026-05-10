"""Cross-modal second-pass verification for high-confidence claims.

Phase 1: opt-in helper. Not wired into pipeline.py — caller invokes manually
on a per-run basis (e.g. from a benchmarking script). Future integration into
the pipeline is gated on cost characterization across 3 dogfood runs.

Why: per `feedback_resolver_priority` and the "Silent failures" rule in
CLAUDE.md, confident-but-wrong outputs are the worst failure mode. A single
verifier has no signal to detect its own overconfidence. A second model with
a different inductive bias is the cheapest disagreement detector available.

Gate (cost-controlled):
- primary verdict in {supported, unsupported}
- AND primary confidence > threshold (default 0.7)
- AND verification_depth == "abstract"

Outside the gate: returns the primary unchanged with provenance=None — no
LLM call made, no provenance step emitted.

On disagreement (status mismatch): downgrades primary's confidence to
min(primary, 0.5), appends a "[CROSS-MODAL DISAGREEMENT]" note to the
explanation, and emits a ProvenanceStep with confidence=None to signal
the unresolved state.

On agreement: returns the primary unchanged but emits a ProvenanceStep
with confidence=primary.confidence to record that cross-modal ran and
agreed (audit value: lets you measure agreement rate).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import replace
from typing import Any

import anthropic
import structlog
from anthropic.types import TextBlock

from src.models import (
    Claim,
    ProvenanceStep,
    VerificationResult,
    VerificationStatus,
)
from src.verify_prompts import (
    _SYSTEM_PROMPT,
    _VALID_STATUSES,
    _hash,
    _parse_cache_hit,
    _strip_fences,
)

logger: structlog.BoundLogger = structlog.get_logger(__name__)

DEFAULT_SECOND_MODEL_ID = "claude-haiku-4-5-20251001"
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DOWNGRADED_CONFIDENCE = 0.5

_TARGETED_STATUSES: set[VerificationStatus] = {"supported", "unsupported"}


def cross_modal_check(
    claim: Claim,
    abstract: str,
    primary_result: VerificationResult,
    *,
    second_model_id: str = DEFAULT_SECOND_MODEL_ID,
    api_key: str | None = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> tuple[VerificationResult, ProvenanceStep | None]:
    """Run a second-pass verify with a different model on confident verdicts.

    Returns:
        (result, step) where:
          - result is the (possibly downgraded) VerificationResult
          - step is a ProvenanceStep with operation="verify_cross_modal", or
            None if the gate did not fire (no LLM call was made)

    Gate: primary.status in {supported, unsupported}
          AND primary.confidence > confidence_threshold
          AND primary.verification_depth == "abstract"
    """
    if primary_result.status not in _TARGETED_STATUSES:
        return primary_result, None
    if primary_result.confidence <= confidence_threshold:
        return primary_result, None
    if primary_result.verification_depth != "abstract":
        return primary_result, None

    ts = time.time()
    effective_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=effective_key)
    user_message = f"<claim>{claim.claim_text}</claim>\n<source>{abstract}</source>"

    second_status: VerificationStatus | None = None
    second_confidence: float | None = None
    parse_error: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cache_hit: bool | None = None

    try:
        response = client.messages.create(
            model=second_model_id,
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
        tokens_in = response.usage.input_tokens
        tokens_out = response.usage.output_tokens
        cache_hit = _parse_cache_hit(response.usage)

        first_block = response.content[0] if response.content else None
        response_text = first_block.text if isinstance(first_block, TextBlock) else ""
        parsed: dict[str, Any] = json.loads(_strip_fences(response_text))
        raw_status = str(parsed["status"])
        if raw_status not in _VALID_STATUSES:
            raise ValueError(f"Invalid status: {raw_status}")
        second_status = raw_status  # type: ignore[assignment]
        second_confidence = float(parsed["confidence"])
    except (
        anthropic.APIError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        parse_error = str(exc)
        logger.error(
            "cross_modal_error",
            claim_id=claim.claim_id,
            second_model_id=second_model_id,
            error=parse_error,
        )

    logger.info(
        "cross_modal_call",
        claim_id=claim.claim_id,
        second_model_id=second_model_id,
        primary_status=primary_result.status,
        primary_confidence=primary_result.confidence,
        second_status=second_status,
        second_confidence=second_confidence,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cache_hit=cache_hit,
    )

    if parse_error is not None or second_status is None:
        # API or parse error: emit step with confidence=None, return primary unchanged.
        # Do NOT downgrade — agreement is undetermined.
        step = ProvenanceStep(
            step_id=str(uuid.uuid4()),
            claim_id=claim.claim_id,
            operation="verify_cross_modal",
            input_hash=_hash(repr((claim.claim_id, abstract, primary_result.status))),
            output_hash=_hash(repr(("error", parse_error))),
            model_id=second_model_id,
            timestamp=ts,
            tokens_in=tokens_in or None,
            tokens_out=tokens_out or None,
            cache_hit=cache_hit,
            confidence=None,
        )
        return primary_result, step

    if second_status == primary_result.status:
        step = ProvenanceStep(
            step_id=str(uuid.uuid4()),
            claim_id=claim.claim_id,
            operation="verify_cross_modal",
            input_hash=_hash(repr((claim.claim_id, abstract, primary_result.status))),
            output_hash=_hash(repr(("agree", second_status, second_confidence))),
            model_id=second_model_id,
            timestamp=ts,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cache_hit=cache_hit,
            confidence=primary_result.confidence,
        )
        return primary_result, step

    new_confidence = min(primary_result.confidence, DOWNGRADED_CONFIDENCE)
    note = (
        f"\n[CROSS-MODAL DISAGREEMENT: secondary={second_status} "
        f"conf={second_confidence:.2f}; primary={primary_result.status} "
        f"conf={primary_result.confidence:.2f}; primary status preserved, "
        f"confidence downgraded to {new_confidence:.2f}]"
    )
    downgraded = replace(
        primary_result,
        confidence=new_confidence,
        explanation=primary_result.explanation + note,
    )
    step = ProvenanceStep(
        step_id=str(uuid.uuid4()),
        claim_id=claim.claim_id,
        operation="verify_cross_modal",
        input_hash=_hash(repr((claim.claim_id, abstract, primary_result.status))),
        output_hash=_hash(repr(("disagree", second_status, second_confidence))),
        model_id=second_model_id,
        timestamp=ts,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cache_hit=cache_hit,
        confidence=None,
    )
    return downgraded, step
