"""LLM-based single-claim verification against source abstract."""

from __future__ import annotations

import dataclasses
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
    VerificationResult,
    safe_verification_result,
)
from src.verify_citing_context import verify_claim_citing_context
from src.verify_fulltext import (
    _MAX_FALLBACK_PASSAGE_CHARS,
    _truncate_passage,
    verify_claim_fulltext,
)
from src.verify_multi import (
    _aggregate_multi_source_verdicts,
    verify_claim_multi_source,
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
from src.verify_title_only import verify_claim_title_only

# Re-export all symbols that tests import from this module path.
__all__ = [
    "MODEL_ID",
    "_CITING_CONTEXT_MAX_CONFIDENCE",
    "_CITING_CONTEXT_SYSTEM_PROMPT",
    "_CITING_CONTEXT_WINDOW_CHARS",
    "_FULLTEXT_SYSTEM_PROMPT",
    "_MAX_FALLBACK_PASSAGE_CHARS",
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
    "_truncate_passage",
    "verify_claim",
    "verify_claim_citing_context",
    "verify_claim_fulltext",
    "verify_claim_fulltext_with_numeric",
    "verify_claim_multi_source",
    "verify_claim_title_only",
]

logger: structlog.BoundLogger = structlog.get_logger(__name__)


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
        # confidence is float | None: the LLM may emit null when it picks
        # status="unverifiable" on its own.
        raw_confidence = parsed.get("confidence")
        confidence_val: float | None = None if raw_confidence is None else float(raw_confidence)
        # Route through the helper so that
        # (supported|unsupported) + abstract_only + numeric claim_text
        # is downgraded to (unverifiable, None). evidence_quality is
        # abstract_only here because verify_claim only ever sees the abstract.
        # Reason is explicit at the call site rather than relying on the
        # helper's default, so the contract is readable in place.
        result = safe_verification_result(
            status=status_raw,
            confidence=confidence_val,
            explanation=str(parsed["explanation"]),
            evidence_quality="abstract_only",
            claim_text=claim.claim_text,
            unverifiable_reason="numeric_claim_abstract_only",
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.error(
            "verify_parse_error",
            claim_id=claim.claim_id,
            raw_response=response_text[:200],
            error=str(exc),
        )
        result = _PARSE_ERROR_RESULT

    # Propagate the helper-set unverifiable_reason from the result to the
    # provenance step so both records stay consistent.
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
        unverifiable_reason=result.unverifiable_reason,
    )

    return result, step
