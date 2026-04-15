"""LLM-based single-claim verification against source abstract."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from typing import Any

import anthropic
import structlog
from anthropic.types import TextBlock, Usage

from src.models import Claim, ProvenanceStep, ResolvedSource, VerificationResult, VerificationStatus

logger: structlog.BoundLogger = structlog.get_logger(__name__)

MODEL_ID = "claude-sonnet-4-6"

_SYSTEM_PROMPT = """\
You are a scientific claim verifier. Your task is to determine whether a source abstract supports, contradicts, or does not address a given scientific claim.

Verification statuses:
- supported: The abstract explicitly provides evidence that supports the claim. The claim's core assertion is consistent with what the abstract states.
- unsupported: The abstract explicitly contradicts the claim, or the abstract addresses the same topic but the claim's assertion is inconsistent with the abstract's findings.
- not_addressed: The abstract does not contain relevant information about the claim's subject matter. The abstract is about a different topic, or the specific assertion in the claim is not mentioned.
- partially_supported: The abstract provides some support for the claim but not complete support — for example, if the claim states a stronger effect than the abstract reports, or if the abstract's findings are mixed.

Guidelines:
- Base your verdict ONLY on the abstract text provided. Do not use outside knowledge.
- If the abstract is very short or general, err toward not_addressed rather than guessing.
- Confidence: 0.9-1.0 for clear cases, 0.6-0.8 for moderate certainty, 0.4-0.6 for uncertain.

Return ONLY a JSON object:
{
  "status": "supported|unsupported|not_addressed|partially_supported",
  "explanation": "One or two sentences explaining your verdict, citing specific evidence from the abstract.",
  "confidence": 0.85
}

Your response must be valid JSON only — no explanatory text, no markdown code blocks, no additional commentary.

Remember:
- "supported" requires explicit positive evidence in the abstract.
- "unsupported" requires the abstract to specifically contradict the claim.
- "not_addressed" is appropriate when the abstract does not discuss the claim's topic at all, or discusses it without addressing the specific assertion.
- "partially_supported" is for cases where the abstract provides some but not complete support.
- Always cite the specific sentences or phrases from the abstract that justify your verdict.
- Confidence should reflect your certainty, not the strength of the claim.
"""


def _hash(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def _strip_fences(text: str) -> str:
    """Strip markdown code fences (```json ... ``` or ``` ... ```) from LLM output."""
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[: stripped.rfind("```")].rstrip()
    return stripped


def _parse_cache_hit(usage: Usage) -> bool | None:
    cache_read: int = usage.cache_read_input_tokens or 0
    cache_creation: int = usage.cache_creation_input_tokens or 0
    if cache_read > 0:
        return True
    if cache_creation > 0:
        return False
    return None


def _make_short_circuit_step(
    claim: Claim,
    source: ResolvedSource,
) -> ProvenanceStep:
    return ProvenanceStep(
        step_id=str(uuid.uuid4()),
        claim_id=claim.claim_id,
        operation="verify",
        input_hash=_hash(repr((claim, source))),
        output_hash=_hash(repr("not_addressed")),
        model_id=None,
        timestamp=time.time(),
        tokens_in=None,
        tokens_out=None,
        cache_hit=None,
        confidence=1.0,
    )


_SHORT_CIRCUIT_RESULT = VerificationResult(
    status="not_addressed",
    explanation="Source not found or abstract unavailable.",
    confidence=1.0,
)

_PARSE_ERROR_RESULT = VerificationResult(
    status="not_addressed",
    explanation="Parse error.",
    confidence=0.0,
)

_VALID_STATUSES: set[str] = {"supported", "unsupported", "not_addressed", "partially_supported"}


def verify_claim(
    claim: Claim,
    source: ResolvedSource,
    *,
    model_id: str = MODEL_ID,
    api_key: str | None = None,
) -> tuple[VerificationResult, ProvenanceStep]:
    """Verify a single claim against its resolved source abstract via Claude API.

    Short-circuits (no LLM call) when source.found=False or source.abstract is None.
    System prompt >1024 tokens → cache_control={"type": "ephemeral"}.
    Claim wrapped in <claim>...</claim>; abstract in <source>...</source>.
    Logs tokens_in, tokens_out, cache_hit, model_id via structlog on every LLM call.
    """
    if not source.found or source.abstract is None:
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
