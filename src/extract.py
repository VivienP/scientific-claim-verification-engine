"""LLM-based claim extraction from free-form scientific text."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from typing import Any  # Any used for json.loads() return type only

import anthropic
import structlog
from anthropic.types import TextBlock, Usage

from src.models import Claim, ProvenanceStep

logger: structlog.BoundLogger = structlog.get_logger(__name__)

MODEL_ID = "claude-sonnet-4-6"

_SYSTEM_PROMPT = """\
You are a scientific claim extractor. Your task is to identify verifiable factual claims in scientific text that cite specific sources.

A verifiable claim must:
1. Make a specific, testable assertion about a scientific fact, method, or result
2. Be attributed to a specific cited source (author name(s) and/or year visible nearby)
3. Be falsifiable — it could in principle be checked against the cited source

Claim types:
- factual_numeric: claims involving specific numbers, statistics, percentages, p-values, effect sizes
- factual_qualitative: claims about categorical outcomes, findings, or relationships (no specific numbers)
- methodological: claims about how a study was conducted, what methods were used, sample sizes
- causal: claims asserting that X causes Y or that X leads to Y

For each claim, extract:
- claim_text: the exact claim as stated (verbatim or minimally paraphrased)
- cited_authors: list of author last names mentioned near the claim (empty list if none)
- cited_year: the year cited (integer or null)
- claim_type: one of the four types above

Return ONLY a JSON object in this exact format:
{
  "claims": [
    {
      "claim_text": "...",
      "cited_authors": ["Smith", "Jones"],
      "cited_year": 2019,
      "claim_type": "factual_numeric"
    }
  ]
}

If there are no verifiable claims, return {"claims": []}.
Do not include claims that have no citation anchor.
Do not hallucinate citations.

Additional context on verifiability:
- A claim is verifiable if we can check it against the cited source's abstract or full text.
- Claims that merely describe the general topic of a paper (without a specific assertion) are not verifiable.
- Claims with very vague attributions (e.g., "some studies suggest") are not verifiable — there must be a specific author/year anchor.
- For numerical claims, the specific number (percentage, p-value, effect size, sample size) must be present.
- For qualitative claims, the specific finding or relationship must be stated (e.g., "increased", "decreased", "no effect").
- For methodological claims, the specific method or design must be named (e.g., "randomized controlled trial", "cross-sectional study").
- For causal claims, the direction of causation must be explicit.

Your response must be valid JSON only — no explanatory text, no markdown, no code blocks.
"""


def _hash(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def _strip_fences(text: str) -> str:
    """Strip markdown code fences (```json ... ``` or ``` ... ```) from LLM output."""
    stripped = text.strip()
    if stripped.startswith("```"):
        # Remove opening fence line
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1 :]
        # Remove closing fence
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


def extract_claims(
    text: str,
    *,
    model_id: str = MODEL_ID,
    api_key: str | None = None,
) -> tuple[list[Claim], ProvenanceStep]:
    """Extract verifiable scientific claims from free-form scientific text.

    Uses Claude API with structured XML-tagged prompt.
    System prompt uses cache_control={"type": "ephemeral"} (>1024 tokens).
    Input text wrapped in <text>...</text> to prevent prompt injection.
    On malformed LLM response: returns ([], provenance_step), logs structlog.error.
    ProvenanceStep.claim_id = "__extract__:{sha256(text)[:8]}".
    """
    ts = time.time()
    claim_id = f"__extract__:{_hash(text)[:8]}"
    input_hash = _hash(repr(text))

    effective_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=effective_key)

    response = client.messages.create(
        model=model_id,
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": f"<text>{text}</text>"}],
    )

    tokens_in: int = response.usage.input_tokens
    tokens_out: int = response.usage.output_tokens
    cache_hit = _parse_cache_hit(response.usage)

    logger.info(
        "extract_llm_call",
        model_id=model_id,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cache_hit=cache_hit,
    )

    first_block = response.content[0]
    response_text = first_block.text if isinstance(first_block, TextBlock) else ""
    claims: list[Claim] = []

    try:
        parsed: dict[str, Any] = json.loads(_strip_fences(response_text))
        raw_claims: list[dict[str, Any]] = parsed["claims"]
        for raw in raw_claims:
            claims.append(
                Claim(
                    claim_id=str(uuid.uuid4()),
                    claim_text=str(raw["claim_text"]),
                    cited_authors=list(raw.get("cited_authors", [])),
                    cited_year=int(raw["cited_year"])
                    if raw.get("cited_year") is not None
                    else None,
                    claim_type=str(raw["claim_type"])
                    if raw.get("claim_type")
                    else "factual_qualitative",  # type: ignore[arg-type]
                )
            )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.error(
            "extract_parse_error",
            raw_response=response_text[:200],
            error=str(exc),
        )
        claims = []

    output_hash = _hash(repr(claims))

    step = ProvenanceStep(
        step_id=str(uuid.uuid4()),
        claim_id=claim_id,
        operation="extract",
        input_hash=input_hash,
        output_hash=output_hash,
        model_id=model_id,
        timestamp=ts,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cache_hit=cache_hit,
        confidence=None,
    )

    return claims, step
