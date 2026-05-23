"""LLM-based claim extraction from free-form scientific text."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from typing import Any  # Any used for json.loads() return type only

import anthropic
import structlog
from anthropic.types import TextBlock, Usage

from src.models import Claim, ProvenanceStep
from src.prompts import load_prompt

logger: structlog.BoundLogger = structlog.get_logger(__name__)

MODEL_ID = "claude-sonnet-4-6"
_OUTPUT_FLOOR = 4096
_OUTPUT_CEILING = 16384

_SYSTEM_PROMPT = load_prompt("extract_v1")


def _scale_max_output_tokens(text: str) -> int:
    """Pick an output-token budget proportional to input length.

    Scientific PDFs run ~3 chars per token (denser than typical prose at ~4),
    and citation-anchored extraction on dense lit reviews emits roughly
    25-35% of input tokens as output. The formula targets ~40% of the
    char-derived input estimate to keep a safety margin against truncation.

    Floor is 4096 (preserves prior default for short inputs); ceiling is
    16384 (bounds cost on pathological inputs and stays within Sonnet 4.6's
    practical output limit).
    """
    return min(_OUTPUT_CEILING, max(_OUTPUT_FLOOR, len(text) * 4 // 30))


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


_CITATION_BRACKET_RE = re.compile(r"\[([0-9,\-\s]+)\]")


def _parse_citation_markers(raw_markers: object, claim_text: str) -> list[int]:
    """Return sorted citation marker integers from JSON or bracketed claim text."""
    markers: list[int] = []
    if isinstance(raw_markers, list):
        for item in raw_markers:
            try:
                marker = int(item)
            except (TypeError, ValueError):
                continue
            if marker > 0:
                markers.append(marker)
    if markers:
        return sorted(set(markers))

    for match in _CITATION_BRACKET_RE.finditer(claim_text):
        for part in match.group(1).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_raw, end_raw = [p.strip() for p in part.split("-", 1)]
                try:
                    start = int(start_raw)
                    end = int(end_raw)
                except ValueError:
                    continue
                if 0 < start <= end and end - start <= 50:
                    markers.extend(range(start, end + 1))
            else:
                try:
                    marker = int(part)
                except ValueError:
                    continue
                if marker > 0:
                    markers.append(marker)
    return sorted(set(markers))


def _parse_cache_hit(usage: Usage) -> bool | None:
    cache_read: int = usage.cache_read_input_tokens or 0
    cache_creation: int = usage.cache_creation_input_tokens or 0
    if cache_read > 0:
        return True
    if cache_creation > 0:
        return False
    return None


_CLAIMS_ARRAY_START_RE = re.compile(r'"claims"\s*:\s*\[')


def _attempt_partial_recovery(response_text: str) -> list[dict[str, Any]]:
    """Salvage as many complete claim objects as possible from a truncated response.

    Premium Systematic Review outputs occasionally exceed ``max_output_tokens``,
    leaving the JSON payload truncated mid-object. The default ``json.loads``
    failure discards every claim — including all the ones the LLM emitted
    completely before truncation. This helper iterates objects via
    ``json.JSONDecoder.raw_decode`` and stops cleanly at the first un-parseable
    position, returning whatever was fully decoded.

    Returns an empty list when no ``"claims": [`` array marker is found, which
    is the right behavior for genuinely malformed responses (vs. truncated ones).
    """
    cleaned = _strip_fences(response_text)
    array_match = _CLAIMS_ARRAY_START_RE.search(cleaned)
    if not array_match:
        return []
    decoder = json.JSONDecoder()
    pos = array_match.end()
    recovered: list[dict[str, Any]] = []
    while pos < len(cleaned):
        # Skip whitespace and inter-object commas. Stop on closing bracket.
        while pos < len(cleaned) and cleaned[pos] in " \t\n\r,":
            pos += 1
        if pos >= len(cleaned) or cleaned[pos] == "]":
            break
        try:
            obj, end_pos = decoder.raw_decode(cleaned, pos)
        except json.JSONDecodeError as exc:
            # Expected exit for truncated payloads — the recovered count is
            # logged by the caller as ``extract_partial_recovery``. Debug-level
            # here so the structlog rule is satisfied without spamming a
            # warning on every truncation (which is the function's normal exit).
            logger.debug(
                "extract_partial_recovery_truncation",
                position=pos,
                recovered_so_far=len(recovered),
                error=str(exc),
            )
            break
        if isinstance(obj, dict):
            recovered.append(obj)
        pos = end_pos
    return recovered


def extract_claims(
    text: str,
    *,
    model_id: str = MODEL_ID,
    api_key: str | None = None,
    max_output_tokens: int | None = None,
) -> tuple[list[Claim], ProvenanceStep]:
    """Extract verifiable scientific claims from free-form scientific text.

    Uses Claude API with structured XML-tagged prompt.
    System prompt uses cache_control={"type": "ephemeral"} (>1024 tokens).
    Input text wrapped in <text>...</text> to prevent prompt injection.
    On malformed LLM response: returns ([], provenance_step), logs structlog.error.
    ProvenanceStep.claim_id = "__extract__:{sha256(text)[:8]}".

    max_output_tokens caps the LLM JSON response length. When None (default),
    the budget auto-scales with input length via _scale_max_output_tokens()
    to avoid truncation on dense lit-review inputs. Pass an explicit integer
    to override the heuristic.
    """
    ts = time.time()
    claim_id = f"__extract__:{_hash(text)[:8]}"
    input_hash = _hash(repr(text))

    if max_output_tokens is None:
        max_output_tokens = _scale_max_output_tokens(text)

    effective_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=effective_key)

    # Streaming avoids connection drops on long generations (>4096 output tokens).
    with client.messages.stream(
        model=model_id,
        max_tokens=max_output_tokens,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": f"<text>{text}</text>"}],
    ) as stream:
        response = stream.get_final_message()

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
            claim_text = str(raw["claim_text"]).strip()
            if not claim_text:
                logger.warning("extract_empty_claim_text", raw_claim=raw)
                continue
            claims.append(
                Claim(
                    claim_id=str(uuid.uuid4()),
                    claim_text=claim_text,
                    cited_authors=list(raw.get("cited_authors", [])),
                    cited_year=int(raw["cited_year"])
                    if raw.get("cited_year") is not None
                    else None,
                    claim_type=str(raw["claim_type"])
                    if raw.get("claim_type")
                    else "factual_qualitative",  # type: ignore[arg-type]
                    citation_markers=_parse_citation_markers(
                        raw.get("citation_markers"), claim_text
                    ),
                )
            )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.error(
            "extract_parse_error",
            raw_response=response_text[:200],
            error=str(exc),
        )
        # Reset any claims appended by the happy-path loop before it raised,
        # then attempt partial recovery. Better to verify N claims out of M
        # than 0 out of M when the only damage is a truncated trailing object.
        claims = []
        recovered = _attempt_partial_recovery(response_text)
        if recovered:
            logger.warning(
                "extract_partial_recovery",
                recovered_count=len(recovered),
                response_length=len(response_text),
            )
            for raw in recovered:
                raw_text = raw.get("claim_text")
                if raw_text is None:
                    continue
                claim_text = str(raw_text).strip()
                if not claim_text:
                    continue
                try:
                    cited_year = (
                        int(raw["cited_year"]) if raw.get("cited_year") is not None else None
                    )
                    claim_type = (
                        str(raw["claim_type"]) if raw.get("claim_type") else "factual_qualitative"
                    )
                    claims.append(
                        Claim(
                            claim_id=str(uuid.uuid4()),
                            claim_text=claim_text,
                            cited_authors=list(raw.get("cited_authors", [])),
                            cited_year=cited_year,
                            claim_type=claim_type,  # type: ignore[arg-type]
                            citation_markers=_parse_citation_markers(
                                raw.get("citation_markers"), claim_text
                            ),
                        )
                    )
                except (ValueError, TypeError) as exc:
                    # Partial-recovery only — log and drop the malformed
                    # object rather than aborting the whole batch. CLAUDE.md
                    # requires every except to log; without this, a recovered
                    # claim with e.g. cited_year=={} silently disappears,
                    # masking the truncation damage we run partial recovery
                    # to surface.
                    logger.warning(
                        "extract_partial_recovery_skip_malformed_claim",
                        claim_text_preview=claim_text[:80],
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    continue

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
