"""LLM-based claim extraction from free-form scientific text."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import unicodedata
import uuid
from typing import Any  # Any used for json.loads() return type only

import anthropic
import structlog
from anthropic.types import TextBlock, Usage

from src.models import Claim, ClaimDirection, ProvenanceStep
from src.prompts import load_prompt

logger: structlog.BoundLogger = structlog.get_logger(__name__)

MODEL_ID = "claude-sonnet-4-6"
_OUTPUT_FLOOR = 4096
_OUTPUT_CEILING = 32768
_VALID_DIRECTIONS = frozenset(("increase", "decrease", "no_effect", "unclear"))

_SYSTEM_PROMPT = load_prompt("extract_v2")


def _scale_max_output_tokens(text: str) -> int:
    """Pick an output-token budget proportional to input length.

    Calibrated against v2-prompt extraction on real Elicit literature
    reviews. The v2 prompt emits 10 structured fields per claim, so output
    density is roughly 2.5x v1: dense lit reviews produce 300-370 output
    tokens per claim. The 12/30 multiplier (≈0.4 of input chars) covers
    both observed PDFs with margin:
      - PDF 2 (59,536 chars, ~53 claims): budget ≈23,800; need ≈16,000
      - PDF 1 (78,395 chars, ~75 claims): budget ≈31,300; need ≈27,000

    Floor is 4096 (preserves prior default for short inputs); ceiling is
    32768 (Sonnet 4.6's practical output limit, raised from 16384 after
    Phase 2.5.1 showed PDF 1 v2 truncating at the lower cap).
    """
    return min(_OUTPUT_CEILING, max(_OUTPUT_FLOOR, len(text) * 12 // 30))


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


def _str_or_none(value: object) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _parse_direction(value: object) -> ClaimDirection | None:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in _VALID_DIRECTIONS:
        return s  # type: ignore[return-value]
    return None


def _parse_confidence(value: object) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if 0.0 <= f <= 1.0:
        return f
    return None


_PUNCTUATION_FOLD: dict[str, str] = {
    chr(0x2014): "-",  # em-dash
    chr(0x2013): "-",  # en-dash
    chr(0x2212): "-",  # minus sign — PDFs often substitute for hyphen
    chr(0x2018): "'",  # left single quotation mark
    chr(0x2019): "'",  # right single quotation mark
    chr(0x201C): '"',  # left double quotation mark
    chr(0x201D): '"',  # right double quotation mark
    chr(0x00A0): " ",  # no-break space
    chr(0x2009): " ",  # thin space
    chr(0x200B): "",  # zero-width space
}


def _normalize_for_quote_match(text: str) -> str:
    """Fold Unicode variants that PDF text extraction commonly substitutes.

    pymupdf extracts em-dashes, smart quotes, and other typographic Unicode
    that the LLM may emit as their ASCII equivalents (or vice versa).
    Naked substring match rejects character-equivalent quotes; folding both
    sides through NFKC plus a small punctuation table makes the match
    robust to those substitutions while still rejecting genuine paraphrases.
    """
    normalized = unicodedata.normalize("NFKC", text)
    for src, dst in _PUNCTUATION_FOLD.items():
        normalized = normalized.replace(src, dst)
    return normalized


def _validate_source_quote(quote: object, source_text: str) -> str | None:
    """Return quote only when it appears in source_text (modulo Unicode quirks).

    LLMs sometimes paraphrase the quote rather than copy it. Downstream
    evidence anchoring relies on `quote in input_text` being a reliable
    contract, so paraphrased quotes are dropped (logged) rather than passed
    through. PDF text extraction often substitutes typographic Unicode
    (em-dashes, smart quotes, no-break spaces) that the LLM may not
    reproduce, so the substring check happens on a normalized form. The
    returned string is the LLM's original wording so downstream sees what
    the LLM actually said, not a normalized version.
    """
    if quote is None:
        return None
    q = str(quote).strip()
    if not q:
        return None
    if _normalize_for_quote_match(q) in _normalize_for_quote_match(source_text):
        return q
    # ASCII-safe preview so a non-UTF-8 console (e.g. Windows cp1252) does
    # not crash the whole extraction on log emit.
    preview = q[:80].encode("ascii", errors="replace").decode("ascii")
    logger.warning("extract_source_quote_not_in_input", quote_preview=preview)
    return None


def _extract_optional_fields(raw: dict[str, Any], source_text: str) -> dict[str, Any]:
    """Pull v2 structured fields from a raw LLM claim dict.

    Returns a kwargs dict suitable for `Claim(**dict)` construction. v1-style
    responses (without these fields) yield only None values, so the Claim
    defaults take over and nothing breaks.
    """
    return {
        "source_quote": _validate_source_quote(raw.get("source_quote"), source_text),
        "subject": _str_or_none(raw.get("subject")),
        "population": _str_or_none(raw.get("population")),
        "intervention": _str_or_none(raw.get("intervention")),
        "comparator": _str_or_none(raw.get("comparator")),
        "outcome": _str_or_none(raw.get("outcome")),
        "direction": _parse_direction(raw.get("direction")),
        "numeric_value": _str_or_none(raw.get("numeric_value")),
        "time_horizon": _str_or_none(raw.get("time_horizon")),
        "extraction_confidence": _parse_confidence(raw.get("extraction_confidence")),
    }


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
                    **_extract_optional_fields(raw, text),
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
                            **_extract_optional_fields(raw, text),
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
