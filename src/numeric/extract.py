"""LLM-driven extraction of numeric assertions from claim text.

Returns structured NumericAssertion records. Backed by the Anthropic SDK with
prompt caching on the system prompt (>1024 tokens).

Span anchoring is deterministic: after the LLM emits ``raw_text`` for each
assertion, this module derives ``span_start`` / ``span_end`` via
``claim_text.find(raw_text)``. The LLM is not asked for character offsets —
its job is to identify which substrings matter; the Python side owns
positions and sentence ids.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from typing import Any

import anthropic
import structlog
from anthropic.types import TextBlock, Usage

from src.models import ProvenanceStep
from src.numeric.checks import NumericAssertion, NumericRole
from src.prompt_guard import PROMPT_INJECTION_GUARD

logger: structlog.BoundLogger = structlog.get_logger(__name__)

MODEL_ID = "claude-sonnet-4-6"

_VALID_ROLES: set[str] = {"primary", "ci_low", "ci_high", "comparator", "p_value", "n"}

_SYSTEM_PROMPT = (
    PROMPT_INJECTION_GUARD
    + "\n"
    + """\
You are a structured-data extractor for scientific numeric assertions.

Your task: given a scientific claim that may contain odds ratios, confidence intervals, p-values, sample sizes, percentages, or other quantitative reports, extract each numeric assertion as a structured record so a downstream deterministic engine can run consistency checks.

For each numeric value in the claim, emit one record with these fields:
- raw_text: the exact substring from the claim (e.g. "OR 40.53", "95% CI 23.58-73.71", "p < 0.0001")
- value: the numeric value as a float (for ranges, emit one record per endpoint)
- unit: a short unit label or null (examples: "%", "nM", "uM", "mg/kg"; null for dimensionless ratios like odds ratios, p-values, sample sizes)
- role: one of:
    "primary"     — the headline statistic (the OR itself, the main effect size, the headline percentage)
    "ci_low"      — lower bound of a confidence interval
    "ci_high"     — upper bound of a confidence interval
    "comparator"  — a value being compared against the primary (e.g. the control percentage when claim says "39% vs 11.5%")
    "p_value"     — a reported p-value
    "n"           — a reported sample size
- context: a short phrase from the claim describing what the value refers to (e.g. "odds ratio for ARM in A+T- vs A-T-")

CRITICAL RULES:
- Extract numbers verbatim. Do not paraphrase, round, or normalize units.
- For confidence intervals, ALWAYS emit both ci_low and ci_high as separate records, in that order.
- For percentages reported with comparator (e.g. "77.5% vs 7.8%"), emit the first as "primary" and the second as "comparator".
- If a claim contains no numeric assertions, return an empty list.
- If a number's role is ambiguous, prefer "primary".
- Preserve sign and decimal precision exactly as written in the claim.

Return ONLY a JSON object with this exact schema:
{
  "assertions": [
    {
      "raw_text": "...",
      "value": 40.53,
      "unit": null,
      "role": "primary",
      "context": "..."
    }
  ]
}

Your response must be valid JSON only — no explanatory text outside the JSON, no markdown code blocks, no additional commentary.

Examples of correct extraction:

Claim: "ARM were 77.5% in A+T- vs 7.8% in A-T- (OR 40.53, 95% CI 23.58-73.71)"
Extraction:
{"assertions": [
  {"raw_text": "77.5%", "value": 77.5, "unit": "%", "role": "primary", "context": "ARM percentage in A+T- group"},
  {"raw_text": "7.8%", "value": 7.8, "unit": "%", "role": "comparator", "context": "ARM percentage in A-T- group"},
  {"raw_text": "OR 40.53", "value": 40.53, "unit": null, "role": "primary", "context": "odds ratio ARM A+T- vs A-T-"},
  {"raw_text": "23.58", "value": 23.58, "unit": null, "role": "ci_low", "context": "95% CI lower bound for OR 40.53"},
  {"raw_text": "73.71", "value": 73.71, "unit": null, "role": "ci_high", "context": "95% CI upper bound for OR 40.53"}
]}

Claim: "TREM2 is implicated in microglial activation."
Extraction:
{"assertions": []}

Reminder: emit JSON only, no commentary, no markdown.
"""
)


_SENTENCE_BOUNDARY: re.Pattern[str] = re.compile(r"[.;!?](?=\s|$)")


def _segment_sentences(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end_exclusive)`` char offsets for each sentence in ``text``.

    Splits on ``.``, ``;``, ``?``, ``!`` followed by whitespace or end-of-string.
    Pure deterministic function — used by ``_derive_sentence_id`` to anchor
    a span to a sentence number without consulting the LLM.
    """
    segments: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        end = match.end()
        segments.append((start, end))
        j = end
        while j < len(text) and text[j].isspace():
            j += 1
        start = j
    if start < len(text):
        segments.append((start, len(text)))
    return segments


def _derive_sentence_id(offset: int | None, segments: list[tuple[int, int]]) -> int | None:
    """Return 0-indexed sentence containing ``offset`` or ``None``."""
    if offset is None:
        return None
    for i, (s, e) in enumerate(segments):
        if s <= offset < e:
            return i
    return None


def _derive_span(claim_text: str, raw_text: str) -> tuple[int | None, int | None]:
    """Locate ``raw_text`` inside ``claim_text`` deterministically.

    Returns ``(span_start, span_end)`` when ``raw_text`` appears exactly
    once. Returns ``(None, None)`` when the substring is absent or
    appears multiple times — the span-anchored pairing path then sees
    ``span_start is None`` and the engine falls back to substring/window
    matching. Pure function.
    """
    if not raw_text:
        return (None, None)
    first = claim_text.find(raw_text)
    if first == -1:
        return (None, None)
    second = claim_text.find(raw_text, first + 1)
    if second != -1:
        return (None, None)
    return (first, first + len(raw_text))


def _hash(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def _strip_fences(text: str) -> str:
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


def extract_numeric_assertions(
    claim_text: str,
    *,
    claim_id: str = "__numeric_extract__",
    model_id: str = MODEL_ID,
    api_key: str | None = None,
) -> tuple[list[NumericAssertion], ProvenanceStep]:
    """Extract numeric assertions from a claim string via Claude.

    Returns (assertions, provenance_step). assertions is empty if the claim
    contains no numerics or if the LLM response is malformed (graceful fallback).
    Never raises.
    """
    ts = time.time()
    effective_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=effective_key)

    user_message = f"<claim>{claim_text}</claim>"

    response = client.messages.create(
        model=model_id,
        max_tokens=1024,
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
        "numeric_extract_llm_call",
        model_id=model_id,
        claim_id=claim_id,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cache_hit=cache_hit,
    )

    first_block = response.content[0] if response.content else None
    response_text = first_block.text if isinstance(first_block, TextBlock) else ""

    assertions: list[NumericAssertion] = []
    sentence_segments = _segment_sentences(claim_text)
    try:
        parsed: dict[str, Any] = json.loads(_strip_fences(response_text))
        raw_list = parsed.get("assertions", [])
        if not isinstance(raw_list, list):
            raise ValueError("assertions field is not a list")
        for entry in raw_list:
            if not isinstance(entry, dict):
                continue
            role_raw = str(entry.get("role", ""))
            if role_raw not in _VALID_ROLES:
                continue
            try:
                value = float(entry["value"])
            except (KeyError, TypeError, ValueError):
                continue
            role: NumericRole = role_raw  # type: ignore[assignment]
            unit_raw = entry.get("unit")
            unit: str | None = str(unit_raw) if unit_raw not in (None, "") else None
            raw_text = str(entry.get("raw_text", ""))
            span_start, span_end = _derive_span(claim_text, raw_text)
            sentence_id = _derive_sentence_id(span_start, sentence_segments)
            assertions.append(
                NumericAssertion(
                    raw_text=raw_text,
                    value=value,
                    unit=unit,
                    role=role,
                    context=str(entry.get("context", "")),
                    span_start=span_start,
                    span_end=span_end,
                    sentence_id=sentence_id,
                )
            )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.error(
            "numeric_extract_parse_error",
            claim_id=claim_id,
            raw_response=response_text[:200],
            error=str(exc),
        )
        assertions = []

    step = ProvenanceStep(
        step_id=str(uuid.uuid4()),
        claim_id=claim_id,
        operation="numeric_extract",
        input_hash=_hash(repr(claim_text)),
        output_hash=_hash(repr(assertions)),
        model_id=model_id,
        timestamp=ts,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cache_hit=cache_hit,
        confidence=None,
    )

    return assertions, step
