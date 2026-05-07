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
from anthropic.types import TextBlock, Usage

from src.models import (
    Claim,
    PaperChunk,
    ProvenanceStep,
    ResolvedSource,
    VerificationResult,
    VerificationStatus,
)

logger: structlog.BoundLogger = structlog.get_logger(__name__)

MODEL_ID = "claude-sonnet-4-6"

_SYSTEM_PROMPT = """\
You are a scientific claim verifier. Your task is to determine whether a source abstract supports, contradicts, or does not address a given scientific claim. The user is auditing whether the cited source actually backs the claim it is attached to, so the distinction between "wrong citation" and "wrong topic" matters.

Verification statuses:
- supported: The abstract explicitly provides evidence consistent with the claim's core assertion.
- unsupported: Use this when EITHER (a) the abstract explicitly contradicts the claim, OR (b) the abstract addresses the topic of the claim but does not contain the specific content the claim asserts (absence-of-support on an on-topic source). Case (b) is the critical "wrong-citation" signal — the source is about the right area but is the wrong citation for THIS claim.
- not_addressed: The abstract is about a fundamentally different subject; the topic the claim is making a statement about is not the subject of the source. This is the "wrong topic / off-domain" signal, not "evidence is unclear".
- partially_supported: The abstract provides some support for the claim but not complete support — the claim asserts a stronger effect than the abstract reports, the abstract reports a related quantity rather than the specific one claimed, the abstract's findings are mixed, or the abstract supports the directional claim but not the magnitude.

The unsupported vs not_addressed distinction:
- If the source is a study of dermal interstitial fluid lactate, and the claim is about dermal interstitial fluid lactate, but the specific value or relationship the claim asserts is not present in the abstract → unsupported (case b).
- If the source is a study of amino acid immunology, and the claim is about lactate concentration ratios → not_addressed.
- In other words: not_addressed = "this is the wrong source domain entirely". unsupported = "this is the right source domain but the source does not contain the specific evidence the claim asserts".

Guidelines:
- Base your verdict ONLY on the abstract text provided. Do not use outside knowledge.
- Do not default to not_addressed when the source is on-topic but lacks the specific evidence — that case is unsupported.
- Reserve not_addressed for genuine topic mismatch. A source that addresses the claim's general subject but is silent on the specific assertion is unsupported, not not_addressed.
- Confidence: 0.9-1.0 for clear cases, 0.6-0.8 for moderate certainty, 0.4-0.6 for uncertain.

Return ONLY a JSON object:
{
  "status": "supported|unsupported|not_addressed|partially_supported",
  "explanation": "One or two sentences explaining your verdict, citing specific evidence from the abstract. When the verdict is unsupported, state explicitly whether the source contradicts the claim or is silent on the specific assertion.",
  "confidence": 0.85
}

Your response must be valid JSON only — no explanatory text, no markdown code blocks, no additional commentary.

Remember:
- "supported" requires explicit positive evidence in the abstract.
- "unsupported" covers both contradiction and on-topic absence-of-support.
- "not_addressed" is reserved for off-topic / wrong-domain abstracts only.
- "partially_supported" is for cases where the abstract provides some but not complete support, or supports the direction but not the magnitude.
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
    evidence_quality="no_evidence",
)

_PARSE_ERROR_RESULT = VerificationResult(
    status="not_addressed",
    explanation="Parse error.",
    confidence=0.0,
    evidence_quality="no_evidence",
)

_VALID_STATUSES: set[str] = {"supported", "unsupported", "not_addressed", "partially_supported"}


_FULLTEXT_SYSTEM_PROMPT = """\
You are a scientific claim verifier operating in full-text mode. Your task is to determine whether the provided source passages support, contradict, or do not address a given scientific claim. The user is auditing whether the cited source actually backs the claim it is attached to, so the distinction between "wrong citation" and "wrong topic" matters.

You will receive a claim and a set of passages selected from the source paper using BM25 relevance ranking. Each passage is labeled with the section it came from (introduction, methods, results, discussion, or other) so you can weigh evidence appropriately:
- Claims about study design should be verified against Methods passages.
- Claims about quantitative outcomes should be verified against Results passages.
- Interpretive or causal claims should be verified against Discussion passages.
- Background statements may be verified against Introduction passages.

Verification statuses:
- supported: At least one passage explicitly provides evidence consistent with the claim's core assertion. Quote the exact sentence(s) from the passage that justify this verdict.
- unsupported: Use this when EITHER (a) at least one passage explicitly contradicts the claim, OR (b) the passages address the topic of the claim but do not contain the specific content the claim asserts (absence-of-support on on-topic passages). Case (b) is the critical "wrong-citation" signal — the source is about the right area but is the wrong citation for THIS specific claim.
- not_addressed: The passages are about a fundamentally different subject; the topic the claim is making a statement about is not what these passages cover. This is the "wrong topic / off-domain" signal, not "evidence is unclear".
- partially_supported: The passages provide some support for the claim but not complete support — the claim asserts a stronger effect than the passages report, the passages report a related quantity rather than the specific one claimed, the passages' findings are mixed, or the passages support the directional claim but not the magnitude.

The unsupported vs not_addressed distinction:
- If the passages are from a paper on dermal interstitial fluid lactate and the claim is about dermal ISF lactate, but the specific value or relationship the claim asserts is not in the passages → unsupported (case b).
- If the passages are from a paper on cosmology and the claim is about lactate kinetics → not_addressed.
- In other words: not_addressed = "wrong source domain entirely". unsupported = "right source domain but the passages do not contain the specific evidence the claim asserts".

Guidelines:
- Base your verdict ONLY on the provided passages. Do not use outside knowledge of the paper or domain.
- Do not default to not_addressed when the passages are on-topic but lack the specific evidence — that case is unsupported.
- Reserve not_addressed for genuine topic mismatch (e.g., the BM25 selector returned passages from a paper that is in a totally different field from the claim).
- Identify the section that contains the strongest evidence for your verdict (use the section attribute of the most relevant passage). Lowercase: "introduction", "methods", "results", "discussion", or "other".
- Extract verbatim sentences from the passages — at most three — into source_passages. Do NOT paraphrase. If the passages contain no relevant evidence, return an empty list.
- Confidence: 0.9-1.0 for clear-cut cases with explicit textual evidence, 0.6-0.8 for moderate certainty, 0.4-0.6 for uncertain verdicts.

Return ONLY a JSON object with this exact schema:
{
  "status": "supported|unsupported|not_addressed|partially_supported",
  "explanation": "One or two sentences explaining your verdict, citing specific evidence from the passages. When the verdict is unsupported, state explicitly whether the passages contradict the claim or are silent on the specific assertion.",
  "confidence": 0.85,
  "source_passages": ["exact sentence quoted from a passage", "another exact sentence"],
  "source_section": "results"
}

Your response must be valid JSON only — no explanatory text outside the JSON, no markdown code blocks, no additional commentary.

Reminder of how to weigh evidence:
- "supported" requires explicit textual evidence in at least one passage.
- "unsupported" covers both contradiction and on-topic absence-of-support.
- "not_addressed" is reserved for off-topic / wrong-domain passages only.
- "partially_supported" is for cases where the evidence is real but qualified, mixed, or weaker than the claim suggests.
- source_passages must contain verbatim quotes pulled directly from the passages provided. Never paraphrase or invent text.
- source_section should match the section attribute of the passage(s) you cite. If you cite multiple passages from different sections, choose the one whose section best characterizes the evidence (Results for outcome data, Methods for design, Discussion for interpretation).
- Confidence should reflect your certainty in the verdict, not the strength or specificity of the claim itself.
"""


def _build_passages_block(passages: list[PaperChunk]) -> str:
    parts: list[str] = []
    for chunk in passages:
        parts.append(f'<passage section="{chunk.section}">\n{chunk.text}\n</passage>')
    return "<passages>\n" + "\n".join(parts) + "\n</passages>"


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
        max_tokens=1024,
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
