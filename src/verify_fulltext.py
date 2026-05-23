"""Full-text verifier: BM25-passage path and numeric-augmented variant."""

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
    EvidenceQuality,
    PaperChunk,
    ProvenanceStep,
    ResolvedSource,
    VerificationResult,
    VerificationStatus,
    safe_verification_result,
)
from src.verify_prompts import (
    _FULLTEXT_SYSTEM_PROMPT,
    _VALID_STATUSES,
    MODEL_ID,
    _build_passages_block,
    _hash,
    _parse_cache_hit,
    _strip_fences,
)

logger: structlog.BoundLogger = structlog.get_logger(__name__)

# Cap each fallback passage at this many characters when populating
# ``source_passages`` from the BM25-selected chunks. The full text is already
# in ``provenance.jsonl`` (input_hash references it); this is just the visible
# audit-trail surface that lands in report.json and the copilot HTML report.
# 800 chars is enough to show ~150 words of context per passage — long enough
# to be useful, short enough to keep a 3-passage block under one screen.
_MAX_FALLBACK_PASSAGE_CHARS = 800


def _truncate_passage(text: str, limit: int = _MAX_FALLBACK_PASSAGE_CHARS) -> str:
    """Truncate a passage to ``limit`` characters with an ellipsis suffix.

    Pure function. Returns ``text`` unchanged if it already fits.
    """
    text = text.strip()
    if len(text) <= limit:
        return text
    # Try to break on a word boundary in the last 80 chars so we don't end
    # mid-word. Falls back to hard truncate if no boundary is found.
    boundary = text.rfind(" ", limit - 80, limit)
    cut = boundary if boundary > 0 else limit
    return text[:cut].rstrip() + "…"


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
        # Empty-passages contract: the pipeline now owns this routing decision
        # (``src/pipeline.py::verify_one_claim`` falls back to the abstract
        # verifier when BM25 returns no chunks). A defensive caller that still
        # passes an empty list lands here; we emit a deterministic
        # ``unverifiable + fulltext_unavailable`` verdict without an LLM call
        # so a degenerate input never leaks a confident verdict.
        ts_empty = time.time()
        empty_result = VerificationResult(
            status="unverifiable",
            explanation=(
                "verify_claim_fulltext invoked with empty passages — the pipeline "
                "owns empty-passages routing; this branch is a defensive no-LLM "
                "fallback. No verifier call was made."
            ),
            confidence=None,
            source_passages=[],
            source_section=None,
            fulltext_available=False,
            verification_depth="fulltext",
            retrieval_status="fulltext_unavailable",
            evidence_quality="no_evidence",
            retraction_status=source.retraction_status,
            unverifiable_reason="fulltext_unavailable",
        )
        empty_step = ProvenanceStep(
            step_id=str(uuid.uuid4()),
            claim_id=claim.claim_id,
            operation="verify",
            input_hash=_hash(repr((claim, source, passages))),
            output_hash=_hash(repr(empty_result)),
            model_id=None,
            timestamp=ts_empty,
            tokens_in=None,
            tokens_out=None,
            cache_hit=None,
            confidence=None,
            unverifiable_reason="fulltext_unavailable",
        )
        return empty_result, empty_step

    ts = time.time()
    effective_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=effective_key)

    # Inject source_quote focal anchor before the passages block when present.
    # Same contract as verify.py 3.1: user message only, system prompt unchanged.
    if claim.source_quote is not None:
        user_message = (
            f"<claim>{claim.claim_text}</claim>\n"
            f"<source_quote>{claim.source_quote}</source_quote>\n" + _build_passages_block(passages)
        )
    else:
        user_message = f"<claim>{claim.claim_text}</claim>\n" + _build_passages_block(passages)

    response = client.messages.create(
        model=model_id,
        # max_tokens=2048: full-text verifier may quote up to 3 source_passages
        # plus a multi-sentence explanation. Observed parse errors on claim 003
        # were caused by truncated JSON when 1024 was insufficient.
        max_tokens=2048,
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
        llm_quoted_passages = (
            [str(p) for p in raw_passages] if isinstance(raw_passages, list) else []
        )
        source_section_raw = parsed.get("source_section")
        source_section = str(source_section_raw) if source_section_raw else None

        # CTran-transparency fallback: when the LLM returns no quoted passages
        # The model chose not to quote; populate source_passages with the BM25 chunks shown.
        # Preserves the audit trail: auditor sees what the verifier examined, not just quotes.
        if llm_quoted_passages:
            source_passages = llm_quoted_passages
            evidence_quality: EvidenceQuality = "quoted_passage"
        else:
            source_passages = [_truncate_passage(p.text) for p in passages]
            evidence_quality = "passages_searched_no_quote"

        # Handle null confidence on the fulltext path too — if the LLM ever
        # returns status="unverifiable" with confidence=null (forward compat,
        # spec emission-gates §7.11: "all emission sites must handle this case"),
        # float(parsed["confidence"]) would crash. verify.py's parse boundary
        # already handles this; mirror the pattern here.
        raw_confidence_ft = parsed.get("confidence")
        confidence_val_ft: float | None = (
            None if raw_confidence_ft is None else float(raw_confidence_ft)
        )
        # Route through safe_verification_result to apply the extraction_confidence
        # cap (Gate 1) consistently across all verifier paths (C2).
        # Gate 2 (evidence-depth downgrade) does NOT fire here because
        # evidence_quality is fulltext-grade ({quoted_passage,
        # passages_searched_no_quote}), which is NOT in _INSUFFICIENT_EVIDENCE_SET.
        # Gate 1 (extraction_confidence cap) fires independently of evidence depth:
        # a low-extraction-confidence claim on fulltext evidence is still capped to
        # partially_supported. This is the correct behaviour per spec edge case 7.5.
        # If status=="unverifiable" the schema invariant requires confidence is None,
        # which the parse above ensures; safe_verification_result preserves this.
        result = safe_verification_result(
            status=status,
            explanation=str(parsed["explanation"]),
            confidence=None if status == "unverifiable" else confidence_val_ft,
            source_passages=source_passages,
            source_section=source_section,
            fulltext_available=True,
            verification_depth="fulltext",
            retrieval_status="passage_found",
            evidence_quality=evidence_quality,
            retraction_status=source.retraction_status,
            extraction_confidence=claim.extraction_confidence,
            claim_text=claim.claim_text,
            unverifiable_reason="insufficient_evidence_depth",
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.error(
            "verify_fulltext_parse_error",
            claim_id=claim.claim_id,
            raw_response=response_text[:200],
            error=str(exc),
        )
        # Same audit-trail fallback as the success path: surface the BM25
        # passages that were considered, so a parse error does not erase
        # the evidence the verifier saw.
        result = VerificationResult(
            status="not_addressed",
            explanation="Parse error.",
            confidence=0.0,
            source_passages=[_truncate_passage(p.text) for p in passages],
            source_section=None,
            fulltext_available=True,
            verification_depth="fulltext",
            retrieval_status="passage_found",
            evidence_quality="passages_searched_no_quote",
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
        unverifiable_reason=result.unverifiable_reason,
    )

    return result, step
