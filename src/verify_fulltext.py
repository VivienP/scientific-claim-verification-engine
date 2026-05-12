"""Full-text verifier: BM25-passage path and numeric-augmented variant."""

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
    EvidenceQuality,
    PaperChunk,
    ProvenanceStep,
    ResolvedSource,
    VerificationResult,
    VerificationStatus,
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
        from src.verify import verify_claim

        abstract_result, step = verify_claim(claim, source, model_id=model_id, api_key=api_key)
        # A2 fix: was fulltext_available=True (lie) and retrieval_status="no_passage_found"
        # (misleading). After A2: verify_claim routes through safe_verification_result,
        # so abstract_result already has the correct status (unverifiable if the LLM
        # tried to emit supported/unsupported on abstract_only evidence).
        # We only override metadata fields here -- NOT status/confidence/evidence_quality.
        # dataclasses.replace calls __post_init__, but since we leave status/confidence/
        # evidence_quality unchanged (already valid from verify_claim), no invariant
        # violation occurs.
        # F1: when the inner verify_claim downgraded to unverifiable, the
        # proximate cause for THIS outer fallback path is "we tried to fetch
        # fulltext and couldn't" — override the inner reason
        # ("numeric_claim_abstract_only") with "fulltext_unavailable" because
        # for the outer caller, the missing-fulltext is the more actionable
        # framing. The explanation from the helper already captured the
        # original LLM verdict; we just refine the reason classification.
        from src.models import UnverifiableReason

        outer_reason: UnverifiableReason | None = (
            "fulltext_unavailable" if abstract_result.status == "unverifiable" else None
        )
        result = dataclasses.replace(
            abstract_result,
            fulltext_available=False,  # was: True (BUG)
            verification_depth="abstract",
            retrieval_status="fulltext_unavailable",  # was: "no_passage_found"
            retraction_status=source.retraction_status,
            unverifiable_reason=(
                outer_reason if outer_reason is not None else abstract_result.unverifiable_reason
            ),
        )
        return (
            result,
            dataclasses.replace(
                step,
                input_hash=_hash(repr((claim, source, passages))),
                output_hash=_hash(repr(result)),
                confidence=result.confidence,
                unverifiable_reason=outer_reason,
            ),
        )

    ts = time.time()
    effective_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=effective_key)

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
        # (typically because the verdict is unsupported/not_addressed and the
        # model chose not to quote), populate source_passages with the BM25
        # passages that WERE shown to the verifier. This preserves the audit
        # trail — an auditor inspecting the run can see which passages the
        # verifier examined, not just which it agreed with. Phase A.2 fix
        # (was the dominant CTran-failure mode at 50% of failures across the
        # 5-document benchmark; see reports/phase_a2/ctran_failure_matrix.md).
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
        # Direct VerificationResult construction here is permitted by
        # .claude/rules/no-confident-verdict-without-evidence.md: the
        # evidence_quality is fulltext-grade ({quoted_passage,
        # passages_searched_no_quote}) and is NOT in the helper's
        # INSUFFICIENT set, so safe_verification_result would pass it
        # through unchanged. Routing through the helper would be a no-op.
        # If status=="unverifiable" the schema invariant requires
        # confidence is None, which the parse above ensures.
        result = VerificationResult(
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
    )

    return result, step
