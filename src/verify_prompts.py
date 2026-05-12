"""Prompt constants, pure helpers, and shared constants for the verify layer.

Nothing in this module makes LLM calls or performs I/O.  All symbols are
imported by src/verify.py; callers that previously imported from src.verify
continue to work via the re-exports defined there.
"""

from __future__ import annotations

import hashlib
import time
import uuid

import structlog
from anthropic.types import Usage

from src.models import (
    Claim,
    PaperChunk,
    ProvenanceStep,
    ResolvedSource,
    UnverifiableReason,
    VerificationResult,
    safe_verification_result,
)
from src.prompts import load_prompt

logger: structlog.BoundLogger = structlog.get_logger(__name__)

MODEL_ID = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# System prompts (all > 1024 tokens → prompt-cached at call sites).
# Bodies live as markdown in src/prompts/{name}_v{N}.md; load_prompt()
# prepends PROMPT_INJECTION_GUARD. See src/prompts/__init__.py.
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = load_prompt("verify_v1")
_FULLTEXT_SYSTEM_PROMPT = load_prompt("verify_fulltext_v1")
_TITLE_ONLY_SYSTEM_PROMPT = load_prompt("verify_title_only_v1")
_CITING_CONTEXT_SYSTEM_PROMPT = load_prompt("verify_citing_context_v1")

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_VALID_STATUSES: set[str] = {
    "supported",
    "unsupported",
    "not_addressed",
    "partially_supported",
    "unverifiable",
}

_TITLE_ONLY_MIN_TITLE_LENGTH = 20
_TITLE_ONLY_MAX_CONFIDENCE = 0.7

_CITING_CONTEXT_WINDOW_CHARS = 600
_CITING_CONTEXT_MAX_CONFIDENCE = 0.6  # weaker than title-only: internal consistency only

# ---------------------------------------------------------------------------
# Short-circuit constants (used by verify_claim when source not found)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Pure helper functions (no I/O, no side effects)
# ---------------------------------------------------------------------------


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


def _build_passages_block(passages: list[PaperChunk]) -> str:
    parts: list[str] = []
    for chunk in passages:
        parts.append(f'<passage section="{chunk.section}">\n{chunk.text}\n</passage>')
    return "<passages>\n" + "\n".join(parts) + "\n</passages>"


def _extract_citing_context_window(
    text: str,
    claim_text: str,
    *,
    window_chars: int = _CITING_CONTEXT_WINDOW_CHARS,
) -> str:
    """Locate the claim within the citing paper text and return a ±window slice.

    Searches for the first ~80 chars of the claim text. Falls back to the
    full text (truncated to 2x window) if the claim is not found.
    """
    needle = claim_text[:80].strip()
    if not needle:
        return text[: 2 * window_chars]
    idx = text.find(needle)
    if idx < 0:
        # Try a shorter prefix
        short = claim_text[:40].strip()
        idx = text.find(short) if short else -1
    if idx < 0:
        return text[: 2 * window_chars]
    start = max(0, idx - window_chars)
    end = min(len(text), idx + len(claim_text) + window_chars)
    return text[start:end]


__all__ = [
    "MODEL_ID",
    "_CITING_CONTEXT_MAX_CONFIDENCE",
    "_CITING_CONTEXT_SYSTEM_PROMPT",
    "_CITING_CONTEXT_WINDOW_CHARS",
    "_FULLTEXT_SYSTEM_PROMPT",
    "_PARSE_ERROR_RESULT",
    "_SHORT_CIRCUIT_RESULT",
    "_SYSTEM_PROMPT",
    "_TITLE_ONLY_MAX_CONFIDENCE",
    "_TITLE_ONLY_MIN_TITLE_LENGTH",
    "_TITLE_ONLY_SYSTEM_PROMPT",
    "_VALID_STATUSES",
    "UnverifiableReason",
    "_build_passages_block",
    "_extract_citing_context_window",
    "_hash",
    "_make_short_circuit_step",
    "_parse_cache_hit",
    "_strip_fences",
    "safe_verification_result",
]
