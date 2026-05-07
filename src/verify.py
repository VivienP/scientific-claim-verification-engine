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
    ResolvedSourceSet,
    VerificationResult,
    VerificationStatus,
)

logger: structlog.BoundLogger = structlog.get_logger(__name__)

MODEL_ID = "claude-sonnet-4-6"

_SYSTEM_PROMPT = """\
You are a scientific claim verifier. Your task is to determine whether a source abstract supports, contradicts, or does not address a given scientific claim. The user is auditing whether the cited source actually backs the claim it is attached to.

Verification statuses:
- supported: The abstract explicitly provides evidence consistent with the claim's core assertion AND the specific magnitude / value / direction the claim asserts.
- unsupported: Use this when ANY of: (a) the abstract explicitly contradicts the claim; (b) the abstract addresses the topic of the claim but does not contain the specific content the claim asserts (on-topic absence-of-support); OR (c) the abstract is on a different scientific subject altogether and therefore cannot substantiate the claim's specific assertion (off-topic absence-of-support).
- not_addressed: Reserved for the rare case where no source content is provided at all. You will not normally encounter this case — assume abstract content is present.
- partially_supported: The abstract provides some support but not complete support (see the partial-support rules below).

Clause A — collapse off-topic into unsupported:
When you receive any abstract content, the verdict must be one of `supported` / `partially_supported` / `unsupported`. An off-topic source whose subject is unrelated to the claim is `unsupported` (the cited source does not contain evidence for the specific claim), NOT `not_addressed`. The annotator and audit consumer use `unsupported` for both "right paper, wrong specific evidence" and "wrong paper entirely". Do not split the two into `unsupported` vs `not_addressed`.

Clause B — partial when source covers only part of the claimed quantitative space (apply BEFORE deciding `supported`):

Two directions, both yield `partially_supported`:

(B.1) Claim asserts a RANGE, source reports a SINGLE POINT inside that range.
The source proves the claim is plausible at one point but does not establish the full range. This is `partially_supported`, NOT `supported`, even when the point is squarely in the middle of the claimed range.
- Example: Claim "skin lactate is between 1 and 2.5 mmol/L"; abstract reports "skin lactate = 1.74 mmol/L (n=11)" → `partially_supported`. The single mean is consistent with the range but cannot establish it.
- Example: Claim "depth is 0.6-1.5 mm depending on body site"; abstract reports "1-1.5 mm below the skin surface" → `partially_supported`. Source supports the upper part of the range but not the 0.6 lower bound.

(B.2) Claim asserts a POINT VALUE, source reports a CENTRAL ESTIMATE with explicit uncertainty (95% CI, IQR, SD, range), and the claimed value falls inside the uncertainty band even when differing from the central estimate.
- Example: Claim "lag time is approximately 10 minutes"; abstract reports "lag = 5 min (IQR -4 to 11)" → `partially_supported`. 10 falls inside [-4, 11] despite the central estimate being 5.
- Use `unsupported` only when the claimed value is outside any reported band (e.g., claim "10 min", source "5 min ± 1") or when the source explicitly contradicts the direction.

Clause B applies whenever EITHER direction matches, regardless of the rest of the prompt. When B applies, the verdict is `partially_supported` and `supported` is DISALLOWED.

Clause C — trajectory vs snapshot (apply BEFORE deciding `supported`):

When the claim asserts a directional CHANGE (increase, decrease, slope, trajectory) over a condition (time, intensity, dose, group), and the source reports only static or aggregate values for that quantity (no temporal/intensity decomposition), choose `partially_supported`. A high correlation, mean, or aggregate r-value DOES NOT establish the asserted change.
- Example: Claim "correlation between arterial and capillary lactate increases during exercise"; abstract reports "r = 0.858 to 0.983 across the incremental treadmill protocol" → `partially_supported`. The high r supports a correlation but does NOT establish that it INCREASES from rest to exercise (the source did not compare rest vs exercise side-by-side).
- Example: Claim "X declines over time"; source "mean X = 5.2 across all timepoints" → `partially_supported`.

When C applies, the verdict is `partially_supported` and `supported` is DISALLOWED.

General guidelines:
- Base your verdict ONLY on the abstract text provided. Do not use outside knowledge.
- Partial-support precedence: when the abstract supports ANY concrete part of a multi-part or numeric claim, `partially_supported` takes precedence over both `supported` AND `unsupported`. Never output `unsupported` when the abstract clearly supports a sub-claim, an endpoint of a range, a direction, or a related quantity.
- For range, threshold, lag-time, ratio, and depth claims, default to `partially_supported` when the abstract supports one endpoint, direction, central estimate, related quantity, or qualitative relationship but not the exact magnitude or all conditions in the claim.
- Confidence: 0.9-1.0 for clear cases, 0.6-0.8 for moderate certainty, 0.4-0.6 for uncertain.

Return ONLY a JSON object:
{
  "status": "supported|unsupported|partially_supported",
  "explanation": "One or two sentences explaining your verdict, citing specific evidence from the abstract. When the verdict is unsupported, state explicitly whether the source contradicts the claim, is silent on the specific assertion, or is on a different scientific subject.",
  "confidence": 0.85
}

Your response must be valid JSON only — no explanatory text, no markdown code blocks, no additional commentary.

Remember:
- "supported" requires explicit positive evidence in the abstract that fully matches the claim's specific assertion (including magnitude, direction, and conditions).
- "unsupported" covers contradiction, on-topic absence-of-support, AND off-topic sources. Do not output `not_addressed` when an abstract is provided.
- "partially_supported" applies when: (i) the abstract supports the direction but not the magnitude; (ii) the claimed value falls inside the source's uncertainty band but differs from the central estimate; (iii) the source reports only static values for a directional/trajectory claim; (iv) the abstract supports some but not all parts of a compound claim.
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
- supported: At least one passage explicitly provides evidence consistent with the claim's core assertion AND the specific magnitude / value / direction the claim asserts. Quote the exact sentence(s) from the passage that justify this verdict.
- unsupported: Use this when ANY of: (a) at least one passage explicitly contradicts the claim; (b) the passages address the topic of the claim but do not contain the specific content the claim asserts (on-topic absence-of-support); OR (c) the passages are on a different scientific subject altogether and therefore cannot substantiate the claim's specific assertion (off-topic absence-of-support).
- not_addressed: Reserved for the rare case where no passage content is provided at all. You will not normally encounter this case — assume passages are present.
- partially_supported: The passages provide some support but not complete support (see the partial-support rules below).

Clause A — collapse off-topic into unsupported:
When you receive any passage content, the verdict must be one of `supported` / `partially_supported` / `unsupported`. Off-topic passages whose subject is unrelated to the claim are `unsupported` (the cited source does not contain evidence for the specific claim), NOT `not_addressed`. Do not split "right paper, wrong specific evidence" and "wrong paper entirely" into two different verdicts.

Clause B — partial when source covers only part of the claimed quantitative space (apply BEFORE deciding `supported`):

Two directions, both yield `partially_supported`:

(B.1) Claim asserts a RANGE, passages report a SINGLE POINT inside that range.
The passage proves the claim is plausible at one point but does not establish the full range. This is `partially_supported`, NOT `supported`.
- Example: Claim "skin lactate is between 1 and 2.5 mmol/L"; passage reports "skin lactate = 1.74 mmol/L" → `partially_supported`.
- Example: Claim "depth is 0.6-1.5 mm depending on body site"; passage reports "1-1.5 mm below the skin surface" → `partially_supported`.

(B.2) Claim asserts a POINT VALUE, passages report a CENTRAL ESTIMATE with explicit uncertainty (95% CI, IQR, SD, range), and the claimed value falls inside the uncertainty band even when differing from the central estimate.
- Example: Claim "lag is approximately 10 minutes"; passage reports "lag = 5 min (IQR -4 to 11)" → `partially_supported`.
- Use `unsupported` only when the claimed value is outside any reported band, or when passages explicitly contradict the direction.

Clause B applies whenever EITHER direction matches. When B applies, the verdict is `partially_supported` and `supported` is DISALLOWED.

Clause C — trajectory vs snapshot (apply BEFORE deciding `supported`):

When the claim asserts a directional CHANGE over a condition (time, intensity, dose, group), and the passages report only static or aggregate values for that quantity (no temporal/intensity decomposition), choose `partially_supported`. A high correlation, mean, or aggregate r-value DOES NOT establish the asserted change.
- Example: Claim "correlation between arterial and capillary lactate increases during exercise"; passage reports "r = 0.858 to 0.983 across the incremental treadmill protocol" → `partially_supported`. The high r supports a correlation but does NOT establish that it INCREASES from rest to exercise.

When C applies, the verdict is `partially_supported` and `supported` is DISALLOWED.

General guidelines:
- Base your verdict ONLY on the provided passages. Do not use outside knowledge of the paper or domain.
- Partial-support precedence: when the passages support ANY concrete part of a multi-part or numeric claim, `partially_supported` takes precedence over both `supported` AND `unsupported`. Never output `unsupported` when the passages clearly support a sub-claim, an endpoint of a range, a direction, or a related quantity.
- For range, threshold, lag-time, ratio, and depth claims, default to `partially_supported` when the passages support one endpoint, direction, central estimate, related quantity, or qualitative relationship but not the exact magnitude or all conditions in the claim.
- Identify the section that contains the strongest evidence for your verdict (use the section attribute of the most relevant passage). Lowercase: "introduction", "methods", "results", "discussion", or "other".
- Extract verbatim sentences from the passages — at most three — into source_passages. Do NOT paraphrase. If the passages contain no relevant evidence, return an empty list.
- Confidence: 0.9-1.0 for clear-cut cases with explicit textual evidence, 0.6-0.8 for moderate certainty, 0.4-0.6 for uncertain verdicts.

Return ONLY a JSON object with this exact schema:
{
  "status": "supported|unsupported|partially_supported",
  "explanation": "One or two sentences explaining your verdict, citing specific evidence from the passages. When the verdict is unsupported, state explicitly whether the passages contradict the claim, are silent on the specific assertion, or are on a different scientific subject.",
  "confidence": 0.85,
  "source_passages": ["exact sentence quoted from a passage", "another exact sentence"],
  "source_section": "results"
}

Your response must be valid JSON only — no explanatory text outside the JSON, no markdown code blocks, no additional commentary.

Reminder of how to weigh evidence:
- "supported" requires explicit textual evidence in at least one passage that fully matches the claim's specific assertion (including magnitude, direction, and conditions).
- "unsupported" covers contradiction, on-topic absence-of-support, AND off-topic passages. Do not output `not_addressed` when passages are provided.
- "partially_supported" applies when: (i) the passages support the direction but not the magnitude; (ii) the claimed value falls inside the source's uncertainty band but differs from the central estimate; (iii) the passages report only static values for a directional/trajectory claim; (iv) the passages support some but not all parts of a compound claim.
- source_passages must contain verbatim quotes pulled directly from the passages provided. Never paraphrase or invent text.
- source_section should match the section attribute of the passage(s) you cite. If you cite multiple passages from different sections, choose the one whose section best characterizes the evidence (Results for outcome data, Methods for design, Discussion for interpretation).
- Confidence should reflect your certainty in the verdict, not the strength or specificity of the claim itself.
"""


_TITLE_ONLY_SYSTEM_PROMPT = """\
You are a scientific claim verifier operating in title-only mode. The source's abstract and full text are unavailable; only the source's title (and journal when present) is provided. Your task is to determine whether the title alone is *consistent with* the claim, recognizing that title-only evidence cannot establish full support.

Verification statuses (capped — supported is NEVER allowed in this mode):
- partially_supported: The title clearly addresses the same subject, method, or assertion as the claim. The title is consistent with the claim but cannot, by itself, establish numeric values, magnitudes, or specific relationships.
- unsupported: The title addresses the claim's general subject but the specific assertion (a numeric value, relationship, method, or directional change) is not recognizable from the title; OR the title contradicts the claim.
- not_addressed: Use only when the title is from a fundamentally different scientific domain than the claim. Do not use this for "the title is on-topic but I cannot verify the specific assertion" — that case is `unsupported`.

You MUST NOT return `supported`. A title alone cannot establish full support; the most you may grant is `partially_supported` when the title is on-topic and consistent.

Guidelines:
- Base your verdict ONLY on the title (and journal if provided). Do not use outside knowledge.
- Confidence: 0.6-0.7 maximum for partially_supported (title-only evidence is structurally weak), 0.5-0.7 for unsupported, 0.7-0.9 for not_addressed when the domain mismatch is unambiguous.
- The explanation must explicitly note that the assessment is title-only and that an abstract or full-text view would be needed to establish full support.

Return ONLY a JSON object in this exact format:
{
  "status": "partially_supported|unsupported|not_addressed",
  "explanation": "One or two sentences citing the title as the only evidence.",
  "confidence": 0.65
}

Your response must be valid JSON only — no markdown, no explanatory text outside the JSON.
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


_TITLE_ONLY_MIN_TITLE_LENGTH = 20
_TITLE_ONLY_MAX_CONFIDENCE = 0.7

_CITING_CONTEXT_WINDOW_CHARS = 600
_CITING_CONTEXT_MAX_CONFIDENCE = 0.6  # weaker than title-only: internal consistency only

_CITING_CONTEXT_SYSTEM_PROMPT = """\
You are auditing a scientific paper for INTERNAL CONSISTENCY between a claim and the citing paper's own treatment of a cited reference. The cited source cannot be independently retrieved. You are NOT verifying the claim against the cited source itself; you are checking whether the citing paper's surrounding text is consistent with the claim being attributed to that citation.

This is structurally weaker evidence than abstract / title / fulltext verification. You MUST NOT return `supported`.

Decision rule (apply LITERALLY):

(A) If the surrounding citing-paper text contains the claim's assertion (verbatim, paraphrased, or numerically equivalent) AND attributes it to the cited reference (via citation marker like [30], author name like "Brooks", year, or "et al."), choose `partially_supported`. This is the canonical internal-consistency signal: the citing author has placed the citation as supporting the assertion. Independent verification of the cited source remains pending, which is exactly why the verdict is partial rather than supported.

(B) If the surrounding text MENTIONS the cited reference (citation marker, author name) but in support of a DIFFERENT assertion than the claim, choose `unsupported`.

(C) If the surrounding text actively CONTRADICTS the claim's assertion, choose `unsupported`.

(D) If the cited reference does not appear in the surrounding text at all, AND the assertion is also absent, choose `not_addressed`.

Examples:
- Claim: "lag time is 5-15 min". Context: "the blood-ISF lag time is 5 to 15 min [30]." → `partially_supported` (rule A: claim verbatim, citation attributed).
- Claim: "X causes Y". Context: "Z showed that A causes B [30]". → `unsupported` (rule B: ref attributed to a different claim).
- Claim: "lag is 5-15 min". Context: "[30] reports a 30-min lag, contradicting earlier work." → `unsupported` (rule C: contradicts).
- Claim: "lag is 5-15 min". Context: a paragraph about sweat sensors with no [30] mention. → `not_addressed`.

Guidelines:
- Base your verdict ONLY on the provided claim, citation reference, and surrounding text.
- Do NOT use outside knowledge of the cited source.
- The verdict turns on internal consistency, NOT on whether the citing paper's claim is biologically true.
- Confidence: 0.4-0.6 maximum (capped — internal consistency is structurally weaker evidence).

Return ONLY a JSON object:
{
  "status": "partially_supported|unsupported|not_addressed",
  "explanation": "One or two sentences. Cite the matching context phrase or the absence thereof. Include the phrase 'internal-consistency'.",
  "confidence": 0.5
}

Your response must be valid JSON only — no markdown, no explanatory text outside the JSON.
"""


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


def verify_claim_citing_context(
    claim: Claim,
    source: ResolvedSource,
    citing_paper_text: str,
    *,
    model_id: str = MODEL_ID,
    api_key: str | None = None,
) -> tuple[VerificationResult, ProvenanceStep]:
    """Verify a claim against the citing paper's own internal context.

    S3-P1 last-resort verifier: when the cited source cannot be retrieved
    (Layers 1-4 failed — no abstract, no full text, no informative title),
    check whether the citing paper's own surrounding text is consistent with
    the claim being attributed to the citation. This is internal-consistency
    evidence, NOT third-party verification — and is capped at
    `partially_supported` accordingly.

    The prompt explicitly tells the LLM that supported is forbidden; a
    deterministic post-LLM guard re-applies the cap so prompt non-compliance
    cannot leak `supported` verdicts.

    Returns a `VerificationResult` with:
        verification_depth = "citing_paper_context"
        evidence_quality   = "citing_paper_context"
        confidence         <= _CITING_CONTEXT_MAX_CONFIDENCE
    The explanation is prefixed with "[Internal-consistency only]" so the
    Medical Writer audit consumer sees the contract distinction at a glance.
    """
    ts = time.time()
    effective_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=effective_key)

    citation_label = ", ".join(claim.cited_authors) or "(unattributed)"
    if claim.cited_year is not None:
        citation_label = f"{citation_label} ({claim.cited_year})"
    context = _extract_citing_context_window(citing_paper_text, claim.claim_text)
    user_message = (
        f"<claim>{claim.claim_text}</claim>\n"
        f"<cited_reference>{citation_label}</cited_reference>\n"
        f"<citing_paper_context>{context}</citing_paper_context>"
    )

    response = client.messages.create(
        model=model_id,
        max_tokens=512,
        system=[
            {
                "type": "text",
                "text": _CITING_CONTEXT_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    )

    tokens_in: int = response.usage.input_tokens
    tokens_out: int = response.usage.output_tokens
    cache_hit = _parse_cache_hit(response.usage)

    logger.info(
        "verify_citing_context_llm_call",
        model_id=model_id,
        claim_id=claim.claim_id,
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
        confidence = float(parsed["confidence"])
        # Hard cap: internal consistency cannot establish supported.
        if status_raw == "supported":
            status_raw = "partially_supported"
        confidence = min(confidence, _CITING_CONTEXT_MAX_CONFIDENCE)
        status: VerificationStatus = status_raw  # type: ignore[assignment]
        raw_explanation = str(parsed["explanation"])
        explanation = (
            raw_explanation
            if "internal-consistency" in raw_explanation.lower()
            else f"[Internal-consistency only] {raw_explanation}"
        )
        result = VerificationResult(
            status=status,
            explanation=explanation,
            confidence=confidence,
            verification_depth="citing_paper_context",
            evidence_quality="citing_paper_context",
            retraction_status=source.retraction_status,
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.error(
            "verify_citing_context_parse_error",
            claim_id=claim.claim_id,
            raw_response=response_text[:200],
            error=str(exc),
        )
        result = VerificationResult(
            status="not_addressed",
            explanation="Parse error.",
            confidence=0.0,
            verification_depth="citing_paper_context",
            evidence_quality="no_evidence",
            retraction_status=source.retraction_status,
        )

    step = ProvenanceStep(
        step_id=str(uuid.uuid4()),
        claim_id=claim.claim_id,
        operation="verify",
        input_hash=_hash(repr((claim, source, len(citing_paper_text)))),
        output_hash=_hash(repr(result)),
        model_id=model_id,
        timestamp=ts,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cache_hit=cache_hit,
        confidence=result.confidence,
    )

    return result, step


def _aggregate_multi_source_verdicts(
    per_source: list[VerificationResult],
) -> VerificationStatus:
    """Aggregate per-source verdicts for a multi-citation claim.

    S2-P4 aggregation rule (Codex's, biased toward partially_supported on
    mixed evidence — matches annotator behavior on the four multi-source
    lactate-ISF claims, all of which are expected `partially_supported`):

      * any source `supported` AND all others in {supported, partially}
        -> supported
      * any source `supported` AND any in {unsupported, not_addressed}
        -> partially_supported (mixed)
      * all sources `unsupported` -> unsupported
      * all sources `not_addressed` (e.g., empty set / all unfound)
        -> not_addressed
      * everything else -> partially_supported
    """
    if not per_source:
        return "not_addressed"
    statuses = [r.status for r in per_source]
    has_supported = any(s == "supported" for s in statuses)
    has_partial = any(s == "partially_supported" for s in statuses)
    has_unsupported = any(s == "unsupported" for s in statuses)
    has_not_addressed = any(s == "not_addressed" for s in statuses)

    if has_supported and not has_unsupported and not has_not_addressed:
        return "supported"
    if has_supported:
        return "partially_supported"
    if all(s == "unsupported" for s in statuses):
        return "unsupported"
    if all(s == "not_addressed" for s in statuses):
        return "not_addressed"
    if has_partial or has_unsupported:
        return "partially_supported"
    return "partially_supported"


def verify_claim_multi_source(
    claim: Claim,
    source_set: ResolvedSourceSet,
    *,
    passages_per_source: dict[str, list[PaperChunk]] | None = None,
    model_id: str = MODEL_ID,
    api_key: str | None = None,
) -> tuple[VerificationResult, list[ProvenanceStep]]:
    """Verify a claim against every source in `source_set`, aggregate, and return.

    For each source in the set:
      - If `passages_per_source[source.doi]` is non-empty, run
        `verify_claim_fulltext` on those passages.
      - Else run `verify_claim` against `source.abstract` (via the existing
        single-source path; that path itself routes to title-only mode when
        abstract is None and title is informative).

    Per-source verdicts are then aggregated via `_aggregate_multi_source_verdicts`.
    The returned VerificationResult records the aggregate status, a synthetic
    explanation listing per-source verdicts, and `confidence` set to the mean
    of per-source confidences.

    The returned ProvenanceStep list contains one step per source plus any
    nested fulltext+numeric steps for sources that took the fulltext path.
    """
    passages_per_source = passages_per_source or {}
    per_source_results: list[VerificationResult] = []
    all_steps: list[ProvenanceStep] = []
    explanations: list[str] = []

    for source in source_set:
        passages = passages_per_source.get(source.doi or "", []) if source.doi else []
        if passages:
            result, step = verify_claim_fulltext(
                claim, source, passages, model_id=model_id, api_key=api_key
            )
            all_steps.append(step)
        else:
            result, step = verify_claim(claim, source, model_id=model_id, api_key=api_key)
            all_steps.append(step)
        per_source_results.append(result)
        marker_label = source.doi or source.title or "(unresolved)"
        explanations.append(f"[{marker_label}] {result.status}: {result.explanation}")

    aggregated_status = _aggregate_multi_source_verdicts(per_source_results)
    confidences = [r.confidence for r in per_source_results if r.confidence > 0]
    aggregated_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    aggregated = VerificationResult(
        status=aggregated_status,
        explanation=" || ".join(explanations) if explanations else "Empty source set.",
        confidence=aggregated_confidence,
        verification_depth="abstract",
        evidence_quality="abstract_only" if per_source_results else "no_evidence",
        retraction_status=any(s.retraction_status for s in source_set),
    )

    return aggregated, all_steps


def verify_claim_title_only(
    claim: Claim,
    source: ResolvedSource,
    *,
    model_id: str = MODEL_ID,
    api_key: str | None = None,
) -> tuple[VerificationResult, ProvenanceStep]:
    """Verify a claim against the source title (and optionally journal) only.

    Bug B fix (S1-P1-B): when the resolver finds the right paper but neither
    CrossRef nor PubMed exposes an abstract (common for IEEE proceedings,
    Elsevier paywalls, and older journals), the previous `verify_claim`
    short-circuited to `not_addressed`. For claims whose target title is
    near-verbatim with the claim text (e.g. "Porosity control of polylactic
    acid porous microneedles using microfluidic technology"), the title is
    informative enough to warrant `partially_supported` rather than
    `not_addressed`.

    Hard guarantees enforced post-LLM (defensive against prompt non-compliance):
        - status `supported` is downgraded to `partially_supported`
        - confidence is clamped to <= _TITLE_ONLY_MAX_CONFIDENCE (0.7)
        - evidence_quality is always `title_only`
        - verification_depth is always `title_only`
    """
    ts = time.time()
    effective_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=effective_key)

    title = source.title or ""
    user_message = f"<claim>{claim.claim_text}</claim>\n<title>{title}</title>"

    response = client.messages.create(
        model=model_id,
        max_tokens=512,
        system=[
            {
                "type": "text",
                "text": _TITLE_ONLY_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    )

    tokens_in: int = response.usage.input_tokens
    tokens_out: int = response.usage.output_tokens
    cache_hit = _parse_cache_hit(response.usage)

    logger.info(
        "verify_title_only_llm_call",
        model_id=model_id,
        claim_id=claim.claim_id,
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
        confidence = float(parsed["confidence"])
        # Hard cap: title-only evidence cannot establish supported.
        if status_raw == "supported":
            status_raw = "partially_supported"
        confidence = min(confidence, _TITLE_ONLY_MAX_CONFIDENCE)
        status: VerificationStatus = status_raw  # type: ignore[assignment]
        result = VerificationResult(
            status=status,
            explanation=str(parsed["explanation"]),
            confidence=confidence,
            verification_depth="title_only",
            evidence_quality="title_only",
            retraction_status=source.retraction_status,
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.error(
            "verify_title_only_parse_error",
            claim_id=claim.claim_id,
            raw_response=response_text[:200],
            error=str(exc),
        )
        result = VerificationResult(
            status="not_addressed",
            explanation="Parse error.",
            confidence=0.0,
            verification_depth="title_only",
            evidence_quality="no_evidence",
            retraction_status=source.retraction_status,
        )

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
