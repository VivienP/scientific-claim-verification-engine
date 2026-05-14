"""Numeric engine orchestrator: extract assertions, run applicable check, return result."""

from __future__ import annotations

import hashlib
import re
import time
import uuid

import structlog

from src.models import ProvenanceStep
from src.numeric.checks import (
    NumericAssertion,
    NumericCheckResult,
    check_or_ci_consistency,
    check_p_value_ci_consistency,
)
from src.numeric.extract import MODEL_ID, extract_numeric_assertions

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_RATIO_TERMS = ("odds ratio", "hazard ratio", "risk ratio", "relative risk")

RATIO_KEYWORDS_LONG = (
    "odds ratio",
    "hazard ratio",
    "risk ratio",
    "relative risk",
    "incidence rate ratio",
    "rate ratio",
)
RATIO_KEYWORDS_SHORT: frozenset[str] = frozenset(
    {"or", "hr", "rr", "rrr", "ahr", "ihr", "shr", "irr"}
)
_WORD_RE: re.Pattern[str] = re.compile(r"[A-Za-z]+")

# Maximum gap (in chars) between a primary's span_end and a same-sentence CI's
# span_start that still permits span-anchored pairing. 60 chars is wide enough
# to accept a typical "(95% CI 0.48-0.74)" parenthetical between primary and
# CI, narrow enough to reject CIs that belong to a later primary in a compact
# multi-metric sentence.
MAX_CHARS_BETWEEN_PRIMARY_AND_CI = 60


def _hash(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def _has_ratio_keyword(text: str) -> bool:
    """Match either a long phrase or a standalone short token (case-insensitive,
    word-boundary based to avoid hits in 'PRIOR'/'MONITOR')."""
    t = text.lower()
    if any(k in t for k in RATIO_KEYWORDS_LONG):
        return True
    return bool({w.lower() for w in _WORD_RE.findall(text)} & RATIO_KEYWORDS_SHORT)


def _is_ratio_primary(a: NumericAssertion) -> bool:
    """Return True when the assertion is a ratio-measure primary.

    Exclusions applied before keyword scan:
    - role must be "primary"
    - unit == "%" or "%" in raw_text excludes percentages
    - context containing 'reduction'/'change'/'difference' excludes additive
      descriptors even when the context mentions a ratio term (e.g. '13%
      relative risk reduction').
    """
    if a.role != "primary":
        return False
    if a.unit == "%" or "%" in a.raw_text:
        return False
    if _has_ratio_keyword(a.raw_text):
        return True
    if _has_ratio_keyword(a.context):
        ctx_l = a.context.lower()
        return not ("reduction" in ctx_l or "change" in ctx_l or "difference" in ctx_l)
    return False


def _find_span_anchored_triple(
    assertions: list[NumericAssertion], primary_idx: int
) -> tuple[float, float, float] | None:
    """Tier 1: pair primary + ci_low + ci_high by span+sentence anchoring.

    Returns a triple when the primary has ``span_start`` / ``span_end`` /
    ``sentence_id`` populated AND at least one ``ci_low`` and one ``ci_high``
    share the primary's sentence with a closest-CI char gap within
    ``MAX_CHARS_BETWEEN_PRIMARY_AND_CI``. The closest CI by char distance to
    the primary is picked when multiple same-sentence CIs exist (defensive
    against compact multi-CI sentences).

    Returns None when the primary is missing span info, when no same-sentence
    CIs exist on each side, or when the closest CI exceeds the char-gap
    threshold — the caller falls through to substring / window matching.
    """
    primary = assertions[primary_idx]
    if primary.span_start is None or primary.span_end is None or primary.sentence_id is None:
        return None
    primary_start = primary.span_start
    primary_end = primary.span_end

    def _gap(ci: NumericAssertion) -> int:
        ci_start = ci.span_start
        ci_end = ci.span_end
        if ci_start is None or ci_end is None:
            return 10**9
        if ci_start >= primary_end:
            return ci_start - primary_end
        if ci_end <= primary_start:
            return primary_start - ci_end
        return 0  # overlapping spans — pathological but accept it

    same_sent_lows = [
        a
        for a in assertions
        if a.role == "ci_low" and a.span_start is not None and a.sentence_id == primary.sentence_id
    ]
    same_sent_highs = [
        a
        for a in assertions
        if a.role == "ci_high" and a.span_start is not None and a.sentence_id == primary.sentence_id
    ]
    if not (same_sent_lows and same_sent_highs):
        return None
    closest_low = min(same_sent_lows, key=_gap)
    closest_high = min(same_sent_highs, key=_gap)
    if _gap(closest_low) > MAX_CHARS_BETWEEN_PRIMARY_AND_CI:
        return None
    if _gap(closest_high) > MAX_CHARS_BETWEEN_PRIMARY_AND_CI:
        return None
    return (primary.value, closest_low.value, closest_high.value)


def _find_or_ci_triple(
    assertions: list[NumericAssertion],
) -> tuple[float, float, float] | None:
    """Find a (ratio, ci_low, ci_high) triple via three-tier matching.

    Step 1: pick the first primary that is a ratio measure. No unit=None
    fallback — Bug B fix.

    Step 2: pair the primary with its CI by trying, in order:
      (a) Span-anchored — primary + CIs share a sentence and the closest
          CIs by char gap are within ``MAX_CHARS_BETWEEN_PRIMARY_AND_CI``.
          Requires deterministic span derivation by the extractor.
      (b) Strong substring — CI context contains primary's raw_text
          (case-insensitive). Survives even when spans are unavailable.
      (c) Window — CI inside ``[primary_idx+1, next_primary_idx)``.

    If no tier yields a ci_low + ci_high pair, return None. This avoids
    pairing a ratio primary with CIs that belong to a different statistic when
    multiple primaries are present — Bug A fix.
    """
    primary_idx: int | None = None
    for i, a in enumerate(assertions):
        if _is_ratio_primary(a):
            primary_idx = i
            break

    if primary_idx is None:
        return None

    span_triple = _find_span_anchored_triple(assertions, primary_idx)
    if span_triple is not None:
        return span_triple

    primary = assertions[primary_idx]
    p_raw_l = primary.raw_text.lower()

    next_primary_idx = next(
        (j for j in range(primary_idx + 1, len(assertions)) if assertions[j].role == "primary"),
        len(assertions),
    )
    window = assertions[primary_idx + 1 : next_primary_idx]

    strong_lows = [a for a in assertions if a.role == "ci_low" and p_raw_l in a.context.lower()]
    strong_highs = [a for a in assertions if a.role == "ci_high" and p_raw_l in a.context.lower()]
    if strong_lows and strong_highs:
        return (primary.value, strong_lows[0].value, strong_highs[0].value)

    win_lows = [a for a in window if a.role == "ci_low"]
    win_highs = [a for a in window if a.role == "ci_high"]
    if win_lows and win_highs:
        return (primary.value, win_lows[0].value, win_highs[0].value)

    return None


def _detect_or_ci_pairing_ambiguity(assertions: list[NumericAssertion]) -> bool:
    """True when CIs exist but cross-reference a different primary than the selected one.

    Activates when (a) ≥2 ratio primaries are present, (b) span-anchored
    pairing is unavailable for the first primary, (c) the first primary has
    no own strong-substring CI match, and (d) at least one CI's context names
    a different ratio primary's ``raw_text``. The Zhou/Wang
    compact-multi-metric pattern: two HRs in one sentence, LLM-emitted CI
    contexts that bind the CI to a later primary, and a window-match that
    would either fail (no own CIs in window) or fabricate a triple that
    flags a false-positive inconsistency.

    On detection, ``run_numeric_check`` skips the OR/CI consistency check
    and emits ``NumericCheckResult(ambiguous=True)`` — better to skip than
    to flag a wrong inconsistency. The text-path verdict is preserved.
    """
    ratio_primaries = [(i, a) for i, a in enumerate(assertions) if _is_ratio_primary(a)]
    if len(ratio_primaries) < 2:
        return False
    first_idx = ratio_primaries[0][0]

    if _find_span_anchored_triple(assertions, first_idx) is not None:
        return False

    first_primary = assertions[first_idx]
    p_raw_l = first_primary.raw_text.lower()
    strong_lows = [a for a in assertions if a.role == "ci_low" and p_raw_l in a.context.lower()]
    strong_highs = [a for a in assertions if a.role == "ci_high" and p_raw_l in a.context.lower()]
    if strong_lows and strong_highs:
        # The first primary has its own substring-anchored CIs — pairing is
        # unambiguous via tier 2, no need to fall through to window.
        return False

    cis = [a for a in assertions if a.role in ("ci_low", "ci_high")]
    if not cis:
        return False
    other_primaries = [a for j, a in ratio_primaries if j != first_idx]
    for ci in cis:
        ctx_l = ci.context.lower()
        for other in other_primaries:
            other_raw_l = other.raw_text.lower()
            if other_raw_l and other_raw_l in ctx_l:
                return True
    return False


def _infer_null_value(assertions: list[NumericAssertion]) -> float:
    text = " ".join(f"{a.raw_text} {a.context}" for a in assertions).lower()
    if any(term in text for term in _RATIO_TERMS):
        return 1.0
    return 0.0


def _find_p_value_ci_tuple(
    assertions: list[NumericAssertion],
) -> tuple[float, float, float, float] | None:
    p_value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    for assertion in assertions:
        if assertion.role == "p_value" and p_value is None:
            p_value = assertion.value
        elif assertion.role == "ci_low" and ci_low is None:
            ci_low = assertion.value
        elif assertion.role == "ci_high" and ci_high is None:
            ci_high = assertion.value

    if p_value is None or ci_low is None or ci_high is None:
        return None
    return p_value, ci_low, ci_high, _infer_null_value(assertions)


def run_numeric_check(
    claim_text: str,
    *,
    claim_id: str = "__numeric_check__",
    model_id: str = MODEL_ID,
    api_key: str | None = None,
) -> tuple[NumericCheckResult | None, list[ProvenanceStep]]:
    """Extract numeric assertions and run any applicable deterministic check.

    Runs the first applicable deterministic check. OR/CI consistency is tried
    first because it is most specific; p-value/CI null-crossing is the fallback
    when no OR/CI triple is available. Never raises.
    """
    ts_check = time.time()

    assertions, extract_step = extract_numeric_assertions(
        claim_text, claim_id=claim_id, model_id=model_id, api_key=api_key
    )

    if not assertions:
        logger.debug("numeric_no_assertions", claim_id=claim_id)
        return None, [extract_step]

    # Ambiguity gate: when a compact multi-metric sentence makes window-match
    # unsafe, skip the OR/CI check and surface ``ambiguous=True``. Preserves
    # the text-path verdict; refuses to fabricate a numeric verdict the
    # extractor can't unambiguously support.
    if _detect_or_ci_pairing_ambiguity(assertions):
        ambiguous_result = NumericCheckResult(
            check_type="or_ci_consistency",
            consistent=True,
            extracted=assertions,
            explanation=(
                "ambiguous primary-CI pairing: multiple ratio primaries with overlapping "
                "context references; OR/CI consistency check skipped."
            ),
            ambiguous=True,
        )
        check_step = ProvenanceStep(
            step_id=str(uuid.uuid4()),
            claim_id=claim_id,
            operation="numeric_check",
            input_hash=_hash(repr(assertions)),
            output_hash=_hash(repr(ambiguous_result)),
            model_id=None,
            timestamp=ts_check,
            tokens_in=None,
            tokens_out=None,
            cache_hit=None,
            confidence=None,
        )
        logger.info(
            "numeric_check_ambiguous",
            claim_id=claim_id,
            n_assertions=len(assertions),
        )
        return ambiguous_result, [extract_step, check_step]

    triple = _find_or_ci_triple(assertions)
    input_payload: tuple[float, ...]
    if triple is not None:
        or_value, ci_low, ci_high = triple
        result = check_or_ci_consistency(or_value, ci_low, ci_high, extracted=assertions)
        input_payload = triple
    else:
        p_ci_tuple = _find_p_value_ci_tuple(assertions)
        if p_ci_tuple is None:
            logger.debug(
                "numeric_no_applicable_check",
                claim_id=claim_id,
                n_assertions=len(assertions),
            )
            return None, [extract_step]
        p_value, ci_low, ci_high, null_value = p_ci_tuple
        result = check_p_value_ci_consistency(
            p_value,
            ci_low,
            ci_high,
            null_value=null_value,
            extracted=assertions,
        )
        input_payload = p_ci_tuple

    check_step = ProvenanceStep(
        step_id=str(uuid.uuid4()),
        claim_id=claim_id,
        operation="numeric_check",
        input_hash=_hash(repr(input_payload)),
        output_hash=_hash(repr(result)),
        model_id=None,  # deterministic — no LLM
        timestamp=ts_check,
        tokens_in=None,
        tokens_out=None,
        cache_hit=None,
        confidence=None,
    )

    logger.info(
        "numeric_check_run",
        claim_id=claim_id,
        check_type=result.check_type,
        consistent=result.consistent,
    )

    return result, [extract_step, check_step]
