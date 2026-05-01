"""Numeric engine orchestrator: extract assertions, run applicable check, return result."""

from __future__ import annotations

import hashlib
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


def _hash(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def _find_or_ci_triple(
    assertions: list[NumericAssertion],
) -> tuple[float, float, float] | None:
    """Find the first OR/CI triple in the assertion list.

    Heuristic: a "primary" record whose context mentions "odds ratio" or "OR ",
    plus an immediately following ci_low and ci_high pair. If multiple primary
    records exist, prefer the one whose context contains "odds ratio" or "OR".
    Returns (or_value, ci_low, ci_high) or None.
    """
    primary_idx: int | None = None
    for i, a in enumerate(assertions):
        if a.role != "primary":
            continue
        ctx_lower = a.context.lower()
        raw_lower = a.raw_text.lower()
        if "odds ratio" in ctx_lower or "or " in raw_lower or raw_lower.startswith("or"):
            primary_idx = i
            break

    if primary_idx is None:
        for i, a in enumerate(assertions):
            if a.role == "primary" and a.unit is None:
                primary_idx = i
                break

    if primary_idx is None:
        return None

    or_value = assertions[primary_idx].value

    ci_low: float | None = None
    ci_high: float | None = None
    for a in assertions[primary_idx:]:
        if a.role == "ci_low" and ci_low is None:
            ci_low = a.value
        elif a.role == "ci_high" and ci_high is None:
            ci_high = a.value
        if ci_low is not None and ci_high is not None:
            break

    if ci_low is None or ci_high is None:
        return None

    return or_value, ci_low, ci_high


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
