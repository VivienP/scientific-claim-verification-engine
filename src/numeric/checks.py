"""Pure-Python deterministic numeric checks. Zero LLM, zero scipy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

NumericRole = Literal["primary", "ci_low", "ci_high", "comparator", "p_value", "n"]
CheckType = Literal["or_ci_consistency", "p_value_ci_consistency"]

_MAX_CI_RATIO = 50.0  # plausibility heuristic for OR/CI ratio


@dataclass(frozen=True)
class NumericAssertion:
    """A single numeric assertion extracted from claim or passage text.

    ``raw_text`` is the substring the LLM identified as significant; this is
    the ground-truth field. ``span_start`` / ``span_end`` are character offsets
    derived deterministically in Python via ``claim_text.find(raw_text)`` (the
    LLM is not trusted for character offsets — its job is identifying which
    substrings matter). When ``raw_text`` appears multiple times in the claim,
    the spans stay ``None`` and pairing falls back to substring/window
    heuristics. ``sentence_id`` is a 0-indexed sentence number used by the
    span-anchored pairing path to require co-sentence primary+CI matches.
    """

    raw_text: str
    value: float
    unit: str | None
    role: NumericRole
    context: str
    span_start: int | None = None
    span_end: int | None = None
    sentence_id: int | None = None


@dataclass(frozen=True)
class NumericCheckResult:
    """Result of a deterministic numeric check on a single claim.

    ``ambiguous=True`` signals the checker detected multiple ratio-primaries
    with no unambiguous pairing (span anchoring unavailable AND window-match
    would steal a CI semantically closer to a later primary). In that case the
    check is skipped, ``consistent`` stays ``True`` by convention (we did not
    detect an inconsistency), and the text-path verdict is preserved upstream.
    """

    check_type: CheckType
    consistent: bool
    extracted: list[NumericAssertion] = field(default_factory=list)
    explanation: str = ""
    ambiguous: bool = False


def check_or_ci_consistency(
    or_value: float,
    ci_low: float,
    ci_high: float,
    *,
    extracted: list[NumericAssertion] | None = None,
) -> NumericCheckResult:
    """Verify ci_low <= or_value <= ci_high AND CI ratio is plausible.

    Plausibility heuristic for the MVP:
      - ci_low must be > 0 (OR is on a multiplicative scale)
      - ci_low must be <= ci_high (CI not inverted)
      - For OR >= 1: ratio = ci_high / ci_low must be <= _MAX_CI_RATIO
      - For OR < 1: ratio = ci_low / ci_high must be >= 1/_MAX_CI_RATIO
      - or_value must lie within [ci_low, ci_high]
    """
    extracted = extracted if extracted is not None else []
    failures: list[str] = []

    if ci_low <= 0:
        failures.append(f"ci_low must be > 0 on multiplicative scale (got {ci_low})")
    if ci_low > ci_high:
        failures.append(f"CI inverted: ci_low={ci_low} > ci_high={ci_high}")

    if not failures:
        if not (ci_low <= or_value <= ci_high):
            failures.append(f"OR={or_value} outside CI [{ci_low}, {ci_high}]")

        if or_value >= 1.0:
            ratio = ci_high / ci_low
            if ratio > _MAX_CI_RATIO:
                failures.append(
                    f"CI ratio high/low={ratio:.2f} exceeds plausibility threshold {_MAX_CI_RATIO}"
                )
        else:
            ratio = ci_low / ci_high
            if ratio < 1.0 / _MAX_CI_RATIO:
                failures.append(
                    f"CI ratio low/high={ratio:.4f} below plausibility threshold {1.0 / _MAX_CI_RATIO}"
                )

    consistent = not failures
    if consistent:
        explanation = (
            f"OR/CI internally consistent: {ci_low} <= {or_value} <= {ci_high}, "
            f"CI ratio within plausibility bounds."
        )
    else:
        explanation = "OR/CI inconsistent: " + "; ".join(failures)

    return NumericCheckResult(
        check_type="or_ci_consistency",
        consistent=consistent,
        extracted=extracted,
        explanation=explanation,
    )


def check_p_value_ci_consistency(
    p_value: float,
    ci_low: float,
    ci_high: float,
    *,
    null_value: float = 0.0,
    alpha: float = 0.05,
    extracted: list[NumericAssertion] | None = None,
) -> NumericCheckResult:
    """Check whether a p-value and CI agree about excluding the null value.

    This intentionally does not recompute a p-value. It only checks the
    high-signal contradiction reviewers expect: a significant p-value paired
    with a CI crossing the null, or a non-significant p-value paired with a CI
    excluding the null.
    """
    extracted = extracted if extracted is not None else []
    failures: list[str] = []

    if not (0.0 <= p_value <= 1.0):
        failures.append(f"p-value must be between 0 and 1 (got {p_value})")
    if ci_low > ci_high:
        failures.append(f"CI inverted: ci_low={ci_low} > ci_high={ci_high}")

    if not failures:
        ci_excludes_null = ci_high < null_value or ci_low > null_value
        p_significant = p_value < alpha
        if p_significant and not ci_excludes_null:
            failures.append(
                f"p={p_value} is significant at alpha={alpha}, but CI "
                f"[{ci_low}, {ci_high}] crosses null={null_value}"
            )
        elif not p_significant and ci_excludes_null:
            failures.append(
                f"p={p_value} is not significant at alpha={alpha}, but CI "
                f"[{ci_low}, {ci_high}] excludes null={null_value}"
            )

    consistent = not failures
    if consistent:
        explanation = (
            f"p-value/CI internally consistent: p={p_value}, CI [{ci_low}, {ci_high}], "
            f"null={null_value}, alpha={alpha}."
        )
    else:
        explanation = "p-value/CI inconsistent: " + "; ".join(failures)

    return NumericCheckResult(
        check_type="p_value_ci_consistency",
        consistent=consistent,
        extracted=extracted,
        explanation=explanation,
    )
