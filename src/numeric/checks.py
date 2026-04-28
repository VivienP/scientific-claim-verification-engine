"""Pure-Python deterministic numeric checks. Zero LLM, zero scipy.

MVP: one check only — OR/CI internal consistency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

NumericRole = Literal["primary", "ci_low", "ci_high", "comparator", "p_value", "n"]
CheckType = Literal["or_ci_consistency"]

_MAX_CI_RATIO = 50.0  # plausibility heuristic for OR/CI ratio


@dataclass(frozen=True)
class NumericAssertion:
    """A single numeric assertion extracted from claim or passage text."""

    raw_text: str
    value: float
    unit: str | None
    role: NumericRole
    context: str


@dataclass(frozen=True)
class NumericCheckResult:
    """Result of a deterministic numeric check on a single claim."""

    check_type: CheckType
    consistent: bool
    extracted: list[NumericAssertion] = field(default_factory=list)
    explanation: str = ""


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
