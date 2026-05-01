"""Deterministic numeric verification engine.

Extraction is LLM-driven; comparison is pure Python (zero LLM, zero scipy).
"""

from src.numeric.checks import (
    NumericAssertion,
    NumericCheckResult,
    check_or_ci_consistency,
    check_p_value_ci_consistency,
)
from src.numeric.engine import run_numeric_check

__all__ = [
    "NumericAssertion",
    "NumericCheckResult",
    "check_or_ci_consistency",
    "check_p_value_ci_consistency",
    "run_numeric_check",
]
