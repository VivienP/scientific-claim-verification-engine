"""Deterministic heuristics over claim text. Pure Python, no LLM."""

from __future__ import annotations

import re

# Compiled patterns that match specific numeric assertions typical of
# Results-section content. These are the claim shapes that cannot be
# reliably verified from an abstract alone, because the abstract
# systematically omits exact figures (percentages, p-values, CIs, effect
# sizes, exact n=, timepoint-specific response rates).
_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\d+(?:\.\d+)?\s*%"),  # 20%, 14.5%
    re.compile(r"\bp\s*[<>=]\s*0?\.\d+", re.IGNORECASE),  # p < 0.001, p=0.02
    re.compile(r"\bn\s*=\s*\d+", re.IGNORECASE),  # n=233
    re.compile(r"95\s*%\s*CI", re.IGNORECASE),  # 95% CI
    re.compile(r"\b(?:HR|OR|RR)\s*[=:]?\s*\d", re.IGNORECASE),  # HR 0.55, OR=1.7
    re.compile(r"hazard\s+ratio", re.IGNORECASE),
    re.compile(r"odds\s+ratio", re.IGNORECASE),
    re.compile(r"\b(?:Cohen'?s?\s*d|Hedges'?\s*g)\b", re.IGNORECASE),
    re.compile(r"\bweek\s*\d+", re.IGNORECASE),  # at week 12 (timepoint)
    re.compile(r"\b\d+\s*(?:mg|mcg|µg|ml|kg|points?)", re.IGNORECASE),
)


def _claim_has_specific_numeric(claim_text: str) -> bool:
    """True if the claim contains a specific numeric/Results-section assertion.

    These patterns mark claims that cannot be reliably verified from an
    abstract alone, because the abstract systematically omits exact figures
    (percentages, p-values, CIs, effect sizes, exact n=, timepoint-specific
    response rates).

    Pure deterministic. Same input -> same output, every run. No LLM, no I/O.
    """
    if not claim_text:
        return False
    return any(p.search(claim_text) for p in _PATTERNS)
