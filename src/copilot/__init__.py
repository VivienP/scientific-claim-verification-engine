"""Copilot enrichment layer — additive post-processing above the V1 pipeline.

Architecture: ``CopilotEnricher`` consumes ``ClaimVerification`` objects already
produced by ``run_pipeline()`` and returns ``EnrichedVerification`` objects that
add three categories of signal:

1. ``verdict_rationale`` — 1-sentence compression of why the verdict was reached.
2. Evidence quality fields — ``is_primary_source``, ``study_design``, ``risk_of_bias``,
   ``conflicting_evidence_flag``, ``primary_source_doi`` (pharma mode only).
3. ``recommended_fix`` — CrossRef-verified citation swap or claim reword for
   ``unsupported`` / ``partially_supported`` / ``not_addressed`` verdicts.

Every sub-step emits a ``ProvenanceStep`` appended to ``EnrichedVerification.copilot_steps``.
The V1 pipeline files are untouched.
"""

from src.copilot.models import (
    CopilotFields,
    CopilotMode,
    EnrichedVerification,
    RecommendedFix,
    ReviewDecision,
    ReviewSession,
)

__all__ = [
    "CopilotFields",
    "CopilotMode",
    "EnrichedVerification",
    "RecommendedFix",
    "ReviewDecision",
    "ReviewSession",
]
