"""Frozen dataclasses for the Copilot enrichment layer.

These types sit above ``ClaimVerification`` from ``src/pipeline.py`` and carry
the three categories of copilot signal: rationale, evidence quality, and
actionable remediation. They never mutate V1 types — ``EnrichedVerification``
wraps a ``ClaimVerification`` via composition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from src.models import ProvenanceStep
from src.pipeline import ClaimVerification

# ---------------------------------------------------------------------------
# Mode enum — controls which optional fields are populated by the enricher.
# ---------------------------------------------------------------------------


class CopilotMode(str, Enum):
    """ICP-adaptive schema selector.

    PHARMA:   Full schema — is_primary_source, study_design, risk_of_bias,
              regulatory_risk_level. Targets Medical Writers at Phase II→III biotechs.
    ACADEMIC: Novelty-focused — novelty_claim, reduced fix actions. Targets
              AI Reviewer for ML papers.
    GENERAL:  Minimal schema — verdict_rationale + recommended_fix only. Targets
              R&D Lit Review loop. Fastest and cheapest.
    """

    PHARMA = "pharma"
    ACADEMIC = "academic"
    GENERAL = "general"


# ---------------------------------------------------------------------------
# Type aliases (annotation-only — no runtime overhead).
# ---------------------------------------------------------------------------

StudyDesign = Literal[
    "rct",
    "observational",
    "case_control",
    "animal_model",
    "in_vitro",
    "meta_analysis",
    "systematic_review",
    "narrative_review",
    "guidelines",
    "unknown",
]

RiskOfBias = Literal["low", "medium", "high", "unknown"]
RegulatoryRiskLevel = Literal["high", "medium", "low"]
FixAction = Literal[
    "swap_doi",  # replace with a better source (unsupported / partial)
    "reword",  # make claim more conservative (partial)
    "swap_and_reword",  # both — different source AND conservative reword
    "add_citation",  # find a source that actually covers this claim (not_addressed)
    "remove",  # no source supports the claim as written
]


# ---------------------------------------------------------------------------
# Core copilot dataclasses.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecommendedFix:
    """A CrossRef-verified remediation suggestion for a non-supported claim.

    Safety invariant: ``suggested_doi`` is either ``None`` or a DOI that has
    been resolved through the CrossRef API with score ≥ 0.7. The fix generator
    enforces this via ``_verify_doi()`` before constructing this object.
    """

    action: FixAction
    regulatory_risk_level: RegulatoryRiskLevel | None  # None in ACADEMIC / GENERAL mode
    suggested_doi: str | None  # CrossRef-verified; never hallucinated
    suggested_doi_title: str | None
    reworded_claim: str | None  # if action ∈ {reword, swap_and_reword}
    confidence: float  # 0.0-1.0 from LLM
    provenance_step_id: str  # links to the ProvenanceStep for this fix


@dataclass(frozen=True)
class CopilotFields:
    """All copilot-layer enrichment signals for one claim.

    Fields that are ``None`` are disabled for the active ``CopilotMode``.
    The enricher sets disabled fields to ``None`` and the HTML template
    conditionally renders them, so no downstream code needs mode-switching
    logic beyond the enricher itself.
    """

    # Present in all modes.
    verdict_rationale: str
    recommended_fix: RecommendedFix | None

    # PHARMA mode only (None in ACADEMIC / GENERAL).
    is_primary_source: bool | None
    study_design: StudyDesign | None
    risk_of_bias: RiskOfBias | None
    conflicting_evidence_flag: bool | None
    primary_source_doi: str | None  # lookup result when is_primary_source=False

    # ACADEMIC mode only (None in PHARMA / GENERAL).
    novelty_claim: bool | None


@dataclass(frozen=True)
class EnrichedVerification:
    """V1 ClaimVerification + copilot fields, fully immutable.

    ``base`` is the unchanged V1 result. ``copilot`` holds enrichment signals.
    ``copilot_steps`` are the ProvenanceSteps emitted during enrichment —
    separate from ``base.steps`` which hold V1 pipeline steps.
    """

    base: ClaimVerification
    copilot: CopilotFields
    copilot_steps: tuple[ProvenanceStep, ...]
    mode: CopilotMode


# ---------------------------------------------------------------------------
# HITL audit trail — populated by the HTML report's JS export button.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewDecision:
    """One Medical Writer decision on a single recommended_fix."""

    claim_id: str
    action_taken: Literal["accepted", "rejected", "modified"]
    accepted_doi: str | None  # the DOI the writer chose (may differ from suggestion)
    reviewer_note: str | None
    timestamp: float  # time.time()


@dataclass(frozen=True)
class ReviewSession:
    """Aggregate of all review decisions for one copilot run.

    Written to ``run_dir/review_session.json`` via the HTML report's
    'Export Review Session' button (< 50 lines inline JS, no server required).
    This sidecar JSON is the HITL audit trail required by pharma compliance.
    """

    session_id: str
    run_id: str
    decisions: tuple[ReviewDecision, ...] = field(default_factory=tuple)
    total_fixes_presented: int = 0
    total_fixes_accepted: int = 0
