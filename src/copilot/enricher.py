"""CopilotEnricher — orchestrates all four copilot sub-components.

Per-claim enrichment pipeline (sequential):
  1. verdict_rationale   — always
  2. classify_source     — always
  3. find_primary_source — if is_primary=False and enable_primary_lookup
  4. generate_fix        — if verdict ∈ {unsupported, partial, not_addressed}
                           and enable_recommended_fix

Output: EnrichedVerification wrapping the unchanged ClaimVerification.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import structlog

from src.copilot.fix_generator import generate_fix
from src.copilot.models import (
    CopilotFields,
    CopilotMode,
    EnrichedVerification,
)
from src.copilot.primary_lookup import find_primary_source_doi
from src.copilot.primary_source import SourceClassification, classify_source
from src.copilot.rationale import extract_rationale
from src.models import ProvenanceStep
from src.pipeline import ClaimVerification

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_FIX_VERDICTS = frozenset({"unsupported", "partially_supported", "not_addressed"})


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CopilotConfig:
    """Runtime configuration for the CopilotEnricher."""

    mode: CopilotMode = CopilotMode.PHARMA
    api_key: str | None = None
    enable_primary_lookup: bool = True
    enable_recommended_fix: bool = True
    crossref_verify_threshold: float = 0.7
    db_path: Path | None = None
    http_timeout: float = 10.0


# ---------------------------------------------------------------------------
# Enricher
# ---------------------------------------------------------------------------


class CopilotEnricher:
    """Enrich a list of ClaimVerification objects into EnrichedVerification."""

    def __init__(self, config: CopilotConfig | None = None) -> None:
        self._config = config or CopilotConfig()

    def enrich_one(self, cv: ClaimVerification) -> EnrichedVerification:
        """Enrich a single ClaimVerification. Never raises."""
        config = self._config
        steps: list[ProvenanceStep] = []

        # Step 1 — verdict_rationale (all modes)
        rationale, rationale_step = extract_rationale(cv, api_key=config.api_key)
        steps.append(rationale_step)

        # Step 2 — source classification (pharma mode only)
        clf: SourceClassification | None = None
        primary_source_doi: str | None = None
        primary_source_title: str | None = None

        if config.mode == CopilotMode.PHARMA:
            clf, clf_step = classify_source(cv)
            steps.append(clf_step)

            # Step 3 — primary source lookup (pharma, when secondary)
            if config.enable_primary_lookup and clf is not None and not clf.is_primary_source:
                lookup_doi, lookup_title, lookup_step = find_primary_source_doi(
                    cv.source.doi,
                    claim_year=cv.claim.cited_year,
                    db_path=config.db_path,
                    timeout=config.http_timeout,
                )
                # Patch claim_id into the step (find_primary_source_doi doesn't know it)
                lookup_step_with_claim = ProvenanceStep(
                    step_id=lookup_step.step_id,
                    claim_id=cv.claim.claim_id,
                    operation=lookup_step.operation,
                    input_hash=lookup_step.input_hash,
                    output_hash=lookup_step.output_hash,
                    model_id=lookup_step.model_id,
                    timestamp=lookup_step.timestamp,
                    tokens_in=lookup_step.tokens_in,
                    tokens_out=lookup_step.tokens_out,
                    cache_hit=lookup_step.cache_hit,
                    confidence=lookup_step.confidence,
                )
                steps.append(lookup_step_with_claim)
                primary_source_doi = lookup_doi
                primary_source_title = lookup_title

        # Step 4 — recommended_fix
        fix = None
        if config.enable_recommended_fix and cv.result.status in _FIX_VERDICTS:
            fix, fix_step = generate_fix(
                cv,
                rationale=rationale,
                is_primary_source=clf.is_primary_source if clf else None,
                primary_source_doi=primary_source_doi,
                primary_source_title=primary_source_title,
                api_key=config.api_key,
                db_path=config.db_path,
                timeout=config.http_timeout,
            )
            steps.append(fix_step)

        # Assemble CopilotFields — None for mode-disabled fields
        copilot = _build_fields(config.mode, rationale, clf, primary_source_doi, fix)

        logger.info(
            "copilot_enriched",
            claim_id=cv.claim.claim_id,
            verdict=cv.result.status,
            has_fix=fix is not None,
            mode=config.mode.value,
        )

        return EnrichedVerification(
            base=cv,
            copilot=copilot,
            copilot_steps=tuple(steps),
            mode=config.mode,
        )

    def enrich_all(
        self,
        cvs: list[ClaimVerification],
    ) -> list[EnrichedVerification]:
        """Enrich a list of ClaimVerification objects, one at a time."""
        results: list[EnrichedVerification] = []
        for cv in cvs:
            try:
                enriched = self.enrich_one(cv)
                results.append(enriched)
            except Exception:
                logger.exception("enricher_failed", claim_id=cv.claim.claim_id)
        return results


# ---------------------------------------------------------------------------
# Field assembly per mode
# ---------------------------------------------------------------------------


def _build_fields(
    mode: CopilotMode,
    rationale: str,
    clf: SourceClassification | None,
    primary_source_doi: str | None,
    fix: object,  # RecommendedFix | None — avoid circular imports
) -> CopilotFields:
    from src.copilot.models import RecommendedFix  # local import to satisfy mypy

    validated_fix: RecommendedFix | None = fix if isinstance(fix, RecommendedFix) else None

    if mode == CopilotMode.PHARMA:
        return CopilotFields(
            verdict_rationale=rationale,
            recommended_fix=validated_fix,
            is_primary_source=clf.is_primary_source if clf else None,
            study_design=clf.study_design if clf else None,
            risk_of_bias=clf.risk_of_bias if clf else None,
            conflicting_evidence_flag=None,  # set by enricher if multi-source
            primary_source_doi=primary_source_doi,
            novelty_claim=None,
        )

    if mode == CopilotMode.ACADEMIC:
        return CopilotFields(
            verdict_rationale=rationale,
            recommended_fix=validated_fix,
            is_primary_source=None,
            study_design=None,
            risk_of_bias=None,
            conflicting_evidence_flag=None,
            primary_source_doi=None,
            novelty_claim=None,  # future: add novelty classifier
        )

    # GENERAL
    return CopilotFields(
        verdict_rationale=rationale,
        recommended_fix=validated_fix,
        is_primary_source=None,
        study_design=None,
        risk_of_bias=None,
        conflicting_evidence_flag=None,
        primary_source_doi=None,
        novelty_claim=None,
    )
