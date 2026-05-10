"""is_primary_source classifier — deterministic two-stage heuristic, zero LLM calls.

Stage 1 (optional): CrossRef work-type string, caller-provided.
  - Certain types (book, edited-book, reference-entry) are always secondary.
  - Preprints (posted-content) count as primary (no peer-review flag).
Stage 2: Abstract keyword scan → study_design, is_primary, risk_of_bias.
  - Falls back to safe defaults when abstract is absent.

Emits ProvenanceStep(operation="copilot_primary_source", model_id=None).
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass

import structlog

from src.copilot.models import RiskOfBias, StudyDesign
from src.models import ProvenanceStep
from src.pipeline import ClaimVerification
from src.verify_prompts import _hash

logger: structlog.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Stage 1 — CrossRef work-type sets
# ---------------------------------------------------------------------------

_ALWAYS_SECONDARY_TYPES: frozenset[str] = frozenset(
    {"book", "edited-book", "reference-entry", "proceedings", "dissertation"}
)
_LIKELY_SECONDARY_TYPES: frozenset[str] = frozenset(
    {"book-chapter", "standard", "report", "dataset", "grant"}
)
# posted-content = preprints → treat as primary (no peer-review, but primary research)
_ALWAYS_PRIMARY_TYPES: frozenset[str] = frozenset({"posted-content"})

# ---------------------------------------------------------------------------
# Stage 2 — Abstract regex patterns
# ---------------------------------------------------------------------------

_RCT_RE = re.compile(
    r"\b(randomi[sz]ed\s+controlled\s+trial|RCT|double[- ]blind(ed)?|"
    r"placebo[- ]controlled|randomis(ed|ation)|randomiz(ed|ation))\b",
    re.IGNORECASE,
)
_META_RE = re.compile(
    r"\b(meta[- ]analy(sis|ses)|pooled\s+analy(sis|ses)|network\s+meta[- ]analysis)\b",
    re.IGNORECASE,
)
_SYSTEMATIC_REVIEW_RE = re.compile(
    r"\b(systematic\s+(literature\s+)?review|PRISMA)\b",
    re.IGNORECASE,
)
_NARRATIVE_REVIEW_RE = re.compile(
    r"\b(narrative\s+review|literature\s+review|review\s+article|review\s+of\s+(the\s+)?"
    r"(literature|evidence))\b",
    re.IGNORECASE,
)
_GUIDELINES_RE = re.compile(
    r"\b(clinical\s+(practice\s+)?guidelines?|consensus\s+statement|"
    r"practice\s+recommendations?|expert\s+(panel|consensus))\b",
    re.IGNORECASE,
)
_OBSERVATIONAL_RE = re.compile(
    r"\b(cohort\s+study|prospective\s+(cohort|observational|study)|"
    r"retrospective\s+(cohort|analysis|study)|longitudinal\s+study|"
    r"cross[- ]sectional\s+(study|survey)|population[- ]based\s+study)\b",
    re.IGNORECASE,
)
_CASE_CONTROL_RE = re.compile(
    r"\b(case[- ]control\s+study|matched\s+case[- ]control)\b",
    re.IGNORECASE,
)
_ANIMAL_MODEL_RE = re.compile(
    r"\b(animal\s+model|mouse\s+model|rat\s+model|murine\s+model|"
    r"in\s+vivo|rodent\s+model|non[- ]?human\s+primate)\b",
    re.IGNORECASE,
)
_IN_VITRO_RE = re.compile(
    r"\b(in\s+vitro|cell\s+(culture|line)|cell[- ]based\s+assay|"
    r"HeLa|HEK293|primary\s+cell|organoid)\b",
    re.IGNORECASE,
)
# Combined primary-signal pattern exported for reuse in primary_lookup.py
_PRIMARY_SIGNALS_RE_FOR_LOOKUP = re.compile(
    r"\b(randomi[sz]ed\s+controlled\s+trial|RCT|double[- ]blind(ed)?|"
    r"cohort\s+study|prospective\s+cohort|retrospective\s+(cohort|analysis)|"
    r"cross[- ]sectional|case[- ]control\s+study|"
    r"animal\s+model|mouse\s+model|rat\s+model|in\s+vivo|"
    r"in\s+vitro|cell\s+(culture|line)|"
    r"clinical\s+trial|randomis(ed|ation)|placebo[- ]controlled)\b",
    re.IGNORECASE,
)

# Risk-of-bias signals
_HIGH_QUALITY_RE = re.compile(
    r"\b(double[- ]blind(ed)?|triple[- ]blind(ed)?|randomis(ed|ation)|"
    r"placebo[- ]controlled|allocation\s+concealment|blinding|intent[- ]to[- ]treat)\b",
    re.IGNORECASE,
)
_LOW_QUALITY_RE = re.compile(
    r"\b(case\s+report|pilot\s+(study|trial)|small\s+sample|"
    r"preliminary\s+(study|findings|results)|proof[- ]of[- ]concept|"
    r"exploratory\s+(study|analysis))\b",
    re.IGNORECASE,
)
_SAMPLE_SIZE_RE = re.compile(
    r"""
    (?:
        [nN]\s*=\s*(\d+)            # n = 123 or N=123
        |(\d+)\s+patients?          # 123 patients
        |(\d+)\s+participants?      # 123 participants
        |(\d+)\s+subjects?          # 123 subjects
        |(\d+)\s+individuals?       # 123 individuals
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceClassification:
    """Deterministic study-type assessment for one resolved source."""

    is_primary_source: bool
    study_design: StudyDesign
    risk_of_bias: RiskOfBias


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_max_n(abstract: str) -> int | None:
    """Return the largest sample size found in ``abstract``, or None."""
    matches = _SAMPLE_SIZE_RE.findall(abstract)
    sizes = [int(g) for match in matches for g in match if g]
    return max(sizes) if sizes else None


def _classify_study_design(abstract: str) -> tuple[StudyDesign, bool]:
    """Return (study_design, is_primary_source) from abstract text.

    Priority order: specific design signals beat generic ones. Returns
    ("unknown", False) when the abstract is empty or gives no signal.
    """
    if not abstract:
        return "unknown", False

    if _META_RE.search(abstract):
        return "meta_analysis", False
    if _SYSTEMATIC_REVIEW_RE.search(abstract):
        return "systematic_review", False
    if _NARRATIVE_REVIEW_RE.search(abstract):
        return "narrative_review", False
    if _GUIDELINES_RE.search(abstract):
        return "guidelines", False
    if _RCT_RE.search(abstract):
        return "rct", True
    if _CASE_CONTROL_RE.search(abstract):
        return "case_control", True
    if _OBSERVATIONAL_RE.search(abstract):
        return "observational", True
    if _ANIMAL_MODEL_RE.search(abstract):
        return "animal_model", True
    if _IN_VITRO_RE.search(abstract):
        return "in_vitro", True

    return "unknown", False


def _assess_risk_of_bias(
    abstract: str,
    study_design: StudyDesign,
    is_primary: bool,
) -> RiskOfBias:
    """Heuristic risk-of-bias assessment.

    Only meaningful for primary studies; secondary sources return "unknown"
    because bias assessment of reviews is a different (and out-of-scope) task.
    """
    if not is_primary or not abstract:
        return "unknown"

    if _LOW_QUALITY_RE.search(abstract):
        return "high"

    n = _extract_max_n(abstract)
    if n is not None and n < 10:
        return "high"

    if study_design == "rct" and _HIGH_QUALITY_RE.search(abstract):
        if n is not None and n >= 50:
            return "low"
        return "medium"

    if study_design in {"rct", "observational", "case_control"}:
        return "medium" if n is None or n >= 10 else "high"

    return "unknown"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_source(
    cv: ClaimVerification,
    *,
    crossref_work_type: str | None = None,
) -> tuple[SourceClassification, ProvenanceStep]:
    """Classify the primary source of ``cv`` without any LLM call.

    Args:
        cv: ClaimVerification from V1 pipeline.
        crossref_work_type: Optional CrossRef ``type`` field for stage-1
            classification. Pass None (default) to rely on abstract only.

    Returns:
        (SourceClassification, ProvenanceStep) — both always present.
        model_id=None in ProvenanceStep (deterministic, no LLM).
    """
    claim_id = cv.claim.claim_id
    source = cv.source
    abstract = source.abstract or ""

    ts = time.time()
    input_repr = repr((source.doi, source.abstract, crossref_work_type))
    input_hash = _hash(input_repr)

    # Stage 1 — CrossRef work-type (when provided)
    if crossref_work_type is not None:
        if crossref_work_type in _ALWAYS_SECONDARY_TYPES:
            clf = SourceClassification(
                is_primary_source=False,
                study_design="narrative_review",
                risk_of_bias="unknown",
            )
            logger.info(
                "primary_source_stage1_secondary",
                claim_id=claim_id,
                crossref_type=crossref_work_type,
            )
            return clf, _make_step(claim_id, input_hash, clf, ts)

        if crossref_work_type in _ALWAYS_PRIMARY_TYPES:
            # Preprint: run stage 2 for design/bias but force is_primary=True
            design, _ = _classify_study_design(abstract)
            rob = _assess_risk_of_bias(abstract, design, True)
            clf = SourceClassification(
                is_primary_source=True, study_design=design, risk_of_bias=rob
            )
            logger.info(
                "primary_source_stage1_primary",
                claim_id=claim_id,
                crossref_type=crossref_work_type,
            )
            return clf, _make_step(claim_id, input_hash, clf, ts)

        if crossref_work_type in _LIKELY_SECONDARY_TYPES:
            clf = SourceClassification(
                is_primary_source=False,
                study_design="unknown",
                risk_of_bias="unknown",
            )
            logger.info(
                "primary_source_stage1_likely_secondary",
                claim_id=claim_id,
                crossref_type=crossref_work_type,
            )
            return clf, _make_step(claim_id, input_hash, clf, ts)

    # Stage 2 — Abstract keyword scan
    design, is_primary = _classify_study_design(abstract)
    rob = _assess_risk_of_bias(abstract, design, is_primary)
    clf = SourceClassification(is_primary_source=is_primary, study_design=design, risk_of_bias=rob)

    logger.info(
        "primary_source_classified",
        claim_id=claim_id,
        is_primary=is_primary,
        study_design=design,
        risk_of_bias=rob,
    )
    return clf, _make_step(claim_id, input_hash, clf, ts)


def _make_step(
    claim_id: str, input_hash: str, clf: SourceClassification, ts: float
) -> ProvenanceStep:
    return ProvenanceStep(
        step_id=str(uuid.uuid4()),
        claim_id=claim_id,
        operation="copilot_primary_source",
        input_hash=input_hash,
        output_hash=_hash(repr(clf)),
        model_id=None,
        timestamp=ts,
        tokens_in=None,
        tokens_out=None,
        cache_hit=None,
        confidence=None,
    )
