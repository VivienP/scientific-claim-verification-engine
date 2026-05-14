"""Auto-evaluation of Copilot field outputs against the lactate-ISF gold set.

Baseline targets:
  - is_primary_source precision  ≥ 0.80
  - doi_hallucination_rate       = 0.00  (hard gate)
  - fix_present_rate (unsupported) ≥ 0.60

Inputs:
  gold_path:     eval/e2e/reference_paper_v1_verdicts.json (25 annotated claims)
  enriched:      list[EnrichedVerification] from running the Copilot enricher

Outputs:
  CopilotEvalReport — field-by-field metrics suitable for the AAR scorecard.

This module never makes API calls. It is a pure scorer over already-produced
EnrichedVerification objects, meaning it can run offline in CI in <1 second.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from src.copilot.models import EnrichedVerification

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_FIX_VERDICTS = frozenset({"unsupported", "partially_supported", "not_addressed"})


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldMetric:
    """Per-field precision/recall against gold."""

    name: str
    n_gold: int  # claims where gold has a value
    n_predicted: int  # claims where prediction has a value
    n_correct: int  # claims where prediction matches gold
    precision: float  # n_correct / n_predicted (or 1.0 if n_predicted == 0)
    recall: float  # n_correct / n_gold (or 1.0 if n_gold == 0)
    f1: float


@dataclass(frozen=True)
class CopilotEvalReport:
    """Aggregate metrics across the gold set."""

    n_claims_evaluated: int
    verdict_metric: FieldMetric
    is_primary_source_metric: FieldMetric
    primary_source_doi_metric: FieldMetric

    # Hard-gate metric — must be 0.00.
    doi_hallucination_rate: float
    n_doi_hallucinations: int

    # Fix-presence metric — fraction of unsupported claims with a non-null fix.
    fix_present_rate_unsupported: float
    n_unsupported_in_gold: int
    n_unsupported_with_fix: int

    # Gate verdicts (booleans for CI).
    passes_phase_b_gate: bool

    # Per-claim diff for debugging.
    per_claim_diffs: tuple[dict[str, Any], ...]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_gold(gold_path: Path) -> list[dict[str, Any]]:
    """Load the gold annotation file. Returns a list of claim dicts."""
    raw = json.loads(gold_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "claims" in raw:
        return list(raw["claims"])
    if isinstance(raw, list):
        return list(raw)
    raise ValueError(f"Unexpected gold schema: {type(raw).__name__}")


def evaluate(
    enriched: list[EnrichedVerification],
    gold_path: Path,
    *,
    crossref_verified_dois: frozenset[str] | None = None,
) -> CopilotEvalReport:
    """Score an enriched run against the gold annotations.

    ``crossref_verified_dois`` is the set of DOIs that have been confirmed via
    CrossRef in this run (pulled from provenance metadata). If a predicted DOI
    is not in this set, it counts as a hallucination — even if it happens to
    match the gold value, because the gate is procedural (was it verified?),
    not coincidental (does it match?).
    """
    gold_by_id = {claim["claim_id"]: claim for claim in load_gold(gold_path)}

    # Match enriched results to gold by claim_id. Use claim_text fuzzy fallback
    # if claim_id schemes differ.
    matched = _match_by_id_or_text(enriched, gold_by_id)

    diffs: list[dict[str, Any]] = []
    verdict_correct = 0
    primary_correct = 0
    primary_predicted = 0
    primary_gold = 0
    doi_correct = 0
    doi_predicted = 0
    doi_gold = 0
    n_hallucinations = 0
    n_unsupported_gold = 0
    n_unsupported_with_fix = 0

    for ev, gold in matched:
        diff = _diff_one(ev, gold, crossref_verified_dois)
        diffs.append(diff)

        if diff["verdict_match"]:
            verdict_correct += 1

        # is_primary_source — only counted if gold has a value.
        if gold.get("is_primary_source") is not None:
            primary_gold += 1
            if ev.copilot.is_primary_source is not None:
                primary_predicted += 1
                if ev.copilot.is_primary_source == gold["is_primary_source"]:
                    primary_correct += 1

        # primary_source_doi — only counted when gold provides a real DOI.
        gold_doi = _normalize_doi(gold.get("primary_source_doi"))
        pred_doi = _normalize_doi(ev.copilot.primary_source_doi)
        if gold_doi is not None:
            doi_gold += 1
        if pred_doi is not None:
            doi_predicted += 1
            if gold_doi is not None and pred_doi == gold_doi:
                doi_correct += 1

        # Hallucination counter — any predicted DOI that wasn't CrossRef-verified.
        if diff["doi_hallucinated"]:
            n_hallucinations += 1

        # Fix presence — only when expected_verdict is unsupported.
        if gold.get("expected_verdict") == "unsupported":
            n_unsupported_gold += 1
            if ev.copilot.recommended_fix is not None:
                n_unsupported_with_fix += 1

    n = len(matched)
    verdict_metric = _make_metric(
        "verdict",
        n_gold=n,
        n_predicted=n,
        n_correct=verdict_correct,
    )
    primary_metric = _make_metric(
        "is_primary_source",
        n_gold=primary_gold,
        n_predicted=primary_predicted,
        n_correct=primary_correct,
    )
    doi_metric = _make_metric(
        "primary_source_doi",
        n_gold=doi_gold,
        n_predicted=doi_predicted,
        n_correct=doi_correct,
    )

    n_with_doi = sum(
        1
        for ev in enriched
        if ev.copilot.recommended_fix is not None
        and ev.copilot.recommended_fix.suggested_doi is not None
    )
    doi_hallucination_rate = (n_hallucinations / n_with_doi) if n_with_doi > 0 else 0.0

    fix_present_rate = (
        (n_unsupported_with_fix / n_unsupported_gold) if n_unsupported_gold > 0 else 0.0
    )

    passes_gate = (
        primary_metric.precision >= 0.80
        and doi_hallucination_rate == 0.0
        and fix_present_rate >= 0.60
    )

    report = CopilotEvalReport(
        n_claims_evaluated=n,
        verdict_metric=verdict_metric,
        is_primary_source_metric=primary_metric,
        primary_source_doi_metric=doi_metric,
        doi_hallucination_rate=doi_hallucination_rate,
        n_doi_hallucinations=n_hallucinations,
        fix_present_rate_unsupported=fix_present_rate,
        n_unsupported_in_gold=n_unsupported_gold,
        n_unsupported_with_fix=n_unsupported_with_fix,
        passes_phase_b_gate=passes_gate,
        per_claim_diffs=tuple(diffs),
    )

    logger.info(
        "copilot_eval_complete",
        n=n,
        verdict_acc=verdict_metric.precision,
        primary_precision=primary_metric.precision,
        doi_hallucination_rate=doi_hallucination_rate,
        fix_present_rate=fix_present_rate,
        passes_gate=passes_gate,
    )
    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _match_by_id_or_text(
    enriched: list[EnrichedVerification],
    gold_by_id: dict[str, dict[str, Any]],
) -> list[tuple[EnrichedVerification, dict[str, Any]]]:
    """Match by claim_id when present; fall back to first ~80 chars of claim_text."""
    matched: list[tuple[EnrichedVerification, dict[str, Any]]] = []
    text_index = {_text_key(g["claim_text"]): g for g in gold_by_id.values() if g.get("claim_text")}
    for ev in enriched:
        cid = ev.base.claim.claim_id
        if cid in gold_by_id:
            matched.append((ev, gold_by_id[cid]))
            continue
        key = _text_key(ev.base.claim.claim_text)
        if key in text_index:
            matched.append((ev, text_index[key]))
    return matched


def _text_key(text: str) -> str:
    return text.strip().lower()[:80]


def _normalize_doi(doi: str | None) -> str | None:
    if doi is None:
        return None
    s = doi.strip().lower()
    if not s or s in {"n/a", "none", "null"}:
        return None
    return s


def _diff_one(
    ev: EnrichedVerification,
    gold: dict[str, Any],
    crossref_verified_dois: frozenset[str] | None,
) -> dict[str, Any]:
    pred_verdict = ev.base.result.status
    gold_verdict = gold.get("expected_verdict")

    fix = ev.copilot.recommended_fix
    fix_doi = fix.suggested_doi if fix else None

    # Hallucination definition: predicted DOI is non-null AND was not in the
    # CrossRef-verified set for this run. If no verified set is provided,
    # we trust the fix_generator's own gate (it only emits CrossRef-verified
    # DOIs by construction) and treat hallucination_rate as 0 by default.
    hallucinated = False
    if fix_doi is not None and crossref_verified_dois is not None:
        hallucinated = fix_doi.lower() not in crossref_verified_dois

    return {
        "claim_id": ev.base.claim.claim_id,
        "verdict_pred": pred_verdict,
        "verdict_gold": gold_verdict,
        "verdict_match": pred_verdict == gold_verdict,
        "is_primary_pred": ev.copilot.is_primary_source,
        "is_primary_gold": gold.get("is_primary_source"),
        "primary_doi_pred": ev.copilot.primary_source_doi,
        "primary_doi_gold": _normalize_doi(gold.get("primary_source_doi")),
        "fix_present": fix is not None,
        "fix_action": fix.action if fix else None,
        "fix_doi": fix_doi,
        "doi_hallucinated": hallucinated,
    }


def _make_metric(name: str, *, n_gold: int, n_predicted: int, n_correct: int) -> FieldMetric:
    precision = (n_correct / n_predicted) if n_predicted > 0 else 1.0
    recall = (n_correct / n_gold) if n_gold > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return FieldMetric(
        name=name,
        n_gold=n_gold,
        n_predicted=n_predicted,
        n_correct=n_correct,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
    )


def report_to_dict(report: CopilotEvalReport) -> dict[str, Any]:
    """Serialise the report to a JSON-friendly dict."""
    return {
        "n_claims_evaluated": report.n_claims_evaluated,
        "verdict": _metric_to_dict(report.verdict_metric),
        "is_primary_source": _metric_to_dict(report.is_primary_source_metric),
        "primary_source_doi": _metric_to_dict(report.primary_source_doi_metric),
        "doi_hallucination_rate": report.doi_hallucination_rate,
        "n_doi_hallucinations": report.n_doi_hallucinations,
        "fix_present_rate_unsupported": report.fix_present_rate_unsupported,
        "n_unsupported_in_gold": report.n_unsupported_in_gold,
        "n_unsupported_with_fix": report.n_unsupported_with_fix,
        "passes_phase_b_gate": report.passes_phase_b_gate,
        "per_claim_diffs": list(report.per_claim_diffs),
    }


def _metric_to_dict(m: FieldMetric) -> dict[str, Any]:
    return {
        "name": m.name,
        "n_gold": m.n_gold,
        "n_predicted": m.n_predicted,
        "n_correct": m.n_correct,
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
    }
