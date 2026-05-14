"""Pure-Python measurement logic for the end-to-end benchmark.

This module is library code: alignment heuristics, DOI normalization, and
metric computation. The orchestration (loading the paper, running the
pipeline, writing results) lives in `scripts/measure_e2e_recall.py`.

Splitting library from CLI keeps the testable surface free of pipeline
imports and lets unit tests run offline without Anthropic credentials.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

from eval.e2e.schema import GroundTruthClaim
from src.models import Claim, ResolvedSource, VerificationResult

_LEXICAL_MATCH_FLOOR = 0.50
_LEXICAL_MATCH_STRONG = 0.85
_MATCH_THRESHOLD = 0.55


@dataclass(frozen=True)
class ClaimMatch:
    """One alignment between a ground-truth claim and an extracted claim.

    `score` is the matching strength (0..1). `extracted_claim_id` is None when
    the GT claim was not matched to any extracted claim.
    """

    gt_claim_id: str
    extracted_claim_id: str | None
    score: float


@dataclass(frozen=True)
class Metrics:
    extraction_recall: float
    extraction_precision: float
    resolution_accuracy: float
    e2e_coverage_useful: float
    not_addressed_unknown_cause: float
    counts: dict[str, int]


def normalize_surname(author: str) -> str:
    """Return a normalized surname for fuzzy comparison.

    Handles "Smith", "Smith, J.", "John Smith", "Smith et al." variants.
    """
    cleaned = author.strip().lower()
    cleaned = cleaned.replace(" et al.", "").replace(" et al", "")
    if "," in cleaned:
        return cleaned.split(",", 1)[0].strip()
    parts = cleaned.split()
    return parts[-1] if parts else cleaned


def author_overlap(extracted: list[str], gt: list[str]) -> bool:
    if not gt:
        return True
    extracted_surnames = {normalize_surname(a) for a in extracted if a}
    gt_surnames = {normalize_surname(a) for a in gt if a}
    return bool(extracted_surnames & gt_surnames)


def year_match(extracted: int | None, gt: int | None) -> bool:
    if extracted is None or gt is None:
        return True
    return abs(extracted - gt) <= 1


def score_pair(extracted: Claim, gt: GroundTruthClaim) -> float:
    """Score the alignment between an extracted claim and a GT claim.

    Returns a value in [0, 1]. Higher is better. 0 = not a candidate match.
    """
    lexical = difflib.SequenceMatcher(
        None, extracted.claim_text.lower(), gt.claim_text.lower()
    ).ratio()

    if lexical < _LEXICAL_MATCH_FLOOR:
        return 0.0

    author_ok = author_overlap(extracted.cited_authors, gt.cited_authors)
    year_ok = year_match(extracted.cited_year, gt.cited_year)

    if lexical >= _LEXICAL_MATCH_STRONG:
        return lexical
    if author_ok and year_ok:
        return lexical
    return 0.0


def align_claims(gt_claims: list[GroundTruthClaim], extracted: list[Claim]) -> list[ClaimMatch]:
    """Greedy 1-to-1 alignment of GT claims to extracted claims.

    For each GT claim, picks the highest-scoring unassigned extracted claim
    above _MATCH_THRESHOLD. Returns one ClaimMatch per GT claim (with
    extracted_claim_id=None if no match found).
    """
    used_extracted: set[str] = set()
    matches: list[ClaimMatch] = []

    for gt in gt_claims:
        best_id: str | None = None
        best_score = 0.0
        for ex in extracted:
            if ex.claim_id in used_extracted:
                continue
            score = score_pair(ex, gt)
            if score > best_score:
                best_score = score
                best_id = ex.claim_id

        if best_id is not None and best_score >= _MATCH_THRESHOLD:
            used_extracted.add(best_id)
            matches.append(ClaimMatch(gt.gt_claim_id, best_id, best_score))
        else:
            matches.append(ClaimMatch(gt.gt_claim_id, None, best_score))

    return matches


def normalize_doi(doi: str | None) -> str | None:
    if doi is None:
        return None
    cleaned = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    return cleaned or None


def compute_metrics(
    gt_claims: list[GroundTruthClaim],
    extracted: list[Claim],
    sources: dict[str, ResolvedSource],
    verifications: dict[str, VerificationResult],
    matches: list[ClaimMatch],
) -> Metrics:
    """Compute the 5 metrics from aligned claims + pipeline outputs.

    `not_addressed_unknown_cause` is 1.0 whenever any not_addressed verdict is
    present. The metric reflects the actual fraction with no assigned cause
    once per-cause attribution is available in VerificationResult.
    """
    n_gt = len(gt_claims)
    n_extracted = len(extracted)
    n_matched = sum(1 for m in matches if m.extracted_claim_id is not None)

    extraction_recall = n_matched / n_gt if n_gt else 0.0
    extraction_precision = n_matched / n_extracted if n_extracted else 0.0

    gt_by_id = {gt.gt_claim_id: gt for gt in gt_claims}

    n_secondary_with_doi_gt = sum(
        1
        for gt in gt_claims
        if gt.claim_origin == "secondary" and normalize_doi(gt.ground_truth_doi) is not None
    )

    n_correct_doi = 0
    n_resolution_attempts_for_matched = 0

    for m in matches:
        if m.extracted_claim_id is None:
            continue
        gt = gt_by_id[m.gt_claim_id]
        if gt.claim_origin != "secondary":
            continue
        gt_doi = normalize_doi(gt.ground_truth_doi)
        if gt_doi is None:
            continue
        n_resolution_attempts_for_matched += 1
        resolved = sources.get(m.extracted_claim_id)
        if resolved is None or not resolved.found:
            continue
        if normalize_doi(resolved.doi) == gt_doi:
            n_correct_doi += 1

    resolution_accuracy = (
        n_correct_doi / n_resolution_attempts_for_matched
        if n_resolution_attempts_for_matched
        else 0.0
    )
    e2e_coverage_useful = (
        n_correct_doi / n_secondary_with_doi_gt if n_secondary_with_doi_gt else 0.0
    )

    n_not_addressed = sum(1 for v in verifications.values() if v.status == "not_addressed")
    not_addressed_unknown_cause = 1.0 if n_not_addressed else 0.0

    counts = {
        "n_gt_claims": n_gt,
        "n_extracted_claims": n_extracted,
        "n_matched": n_matched,
        "n_unmatched_gt": n_gt - n_matched,
        "n_unmatched_extracted": n_extracted - n_matched,
        "n_secondary_with_doi_gt": n_secondary_with_doi_gt,
        "n_resolution_attempts_for_matched": n_resolution_attempts_for_matched,
        "n_correct_doi": n_correct_doi,
        "n_not_addressed": n_not_addressed,
    }

    return Metrics(
        extraction_recall=extraction_recall,
        extraction_precision=extraction_precision,
        resolution_accuracy=resolution_accuracy,
        e2e_coverage_useful=e2e_coverage_useful,
        not_addressed_unknown_cause=not_addressed_unknown_cause,
        counts=counts,
    )
