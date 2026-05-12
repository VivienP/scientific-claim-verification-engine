"""Shared frozen dataclasses and type aliases for the verification pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import structlog

if TYPE_CHECKING:
    from src.numeric.checks import NumericCheckResult

logger: structlog.BoundLogger = structlog.get_logger(__name__)

ClaimType = Literal["factual_numeric", "factual_qualitative", "methodological", "causal"]
VerificationStatus = Literal[
    "supported", "unsupported", "not_addressed", "partially_supported", "unverifiable"
]
UnverifiableReason = Literal[
    "insufficient_evidence_depth",
    "fulltext_unavailable",
    "numeric_claim_abstract_only",
    "parse_error",
]

OperationType = Literal[
    "extract",
    "resolve",
    "fetch_fulltext",
    "chunk_paper",
    "select_passages",
    "verify",
    "verify_cross_modal",
    "aggregate",
    "numeric_extract",
    "numeric_check",
    "copilot_rationale",
    "copilot_primary_source",
    "copilot_primary_lookup",
    "copilot_fix",
]
VerifiabilityStatus = Literal["verifiable", "no_citations_found", "low_citation_density"]
SectionLabel = Literal["introduction", "methods", "results", "discussion", "other"]
RetrievalStatus = Literal["passage_found", "no_passage_found", "fulltext_unavailable"]
EvidenceQuality = Literal[
    "quoted_passage",
    # Fulltext + BM25 found passages, LLM saw them but did not quote any.
    "passages_searched_no_quote",
    "abstract_only",
    "title_only",
    "citing_paper_context",
    "no_evidence",
]

# I1 (2026-05-12): fulltext retrieval telemetry.
# FulltextMethod is named here (rather than in src/fetch_fulltext.py) so that
# FetchOutcome below can reference it without creating an import cycle.
# Consumers (fetch_fulltext.py, pipeline.py, report.py) import it from models.
FulltextMethod = Literal[
    "oa_url_pdf",
    "pmc",
    "publisher_html",
    "europepmc_pdf",
    "unpaywall_pdf",
    "abstract_fallback",
]

# Per-attempt failure category. Granular enough that the weekly
# coverage-by-publisher analysis (scripts/analyze_fetch_coverage.py) can group
# attempts by `(publisher_host, reason)` and surface coverage gaps.
FetchFailureReason = Literal[
    "no_identifiers",  # short-circuit: source has no doi/pmcid/oa_url
    "oa_url_not_pdf",  # oa_url returned non-PDF (paywall HTML page)
    "oa_url_pdf_failed",  # oa_url fetched but PDF extraction failed
    "pmc_no_fulltext",  # pmcid set but PMC has no JATS body
    "publisher_html_unknown",  # DOI prefix not in known-publisher map
    "publisher_html_blocked",  # publisher returned 403/Cloudflare/non-HTML
    "europepmc_no_oa",  # Europe PMC has no OA URL for this DOI
    "europepmc_pdf_failed",  # Europe PMC OA URL present, PDF extract failed
    "unpaywall_no_oa",  # Unpaywall has no OA URL
    "unpaywall_pdf_failed",  # Unpaywall OA URL present, PDF extract failed
    "all_paths_exhausted",  # synthetic terminal reason when chain exits
]


@dataclass(frozen=True)
class PaperChunk:
    doi: str
    section: SectionLabel
    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class Claim:
    claim_id: str
    claim_text: str
    cited_authors: list[str]
    cited_year: int | None
    claim_type: ClaimType
    citation_markers: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class ResolvedSource:
    found: bool
    doi: str | None
    title: str | None
    abstract: str | None
    similarity_score: float | None
    title_match_score: float | None = None
    resolution_low_confidence: bool = False
    oa_url: str | None = None
    pmcid: str | None = None
    retraction_status: bool = False


_NOT_FOUND_SOURCE = ResolvedSource(
    found=False, doi=None, title=None, abstract=None, similarity_score=None
)


@dataclass(frozen=True)
class ResolvedSourceSet:
    """A set of ResolvedSource entries for a multi-citation claim.

    S2-P4: claims with `[81-83]` or `[99, 100]` reference multiple bibliography
    entries that may individually support, contradict, or fail to address the
    claim. The verifier needs all of them to aggregate honestly. The single
    `ResolvedSource` API is preserved through `.primary()` for backward compat,
    so the existing benchmark runner keeps working.
    """

    sources: tuple[ResolvedSource, ...]
    citation_markers: tuple[int, ...]

    def primary(self) -> ResolvedSource:
        """First found source in marker order, or `_NOT_FOUND_SOURCE` when empty.

        Contract: when an author writes `[7, 9]`, ref [7] is the primary
        citation by textual intent. `primary()` honors that order. The
        resolver populates ``self.sources`` parallel to ``citation_markers``,
        so iterating sources is equivalent to iterating markers.

        Bug A (Valsci validation run, 2026-05-08): the previous implementation
        used ``max(sources, key=title_match_score)`` and on `[7, 9]`
        (Kinney + Lo) returned Lo because its title-match-score against
        the claim text "Semantic Scholar database" was higher — even
        though the user listed Kinney first. The score-based heuristic
        was a leaky abstraction; marker order is the only ordering the
        author intentionally provided.

        Returns:
        - The first source whose ``found`` is True, walking sources in
          marker order. This is the marker-order primary.
        - When no source resolved, returns the first source (still
          unfound-shaped) so the result preserves marker-order invariants
          for downstream provenance hashes.
        - When the set is empty, returns ``_NOT_FOUND_SOURCE``.

        Used by callers expecting a single ResolvedSource (legacy single-
        source verifier modes, report headline DOI). The multi-source
        verifier sees the full set via iteration and is unaffected.
        """
        if not self.sources:
            return _NOT_FOUND_SOURCE
        for source in self.sources:
            if source.found:
                return source
        return self.sources[0]

    def found_sources(self) -> tuple[ResolvedSource, ...]:
        """All resolved entries with `found=True`. Empty tuple if none resolved."""
        return tuple(s for s in self.sources if s.found)

    def __iter__(self) -> Iterator[ResolvedSource]:
        return iter(self.sources)

    def __len__(self) -> int:
        return len(self.sources)


@dataclass(frozen=True)
class VerificationResult:
    status: VerificationStatus
    explanation: str
    confidence: float | None  # None only when status == "unverifiable"
    source_passages: list[str] = field(default_factory=list)
    source_section: str | None = None
    fulltext_available: bool = False
    verification_depth: Literal["fulltext", "abstract", "title_only", "citing_paper_context"] = (
        "abstract"
    )
    retrieval_status: RetrievalStatus = "fulltext_unavailable"
    evidence_quality: EvidenceQuality = "abstract_only"
    retraction_status: bool = False
    numeric_check: NumericCheckResult | None = None
    # F1 (2026-05-12): mirrors the field on ProvenanceStep. Populated only
    # when status == "unverifiable". Honest about why the pipeline could not
    # produce a confident verdict — communicates "pipeline access limit" to
    # downstream consumers and report.json readers.
    unverifiable_reason: UnverifiableReason | None = None

    def __post_init__(self) -> None:
        # Invariant 1 (only): unverifiable <-> confidence is None.
        # Decision log 2026-05-11: Invariant 2 (evidence-quality coupling) is
        # NOT enforced here. The evidence-quality discrimination depends on
        # claim_text, which is not a VerificationResult field. That enforcement
        # lives in safe_verification_result() at LLM-response parse boundaries.
        # Direct VerificationResult constructions (test fixtures, internal helpers)
        # are permitted to use any (status, evidence_quality) combination.
        if self.status == "unverifiable" and self.confidence is not None:
            raise ValueError("unverifiable status requires confidence=None")
        if self.status != "unverifiable" and self.confidence is None:
            raise ValueError(f"{self.status!r} status requires non-null confidence")
        # F1: soft check — emit a warning if status is unverifiable but no
        # reason is supplied. Don't raise — keeps backward compatibility with
        # existing serialized reports and test fixtures that predate the field.
        if self.status == "unverifiable" and self.unverifiable_reason is None:
            logger.warning(
                "unverifiable_without_reason",
                explanation_preview=self.explanation[:80],
            )


@dataclass(frozen=True)
class FetchAttempt:
    """One attempt in the full-text retrieval chain.

    I1 (2026-05-12): replaces the implicit "we tried, it returned None"
    signal with explicit per-step reasons so that report.json can answer
    "which publishers fail most often, and at which step?" without re-running
    the pipeline. The attempt order in FetchOutcome.attempts matches the
    chain order in fetch_fulltext.fetch_fulltext.
    """

    method: FulltextMethod
    success: bool
    reason: FetchFailureReason | None  # None on success; populated on failure
    elapsed_ms: int


@dataclass(frozen=True)
class FetchOutcome:
    """Structured result of a full-text retrieval attempt.

    Replaces the old tuple[str | None, FulltextMethod] return shape.
    ``text is None`` means the chain exhausted; the reasons in
    ``attempts`` say WHY. ``method`` is the method that ultimately
    succeeded, or ``"abstract_fallback"`` when no step succeeded.
    """

    text: str | None
    method: FulltextMethod
    attempts: tuple[FetchAttempt, ...]
    elapsed_ms_total: int


@dataclass(frozen=True)
class ProvenanceStep:
    step_id: str
    claim_id: str
    operation: OperationType
    input_hash: str
    output_hash: str
    model_id: str | None
    timestamp: float
    tokens_in: int | None
    tokens_out: int | None
    cache_hit: bool | None
    confidence: float | None
    unverifiable_reason: UnverifiableReason | None = None  # NEW (A2): additive, optional


# ---------------------------------------------------------------------------
# Migration helper (A1/A2): use at LLM response parse boundaries.
# Pure callers with known-valid inputs should construct VerificationResult
# directly. This helper is for sites that receive raw LLM output.
# ---------------------------------------------------------------------------

_INSUFFICIENT_EVIDENCE_SET: frozenset[str] = frozenset(
    {"abstract_only", "title_only", "citing_paper_context", "no_evidence"}
)

# F1 (2026-05-12): explanation template used when the helper downgrades a
# confident LLM verdict to unverifiable. The verdict and explanation must
# be consistent — emitting `unverifiable` with the original LLM's "supported"
# narrative attached is misleading. This template is fully deterministic
# (no LLM call) and surfaces the access limit in plain language.
_EXPLANATION_PREFIX = (
    "Pipeline could not verify this claim with the evidence it was able to "
    "retrieve ({evidence_quality}). The verifier LLM emitted {original_verdict!r} "
    "based on the {evidence_quality} alone, but that depth of evidence is "
    "insufficient for a confident verdict on this claim ({reason}). "
    "Original LLM explanation: "
)
_EXPLANATION_ORIGINAL_TRUNCATE = 240


def _build_unverifiable_explanation(
    *,
    reason: UnverifiableReason,
    original_llm_verdict: str,
    original_explanation: str,
    evidence_quality: EvidenceQuality,
) -> str:
    """Construct a structured explanation for a downgraded verdict.

    Pure-Python deterministic function. The verdict + explanation pair stays
    consistent: when the helper downgrades the status, the explanation is
    rewritten so a downstream reader sees the same story in both fields.
    """
    original = (original_explanation or "(no original explanation)").strip()
    if len(original) > _EXPLANATION_ORIGINAL_TRUNCATE:
        original = original[:_EXPLANATION_ORIGINAL_TRUNCATE].rstrip() + "..."
    prefix = _EXPLANATION_PREFIX.format(
        evidence_quality=evidence_quality,
        original_verdict=original_llm_verdict,
        reason=reason,
    )
    return prefix + original


def safe_verification_result(
    *,
    status: str,
    confidence: float | None,
    evidence_quality: EvidenceQuality = "abstract_only",
    claim_text: str | None = None,
    unverifiable_reason: UnverifiableReason | None = None,
    **kwargs: Any,  # noqa: ANN401 — forwarded verbatim to VerificationResult dataclass fields
) -> VerificationResult:
    """Construct a VerificationResult, downgrading to unverifiable when needed.

    Use at LLM-response parse boundaries where the raw parsed status/confidence
    may violate the invariant. Pure callers that know their inputs satisfy the
    invariant should construct VerificationResult directly.

    Downgrade rule (all must hold):
      1. status in {"supported", "unsupported"}
      2. evidence_quality in {abstract_only, title_only,
                              citing_paper_context, no_evidence}
      3. claim_text is None (legacy callers — fail safe) OR
         _claim_has_specific_numeric(claim_text) is True

    Qualitative claims (no specific numeric content) on abstract-only evidence
    pass through unchanged — the abstract is sufficient for 'X reduces Y'-style
    verdicts when it directly addresses the topic.

    On downgrade (F1, 2026-05-12):
    - `unverifiable_reason` is populated on the result (defaults to
      ``"numeric_claim_abstract_only"`` when caller doesn't specify).
    - `explanation` is rewritten by ``_build_unverifiable_explanation`` so the
      verdict and the explanation stay consistent. The original LLM
      explanation is preserved as a truncated suffix for auditability.

    Additional rules:
    - unverifiable + non-None confidence -> confidence forced to None
    - All other combinations pass through unchanged.
    """
    from src.numeric.heuristics import _claim_has_specific_numeric

    if (
        status in ("supported", "unsupported")
        and evidence_quality in _INSUFFICIENT_EVIDENCE_SET
        and (claim_text is None or _claim_has_specific_numeric(claim_text))
    ):
        reason: UnverifiableReason = unverifiable_reason or "numeric_claim_abstract_only"
        original_explanation = str(kwargs.get("explanation", ""))
        new_explanation = _build_unverifiable_explanation(
            reason=reason,
            original_llm_verdict=status,
            original_explanation=original_explanation,
            evidence_quality=evidence_quality,
        )
        return VerificationResult(
            status="unverifiable",
            confidence=None,
            evidence_quality=evidence_quality,
            unverifiable_reason=reason,
            **{**kwargs, "explanation": new_explanation},
        )
    if status == "unverifiable":
        confidence = None
        # Caller emitted unverifiable directly (e.g. parse-error fallback).
        # If caller didn't specify a reason but we can propagate one cheaply,
        # do so. Otherwise leave None and the soft post_init warning fires.
        if unverifiable_reason is not None and "unverifiable_reason" not in kwargs:
            kwargs = {**kwargs, "unverifiable_reason": unverifiable_reason}
    return VerificationResult(
        status=status,  # type: ignore[arg-type]  # caller-validated status string
        confidence=confidence,
        evidence_quality=evidence_quality,
        **kwargs,
    )
