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
ClaimDirection = Literal["increase", "decrease", "no_effect", "unclear"]
VerificationStatus = Literal[
    "supported", "unsupported", "not_addressed", "partially_supported", "unverifiable"
]
UnverifiableReason = Literal[
    "insufficient_evidence_depth",
    "fulltext_unavailable",
    "numeric_claim_abstract_only",
    "parse_error",
    "resolution_low_confidence",
    "resolution_source_disagreement",
    "low_extraction_confidence",  # 3.3: LLM self-reported extraction uncertainty
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

# Fulltext retrieval telemetry. FulltextMethod lives here (not in
# src/fetch_fulltext.py) so FetchOutcome can reference it without creating
# an import cycle; consumers import it from models.
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

# Which retrieval client produced a CandidateResolution. Used by the resolver
# verdict folder to attribute agreement/disagreement signals to specific
# sources, so a disputed verdict surfaces "crossref vs openalex" rather than
# anonymous "source disagreement".
CandidateClient = Literal["crossref", "openalex", "pubmed", "arxiv"]

# How confident the resolver is that the resolved source is the right paper.
# "corroborated" means >=2 clients agree on DOI or on (year, first_author, venue).
# "disputed" means >=2 clients returned different DOIs and disagreement on
#   the agreement-signal fields exceeds the threshold.
# "low_confidence" means a single candidate with weak signals (existing
#   OpenAlex < 0.15 / arXiv similar threshold logic surfaces here).
# "single_source_only" means only one client returned a candidate so the
#   verdict cannot be tested for agreement.
ResolutionStatus = Literal["corroborated", "disputed", "low_confidence", "single_source_only"]

# Outcome of a single PDF download + extraction attempt. Distinguishes the
# typical paywall pattern (non-PDF Content-Type) from real HTTP failures so
# fetch_fulltext.py can attribute FetchAttempt.reason precisely.
PdfFailureReason = Literal[
    "ok",  # text retrieved successfully (text is non-None and above _MIN_TEXT_LENGTH)
    "http_error",  # non-2xx response from publisher endpoint
    "not_a_pdf",  # Content-Type did not include "pdf" — typical paywall HTML page
    "extraction_failed",  # pymupdf raised on the downloaded bytes
    "too_short",  # extracted text below the _MIN_TEXT_LENGTH threshold
    "timeout",  # request exceeded the httpx timeout
]

# Depth of evidence available for verification. Mirrors EvidenceQuality but is
# the policy-input shape: independent of which prompt/verifier was invoked.
EvidenceDepth = Literal["fulltext", "abstract", "title", "none"]

# Whether the pipeline was able to access the cited source at all.
# "available" — fulltext or abstract text retrieved
# "unavailable" — no identifiers / no OA URL / Europe PMC silent
# "blocked" — publisher returned paywall HTML or 403/Cloudflare
# "unresolved" — resolver could not produce a usable ResolvedSource
AccessStatus = Literal["available", "unavailable", "blocked", "unresolved"]


@dataclass(frozen=True)
class CandidateResolution:
    """One client's answer to "what DOI matches this citation?".

    The resolver collects up to one CandidateResolution per client (CrossRef,
    OpenAlex, PubMed, arXiv) and folds them into a ResolutionVerdict. Each
    candidate carries enough metadata for the verdict folder to detect
    cross-client agreement on (year, first_author, venue) when the DOIs
    themselves differ.
    """

    client: CandidateClient
    doi: str | None
    title: str | None
    year: int | None
    first_author: str | None
    venue: str | None  # journal / conference / preprint server


@dataclass(frozen=True)
class ResolutionVerdict:
    """Cross-source resolution verdict for a single citation.

    ``status`` is the policy-input shape consumed by ``assess_evidence_sufficiency``.
    ``candidates`` retains per-client diagnostics for auditor visibility — when a
    verdict is "disputed", the auditor can see exactly which client returned
    which DOI.

    ``agreement_signals`` records which fields agreed across candidates when
    status is "corroborated" (e.g. ``("doi",)`` for an exact match across
    clients, or ``("year", "first_author", "venue")`` for fuzzy agreement).
    """

    status: ResolutionStatus
    candidates: tuple[CandidateResolution, ...] = ()
    agreement_signals: tuple[str, ...] = ()


@dataclass(frozen=True)
class PdfFetchOutcome:
    """Structured result of a single PDF download + extraction attempt.

    Replaces the legacy ``str | None`` return shape of
    ``src/clients/pdf.py::download_and_extract``. ``failure_reason`` is the
    single source of truth that lets ``fetch_fulltext.py`` populate
    ``FetchAttempt.reason`` with publisher-specific signal (paywall HTML page
    is "not_a_pdf"; a real 403 is "http_error"; a malformed PDF that pymupdf
    cannot open is "extraction_failed").
    """

    text: str | None
    failure_reason: PdfFailureReason
    http_status: int | None = None
    content_type: str | None = None


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
    # Structured assertion fields populated by extract_v2; all optional so v1
    # responses and pre-existing fixtures continue to construct cleanly.
    source_quote: str | None = None
    subject: str | None = None
    population: str | None = None
    intervention: str | None = None
    comparator: str | None = None
    outcome: str | None = None
    direction: ClaimDirection | None = None
    numeric_value: str | None = None
    time_horizon: str | None = None
    extraction_confidence: float | None = None


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
    # Cross-source verdict carrying per-client CandidateResolution diagnostics.
    # Optional for back-compat — legacy resolutions and fixtures serialize
    # without this field. Populated by the resolver; consumed by
    # assess_evidence_sufficiency.
    resolution_verdict: ResolutionVerdict | None = None


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

        Why marker order, not score-based selection: title-match-score
        is a leaky abstraction. On ``[7, 9]`` (Kinney + Lo) a score-based
        ``max(sources, key=title_match_score)`` returns Lo when Lo's
        title happens to be more lexically similar to the claim text,
        even though the author wrote Kinney first. Marker order is the
        only ordering the author intentionally provided.

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
    # Populated only when status == "unverifiable"; mirrors the field on
    # ProvenanceStep. Surfaces why the pipeline could not produce a
    # confident verdict at this call site (a pipeline access limit, not
    # an epistemic claim).
    unverifiable_reason: UnverifiableReason | None = None

    def __post_init__(self) -> None:
        # Invariant: unverifiable <-> confidence is None. The orthogonal
        # evidence-quality coupling depends on claim_text (which is not a
        # field here) and is enforced in safe_verification_result() at
        # LLM-response parse boundaries, not in the schema. Direct
        # VerificationResult constructions are permitted to use any
        # (status, evidence_quality) combination.
        if self.status == "unverifiable" and self.confidence is not None:
            raise ValueError("unverifiable status requires confidence=None")
        if self.status != "unverifiable" and self.confidence is None:
            raise ValueError(f"{self.status!r} status requires non-null confidence")
        # Soft check: warn (not raise) when unverifiable is missing a reason,
        # so deserialized reports that predate the field still load.
        if self.status == "unverifiable" and self.unverifiable_reason is None:
            logger.warning(
                "unverifiable_without_reason",
                explanation_preview=self.explanation[:80],
            )


@dataclass(frozen=True)
class FetchAttempt:
    """One attempt in the full-text retrieval chain.

    Each attempt records a specific failure reason so report.json can
    answer "which publishers fail most often, and at which step?"
    without re-running the pipeline. The attempt order in
    FetchOutcome.attempts matches the chain order in fetch_fulltext.
    """

    method: FulltextMethod
    success: bool
    reason: FetchFailureReason | None  # None on success; populated on failure
    elapsed_ms: int


@dataclass(frozen=True)
class FetchOutcome:
    """Structured result of a full-text retrieval attempt.

    ``text`` is None when the chain exhausted; ``attempts`` carries the failure reasons.
    ``method`` names the method that succeeded, or ``"abstract_fallback"`` when none did.
    """

    text: str | None
    method: FulltextMethod
    attempts: tuple[FetchAttempt, ...]
    elapsed_ms_total: int


@dataclass(frozen=True)
class EvidenceBundle:
    """The single contract consumed by ``assess_evidence_sufficiency``.

    The verifier should not own access-policy, resolution-policy, or
    depth-policy decisions; those live in one pure function gated on this
    bundle. The verifier receives the bundle only when the policy returned
    ``Sufficient``, so its only job is the semantic question
    (supported / partially_supported / unsupported / not_addressed).

    Diagnostic fields (``fetch_attempts``, ``resolution_candidates``) are
    carried forward for auditor visibility in ``report.json`` — they are
    NOT consulted by the policy itself.
    """

    text: str | None
    depth: EvidenceDepth
    access_status: AccessStatus
    source_resolution_status: ResolutionStatus
    fetch_attempts: tuple[FetchAttempt, ...] = ()
    resolution_candidates: tuple[CandidateResolution, ...] = ()


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
    unverifiable_reason: UnverifiableReason | None = None


# ---------------------------------------------------------------------------
# Helper for LLM-response parse boundaries. Pure callers with known-valid
# inputs should construct VerificationResult directly; this helper exists
# for sites that receive raw LLM output and need the downgrade rule
# applied deterministically.
# ---------------------------------------------------------------------------

_INSUFFICIENT_EVIDENCE_SET: frozenset[str] = frozenset(
    {"abstract_only", "title_only", "citing_paper_context", "no_evidence"}
)

# Threshold for the extraction-confidence cap gate (3.3). Claims with
# extraction_confidence strictly below this value are capped to
# partially_supported. Set conservatively at 0.5 based on Phase 1 calibration
# (reports/audits/extraction_confidence_calibration/findings.md): the
# empirical distribution on 36 SciFact-derived claims ranged 0.55-0.80, with
# 0 claims below 0.5. The gate is dormant at 0.5 on these calibration inputs;
# tighten to 0.6 or 0.7 after /eval on real pipeline data.
_EXTRACTION_CONFIDENCE_THRESHOLD: float = 0.5

# Explanation template used when the helper downgrades a confident LLM
# verdict to unverifiable. The verdict and explanation must stay
# consistent — emitting `unverifiable` with the original "supported"
# narrative attached would be misleading. Deterministic (no LLM call).
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
    extraction_confidence: float | None = None,  # 3.3: LLM self-reported extraction confidence
    unverifiable_reason: UnverifiableReason | None = None,
    **kwargs: Any,  # noqa: ANN401 — forwarded verbatim to VerificationResult dataclass fields
) -> VerificationResult:
    """Construct a VerificationResult, downgrading to unverifiable when needed.

    Use at LLM-response parse boundaries where the raw parsed status/confidence
    may violate the invariant. Pure callers that know their inputs satisfy the
    invariant should construct VerificationResult directly.

    Gate order (deterministic, runs in this sequence):

    Gate 1 — Extraction-confidence cap (3.3):
      Fires when extraction_confidence is not None AND
      extraction_confidence < _EXTRACTION_CONFIDENCE_THRESHOLD (0.5) AND
      status in {"supported", "unsupported", "partially_supported"}.
      Action: status -> "partially_supported" (if not already);
              confidence -> min(original_confidence, extraction_confidence).
      This is a cap, NOT an unverifiable downgrade. unverifiable_reason is NOT set.
      A structlog event "extraction_confidence_cap" is emitted at the cap site.

    Gate 2 — Evidence-depth downgrade:
      Fires when status in {"supported", "unsupported"} AND
      evidence_quality in {abstract_only, title_only, citing_paper_context, no_evidence} AND
      claim_text is None (legacy callers -- fail safe) OR
      _claim_has_specific_numeric(claim_text) is True.
      Action: status -> "unverifiable", confidence -> None.
      Note: if Gate 1 already fired, status is "partially_supported" which is
      NOT in {"supported", "unsupported"}, so Gate 2 does not fire. This is the
      correct order -- low-extraction-confidence claims on abstract-only evidence
      become partially_supported, not unverifiable (spec edge case 7.5).

    Additional rules:
    - unverifiable + non-None confidence -> confidence forced to None.
    - All other combinations pass through unchanged.
    """
    from src.numeric.heuristics import _claim_has_specific_numeric

    # Gate 1: extraction_confidence cap (3.3).
    # Runs before the evidence-depth gate so partially_supported exempts
    # the claim from Gate 2 (see spec edge case 7.5).
    if (
        extraction_confidence is not None
        and extraction_confidence < _EXTRACTION_CONFIDENCE_THRESHOLD
        and status in ("supported", "unsupported", "partially_supported")
    ):
        capped_confidence = (
            min(confidence, extraction_confidence)
            if confidence is not None
            else extraction_confidence
        )
        logger.info(
            "extraction_confidence_cap",
            original_status=status,
            original_confidence=confidence,
            extraction_confidence=extraction_confidence,
            capped_confidence=capped_confidence,
        )
        status = "partially_supported"
        confidence = capped_confidence

    # Gate 2: evidence-depth downgrade (existing logic, unchanged).
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
