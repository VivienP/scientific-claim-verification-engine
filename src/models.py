"""Shared frozen dataclasses and type aliases for the verification pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from src.numeric.checks import NumericCheckResult

ClaimType = Literal["factual_numeric", "factual_qualitative", "methodological", "causal"]
VerificationStatus = Literal["supported", "unsupported", "not_addressed", "partially_supported"]
OperationType = Literal[
    "extract",
    "resolve",
    "fetch_fulltext",
    "chunk_paper",
    "select_passages",
    "verify",
    "aggregate",
    "numeric_extract",
    "numeric_check",
]
VerifiabilityStatus = Literal["verifiable", "no_citations_found", "low_citation_density"]
SectionLabel = Literal["introduction", "methods", "results", "discussion", "other"]
RetrievalStatus = Literal["passage_found", "no_passage_found", "fulltext_unavailable"]
EvidenceQuality = Literal[
    "quoted_passage",
    "abstract_only",
    "title_only",
    "citing_paper_context",
    "no_evidence",
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
    so the existing benchmark runner and `examples/sample_run.py` keep working.
    """

    sources: tuple[ResolvedSource, ...]
    citation_markers: tuple[int, ...]

    def primary(self) -> ResolvedSource:
        """Highest-confidence resolved source, or `_NOT_FOUND_SOURCE` when empty.

        Selection order: prefer found entries; among those, prefer the highest
        title_match_score. Used by callers expecting a single ResolvedSource.
        """
        if not self.sources:
            return _NOT_FOUND_SOURCE
        return max(
            self.sources,
            key=lambda s: (s.found, s.title_match_score or 0.0, s.similarity_score or 0.0),
        )

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
    confidence: float
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
