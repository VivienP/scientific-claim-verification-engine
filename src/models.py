"""Shared frozen dataclasses and type aliases for the verification pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from src.numeric.checks import NumericCheckResult

ClaimType = Literal["factual_numeric", "factual_qualitative", "methodological", "causal"]
VerificationStatus = Literal["supported", "unsupported", "not_addressed", "partially_supported"]
OperationType = Literal[
    "extract",
    "resolve",
    "verify",
    "aggregate",
    "numeric_extract",
    "numeric_check",
]
VerifiabilityStatus = Literal["verifiable", "no_citations_found", "low_citation_density"]
SectionLabel = Literal["introduction", "methods", "results", "discussion", "other"]
RetrievalStatus = Literal["passage_found", "no_passage_found", "fulltext_unavailable"]
EvidenceQuality = Literal["quoted_passage", "abstract_only", "title_only", "no_evidence"]


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


@dataclass(frozen=True)
class VerificationResult:
    status: VerificationStatus
    explanation: str
    confidence: float
    source_passages: list[str] = field(default_factory=list)
    source_section: str | None = None
    fulltext_available: bool = False
    verification_depth: Literal["fulltext", "abstract", "title_only"] = "abstract"
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
