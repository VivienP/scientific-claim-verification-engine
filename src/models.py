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


@dataclass(frozen=True)
class ResolvedSource:
    found: bool
    doi: str | None
    title: str | None
    abstract: str | None
    similarity_score: float | None
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
    verification_depth: Literal["fulltext", "abstract"] = "abstract"
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
