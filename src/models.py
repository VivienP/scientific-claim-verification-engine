"""Shared frozen dataclasses and type aliases for the verification pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ClaimType = Literal["factual_numeric", "factual_qualitative", "methodological", "causal"]
VerificationStatus = Literal["supported", "unsupported", "not_addressed", "partially_supported"]
OperationType = Literal["extract", "resolve", "verify", "aggregate"]


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


@dataclass(frozen=True)
class VerificationResult:
    status: VerificationStatus
    explanation: str
    confidence: float


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
