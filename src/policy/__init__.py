"""Policy decisions for the verification pipeline.

The ``evidence_sufficiency`` module owns the single decision point that gates
whether the semantic verifier (LLM) is invoked or whether the pipeline emits
a deterministic ``unverifiable`` verdict instead. This package is import-light
and contains zero LLM calls — it consumes structured fields from
``EvidenceBundle``, ``ResolvedSource``, and ``Claim`` only.
"""

from src.policy.evidence_sufficiency import (
    Insufficient,
    SufficiencyDecision,
    Sufficient,
    assess_evidence_sufficiency,
)

__all__ = [
    "Insufficient",
    "Sufficiency",
    "SufficiencyDecision",
    "Sufficient",
    "assess_evidence_sufficiency",
]

Sufficiency = SufficiencyDecision
