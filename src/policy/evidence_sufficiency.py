"""Single decision point: is the evidence sufficient to invoke the verifier?

The function ``assess_evidence_sufficiency`` is the only place where the
pipeline decides between (a) calling the LLM verifier and (b) emitting a
deterministic ``unverifiable`` verdict. The decision is computed from
``EvidenceBundle`` + ``Claim`` only — pure Python, zero LLM, zero I/O.

The output is a tagged union (``Sufficient`` | ``Insufficient(reason)``) so
the pipeline can branch with ``isinstance`` exhaustively. ``Insufficient``
carries the ``UnverifiableReason`` that the pipeline writes into the
emitted ``VerificationResult`` — the policy is the single source of truth
for which reason applies to which input shape.

Policy ordering (highest priority first):
    1. Resolution-status gates. A disputed cross-source resolution means the
       source isn't reliably identified; verifying against a possibly-wrong
       paper would be a silent failure. ``low_confidence`` only blocks
       specific-numeric claims (those are the ones a wrong paper would
       silently contaminate).
    2. Depth gates. A specific-numeric claim cannot be verified from
       abstract / title / no-text — the abstract systematically omits
       Results-section figures.
    3. Access gates. When the source is unresolved or the publisher
       blocked the fetch and no text was retrieved, there's nothing for
       the verifier to read.

When none of the gates fire, the policy returns ``Sufficient`` and the
pipeline dispatches to the appropriate ``verify_*`` function as today.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.models import Claim, EvidenceBundle, ResolvedSource, UnverifiableReason
from src.numeric.heuristics import _claim_has_specific_numeric


@dataclass(frozen=True)
class Sufficient:
    """Policy verdict: evidence is sufficient — dispatch to the verifier."""


@dataclass(frozen=True)
class Insufficient:
    """Policy verdict: evidence is insufficient — emit unverifiable.

    ``reason`` is the concrete ``UnverifiableReason`` enum value the
    pipeline copies into the emitted ``VerificationResult.unverifiable_reason``
    and ``ProvenanceStep.unverifiable_reason``. The verdict and the
    explanation must stay consistent (handled by the pipeline-side
    emission helper).
    """

    reason: UnverifiableReason


SufficiencyDecision = Sufficient | Insufficient


def assess_evidence_sufficiency(
    claim: Claim,
    source: ResolvedSource,
    evidence: EvidenceBundle,
) -> SufficiencyDecision:
    """Decide whether the evidence is sufficient to invoke the semantic verifier.

    Pure function. Same input -> same output, every run. No LLM, no I/O.

    Args:
        claim: the claim under verification. Only ``claim.claim_text`` is read,
            via the deterministic numeric-pattern heuristic.
        source: the resolved source for the claim. Reserved for future policy
            extensions that may consult per-source metadata (e.g. retraction
            status). Currently unread, but the argument is retained so the
            signature is stable across the planned policy additions.
        evidence: the structured ``EvidenceBundle`` the pipeline assembled
            from the resolver verdict + fetch outcome + chunked passages.

    Returns:
        ``Sufficient()`` when the verifier may be invoked.
        ``Insufficient(reason=...)`` when the pipeline must emit a
        deterministic ``unverifiable`` verdict; ``reason`` is the
        ``UnverifiableReason`` to record on the result.
    """
    has_numeric = _claim_has_specific_numeric(claim.claim_text)

    if evidence.source_resolution_status == "disputed":
        return Insufficient(reason="resolution_source_disagreement")
    if evidence.source_resolution_status == "low_confidence" and has_numeric:
        return Insufficient(reason="resolution_low_confidence")

    if evidence.depth in ("title", "none") and has_numeric:
        return Insufficient(reason="numeric_claim_abstract_only")
    if evidence.depth == "abstract" and has_numeric:
        return Insufficient(reason="numeric_claim_abstract_only")

    if evidence.access_status in ("blocked", "unresolved") and evidence.text is None:
        return Insufficient(reason="fulltext_unavailable")

    return Sufficient()
