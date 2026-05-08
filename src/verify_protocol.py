"""VerifierMode Protocol — the structural contract every verifier honours.

Every verification function in :mod:`src.verify` returns the same shape:
``tuple[VerificationResult, ProvenanceStep | list[ProvenanceStep]]``. The
inputs differ by mode: abstract takes ``(claim, source)``, fulltext takes
``(claim, source, passages)``, multi_source takes ``(claim, source_set,
passages_per_source)``, citing_context takes ``(claim, source,
citing_paper_text)``. There is no inheritance hierarchy because each
verifier has a different positional signature; what they share is the
*output* contract and the side-effect contract.

This module formalises the side-effect contract via two structural
Protocols. Importing from here is optional — verifier callers (the
pipeline, the test suite, custom benchmarks) can use the concrete
function signatures directly. The Protocols exist so:

  1. New verifier modes (e.g. a future table-aware verifier) can
     declare conformance via ``VerifierProvenanceContract`` and
     :func:`assert_verifier_contract` will fail at test time if a step
     omits the required fields.
  2. Tools that aggregate verifier outputs (the report builder, the
     AAR scorecard) can accept any conforming verifier without
     importing the concrete functions.

The Protocols also document the *invariants* every verifier must
preserve, in code, where they are checkable:

  * Every step's ``operation`` is ``"verify"`` (single-shot verifiers)
    or ``"aggregate"`` (multi_source's aggregation step). No verifier
    emits ``"resolve"``, ``"fetch_fulltext"``, ``"chunk_paper"``, or
    ``"select_passages"`` — those operations are the pipeline's
    responsibility, not the verifier's.
  * Every LLM-emitting step carries a non-None ``model_id`` and at
    least one of ``tokens_in`` or ``cache_hit``.
  * Every step's ``claim_id`` matches ``claim.claim_id``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from src.models import (
    Claim,
    ProvenanceStep,
    VerificationResult,
)


@runtime_checkable
class SingleStepVerifier(Protocol):
    """Verifier that emits exactly one ProvenanceStep per call.

    Conforming functions: :func:`src.verify.verify_claim`,
    :func:`src.verify.verify_claim_fulltext`,
    :func:`src.verify.verify_claim_title_only`,
    :func:`src.verify.verify_claim_citing_context`.

    The ``__call__`` signature is intentionally loose because each
    verifier has a different positional shape. The Protocol locks only
    the **output** type.
    """

    def __call__(
        self, *args: object, **kwargs: object
    ) -> tuple[VerificationResult, ProvenanceStep]: ...


@runtime_checkable
class MultiStepVerifier(Protocol):
    """Verifier that emits a list of ProvenanceStep per call.

    Conforming functions: :func:`src.verify.verify_claim_multi_source`
    (per-source verify steps + aggregate step),
    :func:`src.verify.verify_claim_fulltext_with_numeric` (verify + numeric
    extraction + numeric check).
    """

    def __call__(
        self, *args: object, **kwargs: object
    ) -> tuple[VerificationResult, list[ProvenanceStep]]: ...


def assert_verifier_steps_valid(
    claim: Claim,
    steps: Iterable[ProvenanceStep],
) -> None:
    """Sanity-check a list of verifier-emitted ProvenanceStep records.

    Raises AssertionError if any step violates the verifier contract:

      * step.claim_id must equal claim.claim_id
      * step.operation must be one of {"verify", "aggregate"}
      * LLM steps (operation="verify" with model_id) must report
        a non-None token count or cache_hit so cost accounting works
      * The aggregate step (operation="aggregate") must have model_id=None
        — aggregation is deterministic, not an LLM call

    Used by :func:`src.pipeline.verify_one_claim` defensively in tests
    and (eventually) by the integration test suite to catch regressions
    where a new verifier mode forgets to populate provenance fields.
    """
    valid_operations = {"verify", "aggregate"}
    for step in steps:
        if step.claim_id != claim.claim_id:
            raise AssertionError(
                f"Verifier step claim_id={step.claim_id!r} != claim.claim_id={claim.claim_id!r}"
            )
        if step.operation not in valid_operations:
            raise AssertionError(
                f"Verifier step operation={step.operation!r} "
                f"must be one of {sorted(valid_operations)}"
            )
        if step.operation == "aggregate" and step.model_id is not None:
            raise AssertionError(
                f"Aggregate step must have model_id=None (deterministic), got {step.model_id!r}"
            )
        if (
            step.operation == "verify"
            and step.model_id is not None
            and step.tokens_in is None
            and step.cache_hit is None
        ):
            raise AssertionError(
                f"LLM verify step model_id={step.model_id!r} "
                f"must report tokens_in or cache_hit for cost accounting"
            )


__all__ = [
    "MultiStepVerifier",
    "SingleStepVerifier",
    "assert_verifier_steps_valid",
]
