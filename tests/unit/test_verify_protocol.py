"""Unit tests for src/verify_protocol.py — verifier contract."""

from __future__ import annotations

import pytest

from src.models import Claim, ProvenanceStep
from src.verify_protocol import (
    MultiStepVerifier,
    SingleStepVerifier,
    assert_verifier_steps_valid,
)


def _claim(claim_id: str = "c1") -> Claim:
    return Claim(
        claim_id=claim_id,
        claim_text="X correlates with Y.",
        cited_authors=["Smith"],
        cited_year=2020,
        claim_type="factual_qualitative",
    )


def _step(
    *,
    claim_id: str = "c1",
    operation: str = "verify",
    model_id: str | None = "claude-sonnet-4-6",
    tokens_in: int | None = 100,
    cache_hit: bool | None = False,
) -> ProvenanceStep:
    return ProvenanceStep(
        step_id="s1",
        claim_id=claim_id,
        operation=operation,  # type: ignore[arg-type]
        input_hash="i",
        output_hash="o",
        model_id=model_id,
        timestamp=0.0,
        tokens_in=tokens_in,
        tokens_out=50,
        cache_hit=cache_hit,
        confidence=0.9,
    )


class TestProtocolStructuralConformance:
    """The verifier functions must satisfy the structural Protocol.

    mypy --strict already enforces this statically; the runtime check
    below is a single belt-and-braces assertion that catches any future
    refactor that accidentally drops a required method from the verifier
    surface.
    """

    def test_all_verifiers_match_their_protocol(self) -> None:
        from src.verify import (
            verify_claim,
            verify_claim_citing_context,
            verify_claim_fulltext,
            verify_claim_fulltext_with_numeric,
            verify_claim_multi_source,
            verify_claim_title_only,
        )

        # Tuple typed as `object` to collapse the union of differing function
        # signatures — only the runtime structural Protocol check matters here.
        single_step: tuple[object, ...] = (
            verify_claim,
            verify_claim_fulltext,
            verify_claim_title_only,
            verify_claim_citing_context,
        )
        multi_step: tuple[object, ...] = (
            verify_claim_multi_source,
            verify_claim_fulltext_with_numeric,
        )

        for fn in single_step:
            assert isinstance(fn, SingleStepVerifier), f"{fn} broke SingleStepVerifier"
        for fn in multi_step:
            assert isinstance(fn, MultiStepVerifier), f"{fn} broke MultiStepVerifier"


class TestAssertVerifierStepsValid:
    def test_passes_for_well_formed_verify_step(self) -> None:
        assert_verifier_steps_valid(_claim(), [_step()])

    def test_passes_for_aggregate_step_with_no_model(self) -> None:
        assert_verifier_steps_valid(
            _claim(),
            [_step(operation="aggregate", model_id=None, tokens_in=None, cache_hit=None)],
        )

    def test_rejects_mismatched_claim_id(self) -> None:
        with pytest.raises(AssertionError, match="claim_id"):
            assert_verifier_steps_valid(_claim("c1"), [_step(claim_id="other")])

    def test_rejects_invalid_operation(self) -> None:
        with pytest.raises(AssertionError, match="operation"):
            assert_verifier_steps_valid(_claim(), [_step(operation="resolve")])

    def test_rejects_aggregate_step_with_model_id(self) -> None:
        with pytest.raises(AssertionError, match="model_id=None"):
            assert_verifier_steps_valid(
                _claim(),
                [_step(operation="aggregate", model_id="some-model")],
            )

    def test_rejects_llm_verify_step_without_token_or_cache_data(self) -> None:
        with pytest.raises(AssertionError, match="tokens_in or cache_hit"):
            assert_verifier_steps_valid(
                _claim(),
                [_step(tokens_in=None, cache_hit=None)],
            )

    def test_accepts_llm_verify_step_with_only_cache_hit(self) -> None:
        # Cached calls report cache_hit=True with tokens_in=None — still valid.
        assert_verifier_steps_valid(_claim(), [_step(tokens_in=None, cache_hit=True)])

    def test_accepts_short_circuit_step_with_no_model(self) -> None:
        # Short-circuit steps from verify_claim (source.found=False) carry
        # model_id=None and no tokens — they are not LLM calls.
        assert_verifier_steps_valid(
            _claim(),
            [_step(model_id=None, tokens_in=None, cache_hit=None)],
        )
