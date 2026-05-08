"""Unit tests for src/aar.py — AAR scorecard computation.

These tests build minimal report + provenance fixtures inline rather
than running the full pipeline, so each metric is tested in isolation.
"""

from __future__ import annotations

import math
from typing import Any

from src.aar import (
    AARScorecard,
    _claim_is_transparent,
    _is_valid_hash,
    compute_aar,
    render_scorecard_markdown,
)


def _claim(
    claim_id: str = "c1",
    *,
    source_passages: list[str] | None = None,
    evidence_quality: str = "abstract_only",
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "claim_text": "X causes Y",
        "verification": {
            "status": "supported",
            "source_passages": source_passages or [],
            "evidence_quality": evidence_quality,
        },
    }


def _step(
    *,
    claim_id: str = "c1",
    operation: str = "verify",
    input_hash: str = "a" * 64,
    output_hash: str = "b" * 64,
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "operation": operation,
        "input_hash": input_hash,
        "output_hash": output_hash,
    }


def _report(claims: list[dict[str, Any]], *, total_cost_usd: float = 1.0) -> dict[str, Any]:
    return {
        "claims": claims,
        "summary": {"total_cost_usd": total_cost_usd},
    }


class TestIsValidHash:
    def test_accepts_64_char_hex(self) -> None:
        assert _is_valid_hash("a" * 64)

    def test_accepts_min_length_16_hex(self) -> None:
        assert _is_valid_hash("0123456789abcdef")

    def test_rejects_short_hash(self) -> None:
        assert not _is_valid_hash("abc123")

    def test_rejects_empty_string(self) -> None:
        assert not _is_valid_hash("")

    def test_rejects_non_hex_chars(self) -> None:
        assert not _is_valid_hash("Z" * 64)

    def test_rejects_non_string(self) -> None:
        assert not _is_valid_hash(None)
        assert not _is_valid_hash(12345)


class TestClaimIsTransparent:
    def test_passages_present_is_transparent(self) -> None:
        assert _claim_is_transparent({"source_passages": ["quote"], "evidence_quality": "anything"})

    def test_abstract_only_is_transparent(self) -> None:
        assert _claim_is_transparent({"source_passages": [], "evidence_quality": "abstract_only"})

    def test_quoted_passage_is_transparent(self) -> None:
        assert _claim_is_transparent({"source_passages": [], "evidence_quality": "quoted_passage"})

    def test_title_only_is_transparent(self) -> None:
        assert _claim_is_transparent({"source_passages": [], "evidence_quality": "title_only"})

    def test_citing_paper_context_is_not_transparent(self) -> None:
        # The cited source itself was not seen — internal-consistency
        # only, capped at partially_supported by the verifier rubric.
        assert not _claim_is_transparent(
            {"source_passages": [], "evidence_quality": "citing_paper_context"}
        )

    def test_no_evidence_is_not_transparent(self) -> None:
        assert not _claim_is_transparent({"source_passages": [], "evidence_quality": "no_evidence"})


class TestPCov:
    def test_one_claim_with_step_yields_full_coverage(self) -> None:
        card = compute_aar(_report([_claim()]), [_step()])
        assert card.pcov == 1.0
        assert card.n_claims_with_provenance == 1

    def test_one_claim_no_step_yields_zero(self) -> None:
        card = compute_aar(_report([_claim()]), [])
        assert card.pcov == 0.0

    def test_zero_claims_yields_zero(self) -> None:
        card = compute_aar(_report([]), [])
        assert card.pcov == 0.0


class TestPSnd:
    def test_all_steps_with_valid_hashes_yield_full_soundness(self) -> None:
        card = compute_aar(_report([_claim()]), [_step(), _step()])
        assert card.psnd == 1.0

    def test_step_with_empty_hash_lowers_soundness(self) -> None:
        card = compute_aar(
            _report([_claim()]),
            [_step(), _step(input_hash="", output_hash="")],
        )
        assert card.psnd == 0.5

    def test_zero_steps_yields_zero(self) -> None:
        card = compute_aar(_report([_claim()]), [])
        assert card.psnd == 0.0


class TestCTran:
    def test_quoted_passage_is_transparent(self) -> None:
        card = compute_aar(
            _report([_claim(source_passages=["quote"])]),
            [_step()],
        )
        assert card.ctran == 1.0

    def test_no_evidence_is_not_transparent(self) -> None:
        card = compute_aar(
            _report([_claim(evidence_quality="no_evidence")]),
            [_step()],
        )
        assert card.ctran == 0.0

    def test_mixed_yields_fraction(self) -> None:
        card = compute_aar(
            _report(
                [
                    _claim("c1", source_passages=["q1"]),
                    _claim("c2", evidence_quality="no_evidence"),
                ]
            ),
            [_step(claim_id="c1"), _step(claim_id="c2")],
        )
        assert card.ctran == 0.5


class TestAEff:
    def test_one_claim_one_dollar_yields_one(self) -> None:
        card = compute_aar(_report([_claim()], total_cost_usd=1.0), [_step()])
        assert card.aeff == 1.0

    def test_two_claims_half_dollar_yields_four(self) -> None:
        card = compute_aar(
            _report([_claim("c1"), _claim("c2")], total_cost_usd=0.5),
            [_step(claim_id="c1"), _step(claim_id="c2")],
        )
        assert card.aeff == 4.0

    def test_zero_cost_yields_infinity(self) -> None:
        card = compute_aar(_report([_claim()], total_cost_usd=0.0), [_step()])
        assert math.isinf(card.aeff)


class TestRenderMarkdown:
    def test_includes_all_four_metric_rows(self) -> None:
        card = AARScorecard(
            pcov=1.0,
            psnd=0.95,
            ctran=0.8,
            aeff=10.0,
            n_claims=20,
            n_steps=80,
            n_claims_with_provenance=20,
            n_steps_with_valid_hashes=76,
            n_claims_with_quoted_evidence=16,
            total_cost_usd=2.0,
        )
        markdown = render_scorecard_markdown(card)
        for metric in ("PCov", "PSnd", "CTran", "AEff"):
            assert metric in markdown
        assert "100.00%" in markdown
        assert "95.00%" in markdown
        assert "80.00%" in markdown
        assert "10.00" in markdown
