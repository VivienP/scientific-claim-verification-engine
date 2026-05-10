"""Unit tests for src/verify_cross_modal.py — mocked Anthropic SDK."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import anthropic
from anthropic.types import TextBlock

from src.models import Claim, ProvenanceStep, VerificationResult


def _text_block(text: str) -> TextBlock:
    return TextBlock(type="text", text=text)


def _make_claim(claim_id: str = "claim-cm") -> Claim:
    return Claim(
        claim_id=claim_id,
        claim_text="Lactate concentration is approximately 1.7 mmol/L.",
        cited_authors=["Smith"],
        cited_year=2020,
        claim_type="factual_numeric",
    )


def _make_primary(
    *,
    status: str = "supported",
    confidence: float = 0.9,
    depth: str = "abstract",
) -> VerificationResult:
    return VerificationResult(
        status=status,  # type: ignore[arg-type]
        explanation="Primary verifier explanation.",
        confidence=confidence,
        verification_depth=depth,  # type: ignore[arg-type]
    )


def _mock_response(text: str, *, cache_read: int = 100, cache_creation: int = 0) -> MagicMock:
    response = MagicMock()
    response.content = [_text_block(text)]
    response.usage.input_tokens = 200
    response.usage.output_tokens = 40
    response.usage.cache_read_input_tokens = cache_read
    response.usage.cache_creation_input_tokens = cache_creation
    return response


class TestCrossModalGate:
    """Gate skips when primary verdict is not in scope — no LLM call."""

    @patch("src.verify_cross_modal.anthropic.Anthropic")
    def test_skips_when_confidence_below_threshold(self, mock_cls: MagicMock) -> None:
        from src.verify_cross_modal import cross_modal_check

        primary = _make_primary(confidence=0.5)
        result, step = cross_modal_check(_make_claim(), "abstract text", primary)

        assert result is primary
        assert step is None
        mock_cls.assert_not_called()

    @patch("src.verify_cross_modal.anthropic.Anthropic")
    def test_skips_when_status_partially_supported(self, mock_cls: MagicMock) -> None:
        from src.verify_cross_modal import cross_modal_check

        primary = _make_primary(status="partially_supported", confidence=0.95)
        result, step = cross_modal_check(_make_claim(), "abstract", primary)

        assert result is primary
        assert step is None
        mock_cls.assert_not_called()

    @patch("src.verify_cross_modal.anthropic.Anthropic")
    def test_skips_when_status_not_addressed(self, mock_cls: MagicMock) -> None:
        from src.verify_cross_modal import cross_modal_check

        primary = _make_primary(status="not_addressed", confidence=0.95)
        result, step = cross_modal_check(_make_claim(), "abstract", primary)

        assert result is primary
        assert step is None
        mock_cls.assert_not_called()

    @patch("src.verify_cross_modal.anthropic.Anthropic")
    def test_skips_when_depth_not_abstract(self, mock_cls: MagicMock) -> None:
        from src.verify_cross_modal import cross_modal_check

        primary = _make_primary(depth="title_only", confidence=0.95)
        result, step = cross_modal_check(_make_claim(), "abstract", primary)

        assert result is primary
        assert step is None
        mock_cls.assert_not_called()

    @patch("src.verify_cross_modal.anthropic.Anthropic")
    def test_skips_at_exact_threshold_boundary(self, mock_cls: MagicMock) -> None:
        """Threshold uses strict `>`, so confidence == 0.7 must NOT fire."""
        from src.verify_cross_modal import cross_modal_check

        primary = _make_primary(confidence=0.7)
        result, step = cross_modal_check(_make_claim(), "abstract", primary)

        assert result is primary
        assert step is None
        mock_cls.assert_not_called()


class TestCrossModalAgreement:
    """Gate fires, second model agrees → primary preserved, step records agreement."""

    @patch("src.verify_cross_modal.anthropic.Anthropic")
    def test_agreement_preserves_primary(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_response(
            '{"status": "supported", "explanation": "Secondary agrees.", "confidence": 0.85}'
        )
        from src.verify_cross_modal import cross_modal_check

        primary = _make_primary(status="supported", confidence=0.9)
        result, step = cross_modal_check(_make_claim(), "abstract", primary)

        assert result.status == "supported"
        assert result.confidence == 0.9
        assert "[CROSS-MODAL DISAGREEMENT" not in result.explanation
        assert isinstance(step, ProvenanceStep)
        assert step.operation == "verify_cross_modal"
        assert step.confidence == 0.9  # agreement records primary's confidence
        assert step.model_id == "claude-haiku-4-5-20251001"


class TestCrossModalDisagreement:
    """Gate fires, second model disagrees → confidence downgraded, explanation annotated."""

    @patch("src.verify_cross_modal.anthropic.Anthropic")
    def test_disagreement_downgrades_confidence(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_response(
            '{"status": "unsupported", "explanation": "Secondary disagrees.", "confidence": 0.8}'
        )
        from src.verify_cross_modal import cross_modal_check

        primary = _make_primary(status="supported", confidence=0.9)
        result, step = cross_modal_check(_make_claim(), "abstract", primary)

        assert result.status == "supported"  # primary preserved (no `uncertain` status)
        assert result.confidence == 0.5  # downgraded
        assert "[CROSS-MODAL DISAGREEMENT" in result.explanation
        assert "secondary=unsupported" in result.explanation
        assert isinstance(step, ProvenanceStep)
        assert step.operation == "verify_cross_modal"
        assert step.confidence is None  # disagreement → confidence undetermined

    @patch("src.verify_cross_modal.anthropic.Anthropic")
    def test_disagreement_does_not_raise_below_dowgrade_floor(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_response(
            '{"status": "unsupported", "explanation": "Disagrees.", "confidence": 0.2}'
        )
        from src.verify_cross_modal import cross_modal_check

        primary = _make_primary(status="supported", confidence=0.75)
        result, _step = cross_modal_check(_make_claim(), "abstract", primary)

        # min(0.75, 0.5) == 0.5
        assert result.confidence == 0.5


class TestCrossModalErrors:
    """API and parse errors return primary unchanged + step with confidence=None."""

    @patch("src.verify_cross_modal.anthropic.Anthropic")
    def test_parse_error_returns_primary_unchanged(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_response("not valid json")
        from src.verify_cross_modal import cross_modal_check

        primary = _make_primary(confidence=0.9)
        result, step = cross_modal_check(_make_claim(), "abstract", primary)

        assert result is primary
        assert isinstance(step, ProvenanceStep)
        assert step.operation == "verify_cross_modal"
        assert step.confidence is None

    @patch("src.verify_cross_modal.anthropic.Anthropic")
    def test_api_error_returns_primary_unchanged(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.side_effect = anthropic.APIError(
            message="boom",
            request=MagicMock(),
            body=None,
        )
        from src.verify_cross_modal import cross_modal_check

        primary = _make_primary(confidence=0.9)
        result, step = cross_modal_check(_make_claim(), "abstract", primary)

        assert result is primary
        assert isinstance(step, ProvenanceStep)
        assert step.operation == "verify_cross_modal"
        assert step.confidence is None

    @patch("src.verify_cross_modal.anthropic.Anthropic")
    def test_invalid_status_in_response_returns_primary_unchanged(
        self, mock_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_response(
            '{"status": "bogus_status", "explanation": "x", "confidence": 0.8}'
        )
        from src.verify_cross_modal import cross_modal_check

        primary = _make_primary(confidence=0.9)
        result, step = cross_modal_check(_make_claim(), "abstract", primary)

        assert result is primary
        assert isinstance(step, ProvenanceStep)
        assert step.confidence is None
