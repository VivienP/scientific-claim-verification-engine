"""Unit tests for A2 changes in src/verify_citing_context.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from anthropic.types import TextBlock

from src.models import Claim, ResolvedSource


def _text_block(text: str) -> TextBlock:
    return TextBlock(type="text", text=text)


def _make_claim(claim_id: str = "claim-cc") -> Claim:
    # Uses a percentage to trigger _claim_has_specific_numeric for A2 downgrade tests.
    return Claim(
        claim_id=claim_id,
        claim_text="Lactate concentrations increase by 20% at maximal exercise.",
        cited_authors=["Brooks"],
        cited_year=1986,
        claim_type="factual_numeric",
    )


def _make_source() -> ResolvedSource:
    return ResolvedSource(found=False, doi=None, title=None, abstract=None, similarity_score=None)


def _citing_text() -> str:
    return (
        "Lactate kinetics in exercise have been studied extensively. "
        "Some unrelated cosmological context here."
    )


class TestA2CitingContextDowngrade:
    """A2: verify_citing_context.py routes unsupported+citing_paper_context
    through safe_verification_result, downgrading to unverifiable."""

    @patch("src.verify_citing_context.anthropic.Anthropic")
    def test_downgrades_confident_unsupported_to_unverifiable(
        self, mock_anthropic_cls: MagicMock
    ) -> None:
        """LLM returning unsupported+0.55 on citing_paper_context -> unverifiable, None."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [
            _text_block(
                '{"status": "unsupported", "explanation": "Internal-consistency only -- '
                'citing paper does not mention the cited reference.", "confidence": 0.55}'
            )
        ]
        mock_response.usage.input_tokens = 200
        mock_response.usage.output_tokens = 40
        mock_response.usage.cache_read_input_tokens = 200
        mock_response.usage.cache_creation_input_tokens = 0
        mock_client.messages.create.return_value = mock_response

        from src.verify_citing_context import verify_claim_citing_context

        result, step = verify_claim_citing_context(_make_claim(), _make_source(), _citing_text())
        assert result.status == "unverifiable"
        assert result.confidence is None
        assert result.evidence_quality == "citing_paper_context"
        assert step.unverifiable_reason == "insufficient_evidence_depth"

    @patch("src.verify_citing_context.anthropic.Anthropic")
    def test_preserves_partially_supported(self, mock_anthropic_cls: MagicMock) -> None:
        """partially_supported on citing_paper_context is valid -- passes through."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [
            _text_block(
                '{"status": "partially_supported", "explanation": "Internal-consistency only -- '
                'some context present.", "confidence": 0.5}'
            )
        ]
        mock_response.usage.input_tokens = 200
        mock_response.usage.output_tokens = 40
        mock_response.usage.cache_read_input_tokens = 200
        mock_response.usage.cache_creation_input_tokens = 0
        mock_client.messages.create.return_value = mock_response

        from src.verify_citing_context import verify_claim_citing_context

        result, _step = verify_claim_citing_context(_make_claim(), _make_source(), _citing_text())
        assert result.status == "partially_supported"
        assert result.confidence is not None
        assert result.evidence_quality == "citing_paper_context"
