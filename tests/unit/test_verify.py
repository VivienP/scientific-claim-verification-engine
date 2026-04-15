"""Unit tests for src/verify.py — mocked Anthropic SDK."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from anthropic.types import TextBlock

from src.models import Claim, ProvenanceStep, ResolvedSource


def _text_block(text: str) -> TextBlock:
    """Create a real TextBlock for use in mock responses."""
    return TextBlock(type="text", text=text)


def _make_claim(claim_id: str = "claim-1") -> Claim:
    return Claim(
        claim_id=claim_id,
        claim_text="Protein folding rates increase with temperature.",
        cited_authors=["Smith"],
        cited_year=2020,
        claim_type="factual_qualitative",
    )


def _make_source(found: bool = True, abstract: str | None = "Abstract text.") -> ResolvedSource:
    return ResolvedSource(
        found=found,
        doi=None,
        title="Test Paper" if found else None,
        abstract=abstract,
        similarity_score=1.0 if found else None,
    )


class TestVerifyClaimHappyPath:
    @patch("src.verify.anthropic.Anthropic")
    def test_supported_status(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [
            _text_block(
                '{"status": "supported", "explanation": "The abstract confirms this.", "confidence": 0.9}'
            )
        ]
        mock_response.usage.input_tokens = 150
        mock_response.usage.output_tokens = 40
        mock_response.usage.cache_read_input_tokens = 150
        mock_response.usage.cache_creation_input_tokens = 0
        mock_client.messages.create.return_value = mock_response

        from src.verify import verify_claim

        result, _step = verify_claim(_make_claim(), _make_source())
        assert result.status == "supported"
        assert result.confidence == 0.9
        assert isinstance(result.explanation, str)

    @patch("src.verify.anthropic.Anthropic")
    def test_unsupported_status(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [
            _text_block(
                '{"status": "unsupported", "explanation": "The abstract contradicts this.", "confidence": 0.85}'
            )
        ]
        mock_response.usage.input_tokens = 150
        mock_response.usage.output_tokens = 40
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 150
        mock_client.messages.create.return_value = mock_response

        from src.verify import verify_claim

        result, _ = verify_claim(_make_claim(), _make_source())
        assert result.status == "unsupported"

    @patch("src.verify.anthropic.Anthropic")
    def test_not_addressed_status(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [
            _text_block(
                '{"status": "not_addressed", "explanation": "Abstract is on different topic.", "confidence": 0.95}'
            )
        ]
        mock_response.usage.input_tokens = 150
        mock_response.usage.output_tokens = 40
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 150
        mock_client.messages.create.return_value = mock_response

        from src.verify import verify_claim

        result, _ = verify_claim(_make_claim(), _make_source())
        assert result.status == "not_addressed"

    @patch("src.verify.anthropic.Anthropic")
    def test_partially_supported_status(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [
            _text_block(
                '{"status": "partially_supported", "explanation": "Partial match.", "confidence": 0.7}'
            )
        ]
        mock_response.usage.input_tokens = 150
        mock_response.usage.output_tokens = 40
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 150
        mock_client.messages.create.return_value = mock_response

        from src.verify import verify_claim

        result, _ = verify_claim(_make_claim(), _make_source())
        assert result.status == "partially_supported"

    @patch("src.verify.anthropic.Anthropic")
    def test_provenance_step_populated(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [
            _text_block('{"status": "supported", "explanation": "ok", "confidence": 0.9}')
        ]
        mock_response.usage.input_tokens = 200
        mock_response.usage.output_tokens = 30
        mock_response.usage.cache_read_input_tokens = 200
        mock_response.usage.cache_creation_input_tokens = 0
        mock_client.messages.create.return_value = mock_response

        from src.verify import verify_claim

        _, step = verify_claim(_make_claim("claim-x"), _make_source())
        assert isinstance(step, ProvenanceStep)
        assert step.operation == "verify"
        assert step.claim_id == "claim-x"
        assert step.tokens_in == 200
        assert step.tokens_out == 30
        assert step.cache_hit is True
        assert step.model_id == "claude-sonnet-4-6"
        assert step.confidence == 0.9


class TestVerifyClaimShortCircuit:
    def test_source_not_found_no_llm_call(self) -> None:
        """EC-2 variant: source.found=False → no Anthropic call."""
        with patch("src.verify.anthropic.Anthropic") as mock_cls:
            from src.verify import verify_claim

            result, step = verify_claim(_make_claim(), _make_source(found=False, abstract=None))
            mock_cls.assert_not_called()

        assert result.status == "not_addressed"
        assert result.confidence == 1.0
        assert step.operation == "verify"
        assert step.tokens_in is None

    def test_abstract_none_no_llm_call(self) -> None:
        """EC-2: Paper found but abstract=None → short-circuit."""
        with patch("src.verify.anthropic.Anthropic") as mock_cls:
            from src.verify import verify_claim

            result, step = verify_claim(_make_claim(), _make_source(found=True, abstract=None))
            mock_cls.assert_not_called()

        assert result.status == "not_addressed"
        assert step.model_id is None

    @patch("src.verify.anthropic.Anthropic")
    def test_malformed_response_returns_not_addressed(self, mock_anthropic_cls: MagicMock) -> None:
        """Malformed LLM response → not_addressed, confidence=0.0, no exception."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [_text_block("This is not valid JSON at all.")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 10
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 100
        mock_client.messages.create.return_value = mock_response

        from src.verify import verify_claim

        result, _step = verify_claim(_make_claim(), _make_source())
        assert result.status == "not_addressed"
        assert result.confidence == 0.0

    @patch("src.verify.anthropic.Anthropic")
    def test_cache_hit_none_when_no_cache_tokens(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [
            _text_block('{"status": "supported", "explanation": "ok", "confidence": 0.9}')
        ]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 20
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 0
        mock_client.messages.create.return_value = mock_response

        from src.verify import verify_claim

        _, step = verify_claim(_make_claim(), _make_source())
        assert step.cache_hit is None
