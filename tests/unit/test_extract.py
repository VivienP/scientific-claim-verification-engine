"""Unit tests for src/extract.py — mocked Anthropic SDK."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from anthropic.types import TextBlock

from src.models import ProvenanceStep


def _text_block(text: str) -> TextBlock:
    """Create a real TextBlock for use in mock responses."""
    return TextBlock(type="text", text=text)


class TestExtractClaimsHappyPath:
    @patch("src.extract.anthropic.Anthropic")
    def test_returns_claims_and_provenance(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [
            _text_block(
                '{"claims": [{"claim_text": "X causes Y", "cited_authors": ["Smith"], "cited_year": 2020, "claim_type": "causal"}]}'
            )
        ]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 100
        mock_client.messages.create.return_value = mock_response

        from src.extract import extract_claims

        claims, _step = extract_claims("Smith (2020) showed that X causes Y.")

        assert len(claims) == 1
        assert claims[0].claim_text == "X causes Y"
        assert claims[0].cited_authors == ["Smith"]
        assert claims[0].cited_year == 2020
        assert claims[0].claim_type == "causal"
        assert isinstance(claims[0].claim_id, str)
        assert len(claims[0].claim_id) > 0

    @patch("src.extract.anthropic.Anthropic")
    def test_provenance_step_populated(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [_text_block('{"claims": []}')]
        mock_response.usage.input_tokens = 200
        mock_response.usage.output_tokens = 10
        mock_response.usage.cache_read_input_tokens = 200
        mock_response.usage.cache_creation_input_tokens = 0
        mock_client.messages.create.return_value = mock_response

        from src.extract import extract_claims

        _, step = extract_claims("No claims here.")

        assert isinstance(step, ProvenanceStep)
        assert step.operation == "extract"
        assert step.tokens_in == 200
        assert step.tokens_out == 10
        assert step.cache_hit is True  # cache_read_input_tokens > 0
        assert step.model_id == "claude-sonnet-4-6"
        assert step.claim_id.startswith("__extract__:")

    @patch("src.extract.anthropic.Anthropic")
    def test_cache_hit_false_when_creation_tokens(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [_text_block('{"claims": []}')]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 5
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 100
        mock_client.messages.create.return_value = mock_response

        from src.extract import extract_claims

        _, step = extract_claims("Test text.")
        assert step.cache_hit is False  # creation tokens > 0, read tokens = 0

    @patch("src.extract.anthropic.Anthropic")
    def test_multiple_claims(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [
            _text_block(
                '{"claims": ['
                '{"claim_text": "Claim one", "cited_authors": ["A"], "cited_year": 2019, "claim_type": "factual_numeric"},'
                '{"claim_text": "Claim two", "cited_authors": ["B"], "cited_year": 2021, "claim_type": "methodological"}'
                "]}"
            )
        ]
        mock_response.usage.input_tokens = 300
        mock_response.usage.output_tokens = 80
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 300
        mock_client.messages.create.return_value = mock_response

        from src.extract import extract_claims

        claims, _step = extract_claims("Two claims in text.")
        assert len(claims) == 2
        assert claims[0].claim_text == "Claim one"
        assert claims[1].claim_text == "Claim two"

    @patch("src.extract.anthropic.Anthropic")
    def test_each_claim_gets_unique_id(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [
            _text_block(
                '{"claims": ['
                '{"claim_text": "A", "cited_authors": [], "cited_year": null, "claim_type": "factual_qualitative"},'
                '{"claim_text": "B", "cited_authors": [], "cited_year": null, "claim_type": "factual_qualitative"}'
                "]}"
            )
        ]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 100
        mock_client.messages.create.return_value = mock_response

        from src.extract import extract_claims

        claims, _ = extract_claims("Two claims.")
        ids = [c.claim_id for c in claims]
        assert len(set(ids)) == 2  # unique IDs


class TestExtractClaimsEdgeCases:
    @patch("src.extract.anthropic.Anthropic")
    def test_empty_text_returns_empty_list(self, mock_anthropic_cls: MagicMock) -> None:
        """EC-5: empty text produces valid empty list."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [_text_block('{"claims": []}')]
        mock_response.usage.input_tokens = 50
        mock_response.usage.output_tokens = 5
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 50
        mock_client.messages.create.return_value = mock_response

        from src.extract import extract_claims

        claims, step = extract_claims("")
        assert claims == []
        assert step.operation == "extract"

    @patch("src.extract.anthropic.Anthropic")
    def test_malformed_response_returns_empty_list(self, mock_anthropic_cls: MagicMock) -> None:
        """EC-3: malformed LLM response returns empty list, no exception."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [_text_block("I cannot help with that")]
        mock_response.usage.input_tokens = 50
        mock_response.usage.output_tokens = 5
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 50
        mock_client.messages.create.return_value = mock_response

        from src.extract import extract_claims

        claims, step = extract_claims("Some scientific text.")
        assert claims == []
        assert step.operation == "extract"

    @patch("src.extract.anthropic.Anthropic")
    def test_missing_claims_key_returns_empty_list(self, mock_anthropic_cls: MagicMock) -> None:
        """EC-3 variant: JSON valid but missing 'claims' key."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [_text_block('{"results": []}')]
        mock_response.usage.input_tokens = 50
        mock_response.usage.output_tokens = 5
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 50
        mock_client.messages.create.return_value = mock_response

        from src.extract import extract_claims

        claims, step = extract_claims("Some text.")
        assert claims == []
        assert step.operation == "extract"

    @patch("src.extract.anthropic.Anthropic")
    def test_markdown_fenced_json_parsed(self, mock_anthropic_cls: MagicMock) -> None:
        """LLM response wrapped in ```json fences is parsed correctly."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        fenced = '```json\n{"claims": [{"claim_text": "X causes Y", "cited_authors": ["Smith"], "cited_year": 2020, "claim_type": "causal"}]}\n```'
        mock_response.content = [_text_block(fenced)]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_response.usage.cache_read_input_tokens = 100
        mock_response.usage.cache_creation_input_tokens = 0
        mock_client.messages.create.return_value = mock_response

        from src.extract import extract_claims

        claims, _step = extract_claims("Some scientific text.")
        assert len(claims) == 1
        assert claims[0].claim_text == "X causes Y"

    @patch("src.extract.anthropic.Anthropic")
    def test_cache_hit_none_when_no_cache_tokens(self, mock_anthropic_cls: MagicMock) -> None:
        """cache_hit=None when both cache_read and cache_creation are 0."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [_text_block('{"claims": []}')]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 5
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 0
        mock_client.messages.create.return_value = mock_response

        from src.extract import extract_claims

        _, step = extract_claims("Test.")
        assert step.cache_hit is None
