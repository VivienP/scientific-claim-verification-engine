"""Unit tests for src/numeric/engine.py — minimal coverage per modification 1.

Only one test: malformed LLM response → graceful fallback to (None, [step]).
Real signal for the engine comes from examples/numeric_worked_example.py end-to-end.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from anthropic.types import TextBlock

from src.numeric.engine import run_numeric_check


def _text_block(text: str) -> TextBlock:
    return TextBlock(type="text", text=text)


class TestRunNumericCheck:
    @patch("src.numeric.extract.anthropic.Anthropic")
    def test_malformed_llm_response_falls_back_to_none(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [_text_block("this is not valid json at all")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 10
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 0
        mock_client.messages.create.return_value = mock_response

        result, steps = run_numeric_check("Some claim with OR 5 (95% CI 2-8)")

        assert result is None
        assert len(steps) == 1
        assert steps[0].operation == "numeric_extract"
