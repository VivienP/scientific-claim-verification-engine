"""Unit tests for src/numeric/engine.py — minimal coverage per modification 1.

Only one test: malformed LLM response → graceful fallback to (None, [step]).
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

    @patch("src.numeric.engine.extract_numeric_assertions")
    def test_p_value_ci_check_runs_when_no_or_ci_triple(self, mock_extract: MagicMock) -> None:
        from src.models import ProvenanceStep
        from src.numeric.checks import NumericAssertion

        mock_extract.return_value = (
            [
                NumericAssertion(
                    raw_text="p = 0.001",
                    value=0.001,
                    unit=None,
                    role="p_value",
                    context="odds ratio difference between groups",
                ),
                NumericAssertion(
                    raw_text="0.8",
                    value=0.8,
                    unit=None,
                    role="ci_low",
                    context="95% CI lower bound for odds ratio",
                ),
                NumericAssertion(
                    raw_text="1.2",
                    value=1.2,
                    unit=None,
                    role="ci_high",
                    context="95% CI upper bound for odds ratio",
                ),
            ],
            ProvenanceStep(
                step_id="ne",
                claim_id="claim-1",
                operation="numeric_extract",
                input_hash="i",
                output_hash="o",
                model_id="m",
                timestamp=0.0,
                tokens_in=10,
                tokens_out=5,
                cache_hit=False,
                confidence=None,
            ),
        )

        result, steps = run_numeric_check("Claim reports p = 0.001, 95% CI 0.8-1.2")

        assert result is not None
        assert result.check_type == "p_value_ci_consistency"
        assert result.consistent is False
        assert len(steps) == 2
        assert steps[1].operation == "numeric_check"
