"""Unit tests for src/copilot/rationale.py — mocked Anthropic SDK, offline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from anthropic.types import TextBlock

from src.copilot.rationale import _enforce_word_limit, _first_sentence, extract_rationale
from src.models import (
    Claim,
    ProvenanceStep,
    ResolvedSource,
    ResolvedSourceSet,
    VerificationResult,
)
from src.pipeline import ClaimVerification

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _text_block(text: str) -> TextBlock:
    return TextBlock(type="text", text=text)


def _make_cv(
    claim_text: str = "Metformin reduces HbA1c by 1.5%.",
    verdict: str = "unsupported",
    explanation: str = "The cited source does not report HbA1c values. No numerical data found.",
) -> ClaimVerification:
    claim = Claim(
        claim_id="cl-001",
        claim_text=claim_text,
        cited_authors=["Smith"],
        cited_year=2022,
        claim_type="factual_numeric",
    )
    source = ResolvedSource(
        found=True,
        doi="10.1234/test",
        title="Test Paper",
        abstract="This is an abstract.",
        similarity_score=0.9,
    )
    source_set = ResolvedSourceSet(sources=(source,), citation_markers=(1,))
    result = VerificationResult(
        status=verdict,  # type: ignore[arg-type]
        explanation=explanation,
        confidence=0.3,
    )
    return ClaimVerification(
        claim=claim,
        source=source,
        source_set=source_set,
        result=result,
        fetch_method="abstract",
    )


def _mock_response(text: str, tokens_in: int = 120, tokens_out: int = 15) -> MagicMock:
    response = MagicMock()
    response.content = [_text_block(text)]
    response.usage.input_tokens = tokens_in
    response.usage.output_tokens = tokens_out
    response.usage.cache_read_input_tokens = 0
    response.usage.cache_creation_input_tokens = tokens_in
    return response


# ---------------------------------------------------------------------------
# Pure helper tests (no mocking needed)
# ---------------------------------------------------------------------------


class TestFirstSentence:
    def test_period_split(self) -> None:
        assert _first_sentence("First sentence. Second one.") == "First sentence."

    def test_exclamation_split(self) -> None:
        assert _first_sentence("Warning! Something happened.") == "Warning!"

    def test_question_split(self) -> None:
        assert _first_sentence("Is this right? Yes.") == "Is this right?"

    def test_no_punctuation_caps_at_200(self) -> None:
        long = "a" * 250
        assert _first_sentence(long) == "a" * 200

    def test_returns_full_short_text(self) -> None:
        assert _first_sentence("Short.") == "Short."


class TestEnforceWordLimit:
    def test_short_text_unchanged(self) -> None:
        text = "The abstract confirms this claim."
        assert _enforce_word_limit(text, limit=30) == text

    def test_truncates_at_word_boundary(self) -> None:
        words = " ".join(f"word{i}" for i in range(50))
        result = _enforce_word_limit(words, limit=30)
        assert len(result.split()) <= 30

    def test_prefers_punctuation_boundary(self) -> None:
        # Has a comma past the halfway mark — should end there
        text = (
            "One two three four five six seven eight nine ten eleven,"
            " twelve thirteen fourteen fifteen."
        )
        result = _enforce_word_limit(text, limit=12)
        # Should end at a punctuation mark, not mid-word
        assert result[-1] in (".", ",", ";")

    def test_exactly_at_limit_unchanged(self) -> None:
        words = " ".join(["word"] * 30)
        assert _enforce_word_limit(words, limit=30) == words


# ---------------------------------------------------------------------------
# extract_rationale — happy path
# ---------------------------------------------------------------------------


class TestExtractRationaleHappyPath:
    @patch("src.copilot.rationale.anthropic.Anthropic")
    def test_returns_llm_rationale(self, mock_cls: MagicMock) -> None:
        rationale_text = "The cited source contains no numerical HbA1c data to support this claim."
        mock_cls.return_value.messages.create.return_value = _mock_response(rationale_text)

        rationale, _step = extract_rationale(_make_cv())

        assert rationale == rationale_text

    @patch("src.copilot.rationale.anthropic.Anthropic")
    def test_rationale_within_word_limit(self, mock_cls: MagicMock) -> None:
        long_llm_output = " ".join(["word"] * 50)
        mock_cls.return_value.messages.create.return_value = _mock_response(long_llm_output)

        rationale, _ = extract_rationale(_make_cv())

        assert len(rationale.split()) <= 30

    @patch("src.copilot.rationale.anthropic.Anthropic")
    def test_provenance_step_operation(self, mock_cls: MagicMock) -> None:
        mock_cls.return_value.messages.create.return_value = _mock_response("Short rationale.")

        _, step = extract_rationale(_make_cv())

        assert isinstance(step, ProvenanceStep)
        assert step.operation == "copilot_rationale"

    @patch("src.copilot.rationale.anthropic.Anthropic")
    def test_provenance_step_model_id(self, mock_cls: MagicMock) -> None:
        mock_cls.return_value.messages.create.return_value = _mock_response("Rationale.")

        _, step = extract_rationale(_make_cv())

        assert step.model_id is not None
        assert "claude" in step.model_id.lower()

    @patch("src.copilot.rationale.anthropic.Anthropic")
    def test_provenance_step_claim_id(self, mock_cls: MagicMock) -> None:
        mock_cls.return_value.messages.create.return_value = _mock_response("Rationale.")
        cv = _make_cv()

        _, step = extract_rationale(cv)

        assert step.claim_id == cv.claim.claim_id

    @patch("src.copilot.rationale.anthropic.Anthropic")
    def test_provenance_step_token_counts(self, mock_cls: MagicMock) -> None:
        mock_cls.return_value.messages.create.return_value = _mock_response(
            "Rationale.", tokens_in=200, tokens_out=12
        )

        _, step = extract_rationale(_make_cv())

        assert step.tokens_in == 200
        assert step.tokens_out == 12

    @patch("src.copilot.rationale.anthropic.Anthropic")
    def test_provenance_step_hashes_are_strings(self, mock_cls: MagicMock) -> None:
        mock_cls.return_value.messages.create.return_value = _mock_response("Rationale.")

        _, step = extract_rationale(_make_cv())

        assert isinstance(step.input_hash, str) and len(step.input_hash) == 64
        assert isinstance(step.output_hash, str) and len(step.output_hash) == 64

    @patch("src.copilot.rationale.anthropic.Anthropic")
    def test_strips_leading_trailing_whitespace(self, mock_cls: MagicMock) -> None:
        mock_cls.return_value.messages.create.return_value = _mock_response(
            "  Rationale with spaces.  "
        )

        rationale, _ = extract_rationale(_make_cv())

        assert rationale == rationale.strip()

    @patch("src.copilot.rationale.anthropic.Anthropic")
    def test_cache_hit_recorded_when_cache_read_tokens_present(self, mock_cls: MagicMock) -> None:
        response = _mock_response("Rationale.")
        response.usage.cache_read_input_tokens = 150
        response.usage.cache_creation_input_tokens = 0
        mock_cls.return_value.messages.create.return_value = response

        _, step = extract_rationale(_make_cv())

        assert step.cache_hit is True


# ---------------------------------------------------------------------------
# extract_rationale — fallback path
# ---------------------------------------------------------------------------


class TestExtractRationaleFallback:
    @patch("src.copilot.rationale.anthropic.Anthropic")
    def test_fallback_on_api_exception(self, mock_cls: MagicMock) -> None:
        mock_cls.return_value.messages.create.side_effect = RuntimeError("API down")
        explanation = "The source abstract shows contradictory evidence. No replication found."
        cv = _make_cv(explanation=explanation)

        rationale, _step = extract_rationale(cv)

        assert rationale == "The source abstract shows contradictory evidence."

    @patch("src.copilot.rationale.anthropic.Anthropic")
    def test_fallback_provenance_step_still_emitted(self, mock_cls: MagicMock) -> None:
        mock_cls.return_value.messages.create.side_effect = ValueError("bad key")

        _, step = extract_rationale(_make_cv())

        assert isinstance(step, ProvenanceStep)
        assert step.operation == "copilot_rationale"

    @patch("src.copilot.rationale.anthropic.Anthropic")
    def test_fallback_tokens_are_zero(self, mock_cls: MagicMock) -> None:
        mock_cls.return_value.messages.create.side_effect = ConnectionError("timeout")

        _, step = extract_rationale(_make_cv())

        assert step.tokens_in == 0
        assert step.tokens_out == 0

    @patch("src.copilot.rationale.anthropic.Anthropic")
    def test_fallback_on_empty_content_list(self, mock_cls: MagicMock) -> None:
        response = MagicMock()
        response.content = []
        response.usage.input_tokens = 100
        response.usage.output_tokens = 0
        response.usage.cache_read_input_tokens = 0
        response.usage.cache_creation_input_tokens = 100
        mock_cls.return_value.messages.create.return_value = response
        explanation = "The cited paper does not address this claim at all."

        rationale, _ = extract_rationale(_make_cv(explanation=explanation))

        assert rationale == "The cited paper does not address this claim at all."

    @patch("src.copilot.rationale.anthropic.Anthropic")
    def test_never_raises(self, mock_cls: MagicMock) -> None:
        mock_cls.side_effect = Exception("even constructor fails")

        # Must not raise — fallback always returns something
        rationale, step = extract_rationale(_make_cv())

        assert isinstance(rationale, str) and len(rationale) > 0
        assert isinstance(step, ProvenanceStep)


# ---------------------------------------------------------------------------
# Determinism / reproducibility
# ---------------------------------------------------------------------------


class TestExtractRationaleDeterminism:
    @patch("src.copilot.rationale.anthropic.Anthropic")
    def test_same_input_same_hashes(self, mock_cls: MagicMock) -> None:
        """Input hash must be identical across two calls with the same ClaimVerification."""
        mock_cls.return_value.messages.create.return_value = _mock_response("Rationale.")
        cv = _make_cv()

        _, step1 = extract_rationale(cv)
        _, step2 = extract_rationale(cv)

        assert step1.input_hash == step2.input_hash
