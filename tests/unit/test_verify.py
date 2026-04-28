"""Unit tests for src/verify.py — mocked Anthropic SDK."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from anthropic.types import TextBlock

from src.models import Claim, PaperChunk, ProvenanceStep, ResolvedSource


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
    def test_markdown_fenced_json_parsed(self, mock_anthropic_cls: MagicMock) -> None:
        """LLM response wrapped in ```json fences is parsed correctly."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        fenced = (
            '```json\n{"status": "supported", "explanation": "Matches.", "confidence": 0.9}\n```'
        )
        mock_response.content = [_text_block(fenced)]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 20
        mock_response.usage.cache_read_input_tokens = 100
        mock_response.usage.cache_creation_input_tokens = 0
        mock_client.messages.create.return_value = mock_response

        from src.verify import verify_claim

        result, _step = verify_claim(_make_claim(), _make_source())
        assert result.status == "supported"
        assert result.confidence == 0.9

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


def _make_passages(n: int = 3) -> list[PaperChunk]:
    return [
        PaperChunk(
            doi="10.1/x",
            section="results" if i == 0 else "introduction",
            text=f"Passage number {i} with substantive textual content for testing purposes.",
            char_start=i * 100,
            char_end=i * 100 + 80,
        )
        for i in range(n)
    ]


def _fulltext_response(
    status: str = "supported",
    section: str = "results",
    passages: list[str] | None = None,
) -> str:
    if passages is None:
        passages = ["Quoted sentence from passage."]
    import json as _json

    body = {
        "status": status,
        "explanation": "Found support in passages.",
        "confidence": 0.9,
        "source_passages": passages,
        "source_section": section,
    }
    return _json.dumps(body)


class TestVerifyClaimFulltext:
    @patch("src.verify.anthropic.Anthropic")
    def test_three_passages_calls_llm_with_passages(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [_text_block(_fulltext_response())]
        mock_response.usage.input_tokens = 500
        mock_response.usage.output_tokens = 80
        mock_response.usage.cache_read_input_tokens = 500
        mock_response.usage.cache_creation_input_tokens = 0
        mock_client.messages.create.return_value = mock_response

        from src.verify import verify_claim_fulltext

        result, _step = verify_claim_fulltext(_make_claim(), _make_source(), _make_passages(3))
        assert result.status == "supported"
        assert result.verification_depth == "fulltext"
        assert result.fulltext_available is True
        assert result.source_passages == ["Quoted sentence from passage."]
        assert result.source_section == "results"

        # Inspect the user message that was sent
        call = mock_client.messages.create.call_args
        user_message = call.kwargs["messages"][0]["content"]
        assert "<passage" in user_message
        assert "Passage number 0" in user_message

    @patch("src.verify.verify_claim")
    def test_empty_passages_falls_back_to_abstract(self, mock_verify: MagicMock) -> None:
        from src.models import VerificationResult

        mock_verify.return_value = (
            VerificationResult(status="supported", explanation="abs", confidence=0.9),
            ProvenanceStep(
                step_id="s",
                claim_id="claim-1",
                operation="verify",
                input_hash="i",
                output_hash="o",
                model_id="m",
                timestamp=0.0,
                tokens_in=10,
                tokens_out=5,
                cache_hit=False,
                confidence=0.9,
            ),
        )

        from src.verify import verify_claim_fulltext

        result, _ = verify_claim_fulltext(_make_claim(), _make_source(), [])
        mock_verify.assert_called_once()
        assert result.verification_depth == "abstract"
        assert result.fulltext_available is False

    @patch("src.verify.anthropic.Anthropic")
    def test_retraction_status_mirrored(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [_text_block(_fulltext_response())]
        mock_response.usage.input_tokens = 500
        mock_response.usage.output_tokens = 80
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 500
        mock_client.messages.create.return_value = mock_response

        retracted_source = ResolvedSource(
            found=True,
            doi="10.1/r",
            title="T",
            abstract="a",
            similarity_score=1.0,
            retraction_status=True,
        )

        from src.verify import verify_claim_fulltext

        result, _ = verify_claim_fulltext(_make_claim(), retracted_source, _make_passages(2))
        assert result.retraction_status is True

    @patch("src.verify.anthropic.Anthropic")
    def test_malformed_response_returns_parse_error(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [_text_block("not json at all")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 10
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 0
        mock_client.messages.create.return_value = mock_response

        from src.verify import verify_claim_fulltext

        result, _ = verify_claim_fulltext(_make_claim(), _make_source(), _make_passages(1))
        assert result.status == "not_addressed"
        assert result.confidence == 0.0
        assert result.verification_depth == "fulltext"

    @patch("src.verify.anthropic.Anthropic")
    def test_cache_control_on_system_prompt(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [_text_block(_fulltext_response())]
        mock_response.usage.input_tokens = 500
        mock_response.usage.output_tokens = 80
        mock_response.usage.cache_read_input_tokens = 500
        mock_response.usage.cache_creation_input_tokens = 0
        mock_client.messages.create.return_value = mock_response

        from src.verify import verify_claim_fulltext

        verify_claim_fulltext(_make_claim(), _make_source(), _make_passages(2))
        call = mock_client.messages.create.call_args
        system_blocks = call.kwargs["system"]
        assert system_blocks[0]["cache_control"] == {"type": "ephemeral"}


class TestVerifyClaimFulltextWithNumeric:
    @patch("src.numeric.engine.run_numeric_check")
    @patch("src.verify.verify_claim_fulltext")
    def test_numeric_check_attached_when_engine_returns_result(
        self,
        mock_verify_ft: MagicMock,
        mock_run_numeric: MagicMock,
    ) -> None:
        from src.models import ProvenanceStep, VerificationResult
        from src.numeric.checks import NumericCheckResult

        ft_result = VerificationResult(
            status="supported",
            explanation="ok",
            confidence=0.9,
            fulltext_available=True,
            verification_depth="fulltext",
        )
        ft_step = ProvenanceStep(
            step_id="vs",
            claim_id="claim-1",
            operation="verify",
            input_hash="i",
            output_hash="o",
            model_id="m",
            timestamp=0.0,
            tokens_in=100,
            tokens_out=20,
            cache_hit=False,
            confidence=0.9,
        )
        mock_verify_ft.return_value = (ft_result, ft_step)

        nc_result = NumericCheckResult(
            check_type="or_ci_consistency",
            consistent=True,
            extracted=[],
            explanation="OR/CI internally consistent.",
        )
        mock_run_numeric.return_value = (
            nc_result,
            [
                ProvenanceStep(
                    step_id="ne",
                    claim_id="claim-1",
                    operation="numeric_extract",
                    input_hash="i",
                    output_hash="o",
                    model_id="m",
                    timestamp=0.0,
                    tokens_in=200,
                    tokens_out=50,
                    cache_hit=False,
                    confidence=None,
                ),
                ProvenanceStep(
                    step_id="nc",
                    claim_id="claim-1",
                    operation="numeric_check",
                    input_hash="i",
                    output_hash="o",
                    model_id=None,
                    timestamp=0.0,
                    tokens_in=None,
                    tokens_out=None,
                    cache_hit=None,
                    confidence=None,
                ),
            ],
        )

        from src.verify import verify_claim_fulltext_with_numeric

        result, steps = verify_claim_fulltext_with_numeric(
            _make_claim(), _make_source(), _make_passages(2)
        )
        assert result.numeric_check is not None
        assert result.numeric_check.consistent is True
        assert len(steps) == 3
        assert steps[0].operation == "verify"
        assert steps[1].operation == "numeric_extract"
        assert steps[2].operation == "numeric_check"

    @patch("src.numeric.engine.run_numeric_check")
    @patch("src.verify.verify_claim_fulltext")
    def test_no_numeric_assertions_returns_none_check(
        self,
        mock_verify_ft: MagicMock,
        mock_run_numeric: MagicMock,
    ) -> None:
        from src.models import ProvenanceStep, VerificationResult

        ft_result = VerificationResult(
            status="supported",
            explanation="ok",
            confidence=0.9,
            fulltext_available=True,
            verification_depth="fulltext",
        )
        ft_step = ProvenanceStep(
            step_id="vs",
            claim_id="claim-1",
            operation="verify",
            input_hash="i",
            output_hash="o",
            model_id="m",
            timestamp=0.0,
            tokens_in=100,
            tokens_out=20,
            cache_hit=False,
            confidence=0.9,
        )
        mock_verify_ft.return_value = (ft_result, ft_step)

        # Engine returns None when no OR/CI triple is found
        mock_run_numeric.return_value = (
            None,
            [
                ProvenanceStep(
                    step_id="ne",
                    claim_id="claim-1",
                    operation="numeric_extract",
                    input_hash="i",
                    output_hash="o",
                    model_id="m",
                    timestamp=0.0,
                    tokens_in=200,
                    tokens_out=50,
                    cache_hit=False,
                    confidence=None,
                ),
            ],
        )

        from src.verify import verify_claim_fulltext_with_numeric

        result, steps = verify_claim_fulltext_with_numeric(
            _make_claim(), _make_source(), _make_passages(2)
        )
        assert result.numeric_check is None
        assert len(steps) == 2
