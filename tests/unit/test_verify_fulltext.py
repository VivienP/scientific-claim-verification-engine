"""Unit tests for fulltext-mode verification in src/verify.py — mocked
Anthropic SDK.

Split from the original tests/unit/test_verify.py (1069 LOC) along the
abstract/fulltext seam. Abstract / title-only / multi-source /
citing-context tests live in tests/unit/test_verify_abstract.py.

Helpers `_text_block`, `_make_claim`, `_make_source` are duplicated from
the abstract test module to keep this file self-contained — preferred
over a shared conftest for ~25 LOC of fixture code.
"""

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
    def test_empty_passages_marks_no_passage_found(self, mock_verify: MagicMock) -> None:
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
        assert result.fulltext_available is True
        assert result.retrieval_status == "no_passage_found"

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
        assert result.retrieval_status == "passage_found"

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


# ---------------------------------------------------------------------------
# Phase A.2 — verifier audit-trail fallback when LLM returns no quoted passages
# (the dominant CTran-failure mode at baseline; see
# reports/phase_a2/ctran_failure_matrix.md)
# ---------------------------------------------------------------------------


class TestFulltextVerifierAuditFallback:
    """When the LLM returns ``source_passages=[]`` we must surface the BM25
    chunks instead, so the audit trail shows what the verifier examined.
    """

    @patch("src.verify.anthropic.Anthropic")
    def test_empty_llm_passages_falls_back_to_bm25_passages(
        self, mock_anthropic_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        # LLM returns a verdict but quotes nothing.
        mock_response.content = [_text_block(_fulltext_response(status="unsupported", passages=[]))]
        mock_response.usage.input_tokens = 500
        mock_response.usage.output_tokens = 30
        mock_response.usage.cache_read_input_tokens = 500
        mock_response.usage.cache_creation_input_tokens = 0
        mock_client.messages.create.return_value = mock_response

        from src.verify import verify_claim_fulltext

        bm25 = _make_passages(3)
        result, _ = verify_claim_fulltext(_make_claim(), _make_source(), bm25)
        # Audit trail: source_passages now mirrors the BM25 input.
        assert len(result.source_passages) == 3
        for chunk, projected in zip(bm25, result.source_passages, strict=True):
            assert chunk.text in projected or projected.startswith(chunk.text[:80])
        # Evidence quality reflects "passages were searched, none quoted".
        assert result.evidence_quality == "passages_searched_no_quote"
        # Status is preserved from the LLM verdict.
        assert result.status == "unsupported"

    @patch("src.verify.anthropic.Anthropic")
    def test_non_empty_llm_passages_take_precedence_over_bm25(
        self, mock_anthropic_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [
            _text_block(_fulltext_response(passages=["LLM-chosen quoted sentence."]))
        ]
        mock_response.usage.input_tokens = 500
        mock_response.usage.output_tokens = 80
        mock_response.usage.cache_read_input_tokens = 500
        mock_response.usage.cache_creation_input_tokens = 0
        mock_client.messages.create.return_value = mock_response

        from src.verify import verify_claim_fulltext

        result, _ = verify_claim_fulltext(_make_claim(), _make_source(), _make_passages(3))
        # The LLM's quote is surfaced; BM25 chunks are NOT mixed in.
        assert result.source_passages == ["LLM-chosen quoted sentence."]
        assert result.evidence_quality == "quoted_passage"

    @patch("src.verify.anthropic.Anthropic")
    def test_parse_error_also_surfaces_bm25_passages(self, mock_anthropic_cls: MagicMock) -> None:
        # Pre-fix: parse error -> source_passages=[] (audit trail erased).
        # Post-fix: parse error -> BM25 passages are surfaced.
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [_text_block("garbage not-json")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 10
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 0
        mock_client.messages.create.return_value = mock_response

        from src.verify import verify_claim_fulltext

        result, _ = verify_claim_fulltext(_make_claim(), _make_source(), _make_passages(2))
        assert result.status == "not_addressed"
        assert len(result.source_passages) == 2
        assert result.evidence_quality == "passages_searched_no_quote"

    def test_truncate_passage_is_a_no_op_under_limit(self) -> None:
        from src.verify import _truncate_passage

        short = "A short passage of ~30 characters."
        assert _truncate_passage(short) == short

    def test_truncate_passage_breaks_on_word_boundary(self) -> None:
        from src.verify import _truncate_passage

        # Build a passage that is comfortably over the default limit and
        # contains spaces so the truncator can pick a word boundary.
        long_text = ("word " * 300).strip()
        out = _truncate_passage(long_text, limit=200)
        assert len(out) <= 200 + 1  # +1 for ellipsis
        assert out.endswith("…")
        # Last visible char before ellipsis is a non-space (we trimmed).
        assert not out[:-1].endswith(" ")

    def test_audit_fallback_makes_claim_transparent(self) -> None:
        """Cross-module integration: a fulltext run that lands in the
        passages_searched_no_quote bucket must register as transparent in
        the AAR scorecard (the whole point of the fix)."""
        from src.aar import _claim_is_transparent

        verification = {
            "source_passages": ["BM25 passage one", "BM25 passage two"],
            "evidence_quality": "passages_searched_no_quote",
        }
        assert _claim_is_transparent(verification) is True

        # And the empty-passage variant: even if downstream code drops
        # source_passages somewhere, the new evidence_quality value alone
        # is enough to count as transparent.
        verification_no_passages = {
            "source_passages": [],
            "evidence_quality": "passages_searched_no_quote",
        }
        assert _claim_is_transparent(verification_no_passages) is True


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
