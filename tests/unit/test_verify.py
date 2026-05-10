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


class TestVerifierPromptRubric:
    def test_partial_support_precedes_unsupported_when_some_specific_evidence_matches(
        self,
    ) -> None:
        from src.verify import _FULLTEXT_SYSTEM_PROMPT, _SYSTEM_PROMPT

        # S1-P3 v2: partial precedence rule may use backticks; check the
        # semantic content rather than a verbatim string.
        for prompt in (_SYSTEM_PROMPT, _FULLTEXT_SYSTEM_PROMPT):
            assert "takes precedence" in prompt
            assert "partially_supported" in prompt
            assert "unsupported" in prompt

    def test_clause_a_collapses_off_topic_into_unsupported_in_both_prompts(self) -> None:
        """S1-P3 Clause A: off-topic source → unsupported (not not_addressed).
        Pinned in both abstract and fulltext prompts to prevent regression.
        """
        from src.verify import _FULLTEXT_SYSTEM_PROMPT, _SYSTEM_PROMPT

        for prompt in (_SYSTEM_PROMPT, _FULLTEXT_SYSTEM_PROMPT):
            assert "Clause A" in prompt
            assert "off-topic" in prompt.lower()
            # Both prompts must instruct the LLM not to emit not_addressed when
            # content is provided.
            assert "not `not_addressed`" in prompt or "NOT `not_addressed`" in prompt

    def test_clause_b_uncertainty_band_inclusion_in_both_prompts(self) -> None:
        """S1-P3 Clause B: range/IQR/CI/SD inclusion bidirectional rule."""
        from src.verify import _FULLTEXT_SYSTEM_PROMPT, _SYSTEM_PROMPT

        for prompt in (_SYSTEM_PROMPT, _FULLTEXT_SYSTEM_PROMPT):
            assert "Clause B" in prompt
            assert "uncertainty" in prompt.lower()
            # Bidirectional: must mention both directions (claim-in-source-range
            # AND source-point-in-claim-range)
            assert "IQR" in prompt or "95% CI" in prompt
            assert "central estimate" in prompt.lower()

    def test_clause_c_trajectory_vs_snapshot_in_both_prompts(self) -> None:
        """S1-P3 Clause C: directional/trajectory claims vs static evidence."""
        from src.verify import _FULLTEXT_SYSTEM_PROMPT, _SYSTEM_PROMPT

        for prompt in (_SYSTEM_PROMPT, _FULLTEXT_SYSTEM_PROMPT):
            assert "Clause C" in prompt
            assert "directional change" in prompt.lower()
            assert "static" in prompt.lower()

    def test_not_addressed_removed_from_response_schema(self) -> None:
        """Clause A side-effect: the JSON schema in both prompts no longer offers
        not_addressed as a valid output when content is present.

        Robust against ordering of supported/unsupported/partially_supported in
        the schema enum — only checks `not_addressed` is not listed as a value.
        """
        import re

        from src.verify import _FULLTEXT_SYSTEM_PROMPT, _SYSTEM_PROMPT

        schema_line_re = re.compile(r'"status":\s*"([^"]+)"')
        for prompt in (_SYSTEM_PROMPT, _FULLTEXT_SYSTEM_PROMPT):
            match = schema_line_re.search(prompt)
            assert match is not None, "schema line missing"
            allowed_values = match.group(1).split("|")
            assert "not_addressed" not in allowed_values
            assert "supported" in allowed_values
            assert "unsupported" in allowed_values
            assert "partially_supported" in allowed_values


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

    def test_abstract_none_short_title_no_llm_call(self) -> None:
        """EC-2: Paper found, abstract=None, title shorter than the title-only
        threshold (20 chars) → short-circuit (no Anthropic call).

        Pinning a short title explicitly so the test does not silently regress
        if `_make_source`'s default title (`"Test Paper"` = 10 chars) is
        lengthened in the future.
        """
        with patch("src.verify.anthropic.Anthropic") as mock_cls:
            from src.verify import verify_claim

            short_title_source = ResolvedSource(
                found=True,
                doi=None,
                title="Short",
                abstract=None,
                similarity_score=1.0,
            )
            result, step = verify_claim(_make_claim(), short_title_source)
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


def _make_title_only_source(
    title: str = "Real-time Continuous Measurement of Lactate via Microneedle Biosensor",
) -> ResolvedSource:
    """Source with no abstract but a long, informative title (title-only mode trigger)."""
    return ResolvedSource(
        found=True,
        doi="10.1/x",
        title=title,
        abstract=None,
        similarity_score=1.0,
        title_match_score=1.0,
    )


class TestVerifyClaimTitleOnly:
    """Bug B fix (S1-P1-B): when source.abstract is None but source.title is
    informative, route to title-only mode. Verdict is hard-capped at
    partially_supported to prevent overclaim from a title alone.
    """

    @patch("src.verify.anthropic.Anthropic")
    def test_title_only_partially_supported_when_title_matches_subject(
        self, mock_anthropic_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [
            _text_block(
                '{"status": "partially_supported", '
                '"explanation": "Title matches the claim subject.", '
                '"confidence": 0.6}'
            )
        ]
        mock_response.usage.input_tokens = 80
        mock_response.usage.output_tokens = 30
        mock_response.usage.cache_read_input_tokens = 80
        mock_response.usage.cache_creation_input_tokens = 0
        mock_client.messages.create.return_value = mock_response

        from src.verify import verify_claim_title_only

        result, step = verify_claim_title_only(_make_claim(), _make_title_only_source())

        assert result.status == "partially_supported"
        assert result.evidence_quality == "title_only"
        assert result.verification_depth == "title_only"
        assert step.operation == "verify"
        assert step.tokens_in == 80

    @patch("src.verify.anthropic.Anthropic")
    def test_title_only_hard_caps_supported_to_partially_supported(
        self, mock_anthropic_cls: MagicMock
    ) -> None:
        """Defensive: even if the LLM ignores the prompt and returns 'supported',
        the deterministic post-LLM cap downgrades it. Confidence is also clamped.
        """
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [
            _text_block('{"status": "supported", "explanation": "I am sure!", "confidence": 0.95}')
        ]
        mock_response.usage.input_tokens = 80
        mock_response.usage.output_tokens = 30
        mock_response.usage.cache_read_input_tokens = 80
        mock_response.usage.cache_creation_input_tokens = 0
        mock_client.messages.create.return_value = mock_response

        from src.verify import verify_claim_title_only

        result, _step = verify_claim_title_only(_make_claim(), _make_title_only_source())

        assert result.status == "partially_supported"
        assert result.confidence <= 0.7
        assert result.evidence_quality == "title_only"

    def test_verify_routes_to_title_only_when_abstract_none_long_title(self) -> None:
        """Integration: verify_claim() routes to title-only mode when
        abstract is None but title is informative (>20 chars).
        """
        with patch("src.verify.anthropic.Anthropic") as mock_anthropic_cls:
            mock_client = MagicMock()
            mock_anthropic_cls.return_value = mock_client
            mock_response = MagicMock()
            mock_response.content = [
                _text_block(
                    '{"status": "partially_supported", '
                    '"explanation": "Title supports.", '
                    '"confidence": 0.6}'
                )
            ]
            mock_response.usage.input_tokens = 80
            mock_response.usage.output_tokens = 30
            mock_response.usage.cache_read_input_tokens = 80
            mock_response.usage.cache_creation_input_tokens = 0
            mock_client.messages.create.return_value = mock_response

            from src.verify import verify_claim

            result, _step = verify_claim(_make_claim(), _make_title_only_source())

        assert result.status == "partially_supported"
        assert result.evidence_quality == "title_only"
        assert result.verification_depth == "title_only"


class TestAggregateMultiSource:
    """S2-P4: per-source aggregation rule for multi-citation claims."""

    @staticmethod
    def _result(status: str, confidence: float = 0.8) -> object:
        from src.models import VerificationResult

        return VerificationResult(
            status=status,  # type: ignore[arg-type]
            explanation="",
            confidence=confidence,
        )

    def test_empty_returns_not_addressed(self) -> None:
        from src.verify import _aggregate_multi_source_verdicts

        assert _aggregate_multi_source_verdicts([]) == "not_addressed"

    def test_all_supported_returns_supported(self) -> None:
        from src.verify import _aggregate_multi_source_verdicts

        results = [self._result("supported"), self._result("supported")]
        assert _aggregate_multi_source_verdicts(results) == "supported"  # type: ignore[arg-type]

    def test_supported_with_partial_returns_supported(self) -> None:
        from src.verify import _aggregate_multi_source_verdicts

        results = [self._result("supported"), self._result("partially_supported")]
        assert _aggregate_multi_source_verdicts(results) == "supported"  # type: ignore[arg-type]

    def test_supported_with_unsupported_returns_partial(self) -> None:
        """Mixed signal: one source supports, another contradicts → partial."""
        from src.verify import _aggregate_multi_source_verdicts

        results = [self._result("supported"), self._result("unsupported")]
        assert _aggregate_multi_source_verdicts(results) == "partially_supported"  # type: ignore[arg-type]

    def test_supported_with_not_addressed_returns_partial(self) -> None:
        from src.verify import _aggregate_multi_source_verdicts

        results = [self._result("supported"), self._result("not_addressed")]
        assert _aggregate_multi_source_verdicts(results) == "partially_supported"  # type: ignore[arg-type]

    def test_all_not_addressed_returns_not_addressed(self) -> None:
        from src.verify import _aggregate_multi_source_verdicts

        results = [self._result("not_addressed"), self._result("not_addressed")]
        assert _aggregate_multi_source_verdicts(results) == "not_addressed"  # type: ignore[arg-type]


class TestVerifyClaimMultiSource:
    """S2-P4: end-to-end verify against ResolvedSourceSet."""

    @patch("src.verify.anthropic.Anthropic")
    def test_aggregates_two_supported_sources(self, mock_anthropic_cls: MagicMock) -> None:
        from src.models import ResolvedSourceSet
        from src.verify import verify_claim_multi_source

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [
            _text_block('{"status": "supported", "explanation": "Matches.", "confidence": 0.9}')
        ]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 30
        mock_response.usage.cache_read_input_tokens = 100
        mock_response.usage.cache_creation_input_tokens = 0
        mock_client.messages.create.return_value = mock_response

        s1 = ResolvedSource(
            found=True, doi="10.1/a", title="A", abstract="abstract A", similarity_score=1.0
        )
        s2 = ResolvedSource(
            found=True, doi="10.1/b", title="B", abstract="abstract B", similarity_score=1.0
        )
        rs_set = ResolvedSourceSet(sources=(s1, s2), citation_markers=(81, 82))

        result, steps = verify_claim_multi_source(_make_claim(), rs_set)

        assert result.status == "supported"
        assert len(steps) == 3  # one verify step per source + one aggregate step
        assert steps[-1].operation == "aggregate"
        assert steps[-1].model_id is None
        assert steps[-1].claim_id == _make_claim().claim_id
        assert "[10.1/a] supported" in result.explanation
        assert "[10.1/b] supported" in result.explanation

    @patch("src.verify.anthropic.Anthropic")
    def test_aggregates_mixed_verdicts_to_partial(self, mock_anthropic_cls: MagicMock) -> None:
        from src.models import ResolvedSourceSet
        from src.verify import verify_claim_multi_source

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        # First source supports; second is unsupported.
        responses = [
            MagicMock(),
            MagicMock(),
        ]
        responses[0].content = [
            _text_block('{"status": "supported", "explanation": "OK.", "confidence": 0.9}')
        ]
        responses[0].usage.input_tokens = 100
        responses[0].usage.output_tokens = 30
        responses[0].usage.cache_read_input_tokens = 100
        responses[0].usage.cache_creation_input_tokens = 0
        responses[1].content = [
            _text_block(
                '{"status": "unsupported", "explanation": "Off-topic.", "confidence": 0.85}'
            )
        ]
        responses[1].usage.input_tokens = 100
        responses[1].usage.output_tokens = 30
        responses[1].usage.cache_read_input_tokens = 100
        responses[1].usage.cache_creation_input_tokens = 0
        mock_client.messages.create.side_effect = responses

        s1 = ResolvedSource(
            found=True, doi="10.1/a", title="A", abstract="abs", similarity_score=1.0
        )
        s2 = ResolvedSource(
            found=True, doi="10.1/b", title="B", abstract="abs", similarity_score=1.0
        )
        rs_set = ResolvedSourceSet(sources=(s1, s2), citation_markers=(81, 82))

        result, steps = verify_claim_multi_source(_make_claim(), rs_set)
        assert result.status == "partially_supported"
        assert steps[-1].operation == "aggregate"


class TestVerifyClaimCitingContext:
    """S3-P1: source-context fallback for unretrievable cited references.

    When the cited source cannot be reached (paywall + identifier-less +
    no abstract), check whether the citing paper's surrounding text is
    consistent with the claim. Hard-capped at partially_supported.
    """

    @staticmethod
    def _citing_text() -> str:
        # Synthetic citing-paper context: ~700 chars surrounding the claim.
        return (
            "Lactate kinetics in exercise have been studied since the 1970s. "
            "Brooks et al. demonstrated that lactic acid accumulates in muscle "
            "and blood beginning around 50-70% of maximal O2 uptake [58], well "
            "before aerobic capacity is fully utilized — a finding that "
            "challenged the classical anaerobic threshold concept. This is the "
            "key claim being analyzed here. Subsequent work has extended these "
            "observations to interstitial fluid measurements (Jansson 1996), "
            "though the mechanism remains debated."
        )

    @patch("src.verify.anthropic.Anthropic")
    def test_partially_supported_when_context_consistent(
        self, mock_anthropic_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [
            _text_block(
                '{"status": "partially_supported", '
                '"explanation": "Internal-consistency only — citing paper '
                'attributes the 50-70% threshold to Brooks.", '
                '"confidence": 0.5}'
            )
        ]
        mock_response.usage.input_tokens = 200
        mock_response.usage.output_tokens = 50
        mock_response.usage.cache_read_input_tokens = 200
        mock_response.usage.cache_creation_input_tokens = 0
        mock_client.messages.create.return_value = mock_response

        from src.verify import verify_claim_citing_context

        source = ResolvedSource(
            found=False, doi=None, title=None, abstract=None, similarity_score=None
        )
        claim = Claim(
            claim_id="c1",
            claim_text="Lactic acid accumulates in muscle and blood beginning around 50-70%",
            cited_authors=["Brooks"],
            cited_year=1986,
            claim_type="factual_numeric",
        )
        result, step = verify_claim_citing_context(claim, source, self._citing_text())

        assert result.status == "partially_supported"
        assert result.evidence_quality == "citing_paper_context"
        assert result.verification_depth == "citing_paper_context"
        assert result.confidence <= 0.6
        assert "internal-consistency" in result.explanation.lower()
        assert step.operation == "verify"

    @patch("src.verify.anthropic.Anthropic")
    def test_unsupported_when_context_contradicts(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [
            _text_block(
                '{"status": "unsupported", '
                '"explanation": "Internal-consistency only — citing paper does not '
                'mention the cited reference in this context.", '
                '"confidence": 0.55}'
            )
        ]
        mock_response.usage.input_tokens = 200
        mock_response.usage.output_tokens = 40
        mock_response.usage.cache_read_input_tokens = 200
        mock_response.usage.cache_creation_input_tokens = 0
        mock_client.messages.create.return_value = mock_response

        from src.verify import verify_claim_citing_context

        source = ResolvedSource(
            found=False, doi=None, title=None, abstract=None, similarity_score=None
        )
        claim = Claim(
            claim_id="c1",
            claim_text="Some unrelated assertion about cosmology.",
            cited_authors=["Brooks"],
            cited_year=1986,
            claim_type="factual_qualitative",
        )
        result, _step = verify_claim_citing_context(claim, source, self._citing_text())
        assert result.status == "unsupported"
        assert result.evidence_quality == "citing_paper_context"

    @patch("src.verify.anthropic.Anthropic")
    def test_hard_caps_supported_to_partially(self, mock_anthropic_cls: MagicMock) -> None:
        """Defensive: even if the LLM ignores the prompt and returns 'supported',
        the deterministic post-LLM cap downgrades it to partially_supported.
        """
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [
            _text_block(
                '{"status": "supported", "explanation": "I am confident.", "confidence": 0.95}'
            )
        ]
        mock_response.usage.input_tokens = 200
        mock_response.usage.output_tokens = 30
        mock_response.usage.cache_read_input_tokens = 200
        mock_response.usage.cache_creation_input_tokens = 0
        mock_client.messages.create.return_value = mock_response

        from src.verify import verify_claim_citing_context

        source = ResolvedSource(
            found=False, doi=None, title=None, abstract=None, similarity_score=None
        )
        claim = Claim(
            claim_id="c1",
            claim_text="Lactic acid accumulates",
            cited_authors=["Brooks"],
            cited_year=1986,
            claim_type="factual_qualitative",
        )
        result, _step = verify_claim_citing_context(claim, source, self._citing_text())

        assert result.status == "partially_supported"
        # Confidence clamped under the citing-context max.
        assert result.confidence <= 0.6
        # Explanation is prefixed with internal-consistency tag if the LLM omits it.
        assert "internal-consistency" in result.explanation.lower()

    def test_extract_window_locates_claim_in_text(self) -> None:
        from src.verify import _extract_citing_context_window

        text = "Lots of preamble. " + ("X" * 800) + " The KEY CLAIM appears here. " + ("Y" * 800)
        window = _extract_citing_context_window(text, "The KEY CLAIM appears here.")
        assert "The KEY CLAIM appears here." in window
        # Window is bounded around the claim, not the full document.
        assert len(window) < len(text)

    def test_extract_window_falls_back_when_claim_not_found(self) -> None:
        from src.verify import _extract_citing_context_window

        text = "Preamble unrelated to the claim being asked about. " * 30
        window = _extract_citing_context_window(text, "Completely unrelated needle.")
        # Falls back to first 2*window of text rather than empty.
        assert len(window) > 0


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
