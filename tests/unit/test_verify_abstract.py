"""Unit tests for abstract / title-only / multi-source / citing-context paths
of src/verify.py — mocked Anthropic SDK.

Split from the original tests/unit/test_verify.py (1069 LOC) along the
abstract/fulltext seam. Fulltext-specific tests live in
tests/unit/test_verify_fulltext.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from anthropic.types import TextBlock

from src.models import Claim, ProvenanceStep, ResolvedSource


def _text_block(text: str) -> TextBlock:
    """Create a real TextBlock for use in mock responses."""
    return TextBlock(type="text", text=text)


def _make_claim(claim_id: str = "claim-1") -> Claim:
    # Numeric claim text: triggers _claim_has_specific_numeric so A2 downgrade tests work.
    return Claim(
        claim_id=claim_id,
        claim_text="Response rates were 20% at 12 weeks.",
        cited_authors=["Smith"],
        cited_year=2020,
        claim_type="factual_numeric",
    )


def _make_qualitative_claim(claim_id: str = "claim-q") -> Claim:
    """Qualitative claim: does NOT trigger numeric heuristic, passes through safe_verification_result."""
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
    def test_supported_status_numeric_claim_downgrades(self, mock_anthropic_cls: MagicMock) -> None:
        """A2: LLM returning 'supported' on abstract-only evidence for a numeric claim is
        downgraded to 'unverifiable'. Numeric claims cannot be reliably confirmed from abstract."""
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
        # A2: numeric claim + supported+abstract_only -> downgraded to unverifiable
        assert result.status == "unverifiable"
        assert result.confidence is None
        assert isinstance(result.explanation, str)

    @patch("src.verify.anthropic.Anthropic")
    def test_supported_status_qualitative_claim_downgrades(
        self, mock_anthropic_cls: MagicMock
    ) -> None:
        """3.4 (Option A MIRROR): qualitative claim on abstract-only evidence is now
        downgraded to unverifiable, same as numeric claims.  The old pass-through
        behaviour was a latent bug (P1-2 incident class)."""
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

        result, _step = verify_claim(_make_qualitative_claim(), _make_source())
        # 3.4: qualitative claim on abstract-only -> downgraded to unverifiable
        assert result.status == "unverifiable"
        assert result.confidence is None

    @patch("src.verify.anthropic.Anthropic")
    def test_provenance_step_populated(self, mock_anthropic_cls: MagicMock) -> None:
        """A2: LLM returning 'supported' on abstract-only evidence for a numeric claim is
        downgraded to 'unverifiable'. ProvenanceStep.confidence must reflect the downgraded value (None)."""
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
        # A2: numeric claim + supported+abstract_only downgraded -> confidence is None in provenance
        assert step.confidence is None


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

    def test_clause_a_distinguishes_contradiction_from_silence_in_both_prompts(self) -> None:
        """Track G (2026-05-12) Clause A: `unsupported` is reserved for EXPLICIT
        CONTRADICTION; silence (on-topic or off-topic) is `not_addressed`.

        This replaces the previous semantics where off-topic abstracts were
        forced into `unsupported`. The old behavior was a silent-failure
        contributor on the Elicit psilocybin replay (qualitative claims where
        the abstract was silent landed as `unsupported` with high confidence).
        The new prompt teaches the LLM to emit `not_addressed` for silence,
        leaving `unsupported` for cases where the abstract directly disagrees
        with the claim.
        """
        from src.verify import _FULLTEXT_SYSTEM_PROMPT, _SYSTEM_PROMPT

        for prompt in (_SYSTEM_PROMPT, _FULLTEXT_SYSTEM_PROMPT):
            assert "Clause A" in prompt
            # Must instruct that contradiction is the trigger for unsupported.
            assert "CONTRADICTION" in prompt or "contradiction" in prompt
            # Must teach that silence -> not_addressed (the inversion of the
            # old "collapse-into-unsupported" behavior).
            silence_phrasing_present = (
                "silent on the specific" in prompt
                or "silence" in prompt.lower()
                or "does not contain the claim's specific assertion" in prompt
            )
            assert silence_phrasing_present, (
                f"Clause A must teach silence -> not_addressed. Prompt: {prompt[:200]}..."
            )
            # off-topic must now route to not_addressed, not unsupported.
            assert "off-topic" in prompt.lower()

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

    def test_not_addressed_is_in_response_schema(self) -> None:
        """Track G (2026-05-12): the JSON schema in both prompts OFFERS
        `not_addressed` as a valid output now that Clause A reserves
        `unsupported` for explicit contradiction.

        This inverts the previous test (which pinned the old "collapse silence
        into unsupported" semantics). The status enum should list all four
        verdict values; the LLM picks based on Clause A's contradiction-vs-
        silence rule.
        """
        import re

        from src.verify import _FULLTEXT_SYSTEM_PROMPT, _SYSTEM_PROMPT

        schema_line_re = re.compile(r'"status":\s*"([^"]+)"')
        for prompt in (_SYSTEM_PROMPT, _FULLTEXT_SYSTEM_PROMPT):
            match = schema_line_re.search(prompt)
            assert match is not None, "schema line missing"
            allowed_values = match.group(1).split("|")
            assert "supported" in allowed_values
            assert "unsupported" in allowed_values
            assert "not_addressed" in allowed_values
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
        """LLM response wrapped in ```json fences is parsed correctly.
        A2: numeric claim + supported+abstract_only is downgraded to unverifiable."""
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
        # A2: fences parsed correctly; numeric claim + supported+abstract_only -> unverifiable
        assert result.status == "unverifiable"
        assert result.confidence is None

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
        assert result.confidence is not None
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

        # A1: supported/unsupported require fulltext-grade evidence
        eq = "quoted_passage" if status in ("supported", "unsupported") else "abstract_only"
        actual_confidence: float | None = None if status == "unverifiable" else confidence
        return VerificationResult(
            status=status,  # type: ignore[arg-type]
            explanation="",
            confidence=actual_confidence,
            evidence_quality=eq,  # type: ignore[arg-type]
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
        """A2: LLM saying 'supported' on abstract-only sources is downgraded to
        'unverifiable' per-source; the aggregate of two unverifiable is unverifiable."""
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

        # A2: both per-source results are unverifiable (supported+abstract_only -> downgraded)
        assert result.status == "unverifiable"
        assert len(steps) == 3  # one verify step per source + one aggregate step
        assert steps[-1].operation == "aggregate"
        assert steps[-1].model_id is None
        assert steps[-1].claim_id == _make_claim().claim_id
        # Explanation mentions per-source status (now unverifiable)
        assert "[10.1/a] unverifiable" in result.explanation
        assert "[10.1/b] unverifiable" in result.explanation

    @patch("src.verify.anthropic.Anthropic")
    def test_aggregates_mixed_verdicts_to_partial(self, mock_anthropic_cls: MagicMock) -> None:
        """A2: both sources on abstract-only evidence get downgraded to unverifiable
        regardless of the LLM's stated status. Aggregate of two unverifiable is unverifiable."""
        from src.models import ResolvedSourceSet
        from src.verify import verify_claim_multi_source

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        # First source returns supported; second returns unsupported.
        # Both are on abstract-only evidence -> A2 downgrades both to unverifiable.
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
        # A2: both per-source results unverifiable -> aggregate is unverifiable
        assert result.status == "unverifiable"
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
        assert result.confidence is not None
        assert result.confidence <= 0.6
        assert "internal-consistency" in result.explanation.lower()
        assert step.operation == "verify"

    @patch("src.verify.anthropic.Anthropic")
    def test_unsupported_when_context_contradicts(self, mock_anthropic_cls: MagicMock) -> None:
        """A2: unsupported+citing_paper_context is downgraded to unverifiable for numeric claims.
        Citing-paper context is insufficient for a confident 'unsupported' verdict on exact figures."""
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
        # Numeric claim: 5 mmol/L is not in patterns, use percentage instead
        claim = Claim(
            claim_id="c1",
            claim_text="Lactic acid accumulates at 50% of maximal exercise capacity.",
            cited_authors=["Brooks"],
            cited_year=1986,
            claim_type="factual_numeric",
        )
        result, _step = verify_claim_citing_context(claim, source, self._citing_text())
        # A2: numeric claim + unsupported+citing_paper_context -> downgraded to unverifiable
        assert result.status == "unverifiable"
        assert result.confidence is None
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
        assert result.confidence is not None
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


class TestA2EmissionGate:
    """A2: verify.py emission gate -- safe_verification_result routes
    (supported|unsupported) on abstract-only evidence to (unverifiable, None).
    """

    @patch("src.verify.anthropic.Anthropic")
    def test_verify_abstract_downgrades_confident_supported_to_unverifiable(
        self, mock_anthropic_cls: MagicMock
    ) -> None:
        """LLM returning supported+0.9 on abstract-only -> unverifiable, None."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [
            _text_block(
                '{"status": "supported", "explanation": "Abstract supports this.", "confidence": 0.9}'
            )
        ]
        mock_response.usage.input_tokens = 150
        mock_response.usage.output_tokens = 40
        mock_response.usage.cache_read_input_tokens = 150
        mock_response.usage.cache_creation_input_tokens = 0
        mock_client.messages.create.return_value = mock_response

        from src.verify import verify_claim

        result, step = verify_claim(_make_claim(), _make_source())
        assert result.status == "unverifiable"
        assert result.confidence is None
        assert result.evidence_quality == "abstract_only"
        assert step.confidence is None
        # F1 (2026-05-12): verify.py now passes
        # unverifiable_reason="numeric_claim_abstract_only" to the helper,
        # propagating to both the result and the step.
        assert step.unverifiable_reason == "numeric_claim_abstract_only"
        assert result.unverifiable_reason == "numeric_claim_abstract_only"

    @patch("src.verify.anthropic.Anthropic")
    def test_verify_abstract_downgrades_confident_unsupported_to_unverifiable(
        self, mock_anthropic_cls: MagicMock
    ) -> None:
        """The Goodwin NEJM 2022 / 20% sustained response case:
        LLM returning unsupported+0.75 on abstract-only -> unverifiable, None.
        """
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [
            _text_block(
                '{"status": "unsupported", "explanation": "The abstract does not '
                'report a specific 20% sustained response rate.", "confidence": 0.75}'
            )
        ]
        mock_response.usage.input_tokens = 150
        mock_response.usage.output_tokens = 40
        mock_response.usage.cache_read_input_tokens = 150
        mock_response.usage.cache_creation_input_tokens = 0
        mock_client.messages.create.return_value = mock_response

        from src.verify import verify_claim

        result, _step = verify_claim(_make_claim(), _make_source())
        assert result.status == "unverifiable"
        assert result.confidence is None
        assert result.evidence_quality == "abstract_only"

    @patch("src.verify.anthropic.Anthropic")
    def test_verify_abstract_preserves_not_addressed(self, mock_anthropic_cls: MagicMock) -> None:
        """not_addressed is exempt from Invariant 2 -- passes through unchanged."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [
            _text_block(
                '{"status": "not_addressed", "explanation": "Source silent.", "confidence": 0.9}'
            )
        ]
        mock_response.usage.input_tokens = 150
        mock_response.usage.output_tokens = 40
        mock_response.usage.cache_read_input_tokens = 150
        mock_response.usage.cache_creation_input_tokens = 0
        mock_client.messages.create.return_value = mock_response

        from src.verify import verify_claim

        result, _step = verify_claim(_make_claim(), _make_source())
        assert result.status == "not_addressed"
        assert result.confidence == 0.9

    @patch("src.verify.anthropic.Anthropic")
    def test_verify_abstract_preserves_partially_supported(
        self, mock_anthropic_cls: MagicMock
    ) -> None:
        """partially_supported is a hedge -- allowed on abstract-only."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [
            _text_block(
                '{"status": "partially_supported", "explanation": "Some match.", "confidence": 0.65}'
            )
        ]
        mock_response.usage.input_tokens = 150
        mock_response.usage.output_tokens = 40
        mock_response.usage.cache_read_input_tokens = 150
        mock_response.usage.cache_creation_input_tokens = 0
        mock_client.messages.create.return_value = mock_response

        from src.verify import verify_claim

        result, _step = verify_claim(_make_claim(), _make_source())
        assert result.status == "partially_supported"
        assert result.confidence == 0.65

    @patch("src.verify.anthropic.Anthropic")
    def test_verify_abstract_accepts_explicit_unverifiable_from_llm(
        self, mock_anthropic_cls: MagicMock
    ) -> None:
        """After A3 prompt, LLM may return unverifiable directly -- pass through."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [
            _text_block(
                '{"status": "unverifiable", "confidence": null, '
                '"explanation": "Cannot determine from abstract alone."}'
            )
        ]
        mock_response.usage.input_tokens = 150
        mock_response.usage.output_tokens = 40
        mock_response.usage.cache_read_input_tokens = 150
        mock_response.usage.cache_creation_input_tokens = 0
        mock_client.messages.create.return_value = mock_response

        from src.verify import verify_claim

        result, _step = verify_claim(_make_claim(), _make_source())
        assert result.status == "unverifiable"
        assert result.confidence is None


# ---------------------------------------------------------------------------
# 3.1: source_quote focal anchor injection in user message (abstract path)
# ---------------------------------------------------------------------------


class TestSourceQuoteAnchor:
    """Spec §3.1 / §9 Test A1, A2: source_quote injected into user message
    when non-null; omitted when None. Behaviour is unchanged from baseline
    when source_quote is None (A2 is a negative assertion).
    """

    @staticmethod
    def _make_claim_with_quote(source_quote: str | None) -> Claim:
        return Claim(
            claim_id="sq-claim-1",
            claim_text="Response rates were 20% at 12 weeks.",
            cited_authors=["Smith"],
            cited_year=2020,
            claim_type="factual_numeric",
            source_quote=source_quote,
        )

    @staticmethod
    def _mock_response() -> MagicMock:
        mock_response = MagicMock()
        mock_response.content = [
            _text_block('{"status": "supported", "explanation": "ok.", "confidence": 0.9}')
        ]
        mock_response.usage.input_tokens = 150
        mock_response.usage.output_tokens = 40
        mock_response.usage.cache_read_input_tokens = 150
        mock_response.usage.cache_creation_input_tokens = 0
        return mock_response

    @patch("src.verify.anthropic.Anthropic")
    def test_verify_abstract_includes_source_quote_anchor(
        self, mock_anthropic_cls: MagicMock
    ) -> None:
        """A1: when claim.source_quote is non-null, the user message must
        contain a <source_quote>...</source_quote> block with the exact text.
        """
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = self._mock_response()

        from src.verify import verify_claim

        quote = "The incidence of sustained response at week 12 was 20%"
        verify_claim(
            self._make_claim_with_quote(quote),
            _make_source(abstract="Abstract text."),
        )

        call = mock_client.messages.create.call_args
        user_message: str = call.kwargs["messages"][0]["content"]
        assert f"<source_quote>{quote}</source_quote>" in user_message

    @patch("src.verify.anthropic.Anthropic")
    def test_verify_abstract_omits_anchor_when_source_quote_none(
        self, mock_anthropic_cls: MagicMock
    ) -> None:
        """A2: when claim.source_quote is None, the user message must NOT
        contain any <source_quote> tag -- behaviour is unchanged from baseline.
        """
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = self._mock_response()

        from src.verify import verify_claim

        verify_claim(
            self._make_claim_with_quote(None),
            _make_source(abstract="Abstract text."),
        )

        call = mock_client.messages.create.call_args
        user_message: str = call.kwargs["messages"][0]["content"]
        assert "<source_quote>" not in user_message
