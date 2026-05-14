"""Unit tests for src/pipeline.py — orchestration contract.

Network and LLM calls are mocked. The intent is to lock in the routing
decision tree of :func:`verify_one_claim` so the benchmark harness, the
demo, and the integration tests cannot drift apart.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.models import (
    Claim,
    FetchAttempt,
    FetchOutcome,
    PaperChunk,
    ProvenanceStep,
    ResolvedSource,
    ResolvedSourceSet,
    VerificationResult,
)
from src.pipeline import (
    ClaimVerification,
    PipelineConfig,
    run_pipeline,
    verify_one_claim,
)


def _claim(claim_id: str = "c1", *, markers: list[int] | None = None) -> Claim:
    return Claim(
        claim_id=claim_id,
        claim_text="X correlates with Y in adult patients.",
        cited_authors=["Smith"],
        cited_year=2020,
        claim_type="factual_qualitative",
        citation_markers=markers or [],
    )


def _source(*, found: bool = True, abstract: str | None = "abstract present") -> ResolvedSource:
    return ResolvedSource(
        found=found,
        doi="10.1/x" if found else None,
        title="Title" if found else None,
        abstract=abstract if found else None,
        similarity_score=1.0 if found else None,
    )


def _step(claim_id: str = "c1") -> ProvenanceStep:
    return ProvenanceStep(
        step_id="s1",
        claim_id=claim_id,
        operation="verify",
        input_hash="i",
        output_hash="o",
        model_id="m",
        timestamp=0.0,
        tokens_in=10,
        tokens_out=5,
        cache_hit=False,
        confidence=0.9,
    )


def _fo(text: str | None, method: str = "abstract_fallback") -> FetchOutcome:
    """I1: build a minimal FetchOutcome for mocks. Attempts intentionally empty;
    routing tests don't assert on per-attempt telemetry, only on (text, method).
    """
    return FetchOutcome(
        text=text,
        method=method,  # type: ignore[arg-type]  # Literal narrowed at call site
        attempts=(),
        elapsed_ms_total=0,
    )


def _result(
    status: str = "supported", evidence_quality: str = "quoted_passage"
) -> VerificationResult:
    # A1: default evidence_quality changed to "quoted_passage" to match default
    # status="supported" (supported/unsupported require fulltext-grade evidence).
    actual_confidence: float | None = None if status == "unverifiable" else 0.9
    return VerificationResult(
        status=status,  # type: ignore[arg-type]
        explanation="ok",
        confidence=actual_confidence,  # type: ignore[arg-type]
        evidence_quality=evidence_quality,  # type: ignore[arg-type]
    )


class TestVerifyOneClaimRouting:
    """Exhaustive routing decision-tree coverage."""

    @patch("src.pipeline.fetch_fulltext")
    @patch("src.pipeline.verify_claim_fulltext_with_numeric")
    def test_fulltext_path_when_full_text_available(
        self, mock_ft_verify: MagicMock, mock_fetch: MagicMock
    ) -> None:
        mock_fetch.return_value = _fo("body of paper " * 50, "oa_url_pdf")
        mock_ft_verify.return_value = (_result("supported", "quoted_passage"), [_step()])
        rs_set = ResolvedSourceSet(sources=(_source(),), citation_markers=())
        cv = verify_one_claim(_claim(), rs_set, config=PipelineConfig())
        assert cv.result.status == "supported"
        assert cv.fetch_method == "oa_url_pdf"
        mock_ft_verify.assert_called_once()

    @patch("src.pipeline.fetch_fulltext")
    @patch("src.pipeline.verify_claim")
    def test_abstract_path_when_no_fulltext_but_abstract_present(
        self, mock_verify: MagicMock, mock_fetch: MagicMock
    ) -> None:
        mock_fetch.return_value = _fo(None, "abstract_fallback")
        mock_verify.return_value = (_result(), _step())
        rs_set = ResolvedSourceSet(sources=(_source(),), citation_markers=())
        cv = verify_one_claim(_claim(), rs_set, config=PipelineConfig())
        assert cv.fetch_method == "abstract_fallback"
        mock_verify.assert_called_once()

    @patch("src.pipeline.fetch_fulltext")
    @patch("src.pipeline.verify_claim_title_only")
    def test_title_only_path_when_no_abstract_but_long_title(
        self, mock_to: MagicMock, mock_fetch: MagicMock
    ) -> None:
        mock_fetch.return_value = _fo(None, "abstract_fallback")
        mock_to.return_value = (_result("partially_supported", "title_only"), _step())
        long_title = "A long descriptive scientific paper title beyond twenty chars"
        src = ResolvedSource(
            found=True,
            doi="10.1/x",
            title=long_title,
            abstract=None,
            similarity_score=1.0,
        )
        rs_set = ResolvedSourceSet(sources=(src,), citation_markers=())
        cv = verify_one_claim(_claim(), rs_set, config=PipelineConfig())
        mock_to.assert_called_once()
        assert cv.result.status == "partially_supported"

    @patch("src.pipeline.fetch_fulltext")
    @patch("src.pipeline.verify_claim")
    def test_short_circuit_when_source_not_found(
        self, mock_verify: MagicMock, mock_fetch: MagicMock
    ) -> None:
        mock_fetch.return_value = _fo(None, "abstract_fallback")
        mock_verify.return_value = (
            _result("not_addressed", "no_evidence"),
            _step(),
        )
        rs_set = ResolvedSourceSet(
            sources=(_source(found=False, abstract=None),), citation_markers=()
        )
        cv = verify_one_claim(_claim(), rs_set, config=PipelineConfig())
        assert cv.result.evidence_quality == "no_evidence"

    @patch("src.pipeline.fetch_fulltext")
    @patch("src.pipeline.verify_claim_multi_source")
    def test_multi_source_path_when_set_has_multiple_found_sources(
        self, mock_multi: MagicMock, mock_fetch: MagicMock
    ) -> None:
        mock_fetch.return_value = _fo(None, "abstract_fallback")
        mock_multi.return_value = (_result(), [_step(), _step()])
        rs_set = ResolvedSourceSet(sources=(_source(), _source()), citation_markers=(81, 82))
        cv = verify_one_claim(_claim(markers=[81, 82]), rs_set, config=PipelineConfig())
        mock_multi.assert_called_once()
        # 2 fetch steps (per source) + 2 verify steps from mock = 4 total.
        # When fetch returns None there is no chunk/select step.
        assert len(cv.steps) == 4
        assert sum(1 for s in cv.steps if s.operation == "fetch_fulltext") == 2
        assert sum(1 for s in cv.steps if s.operation == "verify") == 2

    @patch("src.pipeline.fetch_fulltext")
    @patch("src.pipeline.verify_claim")
    def test_multi_source_disabled_falls_back_to_primary(
        self, mock_verify: MagicMock, mock_fetch: MagicMock
    ) -> None:
        mock_fetch.return_value = _fo(None, "abstract_fallback")
        mock_verify.return_value = (_result(), _step())
        rs_set = ResolvedSourceSet(sources=(_source(), _source()), citation_markers=(81, 82))
        cv = verify_one_claim(
            _claim(markers=[81, 82]),
            rs_set,
            config=PipelineConfig(enable_multi_source=False),
        )
        mock_verify.assert_called_once()
        assert cv.fetch_method == "abstract_fallback"


class TestCitingContextFallback:
    @patch("src.pipeline.fetch_fulltext")
    @patch("src.pipeline.verify_claim")
    @patch("src.pipeline.verify_claim_citing_context")
    def test_fires_when_evidence_quality_is_no_evidence(
        self, mock_cc: MagicMock, mock_verify: MagicMock, mock_fetch: MagicMock
    ) -> None:
        mock_fetch.return_value = _fo(None, "abstract_fallback")
        mock_verify.return_value = (_result("not_addressed", "no_evidence"), _step())
        mock_cc.return_value = (
            _result("partially_supported", "citing_paper_context"),
            _step(),
        )
        rs_set = ResolvedSourceSet(
            sources=(_source(found=False, abstract=None),), citation_markers=()
        )
        cv = verify_one_claim(
            _claim(),
            rs_set,
            citing_paper_text="Surrounding text for context [30] supports the claim.",
            config=PipelineConfig(),
        )
        mock_cc.assert_called_once()
        assert cv.result.status == "partially_supported"
        assert cv.fetch_method == "citing_paper_context"

    @patch("src.pipeline.fetch_fulltext")
    @patch("src.pipeline.verify_claim_fulltext_with_numeric")
    @patch("src.pipeline.verify_claim_citing_context")
    def test_does_not_fire_when_evidence_quality_is_quoted_passage(
        self, mock_cc: MagicMock, mock_ft: MagicMock, mock_fetch: MagicMock
    ) -> None:
        mock_fetch.return_value = _fo("full text body", "oa_url_pdf")
        mock_ft.return_value = (_result("supported", "quoted_passage"), [_step()])
        rs_set = ResolvedSourceSet(sources=(_source(),), citation_markers=())
        verify_one_claim(
            _claim(),
            rs_set,
            citing_paper_text="text",
            config=PipelineConfig(),
        )
        mock_cc.assert_not_called()

    @patch("src.pipeline.fetch_fulltext")
    @patch("src.pipeline.verify_claim")
    @patch("src.pipeline.verify_claim_citing_context")
    def test_disabled_via_config(
        self, mock_cc: MagicMock, mock_verify: MagicMock, mock_fetch: MagicMock
    ) -> None:
        mock_fetch.return_value = _fo(None, "abstract_fallback")
        mock_verify.return_value = (_result("not_addressed", "no_evidence"), _step())
        rs_set = ResolvedSourceSet(
            sources=(_source(found=False, abstract=None),), citation_markers=()
        )
        verify_one_claim(
            _claim(),
            rs_set,
            citing_paper_text="text",
            config=PipelineConfig(enable_citing_context_fallback=False),
        )
        mock_cc.assert_not_called()

    @patch("src.pipeline.fetch_fulltext")
    @patch("src.pipeline.verify_claim")
    @patch("src.pipeline.verify_claim_citing_context")
    def test_does_not_override_supported_cc_verdict(
        self, mock_cc: MagicMock, mock_verify: MagicMock, mock_fetch: MagicMock
    ) -> None:
        """The cc verifier prompts says it must NEVER return `supported`,
        but if a future regression broke that, the pipeline should still
        only override on partially_supported / unsupported. `supported`
        from cc must not propagate.
        """
        mock_fetch.return_value = _fo(None, "abstract_fallback")
        original_result = _result("not_addressed", "no_evidence")
        mock_verify.return_value = (original_result, _step())
        # A2: supported+citing_paper_context downgrades to unverifiable via helper.
        # Pipeline must not override the original not_addressed verdict with this.
        mock_cc.return_value = (
            _result("unverifiable", "citing_paper_context"),
            _step(),
        )
        rs_set = ResolvedSourceSet(
            sources=(_source(found=False, abstract=None),), citation_markers=()
        )
        cv = verify_one_claim(
            _claim(),
            rs_set,
            citing_paper_text="text",
            config=PipelineConfig(),
        )
        assert cv.result.status == "not_addressed"  # unchanged

    @patch("src.pipeline.fetch_fulltext")
    @patch("src.pipeline.verify_claim_fulltext_with_numeric")
    @patch("src.pipeline.verify_claim_citing_context")
    def test_fires_when_evidence_quality_is_passages_searched_no_quote(
        self, mock_cc: MagicMock, mock_ft: MagicMock, mock_fetch: MagicMock
    ) -> None:
        """Phase A.2 regression test: the verifier now emits
        ``passages_searched_no_quote`` instead of ``no_evidence`` when the
        LLM saw passages but didn't quote any. The citing-context fallback
        gate must still trigger in that case — otherwise audit-trail-only
        verdicts (passages shown, none quoted) silently bypass the path
        designed for exactly that situation.
        """
        mock_fetch.return_value = _fo("full text body", "oa_url_pdf")
        # Fulltext path returns the new evidence_quality with non-empty
        # source_passages (BM25 fallback fired in verify_claim_fulltext).
        mock_ft.return_value = (
            _result("not_addressed", "passages_searched_no_quote"),
            [_step()],
        )
        mock_cc.return_value = (
            _result("partially_supported", "citing_paper_context"),
            _step(),
        )
        rs_set = ResolvedSourceSet(sources=(_source(),), citation_markers=())
        cv = verify_one_claim(
            _claim(),
            rs_set,
            citing_paper_text="Surrounding text for context [30] supports the claim.",
            config=PipelineConfig(),
        )
        # Falsifier: pre-fix this would have been `mock_cc.assert_not_called()`
        # because the gate read `evidence_quality == "no_evidence"`. Post-fix,
        # the gate accepts both `no_evidence` and `passages_searched_no_quote`.
        mock_cc.assert_called_once()
        assert cv.result.status == "partially_supported"
        assert cv.fetch_method == "citing_paper_context"


class TestRunPipeline:
    @patch("src.pipeline.extract_claims")
    @patch("src.pipeline.parse_bibliography")
    @patch("src.pipeline.resolve_citations_multi")
    @patch("src.pipeline.fetch_fulltext")
    @patch("src.pipeline.verify_claim")
    def test_orchestrates_extract_resolve_verify(
        self,
        mock_verify: MagicMock,
        mock_fetch: MagicMock,
        mock_resolve: MagicMock,
        mock_parse_bib: MagicMock,
        mock_extract: MagicMock,
    ) -> None:
        mock_extract.return_value = ([_claim("c1")], _step("c1"))
        mock_parse_bib.return_value = {}
        rs_set = ResolvedSourceSet(sources=(_source(),), citation_markers=())
        mock_resolve.return_value = ({"c1": rs_set}, [_step("c1")])
        mock_fetch.return_value = _fo(None, "abstract_fallback")
        mock_verify.return_value = (_result(), _step("c1"))

        cvs, all_steps = run_pipeline("Some text [1].", config=PipelineConfig())
        assert len(cvs) == 1
        assert isinstance(cvs[0], ClaimVerification)
        # extract_step + resolve_step + fetch_step + verify_step = 4
        assert len(all_steps) == 4
        ops = [s.operation for s in all_steps]
        assert "fetch_fulltext" in ops
        mock_extract.assert_called_once()
        mock_parse_bib.assert_called_once()

    @patch("src.pipeline.extract_claims")
    @patch("src.pipeline.parse_bibliography")
    @patch("src.pipeline.resolve_citations_multi")
    @patch("src.pipeline.fetch_fulltext")
    @patch("src.pipeline.verify_claim")
    def test_pre_extracted_claims_skips_extract_phase(
        self,
        mock_verify: MagicMock,
        mock_fetch: MagicMock,
        mock_resolve: MagicMock,
        mock_parse_bib: MagicMock,
        mock_extract: MagicMock,
    ) -> None:
        mock_parse_bib.return_value = {}
        rs_set = ResolvedSourceSet(sources=(_source(),), citation_markers=())
        mock_resolve.return_value = ({"c1": rs_set}, [_step("c1")])
        mock_fetch.return_value = _fo(None, "abstract_fallback")
        mock_verify.return_value = (_result(), _step("c1"))

        run_pipeline(
            "text",
            config=PipelineConfig(),
            pre_extracted_claims=[_claim("c1")],
        )
        mock_extract.assert_not_called()

    @patch("src.pipeline.extract_claims")
    @patch("src.pipeline.parse_bibliography")
    @patch("src.pipeline.resolve_citations_multi")
    @patch("src.pipeline.fetch_fulltext")
    @patch("src.pipeline.verify_claim")
    def test_pre_parsed_bibliography_skips_parse_phase(
        self,
        mock_verify: MagicMock,
        mock_fetch: MagicMock,
        mock_resolve: MagicMock,
        mock_parse_bib: MagicMock,
        mock_extract: MagicMock,
    ) -> None:
        mock_extract.return_value = ([_claim("c1")], _step("c1"))
        rs_set = ResolvedSourceSet(sources=(_source(),), citation_markers=())
        mock_resolve.return_value = ({"c1": rs_set}, [_step("c1")])
        mock_fetch.return_value = _fo(None, "abstract_fallback")
        mock_verify.return_value = (_result(), _step("c1"))

        run_pipeline(
            "text",
            config=PipelineConfig(),
            pre_parsed_bibliography={},
        )
        mock_parse_bib.assert_not_called()


class TestPipelineConfigDefaults:
    def test_pipeline_config_is_frozen(self) -> None:
        cfg = PipelineConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.top_k_passages = 99  # type: ignore[misc]

    def test_default_values(self) -> None:
        cfg = PipelineConfig()
        assert cfg.api_key is None
        assert cfg.db_path is None
        assert cfg.top_k_passages == 3
        assert cfg.enable_multi_source is True
        assert cfg.enable_citing_context_fallback is True


class TestClaimVerificationShape:
    def test_claim_verification_holds_full_audit_trail(self) -> None:
        rs_set = ResolvedSourceSet(sources=(_source(),), citation_markers=())
        cv = ClaimVerification(
            claim=_claim(),
            source=_source(),
            source_set=rs_set,
            result=_result(),
            fetch_method="abstract_fallback",
            passages=(),
            steps=(_step(),),
        )
        assert cv.claim.claim_id == "c1"
        assert cv.source_set.primary().doi == "10.1/x"
        assert len(cv.steps) == 1

    def test_claim_verification_is_frozen(self) -> None:
        rs_set = ResolvedSourceSet(sources=(_source(),), citation_markers=())
        cv = ClaimVerification(
            claim=_claim(),
            source=_source(),
            source_set=rs_set,
            result=_result(),
            fetch_method="abstract_fallback",
        )
        with pytest.raises((AttributeError, TypeError)):
            cv.fetch_method = "other"  # type: ignore[misc]


class TestEvidencePolicyShortCircuit:
    """Lane A: pipeline gates verify_* on assess_evidence_sufficiency.

    On Insufficient, the pipeline emits a deterministic unverifiable verdict
    and DOES NOT invoke the LLM. The verifier mock should NOT be called.
    """

    @patch("src.pipeline.fetch_fulltext")
    @patch("src.pipeline.verify_claim")
    def test_numeric_claim_on_abstract_only_emits_unverifiable_without_llm(
        self, mock_verify: MagicMock, mock_fetch: MagicMock
    ) -> None:
        """Numeric claim + abstract-only source -> policy short-circuit."""
        mock_fetch.return_value = _fo(None, "abstract_fallback")
        numeric_claim = Claim(
            claim_id="num-1",
            claim_text="The HR for MACE was 0.74 (95% CI 0.58-0.95) at week 12.",
            cited_authors=["Smith"],
            cited_year=2022,
            claim_type="factual_numeric",
        )
        rs_set = ResolvedSourceSet(sources=(_source(),), citation_markers=())
        cv = verify_one_claim(numeric_claim, rs_set, config=PipelineConfig())
        # Verifier was NOT called.
        mock_verify.assert_not_called()
        # Result is deterministic unverifiable.
        assert cv.result.status == "unverifiable"
        assert cv.result.confidence is None
        assert cv.result.unverifiable_reason == "numeric_claim_abstract_only"
        # Provenance step is model-free.
        verify_steps = [s for s in cv.steps if s.operation == "verify"]
        assert len(verify_steps) == 1
        assert verify_steps[0].model_id is None
        assert verify_steps[0].tokens_in is None
        assert verify_steps[0].unverifiable_reason == "numeric_claim_abstract_only"

    @patch("src.pipeline.fetch_fulltext")
    @patch("src.pipeline.verify_claim")
    def test_qualitative_claim_on_abstract_dispatches_to_verifier(
        self, mock_verify: MagicMock, mock_fetch: MagicMock
    ) -> None:
        """Qualitative claim + abstract -> policy is Sufficient -> verifier runs."""
        mock_fetch.return_value = _fo(None, "abstract_fallback")
        mock_verify.return_value = (_result("supported", "abstract_only"), _step())
        rs_set = ResolvedSourceSet(sources=(_source(),), citation_markers=())
        cv = verify_one_claim(_claim(), rs_set, config=PipelineConfig())
        # Policy returned Sufficient -> verifier was called.
        mock_verify.assert_called_once()
        assert cv.result.status == "supported"

    @patch("src.pipeline.fetch_fulltext")
    @patch("src.pipeline.verify_claim_fulltext_with_numeric")
    def test_numeric_claim_on_fulltext_dispatches_to_fulltext_verifier(
        self, mock_ft_verify: MagicMock, mock_fetch: MagicMock
    ) -> None:
        """Numeric claim + fulltext -> policy Sufficient -> fulltext verifier runs."""
        mock_fetch.return_value = _fo("body of paper " * 50, "oa_url_pdf")
        mock_ft_verify.return_value = (_result("supported", "quoted_passage"), [_step()])
        numeric_claim = Claim(
            claim_id="num-1",
            claim_text="HR for MACE was 0.74 (95% CI 0.58-0.95) at week 12.",
            cited_authors=["Smith"],
            cited_year=2022,
            claim_type="factual_numeric",
        )
        rs_set = ResolvedSourceSet(sources=(_source(),), citation_markers=())
        cv = verify_one_claim(numeric_claim, rs_set, config=PipelineConfig())
        # Fulltext verifier was called (numeric+fulltext is Sufficient).
        mock_ft_verify.assert_called_once()
        assert cv.result.status == "supported"

    def test_oa_url_not_pdf_builds_blocked_evidence_bundle(self) -> None:
        """Paywall/non-PDF OA responses must reach the access gate."""
        from src.pipeline import _build_evidence_bundle

        source = ResolvedSource(
            found=True,
            doi="10.1/x",
            title="A source with no usable abstract",
            abstract=None,
            similarity_score=1.0,
        )
        outcome = FetchOutcome(
            text=None,
            method="abstract_fallback",
            attempts=(
                FetchAttempt(
                    method="oa_url_pdf",
                    success=False,
                    reason="oa_url_not_pdf",
                    elapsed_ms=1,
                ),
            ),
            elapsed_ms_total=1,
        )

        bundle = _build_evidence_bundle(
            source,
            outcome,
            title_only_min_title_length=100,
        )

        assert bundle.depth == "none"
        assert bundle.access_status == "blocked"

    def test_failed_attempt_with_no_reason_falls_through_to_unavailable(
        self,
    ) -> None:
        """A failed FetchAttempt with reason=None is not enough to claim blocking.

        Only attempts that explicitly mark a paywall/non-PDF response can
        promote access_status to "blocked". An undocumented failure stays
        as "unavailable" so the policy gate remains conservative.
        """
        from src.pipeline import _build_evidence_bundle

        source = ResolvedSource(
            found=True,
            doi="10.1/x",
            title="A source with no usable abstract",
            abstract=None,
            similarity_score=1.0,
        )
        outcome = FetchOutcome(
            text=None,
            method="abstract_fallback",
            attempts=(
                FetchAttempt(
                    method="oa_url_pdf",
                    success=False,
                    reason=None,
                    elapsed_ms=1,
                ),
            ),
            elapsed_ms_total=1,
        )

        bundle = _build_evidence_bundle(
            source,
            outcome,
            title_only_min_title_length=100,
        )

        assert bundle.depth == "none"
        assert bundle.access_status == "unavailable"


class TestProvenanceEmissionForRetrieval:
    """provenance-first.md requires fetch / chunk / select to emit steps."""

    @patch("src.pipeline.fetch_fulltext")
    @patch("src.pipeline.chunk_paper")
    @patch("src.pipeline.select_passages")
    @patch("src.pipeline.verify_claim_fulltext_with_numeric")
    def test_fulltext_path_emits_fetch_chunk_select_verify_steps(
        self,
        mock_ft_verify: MagicMock,
        mock_select: MagicMock,
        mock_chunk: MagicMock,
        mock_fetch: MagicMock,
    ) -> None:
        mock_fetch.return_value = _fo("body", "oa_url_pdf")
        chunk = PaperChunk(
            doi="10.1/x",
            section="results",
            text="results body",
            char_start=0,
            char_end=12,
        )
        mock_chunk.return_value = [chunk]
        mock_select.return_value = [chunk]
        mock_ft_verify.return_value = (_result("supported", "quoted_passage"), [_step()])

        rs_set = ResolvedSourceSet(sources=(_source(),), citation_markers=())
        cv = verify_one_claim(_claim(), rs_set, config=PipelineConfig())

        ops = [s.operation for s in cv.steps]
        assert ops == ["fetch_fulltext", "chunk_paper", "select_passages", "verify"]
        # Deterministic steps carry no model_id.
        for s in cv.steps[:3]:
            assert s.model_id is None
            assert s.tokens_in is None

    @patch("src.pipeline.fetch_fulltext")
    @patch("src.pipeline.verify_claim")
    def test_abstract_path_emits_fetch_then_verify(
        self, mock_verify: MagicMock, mock_fetch: MagicMock
    ) -> None:
        mock_fetch.return_value = _fo(None, "abstract_fallback")
        mock_verify.return_value = (_result(), _step())
        rs_set = ResolvedSourceSet(sources=(_source(),), citation_markers=())
        cv = verify_one_claim(_claim(), rs_set, config=PipelineConfig())
        ops = [s.operation for s in cv.steps]
        assert ops == ["fetch_fulltext", "verify"]


# Avoid unused import warnings for PaperChunk in tests that don't use it directly.
_ = PaperChunk
