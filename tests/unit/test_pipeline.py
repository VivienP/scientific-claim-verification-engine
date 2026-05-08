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


def _result(
    status: str = "supported", evidence_quality: str = "abstract_only"
) -> VerificationResult:
    return VerificationResult(
        status=status,  # type: ignore[arg-type]
        explanation="ok",
        confidence=0.9,
        evidence_quality=evidence_quality,  # type: ignore[arg-type]
    )


class TestVerifyOneClaimRouting:
    """Exhaustive routing decision-tree coverage."""

    @patch("src.pipeline.fetch_fulltext")
    @patch("src.pipeline.verify_claim_fulltext_with_numeric")
    def test_fulltext_path_when_full_text_available(
        self, mock_ft_verify: MagicMock, mock_fetch: MagicMock
    ) -> None:
        mock_fetch.return_value = ("body of paper " * 50, "oa_url_pdf")
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
        mock_fetch.return_value = (None, "abstract_fallback")
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
        mock_fetch.return_value = (None, "abstract_fallback")
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
        mock_fetch.return_value = (None, "abstract_fallback")
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
        mock_fetch.return_value = (None, "abstract_fallback")
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
        mock_fetch.return_value = (None, "abstract_fallback")
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
        mock_fetch.return_value = (None, "abstract_fallback")
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
        mock_fetch.return_value = ("full text body", "oa_url_pdf")
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
        mock_fetch.return_value = (None, "abstract_fallback")
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
        mock_fetch.return_value = (None, "abstract_fallback")
        original_result = _result("not_addressed", "no_evidence")
        mock_verify.return_value = (original_result, _step())
        # Invalid cc output: status=supported is forbidden by the prompt.
        # Pipeline must not override the original verdict with this.
        mock_cc.return_value = (
            _result("supported", "citing_paper_context"),
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
        mock_fetch.return_value = (None, "abstract_fallback")
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
        mock_fetch.return_value = (None, "abstract_fallback")
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
        mock_fetch.return_value = (None, "abstract_fallback")
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
        mock_fetch.return_value = ("body", "oa_url_pdf")
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
        mock_fetch.return_value = (None, "abstract_fallback")
        mock_verify.return_value = (_result(), _step())
        rs_set = ResolvedSourceSet(sources=(_source(),), citation_markers=())
        cv = verify_one_claim(_claim(), rs_set, config=PipelineConfig())
        ops = [s.operation for s in cv.steps]
        assert ops == ["fetch_fulltext", "verify"]


# Avoid unused import warnings for PaperChunk in tests that don't use it directly.
_ = PaperChunk
