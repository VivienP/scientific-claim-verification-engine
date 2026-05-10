"""Unit tests for src/copilot/fix_generator.py — mocked LLM and CrossRef, offline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.copilot.fix_generator import generate_fix
from src.models import (
    Claim,
    ProvenanceStep,
    ResolvedSource,
    ResolvedSourceSet,
    VerificationResult,
)
from src.pipeline import ClaimVerification

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_cv(
    verdict: str = "unsupported",
    explanation: str = "The source does not report any reduction in HbA1c.",
    claim_text: str = "Drug X reduces HbA1c by 1.5% in T2D patients.",
    passages: list[str] | None = None,
) -> ClaimVerification:
    claim = Claim(
        claim_id="cl-fix-001",
        claim_text=claim_text,
        cited_authors=["Jones"],
        cited_year=2022,
        claim_type="factual_numeric",
    )
    source = ResolvedSource(
        found=True,
        doi="10.1234/original",
        title="Original Paper",
        abstract="This is a review.",
        similarity_score=0.7,
    )
    source_set = ResolvedSourceSet(sources=(source,), citation_markers=(1,))
    result = VerificationResult(
        status=verdict,  # type: ignore[arg-type]
        explanation=explanation,
        confidence=0.3,
        source_passages=passages or [],
    )
    return ClaimVerification(
        claim=claim,
        source=source,
        source_set=source_set,
        result=result,
        fetch_method="abstract",
    )


def _mock_llm_response(
    action: str = "swap_doi",
    suggested_doi: str | None = "10.9999/better",
    suggested_doi_title: str | None = "Better Paper",
    reworded_claim: str | None = None,
    confidence: float = 0.85,
) -> MagicMock:
    """Create a mock Anthropic response with a tool_use block."""
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "submit_fix"
    tool_block.input = {
        "action": action,
        "suggested_doi": suggested_doi,
        "suggested_doi_title": suggested_doi_title,
        "reworded_claim": reworded_claim,
        "confidence": confidence,
        "reasoning": "The suggested source directly supports this claim.",
    }

    response = MagicMock()
    response.content = [tool_block]
    response.usage.input_tokens = 300
    response.usage.output_tokens = 60
    response.usage.cache_read_input_tokens = 0
    response.usage.cache_creation_input_tokens = 300
    return response


def _mock_crossref_found(doi: str = "10.9999/better", title: str = "Better Paper") -> MagicMock:
    result = MagicMock()
    result.found = True
    result.doi = doi
    result.title = title
    return result


def _mock_crossref_not_found() -> MagicMock:
    result = MagicMock()
    result.found = False
    result.doi = None
    result.title = None
    return result


# ---------------------------------------------------------------------------
# Routing — verdicts that skip fix generation
# ---------------------------------------------------------------------------


class TestFixGeneratorRouting:
    def test_supported_returns_none(self, tmp_path: Path) -> None:
        cv = _make_cv(verdict="supported")
        fix, _step = generate_fix(cv, db_path=tmp_path / "c.db")
        assert fix is None

    def test_supported_still_emits_step(self, tmp_path: Path) -> None:
        cv = _make_cv(verdict="supported")
        _fix, step = generate_fix(cv, db_path=tmp_path / "c.db")
        assert isinstance(step, ProvenanceStep)
        assert step.operation == "copilot_fix"

    @patch("src.copilot.fix_generator.anthropic.Anthropic")
    @patch("src.copilot.fix_generator.fetch_work_by_doi")
    def test_unsupported_triggers_llm(
        self, mock_crossref: MagicMock, mock_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_cls.return_value.messages.create.return_value = _mock_llm_response()
        mock_crossref.return_value = _mock_crossref_found()

        fix, _ = generate_fix(_make_cv(verdict="unsupported"), db_path=tmp_path / "c.db")

        mock_cls.return_value.messages.create.assert_called_once()
        assert fix is not None

    @patch("src.copilot.fix_generator.anthropic.Anthropic")
    @patch("src.copilot.fix_generator.fetch_work_by_doi")
    def test_partially_supported_triggers_llm(
        self, mock_crossref: MagicMock, mock_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_cls.return_value.messages.create.return_value = _mock_llm_response()
        mock_crossref.return_value = _mock_crossref_found()

        fix, _ = generate_fix(_make_cv(verdict="partially_supported"), db_path=tmp_path / "c.db")

        assert fix is not None

    @patch("src.copilot.fix_generator.anthropic.Anthropic")
    @patch("src.copilot.fix_generator.fetch_work_by_doi")
    def test_not_addressed_triggers_llm(
        self, mock_crossref: MagicMock, mock_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_cls.return_value.messages.create.return_value = _mock_llm_response(
            action="add_citation"
        )
        mock_crossref.return_value = _mock_crossref_found()

        fix, _ = generate_fix(_make_cv(verdict="not_addressed"), db_path=tmp_path / "c.db")

        assert fix is not None


# ---------------------------------------------------------------------------
# CrossRef verification gate
# ---------------------------------------------------------------------------


class TestCrossRefVerificationGate:
    @patch("src.copilot.fix_generator.anthropic.Anthropic")
    @patch("src.copilot.fix_generator.fetch_work_by_doi")
    def test_verified_doi_is_kept(
        self, mock_crossref: MagicMock, mock_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_cls.return_value.messages.create.return_value = _mock_llm_response(
            suggested_doi="10.9999/real"
        )
        mock_crossref.return_value = _mock_crossref_found(doi="10.9999/real")

        fix, _ = generate_fix(_make_cv(), db_path=tmp_path / "c.db")

        assert fix is not None
        assert fix.suggested_doi == "10.9999/real"

    @patch("src.copilot.fix_generator.anthropic.Anthropic")
    @patch("src.copilot.fix_generator.fetch_work_by_doi")
    def test_hallucinated_doi_is_nulled(
        self, mock_crossref: MagicMock, mock_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_cls.return_value.messages.create.return_value = _mock_llm_response(
            suggested_doi="10.9999/fake-hallucinated-doi"
        )
        mock_crossref.return_value = _mock_crossref_not_found()

        fix, _ = generate_fix(_make_cv(), db_path=tmp_path / "c.db")

        assert fix is not None
        assert fix.suggested_doi is None

    @patch("src.copilot.fix_generator.anthropic.Anthropic")
    @patch("src.copilot.fix_generator.fetch_work_by_doi")
    def test_llm_null_doi_stays_none(
        self, mock_crossref: MagicMock, mock_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_cls.return_value.messages.create.return_value = _mock_llm_response(suggested_doi=None)
        # CrossRef should NOT be called when doi is None
        fix, _ = generate_fix(_make_cv(), db_path=tmp_path / "c.db")

        mock_crossref.assert_not_called()
        assert fix is not None
        assert fix.suggested_doi is None


# ---------------------------------------------------------------------------
# RecommendedFix fields
# ---------------------------------------------------------------------------


class TestFixFields:
    @patch("src.copilot.fix_generator.anthropic.Anthropic")
    @patch("src.copilot.fix_generator.fetch_work_by_doi")
    def test_action_populated(
        self, mock_crossref: MagicMock, mock_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_cls.return_value.messages.create.return_value = _mock_llm_response(
            action="reword", suggested_doi=None
        )
        fix, _ = generate_fix(_make_cv(), db_path=tmp_path / "c.db")
        assert fix is not None
        assert fix.action == "reword"

    @patch("src.copilot.fix_generator.anthropic.Anthropic")
    @patch("src.copilot.fix_generator.fetch_work_by_doi")
    def test_reworded_claim_preserved(
        self, mock_crossref: MagicMock, mock_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_cls.return_value.messages.create.return_value = _mock_llm_response(
            action="reword",
            suggested_doi=None,
            reworded_claim="Drug X may modestly reduce HbA1c.",
        )
        fix, _ = generate_fix(_make_cv(), db_path=tmp_path / "c.db")
        assert fix is not None
        assert fix.reworded_claim == "Drug X may modestly reduce HbA1c."

    @patch("src.copilot.fix_generator.anthropic.Anthropic")
    @patch("src.copilot.fix_generator.fetch_work_by_doi")
    def test_confidence_within_range(
        self, mock_crossref: MagicMock, mock_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_cls.return_value.messages.create.return_value = _mock_llm_response(
            confidence=0.7, suggested_doi=None
        )
        fix, _ = generate_fix(_make_cv(), db_path=tmp_path / "c.db")
        assert fix is not None
        assert 0.0 <= fix.confidence <= 1.0

    @patch("src.copilot.fix_generator.anthropic.Anthropic")
    @patch("src.copilot.fix_generator.fetch_work_by_doi")
    def test_provenance_step_id_present(
        self, mock_crossref: MagicMock, mock_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_cls.return_value.messages.create.return_value = _mock_llm_response(suggested_doi=None)
        fix, _ = generate_fix(_make_cv(), db_path=tmp_path / "c.db")
        assert fix is not None
        assert isinstance(fix.provenance_step_id, str) and len(fix.provenance_step_id) > 0


# ---------------------------------------------------------------------------
# ProvenanceStep
# ---------------------------------------------------------------------------


class TestFixProvenanceStep:
    def test_step_operation(self, tmp_path: Path) -> None:
        _, step = generate_fix(_make_cv(verdict="supported"), db_path=tmp_path / "c.db")
        assert step.operation == "copilot_fix"

    @patch("src.copilot.fix_generator.anthropic.Anthropic")
    @patch("src.copilot.fix_generator.fetch_work_by_doi")
    def test_step_token_counts(
        self, mock_crossref: MagicMock, mock_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_cls.return_value.messages.create.return_value = _mock_llm_response(suggested_doi=None)
        _, step = generate_fix(_make_cv(), db_path=tmp_path / "c.db")
        assert step.tokens_in == 300
        assert step.tokens_out == 60

    def test_step_zero_tokens_when_no_llm_call(self, tmp_path: Path) -> None:
        _, step = generate_fix(_make_cv(verdict="supported"), db_path=tmp_path / "c.db")
        assert step.tokens_in == 0
        assert step.tokens_out == 0

    def test_step_hashes_present(self, tmp_path: Path) -> None:
        _, step = generate_fix(_make_cv(verdict="supported"), db_path=tmp_path / "c.db")
        assert len(step.input_hash) == 64
        assert len(step.output_hash) == 64

    def test_same_input_same_input_hash(self, tmp_path: Path) -> None:
        cv = _make_cv(verdict="supported")
        _, step1 = generate_fix(cv, db_path=tmp_path / "c.db")
        _, step2 = generate_fix(cv, db_path=tmp_path / "c.db")
        assert step1.input_hash == step2.input_hash


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


class TestFixGeneratorFailure:
    @patch("src.copilot.fix_generator.anthropic.Anthropic")
    def test_never_raises_on_llm_exception(self, mock_cls: MagicMock, tmp_path: Path) -> None:
        mock_cls.return_value.messages.create.side_effect = RuntimeError("API failure")

        fix, step = generate_fix(_make_cv(), db_path=tmp_path / "c.db")

        assert fix is None
        assert isinstance(step, ProvenanceStep)

    @patch("src.copilot.fix_generator.anthropic.Anthropic")
    def test_empty_content_returns_none(self, mock_cls: MagicMock, tmp_path: Path) -> None:
        response = MagicMock()
        response.content = []
        response.usage.input_tokens = 100
        response.usage.output_tokens = 0
        response.usage.cache_read_input_tokens = 0
        response.usage.cache_creation_input_tokens = 100
        mock_cls.return_value.messages.create.return_value = response

        fix, step = generate_fix(_make_cv(), db_path=tmp_path / "c.db")

        assert fix is None
        assert step.operation == "copilot_fix"

    @patch("src.copilot.fix_generator.anthropic.Anthropic")
    @patch("src.copilot.fix_generator.fetch_work_by_doi")
    def test_crossref_exception_nulls_doi(
        self, mock_crossref: MagicMock, mock_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_cls.return_value.messages.create.return_value = _mock_llm_response(
            suggested_doi="10.9999/valid-looking"
        )
        mock_crossref.side_effect = Exception("CrossRef down")

        fix, _step = generate_fix(_make_cv(), db_path=tmp_path / "c.db")

        # Exception in CrossRef causes the whole generate_fix to return None
        assert fix is None
