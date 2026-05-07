"""Unit tests for src/resolve.py — citation resolution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from pytest_httpx import HTTPXMock

from src.models import Claim, ProvenanceStep, ResolvedSource


def _make_claim(
    claim_id: str = "claim-1",
    cited_authors: list[str] | None = None,
    cited_year: int | None = 2020,
    claim_text: str = "Some claim about X.",
) -> Claim:
    return Claim(
        claim_id=claim_id,
        claim_text=claim_text,
        cited_authors=cited_authors if cited_authors is not None else ["Smith"],
        cited_year=cited_year,
        claim_type="factual_qualitative",
    )


class TestPhase1ResolveBehavior:
    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve._crossref.search_paper")
    @patch("src.resolve.search_paper")
    def test_openalex_miss_then_crossref_hit(
        self,
        mock_oa: MagicMock,
        mock_cf: MagicMock,
        mock_retr: MagicMock,
    ) -> None:
        mock_oa.return_value = ResolvedSource(False, None, None, None, None)
        mock_cf.return_value = ResolvedSource(True, "10.1/x", "T", None, None)
        from src.resolve import resolve_citations

        sources, _ = resolve_citations([_make_claim("c1")])
        assert sources["c1"].found is True
        assert sources["c1"].doi == "10.1/x"
        mock_cf.assert_called_once()

    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve._crossref.search_paper")
    @patch("src.resolve.search_paper")
    def test_openalex_miss_and_crossref_miss(
        self, mock_oa: MagicMock, mock_cf: MagicMock, mock_retr: MagicMock
    ) -> None:
        mock_oa.return_value = ResolvedSource(False, None, None, None, None)
        mock_cf.return_value = ResolvedSource(False, None, None, None, None)
        from src.resolve import resolve_citations

        sources, _ = resolve_citations([_make_claim("c1")])
        assert sources["c1"].found is False

    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve._crossref.search_paper")
    @patch("src.resolve.search_paper")
    def test_openalex_hit_skips_crossref(
        self, mock_oa: MagicMock, mock_cf: MagicMock, mock_retr: MagicMock
    ) -> None:
        mock_oa.return_value = ResolvedSource(True, "10.1/y", "T", "abs", 1.0)
        from src.resolve import resolve_citations

        resolve_citations([_make_claim("c1")])
        mock_cf.assert_not_called()

    @patch("src.resolve._crossref.check_retraction")
    @patch("src.resolve.search_paper")
    def test_retraction_check_called_when_doi_present(
        self, mock_oa: MagicMock, mock_retr: MagicMock
    ) -> None:
        mock_oa.return_value = ResolvedSource(True, "10.1/x", "T", "abs", 1.0)
        mock_retr.return_value = True
        from src.resolve import resolve_citations

        sources, _ = resolve_citations([_make_claim("c1")])
        mock_retr.assert_called_once_with("10.1/x", db_path=None)
        assert sources["c1"].retraction_status is True

    @patch("src.resolve._crossref.check_retraction")
    @patch("src.resolve._crossref.search_paper")
    @patch("src.resolve.search_paper")
    def test_retraction_check_skipped_when_no_doi(
        self, mock_oa: MagicMock, mock_cf: MagicMock, mock_retr: MagicMock
    ) -> None:
        mock_oa.return_value = ResolvedSource(False, None, None, None, None)
        mock_cf.return_value = ResolvedSource(False, None, None, None, None)
        from src.resolve import resolve_citations

        sources, _ = resolve_citations([_make_claim("c1")])
        mock_retr.assert_not_called()
        assert sources["c1"].retraction_status is False


class TestPubmedAbstractEnrichment:
    @patch("src.resolve._pubmed.fetch_abstract_by_doi")
    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve.search_paper")
    def test_enriches_when_doi_present_and_abstract_missing(
        self, mock_oa: MagicMock, mock_retr: MagicMock, mock_pubmed: MagicMock
    ) -> None:
        mock_oa.return_value = ResolvedSource(
            found=True, doi="10.1/x", title="T", abstract=None, similarity_score=0.9
        )
        mock_pubmed.return_value = "PubMed-fetched abstract about lactate kinetics."
        from src.resolve import resolve_citations

        sources, _ = resolve_citations([_make_claim("c1")])
        mock_pubmed.assert_called_once_with("10.1/x", db_path=None)
        assert sources["c1"].abstract == "PubMed-fetched abstract about lactate kinetics."

    @patch("src.resolve._pubmed.fetch_abstract_by_doi")
    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve.search_paper")
    def test_skipped_when_abstract_already_present(
        self, mock_oa: MagicMock, mock_retr: MagicMock, mock_pubmed: MagicMock
    ) -> None:
        mock_oa.return_value = ResolvedSource(
            found=True,
            doi="10.1/x",
            title="T",
            abstract="Already have one.",
            similarity_score=0.9,
        )
        from src.resolve import resolve_citations

        sources, _ = resolve_citations([_make_claim("c1")])
        mock_pubmed.assert_not_called()
        assert sources["c1"].abstract == "Already have one."

    @patch("src.resolve._pubmed.fetch_abstract_by_doi")
    @patch("src.resolve._crossref.search_paper")
    @patch("src.resolve.search_paper")
    def test_skipped_when_source_not_found(
        self, mock_oa: MagicMock, mock_cf: MagicMock, mock_pubmed: MagicMock
    ) -> None:
        mock_oa.return_value = ResolvedSource(False, None, None, None, None)
        mock_cf.return_value = ResolvedSource(False, None, None, None, None)
        from src.resolve import resolve_citations

        resolve_citations([_make_claim("c1")])
        mock_pubmed.assert_not_called()

    @patch("src.resolve._pubmed.fetch_abstract_by_doi")
    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve.search_paper")
    def test_pubmed_miss_leaves_source_unchanged(
        self, mock_oa: MagicMock, mock_retr: MagicMock, mock_pubmed: MagicMock
    ) -> None:
        mock_oa.return_value = ResolvedSource(
            found=True, doi="10.1/x", title="T", abstract=None, similarity_score=0.9
        )
        mock_pubmed.return_value = None
        from src.resolve import resolve_citations

        sources, _ = resolve_citations([_make_claim("c1")])
        mock_pubmed.assert_called_once()
        assert sources["c1"].found is True
        assert sources["c1"].abstract is None


class TestResolveCitations:
    @patch("src.resolve.search_paper")
    def test_happy_path(self, mock_search: MagicMock) -> None:
        mock_search.return_value = ResolvedSource(
            found=True,
            doi=None,
            title="Some Paper",
            abstract="An abstract.",
            similarity_score=1.0,
        )
        from src.resolve import resolve_citations

        claims = [_make_claim("c1"), _make_claim("c2")]
        sources, steps = resolve_citations(claims)

        assert "c1" in sources
        assert "c2" in sources
        assert sources["c1"].found is True
        assert len(steps) == 2

    @patch("src.resolve.search_paper")
    def test_returns_one_step_per_claim(self, mock_search: MagicMock) -> None:
        mock_search.return_value = ResolvedSource(
            found=True, doi=None, title="T", abstract="A", similarity_score=0.9
        )
        from src.resolve import resolve_citations

        claims = [_make_claim(f"c{i}") for i in range(4)]
        _sources, steps = resolve_citations(claims)
        assert len(steps) == 4

    @patch("src.resolve.search_paper")
    def test_step_fields(self, mock_search: MagicMock) -> None:
        mock_search.return_value = ResolvedSource(
            found=True, doi=None, title="T", abstract="A", similarity_score=0.8
        )
        from src.resolve import resolve_citations

        claims = [_make_claim("claim-x")]
        _, steps = resolve_citations(claims)
        step = steps[0]
        assert isinstance(step, ProvenanceStep)
        assert step.operation == "resolve"
        assert step.model_id is None
        assert step.tokens_in is None
        assert step.tokens_out is None
        assert step.cache_hit is None
        assert step.confidence == 0.8  # mirrors similarity_score

    def test_ec1_no_citation_no_http(self) -> None:
        """EC-1: Claim with no cited_authors returns found=False without HTTP call."""
        from src.resolve import resolve_citations

        claim = _make_claim("c1", cited_authors=[])
        with patch("src.resolve.search_paper") as mock_search:
            sources, _steps = resolve_citations([claim])
            mock_search.assert_not_called()

        assert sources["c1"].found is False

    def test_ec1_no_year_no_http(self) -> None:
        """EC-1: Claim with cited_year=None returns found=False without HTTP call."""
        from src.resolve import resolve_citations

        claim = _make_claim("c1", cited_year=None)
        with patch("src.resolve.search_paper") as mock_search:
            sources, _steps = resolve_citations([claim])
            mock_search.assert_not_called()

        assert sources["c1"].found is False

    @patch("src.resolve.search_paper")
    def test_ec4_year_off_by_one_accepted(self, mock_search: MagicMock) -> None:
        """EC-4: Paper indexed under year ±1 is accepted."""
        # Simulate year=2020 in response when claim cites 2019
        mock_search.return_value = ResolvedSource(
            found=True, doi=None, title="Paper", abstract="Abstract.", similarity_score=0.9
        )
        from src.resolve import resolve_citations

        claim = _make_claim("c1", cited_year=2019)
        sources, _ = resolve_citations([claim])
        assert sources["c1"].found is True

    @patch("src.resolve.search_paper")
    def test_empty_claims_list(self, mock_search: MagicMock) -> None:
        from src.resolve import resolve_citations

        sources, steps = resolve_citations([])
        assert sources == {}
        assert steps == []
        mock_search.assert_not_called()

    def test_ec6_429_mid_batch_httpx(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        """EC-6: 429 on first claim exhausts 3 retries → found=False; second claim succeeds."""
        from src.resolve import resolve_citations

        # 3x 429 for the first claim's query, then success for the second claim
        httpx_mock.add_response(status_code=429)
        httpx_mock.add_response(status_code=429)
        httpx_mock.add_response(status_code=429)
        httpx_mock.add_response(
            status_code=200,
            json={
                "results": [
                    {
                        "id": "https://openalex.org/W1",
                        "title": "Some Paper",
                        "abstract_inverted_index": {"An": [0], "abstract.": [1]},
                        "publication_year": 2020,
                        "doi": None,
                        "authorships": [],
                    }
                ]
            },
        )

        with (
            patch("src.clients.openalex.time.sleep"),
            patch(
                "src.resolve._crossref.search_paper",
                return_value=ResolvedSource(False, None, None, None, None),
            ),
        ):
            claims = [_make_claim("c0"), _make_claim("c1")]
            sources, steps = resolve_citations(claims, db_path=tmp_path / "cache.db")

        assert sources["c0"].found is False  # 429 exhausted
        assert sources["c1"].found is True  # success
        assert len(steps) == 2

    def test_result_has_entry_for_every_claim(self) -> None:
        """Every claim gets an entry in sources dict, even if not found."""
        from src.resolve import resolve_citations

        claims = [_make_claim("c1", cited_authors=[]), _make_claim("c2", cited_authors=[])]
        sources, _ = resolve_citations(claims)
        assert "c1" in sources
        assert "c2" in sources
