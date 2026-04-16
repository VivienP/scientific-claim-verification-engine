"""Unit tests for src/clients/openalex.py — mocked httpx calls."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

from pytest_httpx import HTTPXMock

from src.clients.openalex import _reconstruct_abstract, search_paper
from src.models import ResolvedSource

_OA_URL_PATTERN = re.compile(r"https://api\.openalex\.org/works")

_PAPER_RESPONSE = {
    "results": [
        {
            "id": "https://openalex.org/W2963403868",
            "title": "Scaling Laws for Neural Language Models",
            "abstract_inverted_index": {
                "We": [0],
                "study": [1],
                "scaling": [2],
                "laws": [3],
                "for": [4],
                "neural": [5],
                "language": [6],
                "model": [7],
                "performance.": [8],
            },
            "publication_year": 2022,
            "doi": "https://doi.org/10.1234/scaling",
            "authorships": [
                {"author": {"display_name": "Jordan Hoffmann"}},
                {"author": {"display_name": "Sebastian Borgeaud"}},
                {"author": {"display_name": "Arthur Mensch"}},
            ],
        }
    ]
}


class TestReconstructAbstract:
    def test_basic_reconstruction(self) -> None:
        inv_idx = {"Hello": [0], "world": [1]}
        assert _reconstruct_abstract(inv_idx) == "Hello world"

    def test_out_of_order_keys(self) -> None:
        inv_idx = {"world": [1], "Hello": [0]}
        assert _reconstruct_abstract(inv_idx) == "Hello world"

    def test_word_at_multiple_positions(self) -> None:
        inv_idx = {"the": [0, 3], "cat": [1], "sat": [2]}
        assert _reconstruct_abstract(inv_idx) == "the cat sat the"

    def test_none_returns_none(self) -> None:
        assert _reconstruct_abstract(None) is None

    def test_empty_dict_returns_none(self) -> None:
        assert _reconstruct_abstract({}) is None


class TestSearchPaperFound:
    def test_happy_path(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(url=_OA_URL_PATTERN, json=_PAPER_RESPONSE)
        result = search_paper("Hoffmann 2022 scaling laws", db_path=tmp_path / "cache.db")
        assert result.found is True
        assert result.title == "Scaling Laws for Neural Language Models"
        assert result.abstract == "We study scaling laws for neural language model performance."

    def test_doi_stripped(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(url=_OA_URL_PATTERN, json=_PAPER_RESPONSE)
        result = search_paper("Hoffmann 2022 scaling laws", db_path=tmp_path / "cache.db")
        assert result.doi == "10.1234/scaling"

    def test_returns_resolved_source(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(url=_OA_URL_PATTERN, json=_PAPER_RESPONSE)
        result = search_paper("Hoffmann 2022 scaling laws", db_path=tmp_path / "cache.db")
        assert isinstance(result, ResolvedSource)

    def test_similarity_score_exact_year(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(url=_OA_URL_PATTERN, json=_PAPER_RESPONSE)
        result = search_paper("Hoffmann 2022 scaling laws", db_path=tmp_path / "cache.db")
        assert result.similarity_score == 1.0

    def test_similarity_score_year_off_by_one(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        response = {
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "title": "Scaling Laws",
                    "abstract_inverted_index": {"We": [0], "study": [1], "scaling.": [2]},
                    "publication_year": 2023,  # off by 1 from 2022 query
                    "doi": None,
                    "authorships": [{"author": {"display_name": "Jordan Hoffmann"}}],
                }
            ]
        }
        httpx_mock.add_response(url=_OA_URL_PATTERN, json=response)
        result = search_paper("Hoffmann 2022 scaling laws", db_path=tmp_path / "cache.db")
        assert result.found is True
        assert result.similarity_score == 0.9

    def test_null_abstract_returns_found_true(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        """EC-2: Paper found but abstract_inverted_index is null."""
        response = {
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "title": "Some Paper",
                    "abstract_inverted_index": None,
                    "publication_year": 2022,
                    "doi": None,
                    "authorships": [{"author": {"display_name": "John Smith"}}],
                }
            ]
        }
        httpx_mock.add_response(url=_OA_URL_PATTERN, json=response)
        result = search_paper("Smith 2022 some paper", db_path=tmp_path / "cache.db")
        assert result.found is True
        assert result.abstract is None


class TestSearchPaperNotFound:
    def test_empty_results(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(url=_OA_URL_PATTERN, json={"results": []})
        result = search_paper("NoAuthor 9999 nothing", db_path=tmp_path / "cache.db")
        assert result.found is False
        assert result.title is None
        assert result.abstract is None
        assert result.similarity_score is None

    def test_network_error_returns_not_found(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        import httpx

        httpx_mock.add_exception(httpx.ConnectError("Connection refused"))
        result = search_paper("Smith 2020 test", db_path=tmp_path / "cache.db")
        assert result.found is False


class TestRetryOn429:
    def test_retries_on_429_succeeds_second_attempt(
        self, httpx_mock: HTTPXMock, tmp_path: Path
    ) -> None:
        httpx_mock.add_response(url=_OA_URL_PATTERN, status_code=429)
        httpx_mock.add_response(url=_OA_URL_PATTERN, json=_PAPER_RESPONSE)
        with patch("src.clients.openalex.time.sleep"):
            result = search_paper(
                "Hoffmann 2022 scaling laws",
                db_path=tmp_path / "cache.db",
            )
        assert result.found is True

    def test_max_retries_exceeded_returns_not_found(
        self, httpx_mock: HTTPXMock, tmp_path: Path
    ) -> None:
        """EC-6: 3 consecutive 429s return found=False."""
        httpx_mock.add_response(url=_OA_URL_PATTERN, status_code=429)
        httpx_mock.add_response(url=_OA_URL_PATTERN, status_code=429)
        httpx_mock.add_response(url=_OA_URL_PATTERN, status_code=429)
        with patch("src.clients.openalex.time.sleep"):
            result = search_paper("Smith 2020 test", db_path=tmp_path / "cache.db")
        assert result.found is False


class TestCaching:
    def test_cache_hit_skips_http(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        """Second call with same query must not fire HTTP."""
        httpx_mock.add_response(url=_OA_URL_PATTERN, json=_PAPER_RESPONSE)
        db_path = tmp_path / "cache.db"
        result1 = search_paper("Hoffmann 2022 scaling laws", db_path=db_path)
        # Second call — no more HTTP responses registered; would fail if HTTP called
        result2 = search_paper("Hoffmann 2022 scaling laws", db_path=db_path)
        assert result1.found is True
        assert result2.found is True
        assert result1.title == result2.title

    def test_no_year_in_query_returns_score_08(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        """Query without year → similarity_score=0.8."""
        httpx_mock.add_response(url=_OA_URL_PATTERN, json=_PAPER_RESPONSE)
        result = search_paper("scaling laws neural models", db_path=tmp_path / "cache.db")
        assert result.found is True
        assert result.similarity_score == 0.8
