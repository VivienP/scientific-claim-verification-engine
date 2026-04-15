"""Unit tests for src/clients/semantic_scholar.py — mocked httpx calls."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

from pytest_httpx import HTTPXMock

from src.clients.semantic_scholar import search_paper
from src.models import ResolvedSource

_SS_URL_PATTERN = re.compile(r"https://api\.semanticscholar\.org/graph/v1/paper/search")

_PAPER_RESPONSE = {
    "data": [
        {
            "paperId": "abc123",
            "title": "Scaling Laws for Neural Language Models",
            "abstract": "We study scaling laws for neural language model performance.",
            "year": 2022,
            "authors": [
                {"authorId": "1", "name": "Hoffmann, Jordan"},
                {"authorId": "2", "name": "Borgeaud, Sebastian"},
                {"authorId": "3", "name": "Mensch, Arthur"},
            ],
        }
    ]
}


class TestSearchPaperFound:
    def test_happy_path(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(url=_SS_URL_PATTERN, json=_PAPER_RESPONSE)
        result = search_paper("Hoffmann 2022 scaling laws", db_path=tmp_path / "cache.db")
        assert result.found is True
        assert result.title == "Scaling Laws for Neural Language Models"
        assert result.abstract == "We study scaling laws for neural language model performance."

    def test_returns_resolved_source(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(url=_SS_URL_PATTERN, json=_PAPER_RESPONSE)
        result = search_paper("Hoffmann 2022 scaling laws", db_path=tmp_path / "cache.db")
        assert isinstance(result, ResolvedSource)

    def test_similarity_score_exact_year(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(url=_SS_URL_PATTERN, json=_PAPER_RESPONSE)
        result = search_paper("Hoffmann 2022 scaling laws", db_path=tmp_path / "cache.db")
        assert result.similarity_score == 1.0

    def test_similarity_score_year_off_by_one(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        response = {
            "data": [
                {
                    "paperId": "abc123",
                    "title": "Scaling Laws",
                    "abstract": "We study scaling.",
                    "year": 2023,  # off by 1 from 2022 query
                    "authors": [{"authorId": "1", "name": "Hoffmann, Jordan"}],
                }
            ]
        }
        httpx_mock.add_response(url=_SS_URL_PATTERN, json=response)
        result = search_paper("Hoffmann 2022 scaling laws", db_path=tmp_path / "cache.db")
        assert result.found is True
        assert result.similarity_score == 0.9

    def test_null_abstract_returns_found_true(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        """EC-2: Paper found but abstract is null."""
        response = {
            "data": [
                {
                    "paperId": "abc123",
                    "title": "Some Paper",
                    "abstract": None,
                    "year": 2022,
                    "authors": [{"authorId": "1", "name": "Smith, John"}],
                }
            ]
        }
        httpx_mock.add_response(url=_SS_URL_PATTERN, json=response)
        result = search_paper("Smith 2022 some paper", db_path=tmp_path / "cache.db")
        assert result.found is True
        assert result.abstract is None


class TestSearchPaperNotFound:
    def test_empty_results(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(url=_SS_URL_PATTERN, json={"data": []})
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
        # First call: 429, second call: success
        httpx_mock.add_response(url=_SS_URL_PATTERN, status_code=429)
        httpx_mock.add_response(url=_SS_URL_PATTERN, json=_PAPER_RESPONSE)
        with patch("src.clients.semantic_scholar.time.sleep"):
            result = search_paper(
                "Hoffmann 2022 scaling laws",
                db_path=tmp_path / "cache.db",
            )
        assert result.found is True

    def test_max_retries_exceeded_returns_not_found(
        self, httpx_mock: HTTPXMock, tmp_path: Path
    ) -> None:
        """EC-6: 3 consecutive 429s return found=False."""
        httpx_mock.add_response(url=_SS_URL_PATTERN, status_code=429)
        httpx_mock.add_response(url=_SS_URL_PATTERN, status_code=429)
        httpx_mock.add_response(url=_SS_URL_PATTERN, status_code=429)
        with patch("src.clients.semantic_scholar.time.sleep"):
            result = search_paper("Smith 2020 test", db_path=tmp_path / "cache.db")
        assert result.found is False


class TestCaching:
    def test_cache_hit_skips_http(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        """Second call with same query must not fire HTTP."""
        httpx_mock.add_response(url=_SS_URL_PATTERN, json=_PAPER_RESPONSE)
        db_path = tmp_path / "cache.db"
        result1 = search_paper("Hoffmann 2022 scaling laws", db_path=db_path)
        # Second call — no more HTTP responses registered; would fail if HTTP called
        result2 = search_paper("Hoffmann 2022 scaling laws", db_path=db_path)
        assert result1.found is True
        assert result2.found is True
        assert result1.title == result2.title

    def test_no_year_in_query_returns_score_08(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        """Query without year → similarity_score=0.8."""
        httpx_mock.add_response(url=_SS_URL_PATTERN, json=_PAPER_RESPONSE)
        result = search_paper("scaling laws neural models", db_path=tmp_path / "cache.db")
        assert result.found is True
        assert result.similarity_score == 0.8
