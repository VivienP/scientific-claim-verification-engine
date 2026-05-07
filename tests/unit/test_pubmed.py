"""Unit tests for src/clients/pubmed.py — all HTTP mocked via pytest-httpx."""

from __future__ import annotations

from pathlib import Path

from pytest_httpx import HTTPXMock

from src.clients.pubmed import fetch_abstract, fetch_abstract_by_doi, find_pmid_by_doi

_ESEARCH_OK = '{"esearchresult": {"idlist": ["12345678"]}}'
_ESEARCH_EMPTY = '{"esearchresult": {"idlist": []}}'

_EFETCH_TEXT = """1. Sample Journal. 2007 Jul;1(4):558-569.

Blood lactate measurements and analysis during exercise: a guide for clinicians.

Goodwin ML, Harris JE, Hernandez A, Gladden LB.

Author information:
(1)Department of Kinesiology, Auburn University, Auburn, AL.

The whole-blood-to-plasma lactate ratio is expected to vary from 63% (at 55%
hematocrit) to 81% (at 25% hematocrit), depending on plasma water content
and erythrocyte mass. This relationship has direct implications for clinical
sample interpretation.

Copyright © 2007 Diabetes Technology Society.

DOI: 10.1177/193229680700100414
PMID: 12345678
"""


class TestFindPmidByDoi:
    def test_happy_path(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(text=_ESEARCH_OK)
        pmid = find_pmid_by_doi("10.1177/193229680700100414", db_path=tmp_path / "c.db")
        assert pmid == "12345678"

    def test_empty_result(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(text=_ESEARCH_EMPTY)
        assert find_pmid_by_doi("10.1234/missing", db_path=tmp_path / "c.db") is None

    def test_blank_doi_short_circuits(self, tmp_path: Path) -> None:
        # No httpx_mock setup — the function must not call out.
        assert find_pmid_by_doi("", db_path=tmp_path / "c.db") is None

    def test_http_error(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(status_code=500)
        assert find_pmid_by_doi("10.1234/abc", db_path=tmp_path / "c.db") is None

    def test_network_error(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        import httpx as _httpx

        httpx_mock.add_exception(_httpx.ConnectError("refused"))
        assert find_pmid_by_doi("10.1234/abc", db_path=tmp_path / "c.db") is None

    def test_cache_hit_skips_http(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        db = tmp_path / "c.db"
        httpx_mock.add_response(text=_ESEARCH_OK)
        a = find_pmid_by_doi("10.1177/193229680700100414", db_path=db)
        b = find_pmid_by_doi("10.1177/193229680700100414", db_path=db)
        assert a == b == "12345678"
        assert len(httpx_mock.get_requests()) == 1

    def test_negative_cached(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        db = tmp_path / "c.db"
        httpx_mock.add_response(text=_ESEARCH_EMPTY)
        assert find_pmid_by_doi("10.1234/none", db_path=db) is None
        # Second call should be served from cache, no second request.
        assert find_pmid_by_doi("10.1234/none", db_path=db) is None
        assert len(httpx_mock.get_requests()) == 1


class TestFetchAbstract:
    def test_extracts_body_and_strips_metadata(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(text=_EFETCH_TEXT)
        abstract = fetch_abstract("12345678", db_path=tmp_path / "c.db")
        assert abstract is not None
        assert "63%" in abstract
        assert "81%" in abstract
        assert "PMID:" not in abstract
        assert "DOI:" not in abstract
        assert "Author information" not in abstract
        assert "Copyright" not in abstract

    def test_blank_pmid_short_circuits(self, tmp_path: Path) -> None:
        assert fetch_abstract("", db_path=tmp_path / "c.db") is None

    def test_too_short_returns_none(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(text="1. Brief.\n\nShort.\n\nPMID: 1\n")
        assert fetch_abstract("1", db_path=tmp_path / "c.db") is None

    def test_http_error(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(status_code=429)
        assert fetch_abstract("99999", db_path=tmp_path / "c.db") is None

    def test_cache_hit_skips_http(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        db = tmp_path / "c.db"
        httpx_mock.add_response(text=_EFETCH_TEXT)
        first = fetch_abstract("12345678", db_path=db)
        second = fetch_abstract("12345678", db_path=db)
        assert first == second
        assert len(httpx_mock.get_requests()) == 1


class TestFetchAbstractByDoi:
    def test_happy_path_two_calls(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        # First call hits esearch (DOI -> PMID), second call hits efetch (PMID -> abstract).
        # pytest-httpx serves responses in registration order.
        httpx_mock.add_response(text=_ESEARCH_OK)
        httpx_mock.add_response(text=_EFETCH_TEXT)
        abstract = fetch_abstract_by_doi("10.1177/193229680700100414", db_path=tmp_path / "c.db")
        assert abstract is not None
        assert "63%" in abstract
        # Confirm both endpoints were hit.
        urls = [str(r.url) for r in httpx_mock.get_requests()]
        assert any("esearch" in u for u in urls)
        assert any("efetch" in u for u in urls)

    def test_pmid_not_found(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(text=_ESEARCH_EMPTY)
        # No efetch mock — should not be called when esearch returns empty.
        assert fetch_abstract_by_doi("10.1234/missing", db_path=tmp_path / "c.db") is None
        urls = [str(r.url) for r in httpx_mock.get_requests()]
        assert all("efetch" not in u for u in urls)
