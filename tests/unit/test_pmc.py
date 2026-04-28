"""Unit tests for src/clients/pmc.py — all HTTP mocked via pytest-httpx."""

from __future__ import annotations

from pathlib import Path

from pytest_httpx import HTTPXMock

from src.clients.pmc import fetch_fulltext

_PMC_XML = """<?xml version="1.0" ?>
<pmc-articleset>
  <article>
    <front>
      <article-meta>
        <title-group><article-title>Sample Title</article-title></title-group>
        <abstract><p>This is the abstract of the paper.</p></abstract>
      </article-meta>
    </front>
    <body>
      <sec>
        <title>Introduction</title>
        <p>Background and motivation for the study.</p>
      </sec>
      <sec>
        <title>Results</title>
        <p>We observed a significant effect with p &lt; 0.001.</p>
      </sec>
    </body>
  </article>
</pmc-articleset>
"""


class TestFetchFulltext:
    def test_happy_path(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(text=_PMC_XML)
        text = fetch_fulltext("PMC1234567", db_path=tmp_path / "c.db")
        assert text is not None
        assert "Sample Title" in text
        assert "Introduction" in text
        assert "p < 0.001" in text

    def test_pmc_prefix_stripped_in_url(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(text=_PMC_XML)
        fetch_fulltext("PMC1234567", db_path=tmp_path / "c.db")
        request = httpx_mock.get_request()
        assert request is not None
        assert "id=1234567" in str(request.url)
        assert "PMC1234567" not in str(request.url)

    def test_404_returns_none(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(status_code=404)
        assert fetch_fulltext("PMC9999999", db_path=tmp_path / "c.db") is None

    def test_network_error(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        import httpx as _httpx

        httpx_mock.add_exception(_httpx.ConnectError("refused"))
        assert fetch_fulltext("PMC1234567", db_path=tmp_path / "c.db") is None

    def test_malformed_xml_returns_none(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(text="<not valid xml")
        assert fetch_fulltext("PMC1234567", db_path=tmp_path / "c.db") is None

    def test_cache_hit_skips_http(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        db = tmp_path / "c.db"
        httpx_mock.add_response(text=_PMC_XML)
        t1 = fetch_fulltext("PMC1234567", db_path=db)
        t2 = fetch_fulltext("PMC1234567", db_path=db)
        assert t1 == t2
        assert len(httpx_mock.get_requests()) == 1
