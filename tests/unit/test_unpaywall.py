"""Unit tests for src/clients/unpaywall.py — all HTTP mocked via pytest-httpx."""

from __future__ import annotations

from pathlib import Path

from pytest_httpx import HTTPXMock

from src.clients.unpaywall import get_oa_url

_DOI = "10.1234/test.2023"


class TestGetOaUrl:
    def test_url_for_pdf_preferred(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(
            json={
                "best_oa_location": {
                    "url_for_pdf": "https://example.org/paper.pdf",
                    "url": "https://example.org/landing",
                }
            }
        )
        url = get_oa_url(_DOI, db_path=tmp_path / "c.db")
        assert url == "https://example.org/paper.pdf"

    def test_falls_back_to_url(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(
            json={
                "best_oa_location": {
                    "url_for_pdf": None,
                    "url": "https://example.org/html-page",
                }
            }
        )
        url = get_oa_url(_DOI, db_path=tmp_path / "c.db")
        assert url == "https://example.org/html-page"

    def test_no_oa_location(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(json={"best_oa_location": None})
        assert get_oa_url(_DOI, db_path=tmp_path / "c.db") is None

    def test_missing_key(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(json={})
        assert get_oa_url(_DOI, db_path=tmp_path / "c.db") is None

    def test_network_error(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        import httpx as _httpx

        httpx_mock.add_exception(_httpx.ConnectError("refused"))
        assert get_oa_url(_DOI, db_path=tmp_path / "c.db") is None

    def test_404(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(status_code=404)
        assert get_oa_url(_DOI, db_path=tmp_path / "c.db") is None

    def test_cache_hit_skips_http(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        db = tmp_path / "c.db"
        httpx_mock.add_response(
            json={"best_oa_location": {"url_for_pdf": "https://x/y.pdf", "url": "https://x/"}}
        )
        u1 = get_oa_url(_DOI, db_path=db)
        u2 = get_oa_url(_DOI, db_path=db)
        assert u1 == u2 == "https://x/y.pdf"
        assert len(httpx_mock.get_requests()) == 1
