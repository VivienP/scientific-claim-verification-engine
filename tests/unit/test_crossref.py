"""Unit tests for src/clients/crossref.py — all HTTP mocked via pytest-httpx."""

from __future__ import annotations

from pathlib import Path

from pytest_httpx import HTTPXMock

from src.clients.crossref import check_retraction, search_paper

_WORKS_URL = "https://api.crossref.org/works"
_DOI = "10.1234/test.2023"

_GOOD_RESPONSE = {
    "message": {
        "items": [
            {
                "DOI": _DOI,
                "title": ["A Great Paper on Things"],
            }
        ]
    }
}

_RETRACTION_RESPONSE = {
    "message": {
        "DOI": _DOI,
        "title": ["A Great Paper on Things"],
        "update-to": [{"type": "retraction", "DOI": "10.1234/retraction.2024"}],
    }
}

_CORRECTION_RESPONSE = {
    "message": {
        "DOI": _DOI,
        "title": ["A Great Paper on Things"],
        "update-to": [{"type": "correction", "DOI": "10.1234/correction.2024"}],
    }
}

_NO_UPDATE_RESPONSE = {
    "message": {
        "DOI": _DOI,
        "title": ["A Great Paper on Things"],
    }
}


class TestSearchPaper:
    def test_happy_path(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(json=_GOOD_RESPONSE)
        result = search_paper("Smith 2023 protein folding", db_path=tmp_path / "c.db")
        assert result.found is True
        assert result.doi == _DOI
        assert result.title == "A Great Paper on Things"
        assert result.abstract is None

    def test_empty_items(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(json={"message": {"items": []}})
        result = search_paper("nobody 1900 nothing", db_path=tmp_path / "c.db")
        assert result.found is False

    def test_missing_items_key(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(json={"message": {}})
        result = search_paper("nobody 1900 nothing", db_path=tmp_path / "c.db")
        assert result.found is False

    def test_network_error(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        import httpx as _httpx

        httpx_mock.add_exception(_httpx.ConnectError("refused"))
        result = search_paper("query", db_path=tmp_path / "c.db")
        assert result.found is False

    def test_429_retry_then_success(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(status_code=429)
        httpx_mock.add_response(json=_GOOD_RESPONSE)
        result = search_paper("Smith 2023", db_path=tmp_path / "c.db")
        assert result.found is True

    def test_cache_hit_skips_http(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        db = tmp_path / "c.db"
        httpx_mock.add_response(json=_GOOD_RESPONSE)
        r1 = search_paper("Smith 2023", db_path=db)
        r2 = search_paper("Smith 2023", db_path=db)
        assert r1.doi == r2.doi
        # Only one HTTP call was registered
        assert len(httpx_mock.get_requests()) == 1

    def test_doi_stripped(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        resp = {"message": {"items": [{"DOI": "https://doi.org/10.5678/foo", "title": ["T"]}]}}
        httpx_mock.add_response(json=resp)
        result = search_paper("query", db_path=tmp_path / "c.db")
        assert result.doi == "10.5678/foo"


class TestCheckRetraction:
    def _url(self) -> str:
        import urllib.parse

        return f"{_WORKS_URL}/{urllib.parse.quote(_DOI, safe='')}"

    def test_retraction_detected(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(json=_RETRACTION_RESPONSE)
        assert check_retraction(_DOI, db_path=tmp_path / "c.db") is True

    def test_correction_not_retraction(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(json=_CORRECTION_RESPONSE)
        assert check_retraction(_DOI, db_path=tmp_path / "c.db") is False

    def test_no_update_to(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(json=_NO_UPDATE_RESPONSE)
        assert check_retraction(_DOI, db_path=tmp_path / "c.db") is False

    def test_network_error_returns_false(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        import httpx as _httpx

        httpx_mock.add_exception(_httpx.ConnectError("refused"))
        assert check_retraction(_DOI, db_path=tmp_path / "c.db") is False

    def test_cache_hit_skips_http(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        db = tmp_path / "c.db"
        httpx_mock.add_response(json=_RETRACTION_RESPONSE)
        r1 = check_retraction(_DOI, db_path=db)
        r2 = check_retraction(_DOI, db_path=db)
        assert r1 is True
        assert r2 is True
        assert len(httpx_mock.get_requests()) == 1
