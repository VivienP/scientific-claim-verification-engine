"""Unit tests for src/clients/pdf.py — all HTTP mocked via pytest-httpx."""

from __future__ import annotations

from pathlib import Path

import fitz
from pytest_httpx import HTTPXMock

from src.clients.pdf import download_and_extract


def _make_pdf_bytes(text: str) -> bytes:
    """Generate a valid in-memory PDF containing the given text."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    pdf_bytes: bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


_LONG_TEXT = "This is the body of a sample paper. " * 20  # > 100 chars after extraction


class TestDownloadAndExtract:
    def test_happy_path(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        pdf_bytes = _make_pdf_bytes(_LONG_TEXT)
        httpx_mock.add_response(
            content=pdf_bytes,
            headers={"Content-Type": "application/pdf"},
        )
        text = download_and_extract("https://example.org/paper.pdf", db_path=tmp_path / "c.db")
        assert text is not None
        assert "sample paper" in text

    def test_non_pdf_content_type(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(
            content=b"<html>not a pdf</html>",
            headers={"Content-Type": "text/html"},
        )
        assert download_and_extract("https://x/y", db_path=tmp_path / "c.db") is None

    def test_short_text_returns_none(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        pdf_bytes = _make_pdf_bytes("hi")
        httpx_mock.add_response(
            content=pdf_bytes,
            headers={"Content-Type": "application/pdf"},
        )
        assert download_and_extract("https://x/y.pdf", db_path=tmp_path / "c.db") is None

    def test_network_error(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        import httpx as _httpx

        httpx_mock.add_exception(_httpx.ConnectError("refused"))
        assert download_and_extract("https://x/y.pdf", db_path=tmp_path / "c.db") is None

    def test_corrupt_pdf_returns_none(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(
            content=b"%PDF-1.4\n garbage \n%%EOF",
            headers={"Content-Type": "application/pdf"},
        )
        assert download_and_extract("https://x/y.pdf", db_path=tmp_path / "c.db") is None

    def test_cache_hit_skips_http(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        db = tmp_path / "c.db"
        pdf_bytes = _make_pdf_bytes(_LONG_TEXT)
        httpx_mock.add_response(
            content=pdf_bytes,
            headers={"Content-Type": "application/pdf"},
        )
        t1 = download_and_extract("https://x/y.pdf", db_path=db)
        t2 = download_and_extract("https://x/y.pdf", db_path=db)
        assert t1 == t2
        assert len(httpx_mock.get_requests()) == 1
