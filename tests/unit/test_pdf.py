"""Unit tests for src/clients/pdf.py — all HTTP mocked via pytest-httpx.

Lane B refactor: ``download_and_extract`` returns ``PdfFetchOutcome`` instead
of ``str | None`` so callers can attribute the failure precisely. Tests
assert on ``outcome.text`` AND ``outcome.failure_reason``.
"""

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
        outcome = download_and_extract("https://example.org/paper.pdf", db_path=tmp_path / "c.db")
        assert outcome.text is not None
        assert outcome.failure_reason == "ok"
        assert "sample paper" in outcome.text

    def test_non_pdf_content_type(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(
            content=b"<html>not a pdf</html>",
            headers={"Content-Type": "text/html"},
        )
        outcome = download_and_extract("https://x/y", db_path=tmp_path / "c.db")
        assert outcome.text is None
        assert outcome.failure_reason == "not_a_pdf"
        assert outcome.content_type is not None
        assert "html" in outcome.content_type

    def test_short_text_returns_too_short(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        pdf_bytes = _make_pdf_bytes("hi")
        httpx_mock.add_response(
            content=pdf_bytes,
            headers={"Content-Type": "application/pdf"},
        )
        outcome = download_and_extract("https://x/y.pdf", db_path=tmp_path / "c.db")
        assert outcome.text is None
        assert outcome.failure_reason == "too_short"

    def test_network_error(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        import httpx as _httpx

        httpx_mock.add_exception(_httpx.ConnectError("refused"))
        outcome = download_and_extract("https://x/y.pdf", db_path=tmp_path / "c.db")
        assert outcome.text is None
        # ConnectError is a RequestError, mapped to http_error.
        assert outcome.failure_reason == "http_error"

    def test_timeout_distinguished_from_other_request_errors(
        self, httpx_mock: HTTPXMock, tmp_path: Path
    ) -> None:
        import httpx as _httpx

        httpx_mock.add_exception(_httpx.ReadTimeout("slow"))
        outcome = download_and_extract("https://x/y.pdf", db_path=tmp_path / "c.db")
        assert outcome.text is None
        assert outcome.failure_reason == "timeout"

    def test_corrupt_pdf_returns_extraction_failed(
        self, httpx_mock: HTTPXMock, tmp_path: Path
    ) -> None:
        httpx_mock.add_response(
            content=b"%PDF-1.4\n garbage \n%%EOF",
            headers={"Content-Type": "application/pdf"},
        )
        outcome = download_and_extract("https://x/y.pdf", db_path=tmp_path / "c.db")
        assert outcome.text is None
        assert outcome.failure_reason == "extraction_failed"

    def test_http_error_carries_status_code(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(status_code=403)
        outcome = download_and_extract("https://x/y.pdf", db_path=tmp_path / "c.db")
        assert outcome.text is None
        assert outcome.failure_reason == "http_error"
        assert outcome.http_status == 403

    def test_cache_hit_returns_ok_outcome(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        db = tmp_path / "c.db"
        pdf_bytes = _make_pdf_bytes(_LONG_TEXT)
        httpx_mock.add_response(
            content=pdf_bytes,
            headers={"Content-Type": "application/pdf"},
        )
        first = download_and_extract("https://x/y.pdf", db_path=db)
        second = download_and_extract("https://x/y.pdf", db_path=db)
        assert first.text is not None
        assert second.text is not None
        assert first.text == second.text
        assert second.failure_reason == "ok"
        assert len(httpx_mock.get_requests()) == 1
