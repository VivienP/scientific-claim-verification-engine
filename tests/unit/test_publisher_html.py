"""Unit tests for src/clients/publisher_html.py — all HTTP mocked via pytest-httpx.

publisher_html is the Track D1 fix for the user-caught NEJM Goodwin 2022 case:
when a paper's OA PDF endpoint is paywalled but its `/doi/full/` HTML page is
publicly readable, this client fetches and extracts the article body so the
verifier gets fulltext-grade evidence instead of falling back to abstract.

The end-to-end integration test that hits the real NEJM URL lives in
tests/integration/test_fetch_nejm_goodwin.py — these unit tests cover the
extraction logic, cache behavior, and failure paths with mocked HTTP.
"""

from __future__ import annotations

from pathlib import Path

from pytest_httpx import HTTPXMock

from src.clients.publisher_html import fetch_via_doi

_NEJM_DOI = "10.1056/NEJMoa2206443"
_NEJM_URL = f"https://www.nejm.org/doi/full/{_NEJM_DOI}"

# Long enough body (~4000 chars after whitespace collapse) to clear _MIN_TEXT_LENGTH.
# Repeats the Goodwin Results-section figure so the test is grounded in the
# actual user-caught case.
_RESULTS_PARAGRAPH = (
    "The incidence of sustained response at week 12 was 20% in the 25-mg group, "
    "5% in the 10-mg group, and 10% in the 1-mg group. "
    "The mean MADRS total score at baseline was 32 or 33 in each group. "
    "Least-squares mean changes from baseline to week 3 in the score were -12.0 for 25 mg, "
    "-7.9 for 10 mg, and -5.4 for 1 mg. "
)
_LONG_BODY = _RESULTS_PARAGRAPH * 12  # ~4400 chars


def _nejm_html(body: str = _LONG_BODY) -> str:
    """Synthetic NEJM-like HTML page. Mirrors the rough structure: article > section.

    Includes a <script> block (must be skipped) and a <nav> (must be skipped).
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <title>Single-Dose Psilocybin for a Treatment-Resistant Episode of Major Depression | NEJM</title>
  <script>window.dataLayer = [];</script>
  <style>body {{ font-family: serif; }}</style>
</head>
<body>
  <nav>Home | Articles | Browse</nav>
  <header>NEJM masthead</header>
  <article>
    <h1>Single-Dose Psilocybin for a Treatment-Resistant Episode of Major Depression</h1>
    <section>
      <h2>Results</h2>
      <p>{body}</p>
    </section>
  </article>
  <footer>Copyright 2022 NEJM</footer>
  <script>tracking.init();</script>
</body>
</html>"""


class TestKnownPublisher:
    """DOI prefix matches a known publisher; HTML fetch + extract succeeds."""

    def test_returns_extracted_text_on_happy_path(
        self,
        httpx_mock: HTTPXMock,
        tmp_path: Path,
    ) -> None:
        httpx_mock.add_response(
            url=_NEJM_URL,
            html=_nejm_html(),
            headers={"Content-Type": "text/html; charset=UTF-8"},
        )
        text = fetch_via_doi(_NEJM_DOI, db_path=tmp_path / "pub.db")

        assert text is not None
        # The Goodwin marquee figure must survive extraction:
        assert "sustained response at week 12 was 20%" in text
        # Script and style content must be skipped:
        assert "window.dataLayer" not in text
        assert "font-family" not in text
        # Header/nav/footer must be skipped:
        assert "Home | Articles" not in text
        assert "NEJM masthead" not in text
        assert "Copyright 2022" not in text

    def test_caches_extracted_text(
        self,
        httpx_mock: HTTPXMock,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "pub.db"
        httpx_mock.add_response(
            url=_NEJM_URL,
            html=_nejm_html(),
            headers={"Content-Type": "text/html"},
        )
        first = fetch_via_doi(_NEJM_DOI, db_path=db_path)
        second = fetch_via_doi(_NEJM_DOI, db_path=db_path)

        assert first is not None
        assert second == first
        # If the cache miss-fires, pytest-httpx will raise on an unexpected
        # second outbound request (it only registered one response).


class TestUnknownPublisher:
    """DOI prefix doesn't match — return None without HTTP, don't cache."""

    def test_returns_none_for_unknown_doi_prefix(self, tmp_path: Path) -> None:
        # No httpx_mock fixture used: any outbound HTTP would fail the test.
        text = fetch_via_doi("10.9999/unknown.2024.001", db_path=tmp_path / "pub.db")
        assert text is None


class TestFailurePaths:
    """HTTP errors, non-HTML responses, and short / paywalled content."""

    def test_http_404_returns_none_and_caches_null(
        self,
        httpx_mock: HTTPXMock,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "pub.db"
        httpx_mock.add_response(url=_NEJM_URL, status_code=404)
        first = fetch_via_doi(_NEJM_DOI, db_path=db_path)
        # Second call must hit the cache (null sentinel) — no second HTTP request.
        second = fetch_via_doi(_NEJM_DOI, db_path=db_path)

        assert first is None
        assert second is None

    def test_non_html_content_type_returns_none(
        self,
        httpx_mock: HTTPXMock,
        tmp_path: Path,
    ) -> None:
        httpx_mock.add_response(
            url=_NEJM_URL,
            content=b"\x25PDF-1.7 fake pdf bytes",
            headers={"Content-Type": "application/pdf"},
        )
        text = fetch_via_doi(_NEJM_DOI, db_path=tmp_path / "pub.db")
        assert text is None

    def test_text_too_short_returns_none_and_caches_null(
        self,
        httpx_mock: HTTPXMock,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "pub.db"
        short_html = "<html><body><article><p>Stub article body.</p></article></body></html>"
        httpx_mock.add_response(
            url=_NEJM_URL,
            html=short_html,
            headers={"Content-Type": "text/html"},
        )
        first = fetch_via_doi(_NEJM_DOI, db_path=db_path)
        # Second call must hit the cache — no second HTTP request.
        second = fetch_via_doi(_NEJM_DOI, db_path=db_path)

        assert first is None
        assert second is None

    def test_paywall_markers_return_none_and_cache_null(
        self,
        httpx_mock: HTTPXMock,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "pub.db"
        paywall_body = (
            "To continue reading this article, please sign in to read. "
            "Subscribe to read the full content. " * 80
        )
        paywall_html = f"<html><body><article><p>{paywall_body}</p></article></body></html>"
        httpx_mock.add_response(
            url=_NEJM_URL,
            html=paywall_html,
            headers={"Content-Type": "text/html"},
        )
        first = fetch_via_doi(_NEJM_DOI, db_path=db_path)
        second = fetch_via_doi(_NEJM_DOI, db_path=db_path)

        assert first is None
        assert second is None
