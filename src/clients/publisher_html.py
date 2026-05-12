"""HTML fulltext fetcher via DOI redirect + per-publisher extraction.

Targets publishers that serve their article HTML publicly even when their PDF
endpoint is paywalled. Used as a fulltext source in `src/fetch_fulltext.py`
for cases like NEJM, whose Semantic Scholar `oa_url` is a paywalled
`/doi/pdf/...?articleTools=true` while the full HTML is publicly readable at
`https://www.nejm.org/doi/full/<doi>`.

**Real-world limit:** publishers with active bot protection (Cloudflare
interstitial, JS challenges) return 403 to any straightforward HTTP client
regardless of User-Agent. NEJM in particular serves a "Just a moment..."
Cloudflare page; its `robots.txt` disallows AI/scraper bots (GPTBot,
ChatGPT-User, CCBot, Google-Extended, PerplexityBot, SemanticScholarBot).
This module handles 403 gracefully (returns None and caches the failure) so
the fetch chain falls through to the abstract fallback, and
`safe_verification_result` correctly emits `unverifiable` for claims that
need Results-section data. Bypassing publisher bot protection would require
a headless browser (Playwright) or paid API auth — out of scope for Phase 1.

This module is therefore most useful for publishers that don't gate
HTML access (preprint servers, many OA journals, some legacy
publishers). The NEJM prefix is kept in `_PUBLISHER_URLS` so the route
is wired and will start working if NEJM ever loosens its bot
protection, if we add Playwright/API support, or if a paper happens to
have a non-Cloudflare entry point.

Approach: pure-stdlib HTML parsing (no bs4 / readability dependency).
Per-publisher URL templates are keyed by DOI prefix. Conservative
text-extraction: collect text from semantic content tags (p, section,
article, h1-h4, li), skip layout/script tags. Reject responses that
look paywalled or are too short.

Caching: SQLite WAL via `_cache.py`, 30-day TTL, negative-result
sentinel for failed fetches so we don't hammer publishers.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

import httpx
import structlog

from src.clients._cache import get, put
from src.clients._common import (
    CACHE_TTL_DEFAULT_SECONDS as _CACHE_TTL_SECONDS,
)
from src.clients._common import (
    make_cache_key,
)

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_HEADERS = {"User-Agent": "ScientificClaimVerifier/0.1"}
_NULL_SENTINEL = "__NULL__"
_MIN_TEXT_LENGTH = 3000
_PAYWALL_MARKERS: tuple[str, ...] = (
    "subscribe to read",
    "sign in to read",
    "purchase access",
    "institutional access",
    "to continue reading",
    "purchase this article",
)

# DOI prefix → publisher article URL template. Add new publishers as
# their HTML extraction is verified against the integration test suite.
# NEJM (10.1056) was the user-caught case; others land here as they
# surface in dogfood runs.
_PUBLISHER_URLS: dict[str, str] = {
    "10.1056/": "https://www.nejm.org/doi/full/{doi}",
}

_CAPTURE_TAGS: frozenset[str] = frozenset(
    {"p", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "dd"}
)
_SKIP_TAGS: frozenset[str] = frozenset(
    {"script", "style", "nav", "header", "footer", "aside", "noscript", "form", "button"}
)


class _BodyTextExtractor(HTMLParser):
    """Pure-stdlib extractor that collects text from semantic content tags.

    Skips script/style/nav/header/footer/aside (layout chrome).
    Inserts a single space between captured blocks so that adjacent
    sentences don't merge. Newlines/whitespace are normalized later by
    the caller.
    """

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth: int = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _CAPTURE_TAGS:
            # Insert a separator at the end of each captured block so
            # adjacent sentences don't run together when whitespace is
            # collapsed.
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self.parts.append(data)


def _cache_key(doi: str) -> str:
    return make_cache_key("publisher_html_v1", doi)


def _resolve_publisher_url(doi: str) -> str | None:
    """Return the publisher article URL for a known DOI prefix, or None."""
    doi_lower = doi.lower()
    for prefix, template in _PUBLISHER_URLS.items():
        if doi_lower.startswith(prefix.lower()):
            return template.format(doi=doi)
    return None


def _looks_paywalled(text: str) -> bool:
    """True if the first 5k chars contain a paywall marker."""
    sample = text[:5000].lower()
    return any(marker in sample for marker in _PAYWALL_MARKERS)


def _normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def fetch_via_doi(
    doi: str,
    *,
    db_path: Path | None = None,
    timeout: float = 30.0,
) -> str | None:
    """Fetch full text from a known publisher's HTML article page.

    Returns:
        Extracted plain text on success.
        None when:
          - the DOI prefix doesn't match a known publisher
          - the HTTP request fails (network, 4xx, 5xx)
          - the response is not HTML
          - the parsed text is below _MIN_TEXT_LENGTH chars
          - the parsed text contains paywall markers in the first 5kb

    Cache key: sha256("publisher_html_v1:{doi}"), 30-day TTL.
    Failed fetches are cached with _NULL_SENTINEL so we don't re-hammer
    publishers on every claim. Unknown-publisher DOIs are NOT cached
    (the publisher map may grow over time).
    Never raises.
    """
    resolved_db_path = db_path if db_path is not None else _default_db_path()
    key = _cache_key(doi)

    cached = get(resolved_db_path, key)
    if cached is not None:
        if cached == _NULL_SENTINEL:
            logger.debug("publisher_html_cache_hit_null", doi=doi)
            return None
        logger.debug("publisher_html_cache_hit", doi=doi)
        return cached

    url = _resolve_publisher_url(doi)
    if url is None:
        logger.debug("publisher_html_unknown_publisher", doi=doi)
        # Don't cache: the publisher map may be extended later.
        return None

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url, headers=_HEADERS)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "publisher_html_http_error",
            doi=doi,
            url=url,
            status=exc.response.status_code,
        )
        put(resolved_db_path, key, _NULL_SENTINEL, _CACHE_TTL_SECONDS)
        return None
    except httpx.RequestError as exc:
        logger.warning("publisher_html_request_error", doi=doi, url=url, error=str(exc))
        return None
    except Exception as exc:
        logger.error("publisher_html_unexpected_error", doi=doi, url=url, error=str(exc))
        return None

    content_type = response.headers.get("Content-Type", "").lower()
    if "html" not in content_type:
        logger.warning(
            "publisher_html_not_html",
            doi=doi,
            url=url,
            content_type=content_type,
        )
        return None

    parser = _BodyTextExtractor()
    try:
        parser.feed(response.text)
    except Exception as exc:
        logger.warning("publisher_html_parse_error", doi=doi, url=url, error=str(exc))
        return None

    text = _normalise_whitespace("".join(parser.parts))

    if len(text) < _MIN_TEXT_LENGTH:
        logger.info("publisher_html_too_short", doi=doi, length=len(text))
        put(resolved_db_path, key, _NULL_SENTINEL, _CACHE_TTL_SECONDS)
        return None

    if _looks_paywalled(text):
        logger.info("publisher_html_paywalled", doi=doi)
        put(resolved_db_path, key, _NULL_SENTINEL, _CACHE_TTL_SECONDS)
        return None

    put(resolved_db_path, key, text, _CACHE_TTL_SECONDS)
    logger.info("publisher_html_success", doi=doi, length=len(text))
    return text


def _default_db_path() -> Path:
    from src.clients._cache import default_db_path

    return default_db_path()
