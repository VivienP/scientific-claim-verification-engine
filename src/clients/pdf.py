"""PDF download and text extraction via pymupdf (fitz)."""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # type: ignore[import-untyped]
import httpx
import structlog

from src.clients._cache import get, put
from src.clients._common import (
    CACHE_TTL_DEFAULT_SECONDS as _CACHE_TTL_SECONDS,
)
from src.clients._common import (
    make_cache_key,
)
from src.models import PdfFetchOutcome

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_MIN_TEXT_LENGTH = 100
_WHITESPACE_RE = re.compile(r"\s+")
_HEADERS = {"User-Agent": "ScientificClaimVerifier/0.1"}


def _cache_key(url: str) -> str:
    return make_cache_key("pdf", url)


def _extract_text(pdf_bytes: bytes) -> str | None:
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            raw_text = "".join(page.get_text() for page in doc)
    except Exception as exc:
        logger.warning("pdf_extract_error", error=str(exc))
        return None

    normalized = _WHITESPACE_RE.sub(" ", raw_text).strip()
    return normalized if normalized else None


def download_and_extract(
    url: str,
    *,
    timeout: float = 30.0,
    db_path: Path | None = None,
) -> PdfFetchOutcome:
    """Download a PDF from a URL and extract plain text via pymupdf.

    Returns a structured ``PdfFetchOutcome`` carrying both the text (when
    extraction succeeded) and a ``failure_reason`` that lets callers attribute
    the failure precisely. ``"not_a_pdf"`` is the typical paywall HTML page
    signal; ``"http_error"`` is a real non-2xx; ``"extraction_failed"`` is a
    malformed PDF; ``"too_short"`` is sub-threshold extracted text.

    Cache key: sha256("pdf:{url}"). TTL 30 days. Cache stores only successful
    extracted text — a cache hit always rehydrates to ``failure_reason="ok"``.

    Never raises.
    """
    resolved_db_path = db_path if db_path is not None else _default_db_path()
    key = _cache_key(url)

    cached = get(resolved_db_path, key)
    if cached is not None:
        logger.debug("pdf_cache_hit", url=url)
        return PdfFetchOutcome(text=cached, failure_reason="ok")

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url, headers=_HEADERS)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.warning("pdf_http_error", url=url, status=exc.response.status_code)
        return PdfFetchOutcome(
            text=None,
            failure_reason="http_error",
            http_status=exc.response.status_code,
        )
    except httpx.TimeoutException as exc:
        logger.warning("pdf_timeout", url=url, error=str(exc))
        return PdfFetchOutcome(text=None, failure_reason="timeout")
    except httpx.RequestError as exc:
        logger.warning("pdf_request_error", url=url, error=str(exc))
        return PdfFetchOutcome(text=None, failure_reason="http_error")
    except Exception as exc:
        logger.error("pdf_unexpected_error", url=url, error=str(exc))
        return PdfFetchOutcome(text=None, failure_reason="http_error")

    content_type = response.headers.get("Content-Type", "").lower()
    if "pdf" not in content_type:
        logger.warning("pdf_not_pdf_content_type", url=url, content_type=content_type)
        return PdfFetchOutcome(
            text=None,
            failure_reason="not_a_pdf",
            http_status=response.status_code,
            content_type=content_type or None,
        )

    text = _extract_text(response.content)
    if text is None:
        logger.warning("pdf_extraction_failed", url=url)
        return PdfFetchOutcome(
            text=None,
            failure_reason="extraction_failed",
            http_status=response.status_code,
            content_type=content_type or None,
        )
    if len(text) < _MIN_TEXT_LENGTH:
        logger.warning("pdf_text_too_short", url=url, length=len(text))
        return PdfFetchOutcome(
            text=None,
            failure_reason="too_short",
            http_status=response.status_code,
            content_type=content_type or None,
        )

    put(resolved_db_path, key, text, _CACHE_TTL_SECONDS)
    logger.info("pdf_extracted", url=url, length=len(text))
    return PdfFetchOutcome(
        text=text,
        failure_reason="ok",
        http_status=response.status_code,
        content_type=content_type or None,
    )


def _default_db_path() -> Path:
    from src.clients._cache import default_db_path

    return default_db_path()
