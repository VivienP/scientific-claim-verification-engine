"""PDF download and text extraction via pymupdf (fitz)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import fitz  # type: ignore[import-untyped]
import httpx
import structlog

from src.clients._cache import get, put

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_CACHE_TTL_SECONDS = 30 * 24 * 3600
_MIN_TEXT_LENGTH = 100
_WHITESPACE_RE = re.compile(r"\s+")
_HEADERS = {"User-Agent": "ScientificClaimVerifier/0.1"}


def _cache_key(url: str) -> str:
    return hashlib.sha256(f"pdf:{url}".encode()).hexdigest()


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
) -> str | None:
    """Download a PDF from a URL and extract plain text via pymupdf.

    Verifies Content-Type contains 'pdf' before parsing. Returns None for
    non-PDF responses, network errors, or extracted text < _MIN_TEXT_LENGTH chars.
    Cache key: sha256("pdf:{url}"). TTL 30 days.
    Never raises.
    """
    resolved_db_path = db_path if db_path is not None else _default_db_path()
    key = _cache_key(url)

    cached = get(resolved_db_path, key)
    if cached is not None:
        logger.debug("pdf_cache_hit", url=url)
        return cached

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url, headers=_HEADERS)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.warning("pdf_http_error", url=url, status=exc.response.status_code)
        return None
    except httpx.RequestError as exc:
        logger.warning("pdf_request_error", url=url, error=str(exc))
        return None
    except Exception as exc:
        logger.error("pdf_unexpected_error", url=url, error=str(exc))
        return None

    content_type = response.headers.get("Content-Type", "").lower()
    if "pdf" not in content_type:
        logger.warning("pdf_not_pdf_content_type", url=url, content_type=content_type)
        return None

    text = _extract_text(response.content)
    if text is None or len(text) < _MIN_TEXT_LENGTH:
        logger.warning("pdf_text_too_short", url=url, length=len(text) if text else 0)
        return None

    put(resolved_db_path, key, text, _CACHE_TTL_SECONDS)
    logger.info("pdf_extracted", url=url, length=len(text))
    return text


def _default_db_path() -> Path:
    from src.clients._cache import default_db_path

    return default_db_path()
