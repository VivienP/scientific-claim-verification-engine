"""Unpaywall API client — DOI → best open-access URL lookup."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import structlog

from src.clients._cache import get, put

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_BASE_URL = "https://api.unpaywall.org/v2"
_EMAIL = "vivienperrelle@gmail.com"
_CACHE_TTL_SECONDS = 30 * 24 * 3600

_HEADERS = {"User-Agent": "ScientificClaimVerifier/0.1"}
_NULL_SENTINEL = "__NULL__"


def _cache_key(doi: str) -> str:
    return hashlib.sha256(f"unpaywall:{doi}".encode()).hexdigest()


def get_oa_url(
    doi: str,
    *,
    email: str = _EMAIL,
    timeout: float = 10.0,
    db_path: Path | None = None,
) -> str | None:
    """Look up the best open-access URL for a DOI via Unpaywall.

    Prefers best_oa_location.url_for_pdf (direct PDF), falls back to .url (HTML).
    Returns None if no OA copy is available or on any error.
    Cache key: sha256("unpaywall:{doi}"). TTL 30 days.
    Never raises.
    """
    resolved_db_path = db_path if db_path is not None else _default_db_path()
    key = _cache_key(doi)

    cached = get(resolved_db_path, key)
    if cached is not None:
        logger.debug("unpaywall_cache_hit", doi=doi)
        return None if cached == _NULL_SENTINEL else json.loads(cached)

    url = f"{_BASE_URL}/{doi}"

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url, params={"email": email}, headers=_HEADERS)
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
    except httpx.HTTPStatusError as exc:
        logger.warning("unpaywall_http_error", doi=doi, status=exc.response.status_code)
        return None
    except httpx.RequestError as exc:
        logger.warning("unpaywall_request_error", doi=doi, error=str(exc))
        return None
    except Exception as exc:
        logger.error("unpaywall_unexpected_error", doi=doi, error=str(exc))
        return None

    location = payload.get("best_oa_location")
    if not isinstance(location, dict):
        put(resolved_db_path, key, _NULL_SENTINEL, _CACHE_TTL_SECONDS)
        return None

    pdf_url = location.get("url_for_pdf")
    html_url = location.get("url")
    chosen: str | None = pdf_url if pdf_url else html_url

    if chosen:
        put(resolved_db_path, key, json.dumps(chosen), _CACHE_TTL_SECONDS)
        logger.info("unpaywall_oa_url_found", doi=doi, is_pdf=bool(pdf_url))
    else:
        put(resolved_db_path, key, _NULL_SENTINEL, _CACHE_TTL_SECONDS)

    return chosen


def _default_db_path() -> Path:
    from src.clients._cache import default_db_path

    return default_db_path()
