"""CrossRef API client — fallback resolver and retraction checker."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
import urllib.parse
from pathlib import Path
from typing import Any

import httpx
import structlog

from src.clients._cache import get, put
from src.models import ResolvedSource

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_BASE_URL = "https://api.crossref.org"
_MAILTO = "vivienperrelle@gmail.com"
_RETRY_MAX = 3
_RETRY_BACKOFF_BASE = 2.0
_CACHE_TTL_SECONDS = 30 * 24 * 3600  # 30 days
_RETRACTION_CACHE_TTL = 7 * 24 * 3600  # 7 days — retractions are stable but faster-changing

_HEADERS = {
    "User-Agent": "ScientificClaimVerifier/0.1",
    "mailto": _MAILTO,
}

_NOT_FOUND = ResolvedSource(found=False, doi=None, title=None, abstract=None, similarity_score=None)


def _cache_key(prefix: str, value: str) -> str:
    return hashlib.sha256(f"{prefix}:{value}".encode()).hexdigest()


def search_paper(
    query: str,
    *,
    timeout: float = 10.0,
    db_path: Path | None = None,
) -> ResolvedSource:
    """Search CrossRef by bibliographic query. Returns ResolvedSource.

    Fallback resolver — called when OpenAlex returns found=False.
    CrossRef has no abstracts; returned ResolvedSource always has abstract=None.
    Never raises. Returns found=False on all errors.
    Cache key: sha256("crossref:{query}"). TTL 30 days.
    Retries on 429 with exponential backoff (max 3, base 2.0s).
    """
    resolved_db_path = db_path if db_path is not None else _default_db_path()
    key = _cache_key("crossref", query)

    cached = get(resolved_db_path, key)
    if cached is not None:
        logger.debug("crossref_cache_hit", query=query)
        data: dict[str, Any] = json.loads(cached)
        return ResolvedSource(**{**dataclasses.asdict(_NOT_FOUND), **data})

    url = f"{_BASE_URL}/works"
    params: dict[str, str | int] = {
        "query.bibliographic": query,
        "rows": 1,
        "mailto": _MAILTO,
    }

    for attempt in range(1, _RETRY_MAX + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.get(url, params=params, headers=_HEADERS)

            if response.status_code == 429:
                wait = _RETRY_BACKOFF_BASE**attempt
                logger.warning("crossref_rate_limited", attempt=attempt, wait_seconds=wait)
                if attempt < _RETRY_MAX:
                    time.sleep(wait)
                    continue
                logger.error("crossref_max_retries_exceeded", query=query)
                return _NOT_FOUND

            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            items: list[dict[str, Any]] = payload.get("message", {}).get("items", [])

            if not items:
                return _NOT_FOUND

            item = items[0]
            doi_raw: str | None = item.get("DOI")
            doi = doi_raw.replace("https://doi.org/", "") if doi_raw else doi_raw
            title_list: list[str] | None = item.get("title")
            title = title_list[0] if title_list else None

            resolved = ResolvedSource(
                found=True,
                doi=doi,
                title=title,
                abstract=None,
                similarity_score=None,
            )
            put(resolved_db_path, key, json.dumps(dataclasses.asdict(resolved)), _CACHE_TTL_SECONDS)
            logger.info("crossref_resolved", doi=doi, title=title)
            return resolved

        except httpx.HTTPStatusError as exc:
            logger.error("crossref_request_error", query=query, error=str(exc))
            return _NOT_FOUND
        except httpx.RequestError as exc:
            logger.error("crossref_request_error", query=query, error=str(exc))
            return _NOT_FOUND
        except Exception as exc:
            logger.error("crossref_unexpected_error", query=query, error=str(exc))
            return _NOT_FOUND

    return _NOT_FOUND


def check_retraction(
    doi: str,
    *,
    timeout: float = 10.0,
    db_path: Path | None = None,
) -> bool:
    """Check if a DOI has a retraction record via CrossRef update-to field.

    Returns True if the work has an update-to entry of type "retraction".
    Returns False if not retracted or on any error.
    Cache key: sha256("crossref:retraction:{doi}"). TTL 7 days.
    """
    resolved_db_path = db_path if db_path is not None else _default_db_path()
    key = _cache_key("crossref:retraction", doi)

    cached = get(resolved_db_path, key)
    if cached is not None:
        logger.debug("crossref_retraction_cache_hit", doi=doi)
        return bool(json.loads(cached))

    encoded_doi = urllib.parse.quote(doi, safe="")
    url = f"{_BASE_URL}/works/{encoded_doi}"

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url, headers=_HEADERS)

        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        updates: list[dict[str, Any]] = payload.get("message", {}).get("update-to", [])
        retracted = any(u.get("type") == "retraction" for u in updates)

        put(resolved_db_path, key, json.dumps(retracted), _RETRACTION_CACHE_TTL)
        if retracted:
            logger.warning("crossref_retraction_found", doi=doi)
        return retracted

    except Exception as exc:
        logger.error("crossref_retraction_check_error", doi=doi, error=str(exc))
        return False


def _default_db_path() -> Path:
    from src.clients._cache import default_db_path

    return default_db_path()
