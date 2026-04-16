"""OpenAlex API client with SQLite caching and exponential backoff."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import httpx
import structlog

from src.clients._cache import get, put
from src.models import ResolvedSource

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_BASE_URL = "https://api.openalex.org"
_RETRY_MAX = 3
_RETRY_BACKOFF_BASE = 2.0  # seconds; attempts: 2s, 4s, 8s
_CACHE_TTL_SECONDS = 30 * 24 * 3600  # 30 days
_MAILTO = "vivienperrelle@gmail.com"  # polite pool — better rate limits

_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


def _cache_key(query: str) -> str:
    return hashlib.sha256(f"openalex:{query}".encode()).hexdigest()


def _extract_year_from_query(query: str) -> int | None:
    match = _YEAR_PATTERN.search(query)
    return int(match.group()) if match else None


def _compute_similarity(result_year: int | None, query_year: int | None) -> float:
    if query_year is None:
        return 0.8
    if result_year is None:
        return 0.8
    diff = abs(result_year - query_year)
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.9
    return 0.8


def _pick_best_result(data: list[dict[str, Any]], query_year: int | None) -> dict[str, Any] | None:
    if not data:
        return None
    if query_year is None:
        return data[0]
    # Prefer exact year match, then ±1, then first result
    for result in data:
        if result.get("publication_year") == query_year:
            return result
    for result in data:
        year = result.get("publication_year")
        if year is not None and abs(year - query_year) <= 1:
            return result
    return data[0]


def _reconstruct_abstract(inv_idx: dict[str, list[int]] | None) -> str | None:
    """Reconstruct plain-text abstract from OpenAlex inverted index format."""
    if not inv_idx:
        return None
    positions: dict[int, str] = {}
    for word, pos_list in inv_idx.items():
        for pos in pos_list:
            positions[pos] = word
    if not positions:
        return None
    return " ".join(positions[i] for i in sorted(positions))


def _build_resolved_source(result: dict[str, Any], query_year: int | None) -> ResolvedSource:
    result_year: int | None = result.get("publication_year")
    abstract: str | None = _reconstruct_abstract(result.get("abstract_inverted_index"))
    doi_raw: str | None = result.get("doi")
    doi: str | None = doi_raw.replace("https://doi.org/", "") if doi_raw else None
    return ResolvedSource(
        found=True,
        doi=doi,
        title=result.get("title"),
        abstract=abstract,
        similarity_score=_compute_similarity(result_year, query_year),
    )


def search_paper(
    query: str,
    *,
    api_key: str | None = None,  # unused; kept for interface compatibility with S2 client
    timeout: float = 10.0,
    db_path: Path | None = None,
) -> ResolvedSource:
    """Search OpenAlex for a paper matching the query string.

    Never raises. Returns ResolvedSource(found=False, ...) on all errors.
    Retries on 429 and connection errors with exponential backoff.
    Results cached in SQLite for 30 days.
    api_key parameter is accepted but unused (OpenAlex is free, no key required).
    """
    resolved_db_path = db_path if db_path is not None else _default_db_path()
    key = _cache_key(query)

    cached = get(resolved_db_path, key)
    if cached is not None:
        logger.debug("cache_hit", query=query)
        data: dict[str, Any] = json.loads(cached)
        return ResolvedSource(**data)

    params: dict[str, str | int] = {
        "search": query,
        "per-page": 5,
        "mailto": _MAILTO,
    }
    headers: dict[str, str] = {"User-Agent": "ScientificClaimVerifier/0.1"}
    query_year = _extract_year_from_query(query)

    _not_found = ResolvedSource(
        found=False, doi=None, title=None, abstract=None, similarity_score=None
    )

    for attempt in range(1, _RETRY_MAX + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.get(
                    f"{_BASE_URL}/works",
                    params=params,
                    headers=headers,
                )

            if response.status_code == 429:
                wait = _RETRY_BACKOFF_BASE**attempt
                logger.warning("rate_limited", attempt=attempt, wait_seconds=wait)
                if attempt < _RETRY_MAX:
                    time.sleep(wait)
                    continue
                else:
                    logger.error("max_retries_exceeded", query=query)
                    return _not_found

            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            results_list: list[dict[str, Any]] = payload.get("results", [])

            best = _pick_best_result(results_list, query_year)
            if best is None:
                return _not_found

            resolved = _build_resolved_source(best, query_year)
            logger.info(
                "paper_resolved",
                title=resolved.title,
                year=best.get("publication_year"),
                similarity_score=resolved.similarity_score,
            )

            put(resolved_db_path, key, json.dumps(dataclasses.asdict(resolved)), _CACHE_TTL_SECONDS)
            return resolved

        except httpx.HTTPStatusError as exc:
            logger.error("request_error", query=query, error=str(exc))
            return _not_found
        except httpx.RequestError as exc:
            logger.error("request_error", query=query, error=str(exc))
            return _not_found
        except Exception as exc:  # catch-all: network/JSON/other unexpected errors; always log
            logger.error("unexpected_error", query=query, error=str(exc))
            return _not_found

    logger.error("max_retries_exceeded", query=query)
    return _not_found


def _default_db_path() -> Path:
    from src.clients._cache import default_db_path

    return default_db_path()
