"""Semantic Scholar API client with SQLite caching and exponential backoff."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx
import structlog

from src.clients._cache import get, put
from src.models import ResolvedSource

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_FIELDS = "paperId,title,abstract,year,authors"
_RETRY_MAX = 3
_RETRY_BACKOFF_BASE = 2.0  # seconds; attempts: 2s, 4s, 8s
_CACHE_TTL_SECONDS = 30 * 24 * 3600  # 30 days

_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


def _cache_key(query: str) -> str:
    return hashlib.sha256(f"semantic_scholar:{query}".encode()).hexdigest()


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
        if result.get("year") == query_year:
            return result
    for result in data:
        year = result.get("year")
        if year is not None and abs(year - query_year) <= 1:
            return result
    return data[0]


def _parse_authors(authors: list[dict[str, Any]]) -> list[str]:
    names = []
    for author in authors[:3]:
        name = author.get("name", "")
        # Extract last name: "Hoffmann, Jordan" → "Hoffmann" or "Jordan Hoffmann" → "Hoffmann"
        if "," in name:
            names.append(name.split(",")[0].strip())
        elif name:
            parts = name.strip().split()
            names.append(parts[-1] if parts else name)
    return names


def _build_resolved_source(result: dict[str, Any], query_year: int | None) -> ResolvedSource:
    result_year: int | None = result.get("year")
    abstract_raw = result.get("abstract")
    abstract: str | None = str(abstract_raw) if abstract_raw else None
    return ResolvedSource(
        found=True,
        doi=None,
        title=result.get("title"),
        abstract=abstract,
        similarity_score=_compute_similarity(result_year, query_year),
    )


def search_paper(
    query: str,
    *,
    api_key: str | None = None,
    timeout: float = 10.0,
    db_path: Path | None = None,
) -> ResolvedSource:
    """Search Semantic Scholar for a paper matching the query string.

    Never raises. Returns ResolvedSource(found=False, ...) on all errors.
    Retries on 429 and connection errors with exponential backoff.
    Results cached in SQLite for 30 days.
    API key read from SEMANTIC_SCHOLAR_API_KEY env var if api_key is None.
    """
    resolved_db_path = db_path if db_path is not None else _default_db_path()
    key = _cache_key(query)

    cached = get(resolved_db_path, key)
    if cached is not None:
        logger.debug("cache_hit", query=query)
        data: dict[str, Any] = json.loads(cached)
        return ResolvedSource(**data)

    effective_api_key = api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    headers: dict[str, str] = {"User-Agent": "ScientificClaimVerifier/0.1"}
    if effective_api_key:
        headers["x-api-key"] = effective_api_key

    params: dict[str, str | int] = {"query": query, "fields": _FIELDS, "limit": 5}
    query_year = _extract_year_from_query(query)

    _not_found = ResolvedSource(
        found=False, doi=None, title=None, abstract=None, similarity_score=None
    )

    for attempt in range(1, _RETRY_MAX + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.get(
                    f"{_BASE_URL}/paper/search",
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
            results_list: list[dict[str, Any]] = payload.get("data", [])

            best = _pick_best_result(results_list, query_year)
            if best is None:
                return _not_found

            resolved = _build_resolved_source(best, query_year)
            logger.info(
                "paper_resolved",
                title=resolved.title,
                year=best.get("year"),
                similarity_score=resolved.similarity_score,
            )

            # Cache the result
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
