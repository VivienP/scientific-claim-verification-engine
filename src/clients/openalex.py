"""OpenAlex API client with SQLite caching and exponential backoff."""

from __future__ import annotations

import dataclasses
import json
import re
import time
from pathlib import Path
from typing import Any

import httpx
import structlog

from src.clients._cache import get, put
from src.clients._common import (
    CACHE_TTL_DEFAULT_SECONDS as _CACHE_TTL_SECONDS,
)
from src.clients._common import (
    RETRY_BACKOFF_BASE as _RETRY_BACKOFF_BASE,
)
from src.clients._common import (
    RETRY_MAX as _RETRY_MAX,
)
from src.clients._common import (
    make_cache_key,
)
from src.models import CandidateResolution, ResolvedSource

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_BASE_URL = "https://api.openalex.org"
_MAILTO = "vivienperrelle@gmail.com"  # polite pool — better rate limits

_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
_LEXICAL_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_LOW_TITLE_MATCH_THRESHOLD = 0.15
_STOPWORDS = {
    "and",
    "are",
    "but",
    "for",
    "from",
    "into",
    "not",
    "of",
    "the",
    "this",
    "that",
    "with",
}


def _cache_key(query: str) -> str:
    return make_cache_key("openalex", query)


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


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in _LEXICAL_TOKEN_PATTERN.findall(text.lower())
        if len(token) > 2 and token not in _STOPWORDS and not _YEAR_PATTERN.fullmatch(token)
    }


def _compute_title_match_score(query: str, result: dict[str, Any]) -> float | None:
    query_tokens = _content_tokens(query)
    if not query_tokens:
        return None

    abstract = _reconstruct_abstract(result.get("abstract_inverted_index")) or ""
    title = str(result.get("title") or "")
    result_tokens = _content_tokens(f"{title} {abstract}")
    if not result_tokens:
        return 0.0

    return len(query_tokens.intersection(result_tokens)) / len(query_tokens)


def _pick_best_result(
    data: list[dict[str, Any]], query_year: int | None, query: str
) -> dict[str, Any] | None:
    if not data:
        return None

    def score(item: tuple[int, dict[str, Any]]) -> tuple[float, float, int]:
        index, result = item
        result_year: int | None = result.get("publication_year")
        year_score = _compute_similarity(result_year, query_year)
        lexical_score = _compute_title_match_score(query, result) or 0.0
        return year_score + lexical_score, year_score, -index

    return max(enumerate(data), key=score)[1]


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


def _extract_pmcid(pmcid_raw: str | None) -> str | None:
    """Normalise OpenAlex PMC URL or bare ID → 'PMC1234567' form."""
    if not pmcid_raw:
        return None
    cleaned = pmcid_raw.rstrip("/").split("/")[-1]
    if cleaned.upper().startswith("PMC"):
        return "PMC" + cleaned[3:]
    if cleaned.isdigit():
        return f"PMC{cleaned}"
    return cleaned


def _first_author_display(result: dict[str, Any]) -> str | None:
    """Return the lowercased family name of the first authorship."""
    authorships = result.get("authorships")
    if not isinstance(authorships, list) or not authorships:
        return None
    first = authorships[0]
    if not isinstance(first, dict):
        return None
    author = first.get("author")
    if not isinstance(author, dict):
        return None
    display = author.get("display_name")
    if isinstance(display, str) and display.strip():
        parts = display.strip().split()
        if parts:
            # OpenAlex stores "Given Family"; surname is the last token by convention.
            return parts[-1].lower()
    return None


def _venue_from_result(result: dict[str, Any]) -> str | None:
    """Return OpenAlex's primary host venue display_name when available."""
    primary = result.get("primary_location") or {}
    if isinstance(primary, dict):
        source = primary.get("source")
        if isinstance(source, dict):
            name = source.get("display_name")
            if isinstance(name, str) and name.strip():
                return name.strip()
    host_venue = result.get("host_venue")
    if isinstance(host_venue, dict):
        name = host_venue.get("display_name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return None


def _candidate_from_result(result: dict[str, Any]) -> CandidateResolution:
    """Build a CandidateResolution from an OpenAlex work record.

    Mirrors the CrossRef ``_candidate_from_message`` shape so the verdict
    folder treats the two clients symmetrically when testing
    (year, first_author, venue) agreement.
    """
    doi_raw: str | None = result.get("doi")
    doi: str | None = doi_raw.replace("https://doi.org/", "") if doi_raw else None
    return CandidateResolution(
        client="openalex",
        doi=doi,
        title=result.get("title"),
        year=result.get("publication_year"),
        first_author=_first_author_display(result),
        venue=_venue_from_result(result),
    )


def _build_resolved_source(
    result: dict[str, Any], query_year: int | None, query: str
) -> ResolvedSource:
    result_year: int | None = result.get("publication_year")
    abstract: str | None = _reconstruct_abstract(result.get("abstract_inverted_index"))
    doi_raw: str | None = result.get("doi")
    doi: str | None = doi_raw.replace("https://doi.org/", "") if doi_raw else None
    title_match_score = _compute_title_match_score(query, result)

    oa_info: dict[str, Any] = result.get("open_access") or {}
    oa_url: str | None = oa_info.get("oa_url") or oa_info.get("pdf_url")

    ids_info: dict[str, Any] = result.get("ids") or {}
    pmcid: str | None = _extract_pmcid(ids_info.get("pmcid"))

    return ResolvedSource(
        found=True,
        doi=doi,
        title=result.get("title"),
        abstract=abstract,
        similarity_score=_compute_similarity(result_year, query_year),
        title_match_score=title_match_score,
        resolution_low_confidence=(
            title_match_score is not None and title_match_score < _LOW_TITLE_MATCH_THRESHOLD
        ),
        oa_url=oa_url,
        pmcid=pmcid,
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
        # Drop unknown keys defensively (e.g. if older code wrote a different schema).
        valid_keys = {f.name for f in dataclasses.fields(ResolvedSource)}
        clean = {k: v for k, v in data.items() if k in valid_keys}
        return ResolvedSource(**clean)

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

            best = _pick_best_result(results_list, query_year, query)
            if best is None:
                return _not_found

            resolved = _build_resolved_source(best, query_year, query)
            logger.info(
                "paper_resolved",
                title=resolved.title,
                year=best.get("publication_year"),
                similarity_score=resolved.similarity_score,
                title_match_score=resolved.title_match_score,
                resolution_low_confidence=resolved.resolution_low_confidence,
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


def find_candidate(
    query: str,
    *,
    timeout: float = 10.0,
    db_path: Path | None = None,
) -> CandidateResolution | None:
    """Return OpenAlex's best candidate for a bibliographic query.

    Independent of ``search_paper`` so the resolver verdict folder can call
    both APIs without coupling the legacy ResolvedSource cache to the new
    candidate cache. Same scoring as ``search_paper``; only the returned
    shape differs.
    """
    resolved_db_path = db_path if db_path is not None else _default_db_path()
    key = make_cache_key("openalex:candidate:v1", query)

    cached = get(resolved_db_path, key)
    if cached is not None:
        logger.debug("openalex_candidate_cache_hit", query=query)
        data: dict[str, Any] = json.loads(cached)
        if not data:
            return None
        return CandidateResolution(**data)

    params: dict[str, str | int] = {
        "search": query,
        "per-page": 5,
        "mailto": _MAILTO,
    }
    headers: dict[str, str] = {"User-Agent": "ScientificClaimVerifier/0.1"}
    query_year = _extract_year_from_query(query)

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
                logger.warning(
                    "openalex_candidate_rate_limited", attempt=attempt, wait_seconds=wait
                )
                if attempt < _RETRY_MAX:
                    time.sleep(wait)
                    continue
                return None
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            results_list: list[dict[str, Any]] = payload.get("results", [])
            best = _pick_best_result(results_list, query_year, query)
            if best is None:
                put(resolved_db_path, key, json.dumps({}), _CACHE_TTL_SECONDS)
                return None
            candidate = _candidate_from_result(best)
            put(
                resolved_db_path,
                key,
                json.dumps(dataclasses.asdict(candidate)),
                _CACHE_TTL_SECONDS,
            )
            return candidate
        except httpx.HTTPStatusError as exc:
            logger.error("openalex_candidate_request_error", query=query, error=str(exc))
            return None
        except httpx.RequestError as exc:
            logger.error("openalex_candidate_request_error", query=query, error=str(exc))
            return None
        except Exception as exc:
            logger.error("openalex_candidate_unexpected_error", query=query, error=str(exc))
            return None
    return None


def _default_db_path() -> Path:
    from src.clients._cache import default_db_path

    return default_db_path()
