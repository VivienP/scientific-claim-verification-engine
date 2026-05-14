"""CrossRef API client — fallback resolver and retraction checker."""

from __future__ import annotations

import dataclasses
import json
import re
import time
import unicodedata
import urllib.parse
from pathlib import Path
from typing import Any

import httpx
import structlog

from src.clients._cache import get, put
from src.clients._common import (
    CACHE_TTL_DEFAULT_SECONDS as _CACHE_TTL_SECONDS,
)
from src.clients._common import (
    RETRACTION_CACHE_TTL_SECONDS as _RETRACTION_CACHE_TTL,
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

_BASE_URL = "https://api.crossref.org"
_MAILTO = "vivienperrelle@gmail.com"

_HEADERS = {
    "User-Agent": "ScientificClaimVerifier/0.1",
    "mailto": _MAILTO,
}

_NOT_FOUND = ResolvedSource(found=False, doi=None, title=None, abstract=None, similarity_score=None)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "on",
    "the",
    "to",
    "with",
}
_SEARCH_TRANSLATION = str.maketrans(
    {
        "\u00a0": " ",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)


def _normalise_doi(doi_raw: str | None) -> str | None:
    if doi_raw is None:
        return None
    return doi_raw.replace("https://doi.org/", "")


def _normalise_search_text(text: str) -> str:
    translated = text.translate(_SEARCH_TRANSLATION)
    decomposed = unicodedata.normalize("NFKD", translated)
    ascii_text = decomposed.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_text.split())


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(_normalise_search_text(text).lower())
        if token not in _STOPWORDS
    }


def _title_from_work_message(message: dict[str, Any]) -> str | None:
    title_list: list[str] | None = message.get("title")
    return title_list[0] if title_list else None


def _work_type_score(work_type: str | None) -> float:
    if work_type == "journal-article":
        return 0.25
    if work_type in {"posted-content", "component", "grant"}:
        return -0.25
    return 0.0


_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _extract_query_year(query: str) -> int | None:
    match = _YEAR_RE.search(query)
    return int(match.group()) if match else None


def _item_authors(item: dict[str, Any]) -> list[str]:
    """Return lowercased family-name surnames from a CrossRef ``author`` array."""
    raw = item.get("author")
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for entry in raw:
        if isinstance(entry, dict):
            family = entry.get("family")
            if isinstance(family, str) and family.strip():
                out.append(_normalise_search_text(family).lower())
    return out


def _item_year(item: dict[str, Any]) -> int | None:
    """Return the publication year from CrossRef's nested ``issued`` field."""
    issued = item.get("issued")
    if not isinstance(issued, dict):
        return None
    parts = issued.get("date-parts")
    if not isinstance(parts, list) or not parts or not isinstance(parts[0], list) or not parts[0]:
        return None
    first = parts[0][0]
    if isinstance(first, int):
        return first
    return None


def _candidate_score(query: str, item: dict[str, Any], index: int) -> tuple[float, float, int]:
    """Score a CrossRef candidate against the query using a multi-signal blend.

    Background. Bug C (S1-P1-C) replaced the asymmetric ``|q & t| / |t|``
    overlap with symmetric Jaccard, which fixed claim-005 (Raa long title
    beating Collange short title). But pure title Jaccard cannot
    distinguish two candidates with similar token overlap when the query
    carries strong author and year signals. Claims 002, 003, 017 in the
    lactate-ISF benchmark were resolving to the wrong DOI because of this.

    S4b-4 multi-signal score: 50% title Jaccard + 30% author overlap +
    15% year proximity + 5% DOI presence + work-type bonus. Author
    overlap is computed against CrossRef's structured ``author`` array
    (family-name field), not against title tokens — so a candidate
    whose surname appears in the query is rewarded even if its title
    shares few tokens with the claim.

    The previous behaviour (title Jaccard) is the limit of this score
    when authors / year are absent from the query: then author_score
    and year_score are 0 and the composite reduces to title Jaccard
    times its 0.5 weight, plus the same DOI / work-type bonuses. The
    relative ordering between candidates with no author/year signal is
    preserved.

    Returns (composite, title_only_jaccard, -index) — the second element
    is kept as a tiebreaker so two candidates with equal composites fall
    back to title overlap (the previous primary signal).
    """
    query_tokens = _content_tokens(query)
    query_lower = _normalise_search_text(query).lower()
    query_year = _extract_query_year(query)

    title_tokens = _content_tokens(_title_from_work_message(item) or "")
    if not query_tokens or not title_tokens:
        title_score = 0.0
    else:
        union_size = len(query_tokens | title_tokens)
        title_score = len(query_tokens & title_tokens) / union_size

    item_authors = _item_authors(item)
    if item_authors:
        matched = sum(1 for a in item_authors if a and a in query_lower)
        author_score = matched / len(item_authors)
    else:
        author_score = 0.0

    year = _item_year(item)
    if query_year is not None and year is not None:
        diff = abs(query_year - year)
        if diff == 0:
            year_score = 1.0
        elif diff == 1:
            year_score = 0.5
        else:
            year_score = 0.0
    else:
        year_score = 0.0

    doi_bonus = 0.05 if item.get("DOI") else 0.0
    composite = (
        0.5 * title_score
        + 0.3 * author_score
        + 0.15 * year_score
        + doi_bonus
        + _work_type_score(item.get("type"))
    )
    return (composite, title_score, -index)


def _pick_best_item(query: str, items: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not items:
        return None
    return max(enumerate(items), key=lambda item: _candidate_score(query, item[1], item[0]))[1]


def _first_author_family(message: dict[str, Any]) -> str | None:
    """Return the lowercased family name of the first author, or ``None``."""
    raw = message.get("author")
    if not isinstance(raw, list) or not raw:
        return None
    first = raw[0]
    if not isinstance(first, dict):
        return None
    family = first.get("family")
    if isinstance(family, str) and family.strip():
        return _normalise_search_text(family).lower()
    return None


def _venue_from_message(message: dict[str, Any]) -> str | None:
    """Return CrossRef's ``container-title`` (journal / conference) if any."""
    raw = message.get("container-title")
    if isinstance(raw, list) and raw:
        first = raw[0]
        if isinstance(first, str) and first.strip():
            return first.strip()
    return None


def _candidate_from_message(message: dict[str, Any]) -> CandidateResolution:
    """Build a CandidateResolution from a CrossRef work message.

    Pure helper. The fields it surfaces (doi/title/year/first_author/venue)
    are the ones the resolver verdict folder consults when checking
    cross-client agreement, so every client must emit them in the same
    normalised form (lowercase ASCII first-author surname).
    """
    return CandidateResolution(
        client="crossref",
        doi=_normalise_doi(message.get("DOI")),
        title=_title_from_work_message(message),
        year=_item_year(message),
        first_author=_first_author_family(message),
        venue=_venue_from_message(message),
    )


def _resolved_from_work_message(message: dict[str, Any]) -> ResolvedSource:
    return ResolvedSource(
        found=True,
        doi=_normalise_doi(message.get("DOI")),
        title=_title_from_work_message(message),
        abstract=None,
        similarity_score=None,
    )


def fetch_work_by_doi(
    doi: str,
    *,
    timeout: float = 10.0,
    db_path: Path | None = None,
) -> ResolvedSource:
    """Fetch a CrossRef work by exact DOI using /works/{doi}."""
    if not doi:
        return _NOT_FOUND
    resolved_db_path = db_path if db_path is not None else _default_db_path()
    normalised = _normalise_doi(doi.strip()) or ""
    key = make_cache_key("crossref:doi", normalised.lower())

    cached = get(resolved_db_path, key)
    if cached is not None:
        logger.debug("crossref_doi_cache_hit", doi=normalised)
        data: dict[str, Any] = json.loads(cached)
        return ResolvedSource(**{**dataclasses.asdict(_NOT_FOUND), **data})

    encoded_doi = urllib.parse.quote(normalised, safe="")
    url = f"{_BASE_URL}/works/{encoded_doi}"

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url, params={"mailto": _MAILTO}, headers=_HEADERS)
        if response.status_code == 404:
            put(
                resolved_db_path,
                key,
                json.dumps(dataclasses.asdict(_NOT_FOUND)),
                _CACHE_TTL_SECONDS,
            )
            return _NOT_FOUND
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        message: dict[str, Any] = payload.get("message", {})
        if not message:
            return _NOT_FOUND
        resolved = _resolved_from_work_message(message)
        put(resolved_db_path, key, json.dumps(dataclasses.asdict(resolved)), _CACHE_TTL_SECONDS)
        logger.info("crossref_doi_resolved", doi=resolved.doi, title=resolved.title)
        return resolved
    except httpx.HTTPStatusError as exc:
        logger.error("crossref_doi_request_error", doi=normalised, error=str(exc))
        return _NOT_FOUND
    except httpx.RequestError as exc:
        logger.error("crossref_doi_request_error", doi=normalised, error=str(exc))
        return _NOT_FOUND
    except Exception as exc:
        logger.error("crossref_doi_unexpected_error", doi=normalised, error=str(exc))
        return _NOT_FOUND


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
    key = make_cache_key("crossref:v2", query)

    cached = get(resolved_db_path, key)
    if cached is not None:
        logger.debug("crossref_cache_hit", query=query)
        data: dict[str, Any] = json.loads(cached)
        return ResolvedSource(**{**dataclasses.asdict(_NOT_FOUND), **data})

    url = f"{_BASE_URL}/works"
    params: dict[str, str | int] = {
        "query.bibliographic": query,
        "rows": 5,
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

            item = _pick_best_item(query, items)
            if item is None:
                return _NOT_FOUND
            resolved = _resolved_from_work_message(item)
            put(resolved_db_path, key, json.dumps(dataclasses.asdict(resolved)), _CACHE_TTL_SECONDS)
            logger.info("crossref_resolved", doi=resolved.doi, title=resolved.title)
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


def find_candidate_by_doi(
    doi: str,
    *,
    timeout: float = 10.0,
    db_path: Path | None = None,
) -> CandidateResolution | None:
    """Return a CandidateResolution for a known DOI, or ``None`` on miss.

    Used by the resolver verdict folder to corroborate a bib-supplied or
    fuzzy-matched DOI against CrossRef's authoritative metadata. Cache key
    is independent of ``fetch_work_by_doi`` because the cached payload shape
    differs — this function stores candidate JSON, not the legacy
    ResolvedSource shape.
    """
    if not doi:
        return None
    resolved_db_path = db_path if db_path is not None else _default_db_path()
    normalised = _normalise_doi(doi.strip()) or ""
    key = make_cache_key("crossref:candidate_doi:v1", normalised.lower())

    cached = get(resolved_db_path, key)
    if cached is not None:
        logger.debug("crossref_candidate_doi_cache_hit", doi=normalised)
        data: dict[str, Any] = json.loads(cached)
        if not data:
            return None
        return CandidateResolution(**data)

    encoded_doi = urllib.parse.quote(normalised, safe="")
    url = f"{_BASE_URL}/works/{encoded_doi}"

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url, params={"mailto": _MAILTO}, headers=_HEADERS)
        if response.status_code == 404:
            put(resolved_db_path, key, json.dumps({}), _CACHE_TTL_SECONDS)
            return None
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        message: dict[str, Any] = payload.get("message", {})
        if not message:
            put(resolved_db_path, key, json.dumps({}), _CACHE_TTL_SECONDS)
            return None
        candidate = _candidate_from_message(message)
        put(resolved_db_path, key, json.dumps(dataclasses.asdict(candidate)), _CACHE_TTL_SECONDS)
        return candidate
    except httpx.HTTPStatusError as exc:
        logger.error("crossref_candidate_doi_request_error", doi=normalised, error=str(exc))
        return None
    except httpx.RequestError as exc:
        logger.error("crossref_candidate_doi_request_error", doi=normalised, error=str(exc))
        return None
    except Exception as exc:
        logger.error("crossref_candidate_doi_unexpected_error", doi=normalised, error=str(exc))
        return None


def find_candidate(
    query: str,
    *,
    timeout: float = 10.0,
    db_path: Path | None = None,
) -> CandidateResolution | None:
    """Return CrossRef's best candidate for a bibliographic query.

    Independent of ``search_paper`` so the resolver can corroborate findings
    without breaking the legacy ResolvedSource cache contract. Same scoring
    logic as ``search_paper`` (``_pick_best_item``); only the returned shape
    differs.
    """
    resolved_db_path = db_path if db_path is not None else _default_db_path()
    key = make_cache_key("crossref:candidate_query:v1", query)

    cached = get(resolved_db_path, key)
    if cached is not None:
        logger.debug("crossref_candidate_query_cache_hit", query=query)
        data: dict[str, Any] = json.loads(cached)
        if not data:
            return None
        return CandidateResolution(**data)

    url = f"{_BASE_URL}/works"
    params: dict[str, str | int] = {
        "query.bibliographic": query,
        "rows": 5,
        "mailto": _MAILTO,
    }

    for attempt in range(1, _RETRY_MAX + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.get(url, params=params, headers=_HEADERS)
            if response.status_code == 429:
                wait = _RETRY_BACKOFF_BASE**attempt
                logger.warning(
                    "crossref_candidate_rate_limited", attempt=attempt, wait_seconds=wait
                )
                if attempt < _RETRY_MAX:
                    time.sleep(wait)
                    continue
                return None
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            items: list[dict[str, Any]] = payload.get("message", {}).get("items", [])
            if not items:
                put(resolved_db_path, key, json.dumps({}), _CACHE_TTL_SECONDS)
                return None
            item = _pick_best_item(query, items)
            if item is None:
                put(resolved_db_path, key, json.dumps({}), _CACHE_TTL_SECONDS)
                return None
            candidate = _candidate_from_message(item)
            put(
                resolved_db_path, key, json.dumps(dataclasses.asdict(candidate)), _CACHE_TTL_SECONDS
            )
            return candidate
        except httpx.HTTPStatusError as exc:
            logger.error("crossref_candidate_request_error", query=query, error=str(exc))
            return None
        except httpx.RequestError as exc:
            logger.error("crossref_candidate_request_error", query=query, error=str(exc))
            return None
        except Exception as exc:
            logger.error("crossref_candidate_unexpected_error", query=query, error=str(exc))
            return None
    return None


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
    key = make_cache_key("crossref:retraction", doi)

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
