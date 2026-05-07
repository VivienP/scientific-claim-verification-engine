"""PubMed E-utilities client for DOI -> PMID lookup and PMID -> abstract fetch.

Used as an abstract-fallback path when CrossRef / OpenAlex / Semantic Scholar
return a paper with no abstract field. PubMed often has the abstract for
biomedical papers even when the publisher metadata does not.

NCBI rate limit is 3 requests/second without an API key. The client caches
both lookup steps (DOI -> PMID, PMID -> abstract) for 30 days in the same
SQLite cache used by other clients.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
import structlog

from src.clients._cache import get, put
from src.clients.pubmed_parser import (
    _WHITESPACE,
    _extract_record_fields,
    _title_overlap_score,
)

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_CACHE_TTL_SECONDS = 30 * 24 * 3600  # 30 days
_MIN_ABSTRACT_LENGTH = 50

_HEADERS = {"User-Agent": "ScientificClaimVerifier/0.1"}


@dataclass(frozen=True)
class PubMedRecord:
    pmid: str
    abstract: str | None
    doi: str | None
    pmcid: str | None


def _doi_cache_key(doi: str) -> str:
    return hashlib.sha256(f"pubmed_doi_to_pmid:{doi.lower()}".encode()).hexdigest()


def _title_cache_key(title: str, year: int | None) -> str:
    normalised = _WHITESPACE.sub(" ", title.lower()).strip()
    return hashlib.sha256(f"pubmed_title_to_pmid_v3:{normalised}:{year or ''}".encode()).hexdigest()


def _pmid_cache_key(pmid: str) -> str:
    return hashlib.sha256(f"pubmed_abstract:{pmid}".encode()).hexdigest()


def _record_cache_key(pmid: str) -> str:
    return hashlib.sha256(f"pubmed_record_v1:{pmid}".encode()).hexdigest()


def _params_with_email() -> dict[str, str]:
    """Add tool / email params that NCBI requests for non-anonymous queries."""
    params: dict[str, str] = {"tool": "ScientificClaimVerifier"}
    email = os.environ.get("UNPAYWALL_EMAIL") or os.environ.get("PUBMED_EMAIL")
    if email:
        params["email"] = email
    return params


def _summary_titles(
    ids: list[str],
    *,
    timeout: float,
) -> dict[str, str]:
    if not ids:
        return {}
    params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "json",
        **_params_with_email(),
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(_ESUMMARY_URL, params=params, headers=_HEADERS)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.warning("pubmed_esummary_http_error", status=exc.response.status_code)
        return {}
    except httpx.RequestError as exc:
        logger.warning("pubmed_esummary_request_error", error=str(exc))
        return {}
    except Exception as exc:
        logger.error("pubmed_esummary_unexpected_error", error=str(exc))
        return {}

    try:
        data: dict[str, Any] = response.json()
        result = data.get("result", {})
        titles: dict[str, str] = {}
        for uid in result.get("uids", []):
            item = result.get(str(uid), {})
            title = item.get("title") if isinstance(item, dict) else None
            if title:
                titles[str(uid)] = str(title)
        return titles
    except (ValueError, AttributeError, TypeError) as exc:
        logger.warning("pubmed_esummary_parse_error", error=str(exc))
        return {}


def _best_title_candidate(target_title: str, ids: list[str], *, timeout: float) -> str:
    if len(ids) <= 1:
        return str(ids[0]) if ids else ""
    titles = _summary_titles([str(pmid) for pmid in ids], timeout=timeout)
    if not titles:
        return str(ids[0])
    return max(
        (str(pmid) for pmid in ids),
        key=lambda pmid: _title_overlap_score(target_title, titles.get(str(pmid), "")),
    )


def find_pmid_by_doi(
    doi: str,
    *,
    timeout: float = 15.0,
    db_path: Path | None = None,
) -> str | None:
    """Return the PMID for a DOI via PubMed esearch. None if not found.

    Cache key: sha256("pubmed_doi_to_pmid:{doi}"). TTL 30 days.
    Negative results (no PMID) are also cached as the literal string "" to
    avoid re-hitting NCBI for known-missing entries.
    Never raises.
    """
    if not doi:
        return None
    resolved_db_path = db_path if db_path is not None else _default_db_path()
    key = _doi_cache_key(doi)

    cached = get(resolved_db_path, key)
    if cached is not None:
        logger.debug("pubmed_doi_cache_hit", doi=doi, pmid=cached or None)
        return cached or None

    params = {
        "db": "pubmed",
        "term": f"{doi.strip()}[doi]",
        "retmode": "json",
        "retmax": "1",
        **_params_with_email(),
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(_ESEARCH_URL, params=params, headers=_HEADERS)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.warning("pubmed_esearch_http_error", doi=doi, status=exc.response.status_code)
        return None
    except httpx.RequestError as exc:
        logger.warning("pubmed_esearch_request_error", doi=doi, error=str(exc))
        return None
    except Exception as exc:
        logger.error("pubmed_esearch_unexpected_error", doi=doi, error=str(exc))
        return None

    try:
        data = response.json()
        ids = data.get("esearchresult", {}).get("idlist", [])
    except (ValueError, AttributeError, TypeError) as exc:
        logger.warning("pubmed_esearch_parse_error", doi=doi, error=str(exc))
        return None

    pmid = str(ids[0]) if ids else ""
    put(resolved_db_path, key, pmid, _CACHE_TTL_SECONDS)
    if pmid:
        logger.info("pubmed_pmid_resolved", doi=doi, pmid=pmid)
        return pmid
    logger.debug("pubmed_pmid_not_found", doi=doi)
    return None


def find_pmid_by_title(
    title: str,
    *,
    year: int | None = None,
    timeout: float = 15.0,
    db_path: Path | None = None,
) -> str | None:
    """Return a PMID for an exact title/year query. None if not found."""
    cleaned_title = _WHITESPACE.sub(" ", title).strip()
    if not cleaned_title:
        return None
    resolved_db_path = db_path if db_path is not None else _default_db_path()
    key = _title_cache_key(cleaned_title, year)

    cached = get(resolved_db_path, key)
    if cached is not None:
        logger.debug("pubmed_title_cache_hit", title=cleaned_title, pmid=cached or None)
        return cached or None

    terms = [f'"{cleaned_title}"[Title]' + (f" AND {year}[dp]" if year is not None else "")]
    broad_term = f"{cleaned_title} AND {year}[dp]" if year is not None else cleaned_title
    if broad_term != terms[0]:
        terms.append(broad_term)

    pmid = ""
    for term in terms:
        params = {
            "db": "pubmed",
            "term": term,
            "retmode": "json",
            "retmax": "5",
            **_params_with_email(),
        }

        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.get(_ESEARCH_URL, params=params, headers=_HEADERS)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "pubmed_title_esearch_http_error",
                title=cleaned_title,
                status=exc.response.status_code,
            )
            return None
        except httpx.RequestError as exc:
            logger.warning(
                "pubmed_title_esearch_request_error",
                title=cleaned_title,
                error=str(exc),
            )
            return None
        except Exception as exc:
            logger.error(
                "pubmed_title_esearch_unexpected_error",
                title=cleaned_title,
                error=str(exc),
            )
            return None

        try:
            data = response.json()
            ids = data.get("esearchresult", {}).get("idlist", [])
        except (ValueError, AttributeError, TypeError) as exc:
            logger.warning("pubmed_title_esearch_parse_error", title=cleaned_title, error=str(exc))
            return None
        pmid = _best_title_candidate(
            cleaned_title,
            [str(candidate) for candidate in ids],
            timeout=timeout,
        )
        if pmid:
            break

    put(resolved_db_path, key, pmid, _CACHE_TTL_SECONDS)
    if pmid:
        logger.info("pubmed_title_resolved", title=cleaned_title, year=year, pmid=pmid)
        return pmid
    logger.debug("pubmed_title_not_found", title=cleaned_title, year=year)
    return None


def _extract_record(raw: str, pmid: str) -> PubMedRecord:
    abstract, doi, pmcid = _extract_record_fields(raw)
    return PubMedRecord(pmid=pmid, abstract=abstract, doi=doi, pmcid=pmcid)


def fetch_record(
    pmid: str,
    *,
    timeout: float = 15.0,
    db_path: Path | None = None,
) -> PubMedRecord | None:
    """Fetch abstract plus DOI/PMCID metadata for a PubMed ID. None on failure.

    Cache key: sha256("pubmed_record_v1:{pmid}"). TTL 30 days.
    Returns None when the response has no usable abstract, DOI, or PMCID.
    Never raises.
    """
    if not pmid:
        return None
    resolved_db_path = db_path if db_path is not None else _default_db_path()
    key = _record_cache_key(pmid)

    cached = get(resolved_db_path, key)
    if cached is not None:
        logger.debug("pubmed_record_cache_hit", pmid=pmid)
        if not cached:
            return None
        data = json.loads(cached)
        return PubMedRecord(**data) if data else None

    params = {
        "db": "pubmed",
        "id": pmid,
        "rettype": "abstract",
        "retmode": "text",
        **_params_with_email(),
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(_EFETCH_URL, params=params, headers=_HEADERS)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.warning("pubmed_efetch_http_error", pmid=pmid, status=exc.response.status_code)
        return None
    except httpx.RequestError as exc:
        logger.warning("pubmed_efetch_request_error", pmid=pmid, error=str(exc))
        return None
    except Exception as exc:
        logger.error("pubmed_efetch_unexpected_error", pmid=pmid, error=str(exc))
        return None

    record = _extract_record(response.text, pmid)
    if record.abstract is not None and len(record.abstract) < _MIN_ABSTRACT_LENGTH:
        record = PubMedRecord(
            pmid=record.pmid,
            abstract=None,
            doi=record.doi,
            pmcid=record.pmcid,
        )
    if record.abstract is None and record.doi is None and record.pmcid is None:
        # Cache the negative as empty string so we don't re-query.
        put(resolved_db_path, key, "", _CACHE_TTL_SECONDS)
        logger.debug(
            "pubmed_abstract_too_short",
            pmid=pmid,
            length=len(record.abstract) if record.abstract else 0,
        )
        return None

    put(resolved_db_path, key, json.dumps(asdict(record)), _CACHE_TTL_SECONDS)
    logger.info(
        "pubmed_record_fetched",
        pmid=pmid,
        length=len(record.abstract) if record.abstract else 0,
    )
    return record


def fetch_abstract(
    pmid: str,
    *,
    timeout: float = 15.0,
    db_path: Path | None = None,
) -> str | None:
    """Fetch the abstract text for a PubMed ID. None on failure or empty."""
    resolved_db_path = db_path if db_path is not None else _default_db_path()
    legacy_key = _pmid_cache_key(pmid)
    cached = get(resolved_db_path, legacy_key)
    if cached is not None:
        logger.debug("pubmed_abstract_cache_hit", pmid=pmid)
        return cached or None
    record = fetch_record(pmid, timeout=timeout, db_path=db_path)
    return record.abstract if record else None


def fetch_abstract_by_doi(
    doi: str,
    *,
    timeout: float = 15.0,
    db_path: Path | None = None,
) -> str | None:
    """Convenience: DOI -> PMID -> abstract. None if any step fails."""
    pmid = find_pmid_by_doi(doi, timeout=timeout, db_path=db_path)
    if pmid is None:
        return None
    return fetch_abstract(pmid, timeout=timeout, db_path=db_path)


def _default_db_path() -> Path:
    from src.clients._cache import default_db_path

    return default_db_path()
