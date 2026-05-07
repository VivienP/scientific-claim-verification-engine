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
import os
import re
from pathlib import Path

import httpx
import structlog

from src.clients._cache import get, put

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_CACHE_TTL_SECONDS = 30 * 24 * 3600  # 30 days
_MIN_ABSTRACT_LENGTH = 50

_HEADERS = {"User-Agent": "ScientificClaimVerifier/0.1"}

# efetch rettype=abstract returns blocks separated by blank lines.
# The abstract body sits between the "Author information:" / affiliation block
# and the trailing PMID line. We strip the metadata lines and normalize whitespace.
_PMID_LINE = re.compile(r"^\s*PMID:\s*\d+.*$", re.MULTILINE)
_DOI_LINE = re.compile(r"^\s*DOI:\s*\S+.*$", re.MULTILINE)
_AUTHOR_INFO = re.compile(r"^Author information:.*?(?=\n\n)", re.MULTILINE | re.DOTALL)
_COPYRIGHT_LINE = re.compile(r"^©.*$|^Copyright .*$", re.MULTILINE)
_WHITESPACE = re.compile(r"\s+")


def _doi_cache_key(doi: str) -> str:
    return hashlib.sha256(f"pubmed_doi_to_pmid:{doi.lower()}".encode()).hexdigest()


def _pmid_cache_key(pmid: str) -> str:
    return hashlib.sha256(f"pubmed_abstract:{pmid}".encode()).hexdigest()


def _params_with_email() -> dict[str, str]:
    """Add tool / email params that NCBI requests for non-anonymous queries."""
    params: dict[str, str] = {"tool": "ScientificClaimVerifier"}
    email = os.environ.get("UNPAYWALL_EMAIL") or os.environ.get("PUBMED_EMAIL")
    if email:
        params["email"] = email
    return params


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


def _strip_metadata(text: str) -> str:
    cleaned = _AUTHOR_INFO.sub("", text)
    cleaned = _PMID_LINE.sub("", cleaned)
    cleaned = _DOI_LINE.sub("", cleaned)
    cleaned = _COPYRIGHT_LINE.sub("", cleaned)
    return cleaned


def _extract_abstract_body(raw: str) -> str | None:
    """Pull the abstract body out of the efetch text response.

    The efetch text response has a header (citation, authors, affiliations),
    then a blank line, then the abstract body, then trailing metadata
    (DOI, PMID, copyright). Concatenate all non-metadata text and normalize.
    """
    cleaned = _strip_metadata(raw)
    blocks = [b.strip() for b in cleaned.split("\n\n") if b.strip()]
    if not blocks:
        return None
    # The first block is typically the citation/header. The abstract body is
    # usually the longest remaining block. Concatenate everything after the
    # first block to keep multi-paragraph abstracts together.
    body_parts = blocks[1:] if len(blocks) > 1 else blocks
    body = " ".join(body_parts)
    body = _WHITESPACE.sub(" ", body).strip()
    return body or None


def fetch_abstract(
    pmid: str,
    *,
    timeout: float = 15.0,
    db_path: Path | None = None,
) -> str | None:
    """Fetch the abstract text for a PubMed ID. None on failure or empty.

    Cache key: sha256("pubmed_abstract:{pmid}"). TTL 30 days.
    Returns None when the response is shorter than _MIN_ABSTRACT_LENGTH.
    Never raises.
    """
    if not pmid:
        return None
    resolved_db_path = db_path if db_path is not None else _default_db_path()
    key = _pmid_cache_key(pmid)

    cached = get(resolved_db_path, key)
    if cached is not None:
        logger.debug("pubmed_abstract_cache_hit", pmid=pmid)
        return cached or None

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

    abstract = _extract_abstract_body(response.text)
    if abstract is None or len(abstract) < _MIN_ABSTRACT_LENGTH:
        # Cache the negative as empty string so we don't re-query.
        put(resolved_db_path, key, "", _CACHE_TTL_SECONDS)
        logger.debug(
            "pubmed_abstract_too_short", pmid=pmid, length=len(abstract) if abstract else 0
        )
        return None

    put(resolved_db_path, key, abstract, _CACHE_TTL_SECONDS)
    logger.info("pubmed_abstract_fetched", pmid=pmid, length=len(abstract))
    return abstract


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
