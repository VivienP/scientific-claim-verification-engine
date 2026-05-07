"""Europe PMC client — DOI -> abstract / PMCID / OA full-text URL.

Europe PMC (https://europepmc.org) is a complementary biomedical literature
service that often exposes OA mirrors and abstracts that CrossRef and NCBI
PubMed miss. The S2-P0 OA discovery probe at
`eval/e2e/probes/_oa_discovery_probe.py` confirmed coverage on every paywalled
DOI in the lactate-ISF benchmark sample (100 % abstract availability,
50 % OA URL availability).

Used as an enrichment layer in `src/resolve.py` (after `_enrich_via_pubmed`)
and as a fulltext source in `src/fetch_fulltext.py` (between PMC and
Unpaywall). The single-call `fetch_record` returns a `EuropePMCRecord`
that callers project to the field they need.

Caching follows the project pattern: SQLite WAL via `src/clients/_cache.py`,
30-day TTL, negative results cached as a JSON null sentinel to avoid
re-querying known-missing DOIs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
import structlog

from src.clients._cache import get, put

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_CACHE_TTL_SECONDS = 30 * 24 * 3600  # 30 days
_NULL_SENTINEL = "__NULL__"
_HEADERS = {"User-Agent": "ScientificClaimVerifier/0.1"}


@dataclass(frozen=True)
class EuropePMCRecord:
    pmcid: str | None
    abstract: str | None
    pdf_url: str | None
    html_url: str | None
    is_open_access: bool


def _cache_key(doi: str) -> str:
    return hashlib.sha256(f"europepmc_v1:{doi}".encode()).hexdigest()


def _normalise_pmcid(raw: str | None) -> str | None:
    if not raw:
        return None
    raw = raw.strip()
    return raw if raw.upper().startswith("PMC") else f"PMC{raw}"


def _extract_record(payload: dict[str, Any]) -> EuropePMCRecord | None:
    """Project a Europe PMC `search` JSON response to a single record.

    Returns None when the response has no result. Selects the first hit when
    multiple are returned (DOI-keyed queries practically always return 1).
    """
    result_list_obj = payload.get("resultList", {})
    result_list = result_list_obj if isinstance(result_list_obj, dict) else {}
    results_obj = result_list.get("result", [])
    results = results_obj if isinstance(results_obj, list) else []
    if not results:
        return None
    item = results[0]
    if not isinstance(item, dict):
        return None

    pmcid = _normalise_pmcid(item.get("pmcid") if isinstance(item.get("pmcid"), str) else None)

    abstract_raw = item.get("abstractText")
    abstract = str(abstract_raw) if abstract_raw else None

    is_oa = item.get("isOpenAccess") == "Y"

    pdf_url: str | None = None
    html_url: str | None = None
    fulltext_obj = item.get("fullTextUrlList", {})
    fulltext_list = fulltext_obj if isinstance(fulltext_obj, dict) else {}
    urls_obj = fulltext_list.get("fullTextUrl", [])
    urls = urls_obj if isinstance(urls_obj, list) else []
    for entry in urls:
        if not isinstance(entry, dict):
            continue
        style = entry.get("documentStyle")
        url_raw = entry.get("url")
        if not isinstance(url_raw, str):
            continue
        if style == "pdf" and pdf_url is None:
            pdf_url = url_raw
        elif style == "html" and html_url is None:
            html_url = url_raw

    return EuropePMCRecord(
        pmcid=pmcid,
        abstract=abstract,
        pdf_url=pdf_url,
        html_url=html_url,
        is_open_access=is_oa,
    )


def fetch_record(
    doi: str,
    *,
    timeout: float = 15.0,
    db_path: Path | None = None,
) -> EuropePMCRecord | None:
    """Return a `EuropePMCRecord` for `doi` or None on miss / error.

    Cache key: sha256("europepmc_v1:{doi}"). TTL 30 days.
    Negative results are cached as a sentinel to avoid re-querying.
    Never raises.
    """
    if not doi:
        return None
    resolved_db_path = db_path if db_path is not None else _default_db_path()
    key = _cache_key(doi)

    cached = get(resolved_db_path, key)
    if cached is not None:
        if cached == _NULL_SENTINEL:
            logger.debug("europepmc_cache_hit_negative", doi=doi)
            return None
        try:
            data: dict[str, Any] = json.loads(cached)
            return EuropePMCRecord(**data)
        except (json.JSONDecodeError, TypeError):
            logger.warning("europepmc_cache_corrupted", doi=doi)

    params = {"query": f"DOI:{doi}", "format": "json", "resultType": "core"}
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(_BASE_URL, params=params, headers=_HEADERS)
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
    except httpx.HTTPStatusError as exc:
        logger.warning("europepmc_http_error", doi=doi, status=exc.response.status_code)
        return None
    except httpx.RequestError as exc:
        logger.warning("europepmc_request_error", doi=doi, error=str(exc))
        return None
    except (ValueError, TypeError) as exc:
        logger.warning("europepmc_parse_error", doi=doi, error=str(exc))
        return None
    except Exception as exc:
        logger.error("europepmc_unexpected_error", doi=doi, error=str(exc))
        return None

    record = _extract_record(payload)
    if record is None:
        put(resolved_db_path, key, _NULL_SENTINEL, _CACHE_TTL_SECONDS)
        logger.debug("europepmc_no_result", doi=doi)
        return None

    put(resolved_db_path, key, json.dumps(asdict(record)), _CACHE_TTL_SECONDS)
    logger.info(
        "europepmc_record_fetched",
        doi=doi,
        pmcid=record.pmcid,
        is_oa=record.is_open_access,
        has_abstract=record.abstract is not None,
        has_pdf=record.pdf_url is not None,
    )
    return record


def fetch_oa_url(
    doi: str,
    *,
    timeout: float = 15.0,
    db_path: Path | None = None,
) -> str | None:
    """Return the best OA URL for `doi` from Europe PMC, prefer PDF over HTML.

    Returns None when no OA mirror exists or on any error.
    """
    record = fetch_record(doi, timeout=timeout, db_path=db_path)
    if record is None:
        return None
    return record.pdf_url or record.html_url


def fetch_abstract(
    doi: str,
    *,
    timeout: float = 15.0,
    db_path: Path | None = None,
) -> str | None:
    """Return the Europe PMC abstract for `doi` or None."""
    record = fetch_record(doi, timeout=timeout, db_path=db_path)
    return record.abstract if record else None


def find_pmcid_by_doi(
    doi: str,
    *,
    timeout: float = 15.0,
    db_path: Path | None = None,
) -> str | None:
    """Return the PMCID for `doi` via Europe PMC. Sometimes hits when NCBI misses."""
    record = fetch_record(doi, timeout=timeout, db_path=db_path)
    return record.pmcid if record else None


def _default_db_path() -> Path:
    from src.clients._cache import default_db_path

    return default_db_path()
