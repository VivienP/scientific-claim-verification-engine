"""PubMed Central full-text retrieval via NCBI efetch API."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
import structlog

from src.clients._cache import get, put
from src.clients._common import (
    CACHE_TTL_DEFAULT_SECONDS as _CACHE_TTL_SECONDS,
)
from src.clients._common import (
    make_cache_key,
)

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_MIN_TEXT_LENGTH = 100
_WHITESPACE_RE = re.compile(r"\s+")

_HEADERS = {"User-Agent": "ScientificClaimVerifier/0.1"}


def _cache_key(pmcid: str) -> str:
    return make_cache_key("pmc", pmcid)


def _normalize_pmcid(pmcid: str) -> str:
    """Strip 'PMC' prefix and any URL wrapping; return numeric ID only."""
    normalized = pmcid.strip()
    if "/" in normalized:
        normalized = normalized.rstrip("/").split("/")[-1]
    if normalized.upper().startswith("PMC"):
        normalized = normalized[3:]
    return normalized


def _xml_to_text(xml_str: str) -> str | None:
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as exc:
        logger.error("pmc_xml_parse_error", error=str(exc))
        return None

    raw = "".join(root.itertext())
    normalized = _WHITESPACE_RE.sub(" ", raw).strip()
    return normalized if normalized else None


def fetch_fulltext(
    pmcid: str,
    *,
    timeout: float = 30.0,
    db_path: Path | None = None,
) -> str | None:
    """Fetch full-text XML from PMC efetch and return plain text.

    pmcid: PMC identifier with or without 'PMC' prefix; URL-form also accepted.
    Returns plain text concatenated from all XML nodes (whitespace-normalized).
    Returns None on any error or if extracted text < _MIN_TEXT_LENGTH chars.
    Cache key: sha256("pmc:{pmcid}"). TTL 30 days.
    Never raises.
    """
    resolved_db_path = db_path if db_path is not None else _default_db_path()
    numeric_id = _normalize_pmcid(pmcid)
    key = _cache_key(numeric_id)

    cached = get(resolved_db_path, key)
    if cached is not None:
        logger.debug("pmc_cache_hit", pmcid=numeric_id)
        return cached

    params = {"db": "pmc", "id": numeric_id, "rettype": "full", "retmode": "xml"}

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(_BASE_URL, params=params, headers=_HEADERS)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.warning("pmc_http_error", pmcid=numeric_id, status=exc.response.status_code)
        return None
    except httpx.RequestError as exc:
        logger.warning("pmc_request_error", pmcid=numeric_id, error=str(exc))
        return None
    except Exception as exc:
        logger.error("pmc_unexpected_error", pmcid=numeric_id, error=str(exc))
        return None

    text = _xml_to_text(response.text)
    if text is None or len(text) < _MIN_TEXT_LENGTH:
        logger.warning("pmc_text_too_short", pmcid=numeric_id, length=len(text) if text else 0)
        return None

    put(resolved_db_path, key, text, _CACHE_TTL_SECONDS)
    logger.info("pmc_fulltext_fetched", pmcid=numeric_id, length=len(text))
    return text


def _default_db_path() -> Path:
    from src.clients._cache import default_db_path

    return default_db_path()
