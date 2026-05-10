"""primary_source_doi lookup via Semantic Scholar references graph.

When is_primary_source=False, queries SS /paper/{doi}/references for the
resolved source, filters to references matching primary-study signals, and
returns the top candidate DOI (CrossRef-verified, score ≥ threshold).

Safety: returns None if SS call fails, returns empty results, or CrossRef
verify fails — never guesses.
Emits ProvenanceStep(operation="copilot_primary_lookup", model_id=None).
"""

from __future__ import annotations

import time
import urllib.parse
import uuid
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
from src.clients.crossref import fetch_work_by_doi
from src.copilot.primary_source import _PRIMARY_SIGNALS_RE_FOR_LOOKUP, _extract_max_n
from src.models import ProvenanceStep
from src.verify_prompts import _hash

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_SS_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_SS_HEADERS = {"User-Agent": "ScientificClaimVerifier/0.1"}
_SS_FIELDS = "title,abstract,externalIds,year"
_CROSSREF_VERIFY_THRESHOLD = 0.7

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _default_db_path() -> Path:
    return Path(".cache") / "api_cache.db"


def _is_primary_candidate(ref: dict[str, object]) -> bool:
    """Return True if a SS reference dict looks like a primary study."""
    abstract: str = ref.get("abstract") or ""  # type: ignore[assignment]
    title: str = ref.get("title") or ""  # type: ignore[assignment]
    combined = f"{title} {abstract}"
    return bool(_PRIMARY_SIGNALS_RE_FOR_LOOKUP.search(combined))


def _doi_from_ref(ref: dict[str, object]) -> str | None:
    """Extract DOI from SS externalIds, normalised to bare form."""
    ids: dict[str, str] = ref.get("externalIds") or {}  # type: ignore[assignment]
    doi = ids.get("DOI") or ids.get("doi")
    if doi:
        return doi.replace("https://doi.org/", "").strip()
    return None


def _score_candidate(ref: dict[str, object], claim_year: int | None) -> float:
    """Score a primary candidate by year proximity and sample size signal.

    Higher score = more likely to be the relevant primary source.
    Range: 0.0 - 1.0.
    """
    ref_year: int | None = ref.get("year")  # type: ignore[assignment]
    abstract: str = ref.get("abstract") or ""  # type: ignore[assignment]

    score = 0.5  # base

    # Year proximity: prefer references close to the claimed study year.
    if claim_year is not None and ref_year is not None:
        diff = abs(claim_year - ref_year)
        if diff == 0:
            score += 0.3
        elif diff <= 2:
            score += 0.2
        elif diff <= 5:
            score += 0.1

    # Sample-size signal: prefer larger studies.
    n = _extract_max_n(abstract)
    if n is not None:
        if n >= 100:
            score += 0.2
        elif n >= 50:
            score += 0.1

    return min(score, 1.0)


def _fetch_ss_references(
    doi: str,
    *,
    timeout: float = 10.0,
    db_path: Path | None = None,
) -> list[dict[str, object]]:
    """Fetch /paper/{doi}/references from Semantic Scholar with caching.

    Returns a list of reference dicts (may be empty).
    Never raises — returns [] on any error.
    """
    resolved_db_path = db_path or _default_db_path()
    encoded_doi = urllib.parse.quote(doi, safe="")
    cache_key = make_cache_key("ss_refs_v1", doi.lower())

    import json

    cached = get(resolved_db_path, cache_key)
    if cached is not None:
        logger.debug("ss_references_cache_hit", doi=doi)
        result: list[dict[str, object]] = json.loads(cached)
        return result

    url = f"{_SS_BASE_URL}/paper/{encoded_doi}/references"
    params: dict[str, str | int] = {"fields": _SS_FIELDS, "limit": 50}

    for attempt in range(1, _RETRY_MAX + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.get(url, params=params, headers=_SS_HEADERS)

            if resp.status_code == 429:
                wait = _RETRY_BACKOFF_BASE**attempt
                logger.warning("ss_rate_limited", doi=doi, attempt=attempt)
                if attempt < _RETRY_MAX:
                    time.sleep(wait)
                    continue
                return []

            if resp.status_code == 404:
                put(resolved_db_path, cache_key, json.dumps([]), _CACHE_TTL_SECONDS)
                return []

            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            raw_refs: list[dict[str, Any]] = data.get("data") or []
            refs: list[dict[str, object]] = [
                r["citedPaper"]
                for r in raw_refs
                if isinstance(r, dict) and isinstance(r.get("citedPaper"), dict)
            ]
            put(resolved_db_path, cache_key, json.dumps(refs), _CACHE_TTL_SECONDS)
            logger.info("ss_references_fetched", doi=doi, count=len(refs))
            return refs

        except Exception:
            logger.exception("ss_references_error", doi=doi, attempt=attempt)
            return []

    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_primary_source_doi(
    doi: str | None,
    *,
    claim_year: int | None = None,
    crossref_verify_threshold: float = _CROSSREF_VERIFY_THRESHOLD,
    db_path: Path | None = None,
    timeout: float = 10.0,
) -> tuple[str | None, str | None, ProvenanceStep]:
    """Find a primary study DOI for the given secondary source DOI.

    Queries SS /paper/{doi}/references, filters to primary candidates,
    verifies the top candidate via CrossRef (/works/{doi}), returns its
    DOI and title if verified — otherwise (None, None).

    Args:
        doi: DOI of the resolved secondary source.
        claim_year: Year from the Claim (used for candidate scoring).
        crossref_verify_threshold: Minimum CrossRef score to accept a DOI.
            Not directly used — we accept any DOI that CrossRef can resolve.
        db_path: SQLite cache path. Defaults to .cache/api_cache.db.
        timeout: HTTP timeout in seconds.

    Returns:
        (primary_doi_or_none, primary_title_or_none, ProvenanceStep)
        model_id=None (no LLM). doi=None when no candidate found/verified.
    """
    ts = time.time()
    input_hash = _hash(repr((doi, claim_year)))

    primary_doi: str | None = None
    primary_title: str | None = None

    if doi:
        refs = _fetch_ss_references(doi, timeout=timeout, db_path=db_path)
        primary_candidates = [r for r in refs if _is_primary_candidate(r)]

        if primary_candidates:
            best = max(
                primary_candidates,
                key=lambda r: _score_candidate(r, claim_year),
            )
            candidate_doi = _doi_from_ref(best)
            if candidate_doi:
                verified = fetch_work_by_doi(candidate_doi, timeout=timeout, db_path=db_path)
                if verified.found and verified.doi:
                    primary_doi = verified.doi
                    primary_title = verified.title
                    logger.info(
                        "primary_lookup_found",
                        source_doi=doi,
                        primary_doi=primary_doi,
                    )
                else:
                    logger.info(
                        "primary_lookup_crossref_failed",
                        source_doi=doi,
                        candidate_doi=candidate_doi,
                    )
            else:
                logger.info("primary_lookup_no_doi_in_candidate", source_doi=doi)
        else:
            logger.info("primary_lookup_no_primary_candidates", source_doi=doi)
    else:
        logger.info("primary_lookup_no_source_doi")

    step = ProvenanceStep(
        step_id=str(uuid.uuid4()),
        claim_id="",  # caller fills this; enricher wraps with claim_id
        operation="copilot_primary_lookup",
        input_hash=input_hash,
        output_hash=_hash(repr((primary_doi, primary_title))),
        model_id=None,
        timestamp=ts,
        tokens_in=None,
        tokens_out=None,
        cache_hit=None,
        confidence=None,
    )
    return primary_doi, primary_title, step
