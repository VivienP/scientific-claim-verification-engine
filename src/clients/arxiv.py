"""arXiv API client — title/author search → ResolvedSource.

Used as a fallback for DOI-less bibliography entries in src/resolve.py,
inserted before CrossRef title-search so ML preprints resolve to their
canonical arXiv ID (10.48550/arXiv.<id>) instead of a mis-matched
journal record.

arXiv rate limit: 1 request / 3 seconds. Respected via exponential
backoff on 429 (same RETRY_MAX / RETRY_BACKOFF_BASE as other clients).
Candidates are scored via the same multi-signal blend as crossref.py
(50 % title Jaccard + 30 % author overlap + 15 % year proximity).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import httpx
import structlog

from src.clients._cache import get, put
from src.clients._common import (
    CACHE_TTL_DEFAULT_SECONDS as _CACHE_TTL_SECONDS,
)
from src.clients._common import (
    RETRY_MAX as _RETRY_MAX,
)
from src.clients._common import (
    make_cache_key,
)
from src.clients.crossref import _candidate_score
from src.models import ResolvedSource

# arXiv documents a strict rate limit of 1 request / 3 seconds and may
# temporarily IP-ban callers that exceed it. Linear 5/10/15s backoff is
# more appropriate than the shared exponential 2/4/8s of other clients
# (which target softer per-minute quotas at CrossRef / OpenAlex).
_ARXIV_BACKOFF_SECONDS = 5.0

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_BASE_URL = "https://export.arxiv.org/api/query"
_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_NOT_FOUND = ResolvedSource(found=False, doi=None, title=None, abstract=None, similarity_score=None)
_MIN_COMPOSITE_SCORE = 0.4
_HEADERS = {"User-Agent": "ScientificClaimVerifier/0.1"}
# Matches new-format arXiv IDs: YYMM.NNNNN (optionally followed by vN)
_ARXIV_ID_RE = re.compile(r"/abs/(.+?)(?:v\d+)?$")
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
_TOKEN_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _extract_arxiv_id(url: str) -> str | None:
    """Extract bare arXiv ID from an abs URL like 'http://arxiv.org/abs/2201.11903v6'."""
    match = _ARXIV_ID_RE.search(url.strip())
    return match.group(1) if match else None


def _parse_feed(xml_text: str) -> list[dict[str, Any]]:
    """Parse an arXiv Atom feed into a list of candidate dicts.

    Each dict contains: arxiv_id (str), title (str), authors (list[str]),
    year (int | None). Returns an empty list on XML parse error or empty feed.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        logger.warning("arxiv_xml_parse_error")
        return []

    entries: list[dict[str, Any]] = []
    for entry in root.findall(f"{_ATOM_NS}entry"):
        id_el = entry.find(f"{_ATOM_NS}id")
        title_el = entry.find(f"{_ATOM_NS}title")
        if id_el is None or title_el is None:
            continue

        arxiv_url = (id_el.text or "").strip()
        arxiv_id = _extract_arxiv_id(arxiv_url)
        if not arxiv_id:
            continue

        title = " ".join((title_el.text or "").split())  # normalise whitespace

        authors: list[str] = []
        for author_el in entry.findall(f"{_ATOM_NS}author"):
            name_el = author_el.find(f"{_ATOM_NS}name")
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        year: int | None = None
        pub_el = entry.find(f"{_ATOM_NS}published")
        if pub_el is not None and pub_el.text:
            with contextlib.suppress(ValueError):
                year = int(pub_el.text[:4])

        entries.append({"arxiv_id": arxiv_id, "title": title, "authors": authors, "year": year})

    return entries


def _last_name(full_name: str) -> str:
    """Return the last token of a full name, lowercased (family name heuristic)."""
    parts = full_name.strip().split()
    return parts[-1].lower() if parts else full_name.lower()


def _to_crossref_item(entry: dict[str, Any]) -> dict[str, Any]:
    """Convert a parsed arXiv entry to a CrossRef-format dict for _candidate_score.

    _candidate_score expects:
      - item["title"]: list[str]
      - item["author"]: list[{"family": str}]
      - item["issued"]: {"date-parts": [[int]]}  (optional)
      - item["DOI"]: str (optional but gives +0.05 bonus)
      - item["type"]: str (absent -> 0.0 bonus, avoids -0.25 posted-content penalty)
    """
    author_list = [{"family": _last_name(name)} for name in entry["authors"]]
    item: dict[str, Any] = {
        "title": [entry["title"]],
        "author": author_list,
        "DOI": f"10.48550/arXiv.{entry['arxiv_id']}",
        # Deliberately omit "type" so work_type_score returns 0.0.
        # arXiv preprints would map to "posted-content" (-0.25), but
        # that penalty would mask legitimate ML preprint matches.
    }
    year = entry.get("year")
    if year is not None:
        item["issued"] = {"date-parts": [[year]]}
    return item


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_paper_by_title_authors(
    title: str,
    authors: list[str],
    year: int | None,
    *,
    db_path: Path | None = None,
    timeout: float = 15.0,
) -> ResolvedSource:
    """Return the best arXiv match for a title-authors-year query.

    Builds an arXiv API search query from the title's salient tokens
    and the first 2 surnames, then ranks results by title Jaccard +
    author overlap + year proximity using the same multi-signal blend
    as _candidate_score in crossref.py (50/30/15 weights). Returns
    ResolvedSource(found=False, ...) on miss or when no candidate
    exceeds a minimum composite score (0.4).
    """
    if not title:
        return _NOT_FOUND

    resolved_db_path = db_path if db_path is not None else _default_db_path()
    key = make_cache_key("arxiv_search", title, str(tuple(authors)), str(year))

    cached = get(resolved_db_path, key)
    if cached is not None:
        logger.debug("arxiv_cache_hit", title=title)
        data: dict[str, Any] = json.loads(cached)
        valid_keys = {f.name for f in dataclasses.fields(ResolvedSource)}
        clean = {k: v for k, v in data.items() if k in valid_keys}
        return ResolvedSource(**clean)

    # Build the arXiv search query: ti:<tokens> AND au:<first_surname>
    title_tokens = [
        t for t in _TOKEN_RE.findall(title.lower()) if t not in _STOPWORDS and len(t) > 1
    ][:6]
    surnames = [_last_name(a) for a in authors[:2] if a and a.lower() != "et al."]

    ti_part = "+".join(title_tokens) if title_tokens else "unknown"
    search_query = f"ti:{ti_part}+AND+au:{surnames[0]}" if surnames else f"ti:{ti_part}"

    url = f"{_BASE_URL}?search_query={search_query}&start=0&max_results=5"

    # Build scoring query: full title + surnames + year (mirrors _build_crossref_query_from_bib)
    scoring_parts = title.split()[:8] + [a for a in authors[:3] if a.lower() != "et al."]
    if year is not None:
        scoring_parts.append(str(year))
    scoring_query = " ".join(scoring_parts)

    for attempt in range(1, _RETRY_MAX + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.get(url, headers=_HEADERS)

            if response.status_code == 429:
                wait = _ARXIV_BACKOFF_SECONDS * attempt
                logger.warning("arxiv_rate_limited", attempt=attempt, wait_seconds=wait)
                if attempt < _RETRY_MAX:
                    time.sleep(wait)
                    continue
                logger.error("arxiv_max_retries_exceeded", title=title)
                return _NOT_FOUND

            response.raise_for_status()
            entries = _parse_feed(response.text)

            if not entries:
                put(
                    resolved_db_path,
                    key,
                    json.dumps(dataclasses.asdict(_NOT_FOUND)),
                    _CACHE_TTL_SECONDS,
                )
                logger.debug("arxiv_no_results", title=title)
                return _NOT_FOUND

            best_composite = 0.0
            best_title_score = 0.0
            best_entry: dict[str, Any] | None = None

            for i, entry in enumerate(entries):
                item = _to_crossref_item(entry)
                composite, title_jaccard, _ = _candidate_score(scoring_query, item, i)
                if composite > best_composite:
                    best_composite = composite
                    best_title_score = title_jaccard
                    best_entry = entry

            if best_entry is None or best_composite < _MIN_COMPOSITE_SCORE:
                logger.debug(
                    "arxiv_no_qualifying_candidate",
                    title=title,
                    best_score=best_composite,
                    threshold=_MIN_COMPOSITE_SCORE,
                )
                put(
                    resolved_db_path,
                    key,
                    json.dumps(dataclasses.asdict(_NOT_FOUND)),
                    _CACHE_TTL_SECONDS,
                )
                return _NOT_FOUND

            arxiv_id: str = best_entry["arxiv_id"]
            doi = f"10.48550/arXiv.{arxiv_id}"
            resolved = ResolvedSource(
                found=True,
                doi=doi,
                title=best_entry["title"],
                abstract=None,
                similarity_score=best_composite,
                title_match_score=best_title_score,
                resolution_low_confidence=best_title_score < 0.15,
            )

            put(
                resolved_db_path,
                key,
                json.dumps(dataclasses.asdict(resolved)),
                _CACHE_TTL_SECONDS,
            )
            logger.info(
                "arxiv_resolved",
                arxiv_id=arxiv_id,
                doi=doi,
                title=resolved.title,
                composite_score=best_composite,
            )
            return resolved

        except httpx.HTTPStatusError as exc:
            logger.error("arxiv_http_error", title=title, status=exc.response.status_code)
            return _NOT_FOUND
        except httpx.RequestError as exc:
            logger.error("arxiv_request_error", title=title, error=str(exc))
            return _NOT_FOUND
        except Exception as exc:
            logger.error("arxiv_unexpected_error", title=title, error=str(exc))
            return _NOT_FOUND

    return _NOT_FOUND


def _default_db_path() -> Path:
    from src.clients._cache import default_db_path

    return default_db_path()
