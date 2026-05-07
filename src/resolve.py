"""Batch citation resolution via OpenAlex with CrossRef fallback and retraction check."""

from __future__ import annotations

import dataclasses
import hashlib
import time
import uuid
from pathlib import Path

import structlog

from src.bibliography import BibEntry
from src.clients import crossref as _crossref
from src.clients import pubmed as _pubmed
from src.clients.openalex import search_paper
from src.models import Claim, ProvenanceStep, ResolvedSource

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_NOT_FOUND = ResolvedSource(found=False, doi=None, title=None, abstract=None, similarity_score=None)


def _hash(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def _build_query(claim: Claim) -> str:
    parts: list[str] = (
        claim.cited_authors[:3]
        + ([str(claim.cited_year)] if claim.cited_year else [])
        + claim.claim_text.split()[:5]
    )
    return " ".join(parts)


def _normalize_surname(name: str) -> str:
    return name.strip().lower().replace(".", "")


def _bib_entry_match_score(claim: Claim, entry: BibEntry) -> int:
    """Score a candidate bibliography entry against a claim.

    Returns: 2 + author_overlap if year matches, 0 + author_overlap otherwise.
    Returns 0 (no match) when there is no author overlap. Treats "et al." as
    a wildcard equal to a single match.
    """
    claim_authors = {_normalize_surname(a) for a in claim.cited_authors if a}
    entry_authors = {_normalize_surname(a) for a in entry.authors if a and a.lower() != "et al."}
    if not claim_authors or not entry_authors:
        return 0
    overlap = len(claim_authors & entry_authors)
    if overlap == 0:
        return 0
    year_bonus = 0
    if claim.cited_year is not None and entry.year is not None:
        if claim.cited_year == entry.year:
            year_bonus = 2
        elif abs(claim.cited_year - entry.year) <= 1:
            year_bonus = 1
    return year_bonus + overlap


def _best_bib_match(claim: Claim, bibliography: dict[int, BibEntry]) -> BibEntry | None:
    """Pick the bibliography entry that best matches a claim's cited authors+year."""
    best: BibEntry | None = None
    best_score = 0
    for entry in bibliography.values():
        score = _bib_entry_match_score(claim, entry)
        if score > best_score:
            best_score = score
            best = entry
    if best_score < 1:
        return None
    return best


def _build_query_from_bib(entry: BibEntry, claim: Claim) -> str:
    """Build a richer search query using bibliography metadata."""
    parts: list[str] = []
    if entry.title:
        # Use first 8 title tokens to keep the query precise.
        parts.extend(entry.title.split()[:8])
    surnames = [a for a in entry.authors if a.lower() != "et al."][:3]
    parts.extend(surnames)
    if entry.year is not None:
        parts.append(str(entry.year))
    if not parts:
        return _build_query(claim)
    return " ".join(parts)


def _enrich_abstract_via_pubmed(
    source: ResolvedSource,
    *,
    db_path: Path | None,
) -> ResolvedSource:
    """Fill in source.abstract from PubMed when upstream resolvers returned null.

    Fires only when source is found, has a DOI, and has no abstract. Looks up
    the PMID via DOI and fetches the PubMed-formatted abstract. Returns the
    source unchanged on any failure path (caching layer in pubmed.py records
    negatives so repeated lookups stay cheap).
    """
    if not source.found or source.doi is None or source.abstract:
        return source
    abstract = _pubmed.fetch_abstract_by_doi(source.doi, db_path=db_path)
    if abstract is None:
        return source
    logger.info("pubmed_abstract_enriched", doi=source.doi, length=len(abstract))
    return dataclasses.replace(source, abstract=abstract)


def _resolve_via_bib_doi(bib_doi: str, *, db_path: Path | None) -> ResolvedSource:
    """Look up a bibliography-provided DOI directly via CrossRef.

    Bypasses the lossy author/year search when the bibliography already
    contains the canonical DOI. Returns _NOT_FOUND if the DOI cannot be
    resolved (CrossRef miss, network error, etc.).
    """
    cf_source = _crossref.search_paper(bib_doi, db_path=db_path)
    if cf_source.found:
        return cf_source
    return _NOT_FOUND


def resolve_citations(
    claims: list[Claim],
    *,
    api_key: str | None = None,
    db_path: Path | None = None,
    bibliography: dict[int, BibEntry] | None = None,
) -> tuple[dict[str, ResolvedSource], list[ProvenanceStep]]:
    """Resolve each claim's cited source via OpenAlex.

    When `bibliography` is provided, each claim is first matched to the
    best-fitting bibliography entry by author + year overlap. If a match
    exists and the bibliography has a DOI, that DOI is used directly via
    CrossRef. Otherwise, a richer query is built from the bibliography's
    title/authors/year before falling back to the existing OpenAlex search
    + CrossRef fallback chain. The bibliography path activates per-reference
    routing for multi-citation manuscripts where the extractor flattened
    several cited refs into a single Claim.

    Returns a dict keyed by claim_id (entry present for EVERY claim, even unresolved).
    Returns one ProvenanceStep per claim (operation="resolve", model_id=None).
    Claims with cited_authors=[] or (cited_year=None and no bib match)
    short-circuit to ResolvedSource(found=False) without HTTP call.
    """
    sources: dict[str, ResolvedSource] = {}
    steps: list[ProvenanceStep] = []

    for claim in claims:
        ts = time.time()
        bib_match: BibEntry | None = None
        if bibliography:
            bib_match = _best_bib_match(claim, bibliography)

        if not claim.cited_authors and bib_match is None:
            source = _NOT_FOUND
            logger.debug("resolve_skipped_no_citation", claim_id=claim.claim_id)
        elif claim.cited_year is None and bib_match is None:
            source = _NOT_FOUND
            logger.debug("resolve_skipped_no_year_no_bib_match", claim_id=claim.claim_id)
        else:
            source = _NOT_FOUND
            if bib_match is not None and bib_match.doi:
                source = _resolve_via_bib_doi(bib_match.doi, db_path=db_path)
                if source.found:
                    logger.info(
                        "resolve_via_bib_doi",
                        claim_id=claim.claim_id,
                        bib_number=bib_match.number,
                        doi=bib_match.doi,
                    )
            if not source.found:
                query = (
                    _build_query_from_bib(bib_match, claim)
                    if bib_match is not None
                    else _build_query(claim)
                )
                source = search_paper(query, api_key=api_key, db_path=db_path)
                if not source.found:
                    cf_source = _crossref.search_paper(query, db_path=db_path)
                    if cf_source.found:
                        source = cf_source
                        logger.info("crossref_fallback_success", claim_id=claim.claim_id)

        if source.doi is not None:
            retracted = _crossref.check_retraction(source.doi, db_path=db_path)
            if retracted:
                logger.warning("retraction_detected", claim_id=claim.claim_id, doi=source.doi)
            source = dataclasses.replace(source, retraction_status=retracted)

        source = _enrich_abstract_via_pubmed(source, db_path=db_path)

        sources[claim.claim_id] = source
        steps.append(
            ProvenanceStep(
                step_id=str(uuid.uuid4()),
                claim_id=claim.claim_id,
                operation="resolve",
                input_hash=_hash(repr(claim)),
                output_hash=_hash(repr(source)),
                model_id=None,
                timestamp=ts,
                tokens_in=None,
                tokens_out=None,
                cache_hit=None,
                confidence=source.similarity_score,
            )
        )

    return sources, steps
