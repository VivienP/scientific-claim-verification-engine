"""Batch citation resolution via OpenAlex with CrossRef fallback and retraction check."""

from __future__ import annotations

import dataclasses
import time
import uuid
from pathlib import Path

import structlog

from src.bibliography import BibEntry
from src.clients import crossref as _crossref
from src.clients import pubmed as _pubmed
from src.clients.openalex import search_paper
from src.models import Claim, ProvenanceStep, ResolvedSource
from src.resolve_utils import (
    _NOT_FOUND,
    _TITLE_MATCH_ACCEPT_THRESHOLD,
    _best_bib_match,
    _bibliography_title_match_score,
    _build_crossref_query_from_bib,
    _build_query,
    _build_query_from_bib,
    _hash,
    _normalize_pmcid,
    _source_from_bib_entry,
)

logger: structlog.BoundLogger = structlog.get_logger(__name__)


def _enrich_via_pubmed(
    source: ResolvedSource,
    *,
    db_path: Path | None,
) -> ResolvedSource:
    """Populate pmcid and (if missing) abstract from PubMed via DOI->PMID->record.

    Bug A fix (S1-P1-A): the previous `_enrich_abstract_via_pubmed` short-circuited
    whenever CrossRef had any abstract, silently dropping the pmcid. PubMed often
    has the pmcid even when CrossRef does not (NIH-deposit OA mirrors), and
    propagating it unblocks the PMC fulltext path in `fetch_fulltext`. This
    function:

    * fires for any found source with a DOI that lacks either abstract OR pmcid;
    * uses `find_pmid_by_doi` -> `fetch_record` (preserving the full record);
    * preserves the existing abstract when CrossRef's is longer than PubMed's,
      and always backfills pmcid when PubMed exposes one.

    Returns source unchanged on any miss (caching layer keeps repeat lookups cheap).
    """
    if not source.found or source.doi is None:
        return source
    if source.abstract and source.pmcid:
        return source
    pmid = _pubmed.find_pmid_by_doi(source.doi, db_path=db_path)
    if pmid is None:
        return source
    record = _pubmed.fetch_record(pmid, db_path=db_path)
    if record is None:
        return source
    new_abstract = _pick_longer_abstract(source.abstract, record.abstract)
    new_pmcid = source.pmcid or _normalize_pmcid(record.pmcid)
    if new_abstract == source.abstract and new_pmcid == source.pmcid:
        return source
    logger.info(
        "pubmed_record_enriched",
        doi=source.doi,
        pmid=pmid,
        abstract_added=source.abstract is None and record.abstract is not None,
        pmcid_added=source.pmcid is None and new_pmcid is not None,
    )
    return dataclasses.replace(source, abstract=new_abstract, pmcid=new_pmcid)


def _pick_longer_abstract(existing: str | None, candidate: str | None) -> str | None:
    """Return whichever abstract is more informative (longer, prefers existing on ties)."""
    if not existing:
        return candidate
    if not candidate:
        return existing
    return existing if len(existing) >= len(candidate) else candidate


def _resolve_via_bib_doi(entry: BibEntry, *, db_path: Path | None) -> ResolvedSource:
    """Look up a bibliography-provided DOI directly via CrossRef.

    Bypasses the lossy author/year search when the bibliography already
    contains the canonical DOI. Returns _NOT_FOUND if the DOI cannot be
    resolved (CrossRef miss, network error, etc.).
    """
    if entry.doi is None:
        return _NOT_FOUND
    cf_source = _crossref.fetch_work_by_doi(entry.doi, db_path=db_path)
    if cf_source.found:
        return dataclasses.replace(
            cf_source,
            pmcid=_normalize_pmcid(entry.pmcid) or cf_source.pmcid,
            similarity_score=1.0,
            title_match_score=1.0 if entry.title else cf_source.title_match_score,
            resolution_low_confidence=False,
        )
    return _source_from_bib_entry(entry)


def _resolve_via_bib_pmid(entry: BibEntry, *, db_path: Path | None) -> ResolvedSource:
    if entry.pmid is None:
        if entry.pmcid is not None:
            return _source_from_bib_entry(entry)
        return _NOT_FOUND
    record = _pubmed.fetch_record(entry.pmid, db_path=db_path)
    if record is None:
        return _source_from_bib_entry(entry)
    return _source_from_bib_entry(
        entry,
        abstract=record.abstract,
        doi=record.doi,
        pmcid=record.pmcid,
    )


def _resolve_via_bib_crossref_title(
    entry: BibEntry, claim: Claim, *, db_path: Path | None
) -> ResolvedSource:
    if not entry.title:
        return _NOT_FOUND
    source = _crossref.search_paper(_build_crossref_query_from_bib(entry, claim), db_path=db_path)
    if not source.found:
        return _NOT_FOUND
    title_score = _bibliography_title_match_score(entry.title, source.title)
    if title_score is None or title_score < _TITLE_MATCH_ACCEPT_THRESHOLD:
        logger.debug(
            "crossref_title_rejected",
            bib_number=entry.number,
            title=source.title,
            title_match_score=title_score,
        )
        return _NOT_FOUND
    return dataclasses.replace(
        source,
        similarity_score=source.similarity_score if source.similarity_score is not None else 1.0,
        title_match_score=title_score,
        resolution_low_confidence=False,
    )


def _resolve_via_pubmed_title(entry: BibEntry, *, db_path: Path | None) -> ResolvedSource:
    if not entry.title:
        return _NOT_FOUND
    pmid = _pubmed.find_pmid_by_title(entry.title, year=entry.year, db_path=db_path)
    if pmid is None:
        return _NOT_FOUND
    record = _pubmed.fetch_record(pmid, db_path=db_path)
    if record is None:
        return _NOT_FOUND
    return _source_from_bib_entry(
        entry,
        abstract=record.abstract,
        doi=record.doi,
        pmcid=record.pmcid,
    )


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
    CrossRef. Otherwise, strong bibliography-title CrossRef and PubMed
    matches are tried before the existing OpenAlex search + CrossRef fallback
    chain. The bibliography path activates per-reference routing for
    multi-citation manuscripts where the extractor flattened several cited
    refs into a single Claim.

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
                source = _resolve_via_bib_doi(bib_match, db_path=db_path)
                if source.found:
                    logger.info(
                        "resolve_via_bib_doi",
                        claim_id=claim.claim_id,
                        bib_number=bib_match.number,
                        doi=bib_match.doi,
                    )
            if not source.found and bib_match is not None and (bib_match.pmid or bib_match.pmcid):
                source = _resolve_via_bib_pmid(bib_match, db_path=db_path)
                if source.found:
                    logger.info(
                        "resolve_via_bib_pmid",
                        claim_id=claim.claim_id,
                        bib_number=bib_match.number,
                        pmid=bib_match.pmid,
                        pmcid=bib_match.pmcid,
                    )
            if not source.found and bib_match is not None and bib_match.title:
                source = _resolve_via_bib_crossref_title(bib_match, claim, db_path=db_path)
                if source.found:
                    logger.info(
                        "resolve_via_bib_crossref_title",
                        claim_id=claim.claim_id,
                        bib_number=bib_match.number,
                        doi=source.doi,
                    )
            if not source.found and bib_match is not None and bib_match.title:
                source = _resolve_via_pubmed_title(bib_match, db_path=db_path)
                if source.found:
                    logger.info(
                        "resolve_via_pubmed_title",
                        claim_id=claim.claim_id,
                        bib_number=bib_match.number,
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

        source = _enrich_via_pubmed(source, db_path=db_path)

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
