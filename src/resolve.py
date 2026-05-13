"""Batch citation resolution via OpenAlex with CrossRef fallback and retraction check."""

from __future__ import annotations

import dataclasses
import time
import uuid
from pathlib import Path

import structlog

from src.bibliography import BibEntry
from src.clients import arxiv as _arxiv
from src.clients import crossref as _crossref
from src.clients import openalex as _openalex
from src.clients import pubmed as _pubmed
from src.clients.openalex import search_paper
from src.models import (
    CandidateResolution,
    Claim,
    ProvenanceStep,
    ResolutionVerdict,
    ResolvedSource,
    ResolvedSourceSet,
)
from src.resolve_enrich import _enrich_via_europepmc, _enrich_via_pubmed
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
    fold_candidates_into_verdict,
)

logger: structlog.BoundLogger = structlog.get_logger(__name__)


def _is_citing_paper_doi(candidate: str | None, citing: str | None) -> bool:
    """Return True iff `candidate` is the citing paper's own DOI.

    Per the DOI handbook §2.4, DOIs are case-insensitive. We compare in
    lowercase so casing drift between bibliography text and resolver
    output cannot let a self-citation slip through.

    A claim cannot legally cite the paper that contains it. Treat any
    such resolution as a structural error and reject. Without this guard,
    OpenAlex/CrossRef searches whose query happens to contain the citing
    paper's name (e.g. claim text "Valsci integrates...") return the
    citing paper itself, and the verifier then compares the claim against
    the citing text, producing tautological 'supported' verdicts.
    """
    if not candidate or not citing:
        return False
    return candidate.strip().lower() == citing.strip().lower()


def _reject_if_citing_paper(
    source: ResolvedSource,
    citing_paper_doi: str | None,
    *,
    claim_id: str,
) -> ResolvedSource:
    """Replace `source` with `_NOT_FOUND` when it matches the citing paper.

    Logs a warning so silent recursion failures cannot hide in the noise.
    """
    if _is_citing_paper_doi(source.doi, citing_paper_doi):
        logger.warning(
            "resolve_rejected_citing_paper_self_match",
            claim_id=claim_id,
            doi=source.doi,
        )
        return _NOT_FOUND
    return source


def _attach_resolution_verdict(
    source: ResolvedSource,
    *,
    used_bib_doi: bool,
    query: str | None,
    db_path: Path | None,
) -> ResolvedSource:
    """Compute a ``ResolutionVerdict`` and attach it to ``source``.

    Two paths:

    * ``used_bib_doi=True`` — the bibliography already supplied a DOI that
      CrossRef confirmed. The bib acts as the second authority; status is
      ``corroborated`` trivially without extra HTTP calls (the cost-
      discipline rule of the verdict design).
    * ``used_bib_doi=False`` — fuzzy fallback path. Collect candidates from
      CrossRef, OpenAlex, and (when a DOI was settled) PubMed via their
      ``find_candidate`` APIs, then fold into a verdict. Disagreement
      between clients on DOI surfaces as ``disputed`` — the signal that the
      downstream policy uses to gate verification.

    A source with ``found=False`` returns unchanged: no verdict is meaningful
    when no candidate exists. When the source carries
    ``resolution_low_confidence=True`` and the fold yields
    ``single_source_only``, the status is upgraded to ``low_confidence`` so
    the policy gate sees the weak-signal flag.
    """
    if not source.found:
        return source

    if used_bib_doi and source.doi is not None:
        bib_candidate = CandidateResolution(
            client="crossref",
            doi=source.doi,
            title=source.title,
            year=None,
            first_author=None,
            venue=None,
        )
        verdict = ResolutionVerdict(
            status="corroborated",
            candidates=(bib_candidate,),
            agreement_signals=("doi",),
        )
        return dataclasses.replace(source, resolution_verdict=verdict)

    candidates: list[CandidateResolution] = []
    if query:
        cf_cand = _crossref.find_candidate(query, db_path=db_path)
        if cf_cand is not None:
            candidates.append(cf_cand)
        oa_cand = _openalex.find_candidate(query, db_path=db_path)
        if oa_cand is not None:
            candidates.append(oa_cand)

    if not candidates:
        return source

    # CrossRef + OpenAlex are sufficient for the cross-source signal: when
    # they agree on DOI the verdict is corroborated; when they disagree it
    # is disputed. PubMed is exposed as ``find_candidate_by_doi`` for
    # callers that want a third opinion, but adding a third HTTP call per
    # claim here doubles resolver latency without changing the policy
    # outcome — the auditor can inspect ``resolution_verdict.candidates``
    # and pull a PubMed candidate by hand when the verdict is disputed.
    verdict = fold_candidates_into_verdict(tuple(candidates))

    if verdict.status == "single_source_only" and source.resolution_low_confidence:
        verdict = ResolutionVerdict(
            status="low_confidence",
            candidates=verdict.candidates,
            agreement_signals=(),
        )

    return dataclasses.replace(source, resolution_verdict=verdict)


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


def _resolve_via_arxiv_title(entry: BibEntry, *, db_path: Path | None) -> ResolvedSource:
    """Search arXiv by title + authors for a DOI-less bibliography entry.

    Inserted before CrossRef title-search so ML/AI preprints (which lack a
    journal DOI in the bibliography) are caught at their natural authority
    rather than mis-matched to an unrelated journal record.
    """
    if not entry.title:
        return _NOT_FOUND
    return _arxiv.find_paper_by_title_authors(
        entry.title,
        entry.authors,
        entry.year,
        db_path=db_path,
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


def _resolve_for_bib_entry(
    bib_entry: BibEntry,
    claim: Claim,
    *,
    api_key: str | None,
    db_path: Path | None,
) -> ResolvedSource:
    """Run the full per-entry resolution + enrichment chain for one bibliography ref.

    S2-P4 helper: each entry in a multi-citation `[81-83]` claim is resolved
    independently through the same DOI -> PMID -> Title -> OpenAlex chain that
    `resolve_citations` runs inline for single-marker claims. Enrichments
    (PubMed record, Europe PMC) and the retraction check are applied per-entry.
    Returns `_NOT_FOUND` when every step misses.
    """
    source = _NOT_FOUND
    used_bib_doi = False
    if bib_entry.doi:
        source = _resolve_via_bib_doi(bib_entry, db_path=db_path)
        if source.found:
            used_bib_doi = True
    if not source.found and (bib_entry.pmid or bib_entry.pmcid):
        source = _resolve_via_bib_pmid(bib_entry, db_path=db_path)
    if not source.found and bib_entry.title:
        source = _resolve_via_arxiv_title(bib_entry, db_path=db_path)
    if not source.found and bib_entry.title:
        source = _resolve_via_bib_crossref_title(bib_entry, claim, db_path=db_path)
    if not source.found and bib_entry.title:
        source = _resolve_via_pubmed_title(bib_entry, db_path=db_path)
    if not source.found:
        query = _build_query_from_bib(bib_entry, claim)
        source = search_paper(query, api_key=api_key, db_path=db_path)
        if not source.found:
            cf_source = _crossref.search_paper(query, db_path=db_path)
            if cf_source.found:
                source = cf_source

    if source.doi is not None:
        retracted = _crossref.check_retraction(source.doi, db_path=db_path)
        source = dataclasses.replace(source, retraction_status=retracted)

    source = _enrich_via_pubmed(source, db_path=db_path)
    source = _enrich_via_europepmc(source, db_path=db_path)
    verdict_query = _build_query_from_bib(bib_entry, claim) if not used_bib_doi else None
    source = _attach_resolution_verdict(
        source,
        used_bib_doi=used_bib_doi,
        query=verdict_query,
        db_path=db_path,
    )
    return source


def resolve_citations_multi(
    claims: list[Claim],
    *,
    api_key: str | None = None,
    db_path: Path | None = None,
    bibliography: dict[int, BibEntry] | None = None,
    citing_paper_doi: str | None = None,
) -> tuple[dict[str, ResolvedSourceSet], list[ProvenanceStep]]:
    """Resolve every bibliography marker on each claim into a ResolvedSourceSet.

    For multi-citation claims `[81-83]` or `[99, 100]`, this resolves each
    bibliography marker independently (via `_resolve_for_bib_entry`). Single-
    citation or markerless claims fall back to the same chain as
    `resolve_citations` and yield a 1-element set.

    Used by the benchmark runner when `len(claim.citation_markers) > 1`. The
    primary single-source API (`resolve_citations`) is unchanged.

    Returns one ProvenanceStep per claim (operation="resolve"). When a multi-
    citation claim is processed, the step's `output_hash` covers the full
    set so re-runs can detect changes in any constituent source.
    """
    sets: dict[str, ResolvedSourceSet] = {}
    steps: list[ProvenanceStep] = []

    for claim in claims:
        ts = time.time()
        markers = list(claim.citation_markers)
        per_marker_sources: list[ResolvedSource] = []

        if bibliography and markers:
            for marker in markers:
                entry = bibliography.get(marker)
                if entry is None:
                    continue
                source = _resolve_for_bib_entry(entry, claim, api_key=api_key, db_path=db_path)
                source = _reject_if_citing_paper(source, citing_paper_doi, claim_id=claim.claim_id)
                per_marker_sources.append(source)
                logger.debug(
                    "multi_resolve_marker",
                    claim_id=claim.claim_id,
                    marker=marker,
                    found=source.found,
                    doi=source.doi,
                )

        if not per_marker_sources:
            # No bibliography or no resolvable markers: fall back to single-source
            # via the existing resolve_citations path so semantics align with
            # the legacy primary-source API.
            singles, single_steps = resolve_citations(
                [claim],
                api_key=api_key,
                db_path=db_path,
                bibliography=bibliography,
                citing_paper_doi=citing_paper_doi,
            )
            per_marker_sources = [singles[claim.claim_id]]
            steps.extend(single_steps)
            sets[claim.claim_id] = ResolvedSourceSet(
                sources=tuple(per_marker_sources),
                citation_markers=tuple(markers),
            )
            continue

        rs_set = ResolvedSourceSet(
            sources=tuple(per_marker_sources),
            citation_markers=tuple(markers),
        )
        sets[claim.claim_id] = rs_set
        steps.append(
            ProvenanceStep(
                step_id=str(uuid.uuid4()),
                claim_id=claim.claim_id,
                operation="resolve",
                input_hash=_hash(repr(claim)),
                output_hash=_hash(repr(rs_set)),
                model_id=None,
                timestamp=ts,
                tokens_in=None,
                tokens_out=None,
                cache_hit=None,
                confidence=rs_set.primary().similarity_score,
            )
        )

    return sets, steps


def resolve_citations(
    claims: list[Claim],
    *,
    api_key: str | None = None,
    db_path: Path | None = None,
    bibliography: dict[int, BibEntry] | None = None,
    citing_paper_doi: str | None = None,
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
        used_bib_doi = False
        if bibliography:
            bib_match = _best_bib_match(claim, bibliography)

        if not claim.cited_authors and bib_match is None:
            source = _NOT_FOUND
            if claim.citation_markers:
                logger.debug(
                    "resolve_skipped_multi_marker_no_author",
                    claim_id=claim.claim_id,
                    markers=list(claim.citation_markers),
                )
            else:
                logger.debug("resolve_skipped_no_citation", claim_id=claim.claim_id)
        elif claim.cited_year is None and bib_match is None:
            source = _NOT_FOUND
            logger.debug("resolve_skipped_no_year_no_bib_match", claim_id=claim.claim_id)
        else:
            source = _NOT_FOUND
            if bib_match is not None and bib_match.doi:
                source = _resolve_via_bib_doi(bib_match, db_path=db_path)
                if source.found:
                    used_bib_doi = True
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
                source = _resolve_via_arxiv_title(bib_match, db_path=db_path)
                if source.found:
                    logger.info(
                        "resolve_via_arxiv_title",
                        claim_id=claim.claim_id,
                        bib_number=bib_match.number,
                        doi=source.doi,
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

        # Reject self-citations (citing paper resolved to itself). Must run
        # AFTER all resolution paths and BEFORE retraction/enrichment so we
        # do not waste HTTP calls on a structurally-impossible match.
        source = _reject_if_citing_paper(source, citing_paper_doi, claim_id=claim.claim_id)

        if source.doi is not None:
            retracted = _crossref.check_retraction(source.doi, db_path=db_path)
            if retracted:
                logger.warning("retraction_detected", claim_id=claim.claim_id, doi=source.doi)
            source = dataclasses.replace(source, retraction_status=retracted)

        source = _enrich_via_pubmed(source, db_path=db_path)
        source = _enrich_via_europepmc(source, db_path=db_path)

        if not used_bib_doi:
            verdict_query: str | None = (
                _build_query_from_bib(bib_match, claim)
                if bib_match is not None
                else _build_query(claim)
            )
        else:
            verdict_query = None
        source = _attach_resolution_verdict(
            source,
            used_bib_doi=used_bib_doi,
            query=verdict_query,
            db_path=db_path,
        )

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
