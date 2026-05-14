"""Pure helpers for citation resolution: text normalisation, scoring, matching, query building."""

from __future__ import annotations

import hashlib
import re
import unicodedata

from src.bibliography import BibEntry
from src.models import CandidateResolution, Claim, ResolutionVerdict, ResolvedSource

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
_MIN_TITLE_TOKENS_FOR_TITLE_ONLY_MATCH = 4
_TITLE_MATCH_ACCEPT_THRESHOLD = 0.75
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


def _hash(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def _normalise_search_text(text: str) -> str:
    translated = text.translate(_SEARCH_TRANSLATION)
    decomposed = unicodedata.normalize("NFKD", translated)
    ascii_text = decomposed.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_text.split())


def _build_query(claim: Claim) -> str:
    parts: list[str] = (
        claim.cited_authors[:3]
        + ([str(claim.cited_year)] if claim.cited_year else [])
        + claim.claim_text.split()[:5]
    )
    return _normalise_search_text(" ".join(parts))


def _normalize_surname(name: str) -> str:
    return name.strip().lower().replace(".", "")


def _normalize_pmcid(pmcid: str | None) -> str | None:
    if not pmcid:
        return None
    cleaned = pmcid.strip().rstrip("/").split("/")[-1]
    if cleaned.upper().startswith("PMC"):
        return "PMC" + cleaned[3:]
    if cleaned.isdigit():
        return f"PMC{cleaned}"
    return cleaned


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
    marker_candidates = [
        bibliography[marker] for marker in claim.citation_markers if marker in bibliography
    ]
    if len(marker_candidates) == 1:
        return marker_candidates[0]

    best: BibEntry | None = None
    best_score = 0
    candidates = marker_candidates or list(bibliography.values())
    for entry in candidates:
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
    return _normalise_search_text(" ".join(parts))


def _build_crossref_query_from_bib(entry: BibEntry, claim: Claim) -> str:
    """Build a CrossRef bibliographic query with the complete reference title."""
    parts: list[str] = []
    if entry.title:
        parts.append(entry.title)
    surnames = [a for a in entry.authors if a.lower() != "et al."][:3]
    parts.extend(surnames)
    if entry.year is not None:
        parts.append(str(entry.year))
    if not parts:
        return _build_query(claim)
    return _normalise_search_text(" ".join(parts))


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(_normalise_search_text(text).lower())
        if token not in _STOPWORDS
    }


def _bibliography_title_match_score(
    entry_title: str | None,
    resolved_title: str | None,
) -> float | None:
    if not entry_title or not resolved_title:
        return None
    if (
        _normalise_search_text(entry_title).lower()
        == _normalise_search_text(resolved_title).lower()
    ):
        return 1.0
    entry_tokens = _content_tokens(entry_title)
    if len(entry_tokens) < _MIN_TITLE_TOKENS_FOR_TITLE_ONLY_MATCH:
        return None
    resolved_tokens = _content_tokens(resolved_title)
    if not resolved_tokens:
        return 0.0
    return len(entry_tokens & resolved_tokens) / len(entry_tokens)


_FUZZY_AGREEMENT_MIN_FIELDS = 2


def _normalised_venue(venue: str | None) -> str | None:
    if not venue:
        return None
    return _normalise_search_text(venue).lower()


def _fuzzy_agreement_signals(
    candidates: tuple[CandidateResolution, ...],
) -> tuple[str, ...]:
    """Return which of (year, first_author, venue) agree across candidates.

    A field counts toward agreement when at least two candidates have a
    non-None value for it AND every such value is equal (after normalisation
    for venue). Year/first_author are compared exactly; venue is collapsed
    via ``_normalised_venue``. Returns the tuple of agreed field names; empty
    when fewer than ``_FUZZY_AGREEMENT_MIN_FIELDS`` (2 of 3) agree.
    """
    agreed: list[str] = []

    years = [c.year for c in candidates if c.year is not None]
    if len(years) >= 2 and len(set(years)) == 1:
        agreed.append("year")

    authors = [c.first_author for c in candidates if c.first_author]
    if len(authors) >= 2 and len(set(authors)) == 1:
        agreed.append("first_author")

    venues = [v for v in (_normalised_venue(c.venue) for c in candidates) if v]
    if len(venues) >= 2 and len(set(venues)) == 1:
        agreed.append("venue")

    if len(agreed) >= _FUZZY_AGREEMENT_MIN_FIELDS:
        return tuple(agreed)
    return ()


def fold_candidates_into_verdict(
    candidates: tuple[CandidateResolution, ...],
) -> ResolutionVerdict:
    """Fold per-client candidates into a ``ResolutionVerdict``.

    Pure function. The verdict statuses encode resolver confidence as a
    policy input:

    * **corroborated** — at least two candidates agree on DOI (case-
      insensitive) OR on (year, first_author, venue) within tolerance.
      Strongest signal that the resolver landed on the right paper.
    * **disputed** — at least two candidates disagree on DOI AND the
      (year, first_author, venue) fuzzy test fails. Policy treats this as
      ``unverifiable`` because a wrong-paper resolution is a worse failure
      mode than no resolution at all.
    * **single_source_only** — fewer than two candidates returned, so
      cross-source agreement cannot be tested. Policy gates fall back to
      depth/access criteria.
    * **low_confidence** is NOT emitted by this function — the caller sets
      it on a single candidate when individual signals are weak (OpenAlex
      title overlap below ``_LOW_TITLE_MATCH_THRESHOLD``).

    ``agreement_signals`` carries the field names that drove a corroborated
    verdict so the auditor can read e.g. ``("doi",)`` (DOIs match) or
    ``("year", "first_author", "venue")`` (preprint vs published-version
    duplicate listing of the same work).
    """
    if len(candidates) < 2:
        return ResolutionVerdict(
            status="single_source_only",
            candidates=candidates,
            agreement_signals=(),
        )

    dois_present = [c.doi.lower() for c in candidates if c.doi]
    doi_unanimous = len(dois_present) >= 2 and len(set(dois_present)) == 1

    fuzzy_signals = _fuzzy_agreement_signals(candidates)

    if doi_unanimous:
        signals = ("doi", *(s for s in fuzzy_signals if s != "doi"))
        return ResolutionVerdict(
            status="corroborated",
            candidates=candidates,
            agreement_signals=signals,
        )

    if fuzzy_signals:
        return ResolutionVerdict(
            status="corroborated",
            candidates=candidates,
            agreement_signals=fuzzy_signals,
        )

    return ResolutionVerdict(
        status="disputed",
        candidates=candidates,
        agreement_signals=(),
    )


def _source_from_bib_entry(
    entry: BibEntry,
    *,
    abstract: str | None = None,
    doi: str | None = None,
    pmcid: str | None = None,
) -> ResolvedSource:
    return ResolvedSource(
        found=True,
        doi=doi or entry.doi,
        title=entry.title or None,
        abstract=abstract,
        similarity_score=1.0,
        title_match_score=1.0 if entry.title else None,
        resolution_low_confidence=False,
        pmcid=_normalize_pmcid(pmcid) or _normalize_pmcid(entry.pmcid),
    )
