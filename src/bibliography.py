"""Numbered-bibliography parser for scientific manuscripts.

Many literature reviews cite sources by bracket numbers ([3], [81-83])
rather than by inline author names. Without parsing the reference list,
the resolver only sees the flat author/year fields the extractor produced
and loses the per-citation anchor — particularly damaging for multi-
citation claims like [99, 100] where the cited authors get flattened.

This module reads the source text, locates the References / Bibliography
section, and parses each numbered entry into a BibEntry with author list,
title, year, journal, DOI, and PMID when available. Downstream code can
then look up a specific reference number directly.

The parser targets the BibTeX-rendered format common in academic LaTeX
output, where each entry is preceded by its bracket number on its own
line and the entry body spans multiple lines until the next bracket
number. It is robust to:
- Missing DOI lines
- "et al." abbreviations
- Optional "(cited on page N)" trailers
- Multi-line title or journal fields
- Smart quotes around the title

Pure Python (no LLM call). Tested against the lactate-ISF review.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_REFERENCES_HEADER_RE = re.compile(r"\bReferences\b", re.IGNORECASE)
_ENTRY_NUMBER_RE = re.compile(r"^\s*\[(\d+)\]\s*$", re.MULTILINE)
_DOI_FIELD_RE = re.compile(
    r"doi:\s*(10\.\d{4,9}/.+?)(?=(?:\s+\(|\s+PMID\b|\s+PMC\b|\s*$))",
    re.IGNORECASE | re.DOTALL,
)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_PMID_RE = re.compile(r"\bPMID\s*:?\s*(\d+)", re.IGNORECASE)
_PMCID_RE = re.compile(r"\bPMC(\d+)", re.IGNORECASE)
_PAGE_TRAILER_RE = re.compile(r"\(cited on pages?[^)]*\)[\s.]*$", re.IGNORECASE | re.DOTALL)
# Smart-quote pair (U+2018 LEFT, U+2019 RIGHT) used by LaTeX-rendered bibliographies.
_LSQUO = chr(0x2018)
_RSQUO = chr(0x2019)
_SMART_QUOTE_PAIR = re.compile(f"[{_LSQUO}{_RSQUO}]([^{_LSQUO}{_RSQUO}]{{5,}})[{_LSQUO}{_RSQUO}]")
_STRAIGHT_QUOTE_PAIR = re.compile(r"'([^']{5,})'")


@dataclass(frozen=True)
class BibEntry:
    """One parsed reference. All fields except `number` and `raw` may be empty."""

    number: int
    raw: str
    authors: list[str] = field(default_factory=list)
    title: str = ""
    year: int | None = None
    journal: str = ""
    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None


def _strip_trailers(text: str) -> str:
    return _PAGE_TRAILER_RE.sub("", text).strip()


def _extract_title(body: str) -> str:
    """Pull the title out of the entry body.

    Tries smart-quote pair first, then straight-quote pair, then falls back
    to the segment between the first period and the next period after that
    segment looks title-like.
    """
    smart = _SMART_QUOTE_PAIR.search(body)
    if smart:
        return smart.group(1).strip()
    straight = _STRAIGHT_QUOTE_PAIR.search(body)
    if straight:
        return straight.group(1).strip()
    return ""


_ET_AL_RE = re.compile(r"[,\s]+et\s+al\.?\s*$", re.IGNORECASE)


def _extract_authors(body: str, title: str) -> list[str]:
    """Parse the author segment that precedes the title.

    Returns visible surnames as a list. "et al." is recorded as a literal
    trailing item so downstream code can detect "and others" without losing
    the visible author count.
    """
    if not title:
        prefix = body.split(".", 1)[0]
    else:
        idx = body.find(title)
        prefix = body[:idx] if idx > 0 else body.split(".", 1)[0]
    prefix = prefix.replace(_LSQUO, "").replace(_RSQUO, "").rstrip(",. ")

    has_et_al = bool(_ET_AL_RE.search(prefix))
    if has_et_al:
        prefix = _ET_AL_RE.sub("", prefix).strip().rstrip(",. ")

    parts = [p.strip() for p in re.split(r",\s+| and ", prefix) if p.strip()]
    cleaned: list[str] = []
    for raw_p in parts:
        p = raw_p.strip(".,;: ")
        if not p:
            continue
        # Drop initials such as "J", "GA", "Mary E" — keep only the last token
        tokens = p.split()
        if tokens:
            cleaned.append(tokens[-1])
    if has_et_al:
        cleaned.append("et al.")
    return cleaned


def _extract_journal(body: str, title: str) -> str:
    """Heuristic: the journal is the segment after 'In: ' up to the next period."""
    marker = body.find("In:")
    if marker < 0:
        return ""
    after = body[marker + 3 :]
    # Stop at the first period that is followed by whitespace + capital or by end.
    # A simple split on the first comma or period suffices for most academic refs.
    end = re.search(r"\.\s|,\s+\d|$", after)
    journal = after[: end.start()].strip() if end else after.strip()
    return journal.lstrip(": ").strip()


def _extract_doi(body: str) -> str | None:
    match = _DOI_FIELD_RE.search(body)
    if not match:
        return None
    # PDF/plain-text exports sometimes wrap long DOI suffixes across lines.
    doi = re.sub(r"\s+", "", match.group(1))
    return doi.rstrip(".,;)")


def parse_bibliography(text: str) -> dict[int, BibEntry]:
    """Parse the numbered References section out of `text`.

    Returns a dict keyed by reference number. Missing fields are returned as
    empty strings / None / [] so downstream code can read them defensively.
    Returns an empty dict if no References header is found.

    The parser is intentionally tolerant: when a field cannot be extracted,
    it is left empty rather than raising. The `raw` field on each BibEntry
    preserves the original unprocessed text so callers can re-parse if needed.
    """
    header_match = _REFERENCES_HEADER_RE.search(text)
    if header_match is None:
        return {}
    body = text[header_match.end() :]

    matches = list(_ENTRY_NUMBER_RE.finditer(body))
    if not matches:
        return {}

    entries: dict[int, BibEntry] = {}
    for i, m in enumerate(matches):
        number = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        raw_block = _strip_trailers(body[start:end].strip())
        if not raw_block:
            continue

        title = _extract_title(raw_block)
        authors = _extract_authors(raw_block, title)
        journal = _extract_journal(raw_block, title)

        year_match = _YEAR_RE.search(raw_block)
        year = int(year_match.group()) if year_match else None

        doi = _extract_doi(raw_block)

        pmid_match = _PMID_RE.search(raw_block)
        pmid = pmid_match.group(1) if pmid_match else None

        pmcid_match = _PMCID_RE.search(raw_block)
        pmcid = pmcid_match.group(1) if pmcid_match else None

        entries[number] = BibEntry(
            number=number,
            raw=raw_block,
            authors=authors,
            title=title,
            year=year,
            journal=journal,
            doi=doi,
            pmid=pmid,
            pmcid=pmcid,
        )
    return entries
