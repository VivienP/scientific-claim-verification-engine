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
# Numbered-entry markers come in three real-world shapes. The parser must
# recognize all three; failing silently on a recognized References section
# is a worse outcome than a noisy mis-parse, because downstream resolvers
# fall back to lossy author-year search and silently mis-resolve every
# citation in the manuscript.
#
#   form A: `[N]` alone on its own line (LaTeX-export, lactate-ISF benchmark)
#   form B: `[N] Author...` inline (some preprints / arXiv listings)
#   form C: `N. Author...` inline (BMC, Nature, NEJM, JAMA, Cell — most journals)
#
# Group 1 captures form A, group 2 captures form B, group 3 captures form C.
# Exactly one group is populated per match; the entry-extraction loop reads
# whichever is non-None.
_ENTRY_NUMBER_RE = re.compile(
    r"^\s*"
    r"(?:"
    r"\[(\d+)\]\s*$"  # form A: [N] alone on its own line
    r"|\[(\d+)\]\s+(?=\S)"  # form B: [N] inline (preprint style)
    r"|(\d+)\.\s+(?=\S)"  # form C: N. inline (journal style)
    r")",
    re.MULTILINE,
)
# DOI extraction handles two real-world syntaxes: explicit `doi:` prefix and
# resolver URL (`https://doi.org/...` or `https://dx.doi.org/...`). Both
# converge on the same DOI suffix capture; trailing punctuation is stripped
# by `_extract_doi`. The lazy `.+?` with a boundary lookahead (whitespace +
# parenthesis, PMID/PMC marker, or end of string) lets us match DOIs that
# span line breaks in line-wrapped PDF exports — `_extract_doi` collapses
# the internal whitespace.
_DOI_FIELD_RE = re.compile(
    r"(?:"
    r"doi:\s*"
    r"|https?://(?:dx\.)?doi\.org/"
    r")"
    r"(10\.\d{4,9}/.+?)"
    r"(?=(?:\s+\(|\s+PMID\b|\s+PMC\b|\s*$))",
    re.IGNORECASE | re.DOTALL,
)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
# Patterns that contain a year-like 4-digit token but are NOT the
# publication year. Stripped from the body before year extraction so
# the actual year (which always appears in body prose, never in IDs/
# URLs) is the only remaining match.
_ARXIV_ID_RE = re.compile(r"arXiv:\d{4}\.\d+", re.IGNORECASE)
_URL_RE = re.compile(r"https?://\S+")
_PMID_RE = re.compile(r"\bPMID\s*:?\s*(\d+)", re.IGNORECASE)
_PMCID_RE = re.compile(r"\bPMC(\d+)", re.IGNORECASE)
_PAGE_TRAILER_RE = re.compile(r"\(cited on pages?[^)]*\)[\s.]*$", re.IGNORECASE | re.DOTALL)
# PDF text extraction (e.g. pymupdf) interleaves page numbers and breaks
# long URLs across line boundaries. Both artefacts corrupt downstream DOI
# extraction. Stripped before any field-level extraction runs.
_PAGE_NUMBER_LINE_RE = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)
# Collapse `https://...\n<continuation>` when the continuation does not
# start a new bibliography entry (`\d+\.` or `[\d+]`). The negative
# lookahead protects entry boundaries.
_URL_LINE_WRAP_RE = re.compile(r"(https?://\S+)\n(?!\s*(?:\[\d+\]|\d+\.\s))(\S+)")
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


def _clean_pdf_artefacts(text: str) -> str:
    """Strip pymupdf-style page number lines and collapse URL line-wraps.

    Two artefacts dominate PDF text extraction of journal bibliographies:
    1. Page numbers appear as digit-only lines between bibliography entries.
       When unstripped, they get included in the previous entry's body and
       contaminate DOI extraction (e.g. `10.1038/s41591-022-01744-z` becomes
       `10.1038/s41591-022-01744-z12`).
    2. Long URLs wrap across line breaks (e.g. `https://doi.or\ng/10.1177/...`).
       The default DOI regex sees these as broken URLs and either misses the
       DOI or captures only a fragment.

    Order matters: strip page numbers first so they cannot be mistaken for
    URL continuations. The URL-wrap regex uses a negative lookahead to avoid
    eating the next entry's leading number.
    """
    cleaned = _PAGE_NUMBER_LINE_RE.sub("", text)
    # Apply URL-wrap collapse iteratively for URLs split into 3+ fragments.
    # Cap iterations defensively to avoid infinite loops on pathological input.
    for _ in range(8):
        new = _URL_LINE_WRAP_RE.sub(r"\1\2", cleaned)
        if new == cleaned:
            break
        cleaned = new
    return cleaned


def _extract_title(body: str) -> str:
    """Pull the title out of the entry body.

    Three real-world shapes are handled:
    1. Smart-quote pair (LaTeX/BibTeX rendering with `'...'`).
    2. Straight-quote pair (occasional plain-text bibliographies).
    3. Unquoted journal format `Authors. Title. Venue. Year;...` — title
       is the segment between the end of the author block (last initial
       followed by `. `) and the next sentence-period-space boundary.

    Returns an empty string when none of the heuristics yield a salient
    title. The DOI is the resolver's gold path; an empty title only
    degrades the secondary CrossRef-title fallback, not the primary path.
    """
    smart = _SMART_QUOTE_PAIR.search(body)
    if smart:
        return smart.group(1).strip()
    straight = _STRAIGHT_QUOTE_PAIR.search(body)
    if straight:
        return straight.group(1).strip()
    return _extract_title_unquoted(body)


# Author block end: last surname followed by 1-3 capitals (initials) and a
# terminating period+space. The captured period+space marks the boundary
# between authors and title.
_AUTHOR_BLOCK_END_RE = re.compile(r"[A-Z][a-z]+\s+[A-Z]{1,3}\.\s")
# Title block end: a period+space followed by a venue marker — a year, a
# journal-name capital pattern (Nat, Sci, Proc, etc.), `In:`, or a
# parenthesized phrase that typically precedes year/volume metadata.
_TITLE_BLOCK_END_RE = re.compile(
    r"\.\s+"
    r"(?:"
    r"\d{4}\b"  # year token
    r"|In:\s"  # explicit `In:` venue marker
    r"|[A-Z][a-z]*\s+[A-Z][a-z]+(?:\s+[A-Z]\w*)?\b"  # capitalized journal name (2-3 tokens)
    r"|(?:Proc|Nat|J|Sci|Adv|Cell|Nature|BMC|PLoS|ACM|IEEE|arXiv)\b"
    r")"
)


def _extract_title_unquoted(body: str) -> str:
    """Heuristic title extraction for the journal-numbered format.

    Targets the shape `Authors. Title. Venue. Year. URL.` Robust to
    parenthesized phrases inside the title (e.g. `(No. arXiv:2402.01788)`)
    that contain spurious periods.
    """
    author_end = _AUTHOR_BLOCK_END_RE.search(body)
    if author_end is None:
        return ""
    title_start = author_end.end()
    rest = body[title_start:]
    venue_match = _TITLE_BLOCK_END_RE.search(rest)
    title = _strip_to_outer_period(rest) if venue_match is None else rest[: venue_match.start()]
    return title.strip().rstrip(".,;: ")


def _strip_to_outer_period(text: str) -> str:
    """Truncate at the first `. ` that is not inside parentheses."""
    depth = 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
        elif ch == "." and depth == 0 and i + 1 < len(text) and text[i + 1] == " ":
            return text[:i]
    return text


_ET_AL_RE = re.compile(r"[,\s]+et\s+al\.?\s*$", re.IGNORECASE)
# An "initials-only" token: 1-3 uppercase letters, optionally followed by
# trailing periods. Used to detect surname-first formats where the last
# whitespace-separated token of an author entry is only initials.
_INITIALS_ONLY_RE = re.compile(r"^[A-Z]{1,3}\.?$")


def _looks_like_initials(token: str) -> bool:
    """True when a whitespace-separated token is initials only (e.g. `S`, `IH`, `J.E.`)."""
    cleaned = token.replace(".", "")
    if not cleaned or len(cleaned) > 3:
        return False
    return cleaned.isupper() and cleaned.isalpha()


def _pick_surname(tokens: list[str]) -> str:
    """Pick the surname from a single author entry's whitespace-tokenized parts.

    Two real-world formats coexist in numbered bibliographies:
    - Surname-last: `J. E. Smith`, `Mary E Doe`, `JR Williams` → take last token.
    - Surname-first: `Smith JE`, `Doe ME`, `Williams JR` → take first token,
      because the last token is initials only.

    Detected by inspecting whether the last token is initials-only. When it
    is, we treat the author entry as surname-first and return the first
    non-initial token instead. This keeps lactate-ISF (surname-last) and
    BMC (surname-first) parses both correct under one rule.
    """
    if not tokens:
        return ""
    if len(tokens) >= 2 and _looks_like_initials(tokens[-1]):
        # Surname-first: walk left to right and take the first non-initials token.
        for tok in tokens:
            if not _looks_like_initials(tok):
                return tok
        # All initials? unlikely but fall through to last token.
    return tokens[-1]


def _extract_authors(body: str, title: str) -> list[str]:
    """Parse the author segment that precedes the title.

    Returns visible surnames as a list. "et al." is recorded as a literal
    trailing item so downstream code can detect "and others" without losing
    the visible author count.

    Handles both surname-last (`J. E. Smith`, lactate-ISF/LaTeX) and
    surname-first (`Smith JE`, BMC/journal) formats — see `_pick_surname`.
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
        tokens = p.split()
        surname = _pick_surname(tokens)
        if surname:
            cleaned.append(surname)
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
    body = _clean_pdf_artefacts(text[header_match.end() :])

    matches = list(_ENTRY_NUMBER_RE.finditer(body))
    if not matches:
        return {}

    entries: dict[int, BibEntry] = {}
    for i, m in enumerate(matches):
        # Exactly one of the three capture groups is non-None per match;
        # pick whichever fired (form A / B / C — see _ENTRY_NUMBER_RE).
        number_str = m.group(1) or m.group(2) or m.group(3)
        if number_str is None:
            continue
        number = int(number_str)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        raw_block = _strip_trailers(body[start:end].strip())
        if not raw_block:
            continue

        title = _extract_title(raw_block)
        authors = _extract_authors(raw_block, title)
        journal = _extract_journal(raw_block, title)

        # Strip arXiv IDs and URLs before scanning for the year. arXiv IDs
        # of the form `arXiv:YYMM.NNNNN` and DOI URLs containing those IDs
        # both contain year-like 4-digit tokens that are not the actual
        # publication year. Stripping them leaves only the prose-embedded
        # year, of which the last match is taken (handles citations that
        # mention multiple years in the body, e.g. "data from 2018-2022").
        year_text = _ARXIV_ID_RE.sub("", raw_block)
        year_text = _URL_RE.sub("", year_text)
        year_matches = list(_YEAR_RE.finditer(year_text))
        year = int(year_matches[-1].group()) if year_matches else None

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
