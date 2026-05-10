"""Unit tests for src/bibliography.py — pure parser, no IO.

The fixture intentionally contains en-dashes in page ranges to mirror
BibTeX-rendered bibliographies. RUF001 (ambiguous Unicode) is disabled
file-wide so the test input matches real-world data. E501 (line length)
is also disabled file-wide because the BMC-format fixtures are real
bibliography entries and breaking them across lines would change the
text the parser sees and defeat the test's purpose.
"""
# ruff: noqa: RUF001, E501

from __future__ import annotations

from src.bibliography import BibEntry, parse_bibliography

_LSQUO = chr(0x2018)
_RSQUO = chr(0x2019)


def _wrap_title(title: str) -> str:
    return f"{_LSQUO}{title}{_RSQUO}"


_SAMPLE = f"""1. Introduction
Some intro text.

References
[1]
Xiaolu Li et al. {_wrap_title("Lactate metabolism in human health and disease")}. In:
Signal transduction 7.1 (2022), p. 305 (cited on page 1).
[3]
Matthew L Goodwin et al. {_wrap_title("Blood lactate measurements")}. In:
Journal of diabetes science and technology 1.4 (2007), pp. 558–569 (cited on pages 1–5).
[4]
I. Jacobs. {_wrap_title("Blood lactate. Implications for training and sports performance")}. In:
Sports Medicine 3.1 (1986), pp. 10–25. doi: 10.2165/00007256-198603010-00003 (cited on page 1).
[83]
JR Williams, N Armstrong, and BJ Kirby. {_wrap_title("The influence of the site of sampling")}. In:
Journal of sports sciences 10.2 (1992), pp. 95–107 (cited on page 5).
[99]
F Birklein, M Weber, and B Neundörfer. {_wrap_title("Increased skin lactate in CRPS")}. In:
Neurology 55.8 (2000), pp. 1213–1215 (cited on page 6).
[105]
AL Krogstad et al. {_wrap_title("Microdialysis methodology for the measurement of dermal ISF")}. In:
British Journal of Dermatology 134.6 (1996), pp. 1005–1012 (cited on page 6).
"""


class TestParseBibliography:
    def test_returns_empty_when_no_references_header(self) -> None:
        text = "1. Intro\nSome text without a references section."
        assert parse_bibliography(text) == {}

    def test_finds_all_numbered_entries(self) -> None:
        entries = parse_bibliography(_SAMPLE)
        assert set(entries.keys()) == {1, 3, 4, 83, 99, 105}

    def test_entry_keyed_by_number_int(self) -> None:
        entries = parse_bibliography(_SAMPLE)
        assert isinstance(entries[3], BibEntry)
        assert entries[3].number == 3

    def test_authors_with_et_al_suffix(self) -> None:
        entries = parse_bibliography(_SAMPLE)
        assert entries[3].authors == ["Goodwin", "et al."]
        assert entries[1].authors == ["Li", "et al."]

    def test_authors_explicit_list_no_et_al(self) -> None:
        entries = parse_bibliography(_SAMPLE)
        assert entries[83].authors == ["Williams", "Armstrong", "Kirby"]
        assert entries[99].authors == ["Birklein", "Weber", "Neundörfer"]

    def test_year_extraction(self) -> None:
        entries = parse_bibliography(_SAMPLE)
        assert entries[3].year == 2007
        assert entries[4].year == 1986

    def test_title_extraction_smart_quotes(self) -> None:
        entries = parse_bibliography(_SAMPLE)
        assert "Blood lactate measurements" in entries[3].title
        assert "Increased skin lactate" in entries[99].title

    def test_doi_extraction_when_present(self) -> None:
        entries = parse_bibliography(_SAMPLE)
        assert entries[4].doi == "10.2165/00007256-198603010-00003"

    def test_line_wrapped_doi_extraction(self) -> None:
        text = (
            "References\n"
            "[23]\n"
            "Boyu Qin et al. 'Porosity control of polylactic acid porous microneedles "
            "using microfluidic technology'. In: 2022 IEEE CPMT Symposium Japan "
            "(2022), pp. 127-130. doi: 10.1109/ICSJ55786.\n"
            "2022.10034733 (cited on page 1).\n"
        )
        entries = parse_bibliography(text)
        assert entries[23].doi == "10.1109/ICSJ55786.2022.10034733"

    def test_journal_extraction(self) -> None:
        entries = parse_bibliography(_SAMPLE)
        # Journal is the segment after 'In:'.
        assert "Journal of sports sciences" in entries[83].journal
        assert "Neurology" in entries[99].journal

    def test_cited_on_page_trailer_stripped(self) -> None:
        entries = parse_bibliography(_SAMPLE)
        # Raw text should not include the (cited on pages...) trailer.
        for e in entries.values():
            assert "cited on page" not in e.raw.lower()

    def test_pmid_extraction(self) -> None:
        text = "References\n[1]\nSmith A. 'Title'. In: Journal 1 (2020). PMID: 12345678\n"
        entries = parse_bibliography(text)
        assert entries[1].pmid == "12345678"

    def test_pmcid_extraction(self) -> None:
        text = "References\n[1]\nSmith A. 'Title'. In: Journal 1 (2020). PMC4306774 (cited).\n"
        entries = parse_bibliography(text)
        assert entries[1].pmcid == "4306774"

    def test_empty_block_skipped(self) -> None:
        text = "References\n[1]\n\n[2]\nSmith. 'Title'. In: J (2020).\n"
        entries = parse_bibliography(text)
        assert 1 not in entries
        assert 2 in entries


# Journal-numbered format ("1. Author A, ..." inline) — used by BMC, Nature,
# Cell, NEJM, JAMA and most biomedical journals. Distinct from the LaTeX
# bracket-on-own-line format above.
_BMC_SAMPLE = """Background

Some background text [1] referencing an entry.

References

1. Agarwal S, Laradji IH, Charlin L, Pal C. LitLLM: a toolkit for scientific literature review (No. arXiv:2402.01788). 2024. arXiv. https://doi.org/10.48550/arXiv.2402.01788.

2. Haddaway NR, Bethel A, Dicks LV, Koricheva J, Macura B, Petrokofsky G, Pullin AS, Savilaakso S, Stewart GB. Eight problems with literature reviews and how to fix them. Nat Ecol Evol. 2020;4(12):1582-9. https://doi.org/10.1038/s41559-020-01295-x.

5. Hirsch JE. An index to quantify an individual's scientific research output. Proc Natl Acad Sci USA. 2005;102(46):16569-72. https://doi.org/10.1073/pnas.0507655102.

10. Nicholson JM, Mordaunt M, Lopez P, Uppala A, Rosati D, Rodrigues NP, Grabitz P, Rife SC. scite: a smart citation index that displays the context of citations and classifies their intent using deep learning. Quant Sci Stud. 2021;2(3):882-98. https://doi.org/10.1162/qss_a_00146.

15. Wei J, Wang X, Schuurmans D, Bosma M, Ichter B, Xia F, Chi EH, Le QV, Zhou D. Chain-of-thought prompting elicits reasoning in large language models. In: Proceedings of the 36th International Conference on neural information processing systems, 2022; 24824-24837.
"""


class TestParseBibliographyJournalNumbered:
    """The BMC/journal '1. Author A, ...' format must parse to the same
    BibEntry shape as the bracket-on-own-line LaTeX format. The DOI is the
    most important field — `_resolve_via_bib_doi` is the gold path that
    bypasses every lossy author/year search.
    """

    def test_finds_all_numbered_entries(self) -> None:
        entries = parse_bibliography(_BMC_SAMPLE)
        assert set(entries.keys()) == {1, 2, 5, 10, 15}

    def test_extracts_doi_from_https_url(self) -> None:
        entries = parse_bibliography(_BMC_SAMPLE)
        assert entries[1].doi == "10.48550/arXiv.2402.01788"
        assert entries[2].doi == "10.1038/s41559-020-01295-x"
        assert entries[5].doi == "10.1073/pnas.0507655102"
        assert entries[10].doi == "10.1162/qss_a_00146"

    def test_extracts_authors(self) -> None:
        entries = parse_bibliography(_BMC_SAMPLE)
        # First entry: "Agarwal S, Laradji IH, Charlin L, Pal C."
        # We expect the surnames preserved (initials dropped per existing convention).
        assert entries[1].authors[0] == "Agarwal"
        assert "Laradji" in entries[1].authors
        assert "Charlin" in entries[1].authors
        assert "Pal" in entries[1].authors
        # Long author list with mixed initials — surnames preserved
        assert entries[2].authors[0] == "Haddaway"
        assert "Stewart" in entries[2].authors

    def test_extracts_year(self) -> None:
        entries = parse_bibliography(_BMC_SAMPLE)
        assert entries[1].year == 2024
        assert entries[2].year == 2020
        assert entries[5].year == 2005
        assert entries[10].year == 2021
        assert entries[15].year == 2022

    def test_extracts_title(self) -> None:
        entries = parse_bibliography(_BMC_SAMPLE)
        # Titles are not in quotes in this format — captured via period-segmented heuristic
        # We accept any extraction that contains the salient title tokens
        assert "litllm" in entries[1].title.lower()
        assert "eight problems" in entries[2].title.lower()
        assert (
            "quantify" in entries[5].title.lower()
            or "h-index" in entries[5].title.lower()
            or "scientific research output" in entries[5].title.lower()
        )
        assert "scite" in entries[10].title.lower()
        assert (
            "chain-of-thought" in entries[15].title.lower()
            or "chain of thought" in entries[15].title.lower()
        )

    def test_does_not_pick_up_inline_bracket_citations_as_entries(self) -> None:
        # The Background section contains a `[1]` citation marker that is
        # NOT a bibliography entry. The parser must not be confused by it.
        # Specifically, the `[1]` in the body text must not produce a
        # spurious entry from the surrounding background prose.
        entries = parse_bibliography(_BMC_SAMPLE)
        # entry 1 must come from the References section (LitLLM), not from
        # the background prose
        assert "litllm" in entries[1].title.lower() or "agarwal" in str(entries[1].authors).lower()


_INLINE_BRACKET_SAMPLE = """References

[1] Smith J, Doe A. A study on something. Journal of Things. 2020;5:100-110. doi: 10.1234/abc.

[2] Roe B. Another paper. Nature. 2021;500:1-5. doi: 10.5678/xyz.
"""


class TestParseBibliographyInlineBracket:
    """The inline bracket format `[N] Author...` — used by some preprints
    and arXiv listings. Number and content on the same line.
    """

    def test_finds_inline_bracket_entries(self) -> None:
        entries = parse_bibliography(_INLINE_BRACKET_SAMPLE)
        assert set(entries.keys()) == {1, 2}

    def test_extracts_inline_bracket_doi(self) -> None:
        entries = parse_bibliography(_INLINE_BRACKET_SAMPLE)
        assert entries[1].doi == "10.1234/abc"
        assert entries[2].doi == "10.5678/xyz"

    def test_extracts_inline_bracket_year(self) -> None:
        entries = parse_bibliography(_INLINE_BRACKET_SAMPLE)
        assert entries[1].year == 2020
        assert entries[2].year == 2021

    def test_extracts_inline_bracket_authors(self) -> None:
        entries = parse_bibliography(_INLINE_BRACKET_SAMPLE)
        assert "Smith" in entries[1].authors
        assert "Doe" in entries[1].authors
        assert "Roe" in entries[2].authors


class TestParseBibliographyRobustness:
    """Cross-format robustness — the parser must not produce false entries
    or silently drop real ones across format variations.
    """

    def test_silent_zero_return_is_a_bug(self) -> None:
        # If the input has a clear References section with numbered entries
        # in a recognizable format, returning {} is a parser bug — not a
        # legitimate "no references found" outcome. This regression test
        # locks in the BMC-format support so future edits can't regress.
        entries = parse_bibliography(_BMC_SAMPLE)
        assert len(entries) > 0, (
            "Parser silently returned no entries on a recognizable BMC-format "
            "bibliography. This is the regression that broke the Valsci validation run."
        )

    def test_mixed_doi_url_styles(self) -> None:
        # `https://doi.org/...`, `doi:...`, and `https://dx.doi.org/...` all valid
        text = """References

1. A. Title A. 2020. doi: 10.1111/aaa.
2. B. Title B. 2021. https://doi.org/10.2222/bbb.
3. C. Title C. 2022. https://dx.doi.org/10.3333/ccc.
"""
        entries = parse_bibliography(text)
        assert entries[1].doi == "10.1111/aaa"
        assert entries[2].doi == "10.2222/bbb"
        assert entries[3].doi == "10.3333/ccc"

    def test_real_valsci_input_parses(self) -> None:
        """End-to-end regression on the actual Valsci validation run input.

        This file is the canonical fixture for the bibliography parser fix —
        if this test fails after a future edit, the BMC-format support
        regressed.
        """
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        valsci = repo_root / "benchmarks" / "real_papers" / "valsci_brice_2025" / "input.txt"
        if not valsci.exists():
            return  # skip when fixture absent (e.g. fresh clone without real_papers fixture)
        text = valsci.read_text(encoding="utf-8")
        entries = parse_bibliography(text)
        # Valsci has 15 numbered references
        assert len(entries) == 15, (
            f"expected 15 entries, got {len(entries)}: {sorted(entries.keys())}"
        )
        # Spot-check a few specific DOIs
        assert entries[2].doi == "10.1038/s41559-020-01295-x"  # Haddaway
        assert entries[5].doi == "10.1073/pnas.0507655102"  # Hirsch H-index
        assert entries[10].doi == "10.1162/qss_a_00146"  # Nicholson Scite

    def test_pymupdf_page_numbers_stripped(self) -> None:
        """PDF text extraction inserts page numbers as digit-only lines between
        entries. They must not get appended to the previous entry's DOI.
        """
        text = (
            "References\n\n"
            "1.\nSmith A (2022) Paper one. Journal A. https://doi.org/10.1111/aaa\n"
            "12\n"
            "2.\nDoe B (2023) Paper two. Journal B. https://doi.org/10.2222/bbb\n"
            "13\n"
        )
        entries = parse_bibliography(text)
        assert entries[1].doi == "10.1111/aaa", (
            f"page number contaminated DOI: got {entries[1].doi!r}"
        )
        assert entries[2].doi == "10.2222/bbb"

    def test_url_line_wrap_collapsed(self) -> None:
        """Long DOI URLs wrap mid-domain or mid-suffix in PDF text extraction.
        The parser must rejoin them for clean DOI extraction.
        """
        text = (
            "References\n\n"
            "1.\nSmith A (2025) Paper. Journal. https://doi.or\ng/10.1177/20451253251377187\n\n"
            "2.\nDoe B (2024) Other. Journal. https://doi.org/10\n.1016/j.medj.2024.01.005\n"
        )
        entries = parse_bibliography(text)
        assert entries[1].doi == "10.1177/20451253251377187", (
            f"wrap-after-domain not joined: got {entries[1].doi!r}"
        )
        assert entries[2].doi == "10.1016/j.medj.2024.01.005", (
            f"wrap-after-prefix not joined: got {entries[2].doi!r}"
        )

    def test_url_wrap_does_not_eat_next_entry(self) -> None:
        """The URL-wrap fix must respect entry boundaries — a numbered entry
        line right after a URL must NOT be treated as URL continuation.
        """
        text = (
            "References\n\n"
            "1.\nSmith A. Paper. https://doi.org/10.1111/aaa\n"
            "2.\nDoe B. Paper. https://doi.org/10.2222/bbb\n"
        )
        entries = parse_bibliography(text)
        # Entry 1's DOI must not absorb '2.' from the next entry start.
        assert entries[1].doi == "10.1111/aaa"
        assert len(entries) == 2

    def test_real_elicit_input_parses(self) -> None:
        """End-to-end regression on the Elicit Report-mode PDF text extraction.

        Elicit Report exports use parens-year (`Author (YEAR) Title`) format
        with page-number lines and URL line-wraps from the PDF text layer.
        All 10 references must resolve to clean DOIs.
        """
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        fixture = repo_root / "benchmarks" / "real_outputs" / "elicit_psilocybin" / "input.txt"
        if not fixture.exists():
            return
        text = fixture.read_text(encoding="utf-8")
        entries = parse_bibliography(text)
        assert len(entries) == 10, (
            f"expected 10 entries, got {len(entries)}: {sorted(entries.keys())}"
        )
        expected_dois = {
            1: "10.1056/NEJMoa2206443",
            2: "10.1016/S2215-0366(16)30065-7",
            3: "10.1177/20451253251377187",
            4: "10.1007/s00213-017-4771-x",
            5: "10.1038/s41591-022-01744-z",
            6: "10.1176/appi.ajp.20231063",
            7: "10.1038/s41386-023-01648-7",
            8: "10.1038/s41598-017-13282-7",
            9: "10.1016/j.medj.2024.01.005",
            10: "10.1016/j.jad.2023.01.108",
        }
        for n, expected in expected_dois.items():
            assert entries[n].doi == expected, (
                f"entry [{n}] doi mismatch: got {entries[n].doi!r}, expected {expected!r}"
            )

    def test_year_extraction_skips_arxiv_id_prefix(self) -> None:
        """arXiv preprint IDs of the form `arXiv:YYMM.NNNNN` start with a
        year-like 4-digit token (e.g. `2005.11401` for May 2020). A naive
        first-match year regex picks the arXiv ID prefix instead of the
        actual publication year that appears later in the citation.
        Regression test for the Valsci validation run, where 3 entries had
        their year mis-extracted from arXiv IDs.
        """
        text = (
            "References\n\n"
            "1. Lewis P. Retrieval-augmented generation for knowledge-intensive NLP "
            "tasks (No. arXiv:2005.11401). 2021. arXiv. https://doi.org/10.48550/arXiv.2005.11401.\n\n"
            "2. Lo K. S2ORC: the semantic scholar open research corpus "
            "(No. arXiv:1911.02782). 2020. arXiv. https://doi.org/10.48550/arXiv.1911.02782.\n\n"
            "3. Wadden D. Fact or Fiction (No. arXiv:2004.14974). 2020. arXiv. "
            "https://doi.org/10.48550/arXiv.2004.14974.\n"
        )
        entries = parse_bibliography(text)
        assert entries[1].year == 2021, (
            f"expected 2021, got {entries[1].year} (likely picked up arXiv ID 2005)"
        )
        assert entries[2].year == 2020, (
            f"expected 2020, got {entries[2].year} (likely picked up arXiv ID 1911)"
        )
        assert entries[3].year == 2020, (
            f"expected 2020, got {entries[3].year} (likely picked up arXiv ID 2004)"
        )
