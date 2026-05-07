"""Unit tests for src/bibliography.py — pure parser, no IO.

The fixture intentionally contains en-dashes in page ranges to mirror
BibTeX-rendered bibliographies. RUF001 (ambiguous Unicode) is disabled
file-wide so the test input matches real-world data.
"""
# ruff: noqa: RUF001

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

    def test_doi_none_when_absent(self) -> None:
        entries = parse_bibliography(_SAMPLE)
        assert entries[3].doi is None
        assert entries[83].doi is None

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

    def test_handles_non_consecutive_numbers(self) -> None:
        text = "References\n[5]\nFoo B. 'X'. In: Y (2010).\n[7]\nBar C. 'X2'. In: Y (2011).\n"
        entries = parse_bibliography(text)
        assert set(entries.keys()) == {5, 7}

    def test_empty_block_skipped(self) -> None:
        text = "References\n[1]\n\n[2]\nSmith. 'Title'. In: J (2020).\n"
        entries = parse_bibliography(text)
        assert 1 not in entries
        assert 2 in entries
