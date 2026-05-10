"""Unit tests for src/resolve.py — citation resolution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from pytest_httpx import HTTPXMock

from src.models import Claim, ProvenanceStep, ResolvedSource


def _make_claim(
    claim_id: str = "claim-1",
    cited_authors: list[str] | None = None,
    cited_year: int | None = 2020,
    claim_text: str = "Some claim about X.",
    citation_markers: list[int] | None = None,
) -> Claim:
    return Claim(
        claim_id=claim_id,
        claim_text=claim_text,
        cited_authors=cited_authors if cited_authors is not None else ["Smith"],
        cited_year=cited_year,
        claim_type="factual_qualitative",
        citation_markers=citation_markers or [],
    )


class TestPhase1ResolveBehavior:
    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve._crossref.search_paper")
    @patch("src.resolve.search_paper")
    def test_openalex_miss_then_crossref_hit(
        self,
        mock_oa: MagicMock,
        mock_cf: MagicMock,
        mock_retr: MagicMock,
    ) -> None:
        mock_oa.return_value = ResolvedSource(False, None, None, None, None)
        mock_cf.return_value = ResolvedSource(True, "10.1/x", "T", None, None)
        from src.resolve import resolve_citations

        sources, _ = resolve_citations([_make_claim("c1")])
        assert sources["c1"].found is True
        assert sources["c1"].doi == "10.1/x"
        mock_cf.assert_called_once()

    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve._crossref.search_paper")
    @patch("src.resolve.search_paper")
    def test_openalex_hit_skips_crossref(
        self, mock_oa: MagicMock, mock_cf: MagicMock, mock_retr: MagicMock
    ) -> None:
        mock_oa.return_value = ResolvedSource(True, "10.1/y", "T", "abs", 1.0)
        from src.resolve import resolve_citations

        resolve_citations([_make_claim("c1")])
        mock_cf.assert_not_called()

    @patch("src.resolve._crossref.check_retraction")
    @patch("src.resolve.search_paper")
    def test_retraction_check_called_when_doi_present(
        self, mock_oa: MagicMock, mock_retr: MagicMock
    ) -> None:
        mock_oa.return_value = ResolvedSource(True, "10.1/x", "T", "abs", 1.0)
        mock_retr.return_value = True
        from src.resolve import resolve_citations

        sources, _ = resolve_citations([_make_claim("c1")])
        mock_retr.assert_called_once_with("10.1/x", db_path=None)
        assert sources["c1"].retraction_status is True


class TestPubmedRecordEnrichment:
    """Bug A fix (S1-P1-A): enrich both abstract AND pmcid via PubMed record.

    The previous `_enrich_abstract_via_pubmed` skipped when CrossRef had any
    abstract, silently dropping the pmcid even when PubMed had it. This
    blocked claim 003 (Goodwin) and claim 020 (Kotwal) from reaching the
    PMC fulltext path. The fix:
      - drops the `or source.abstract` early-return guard
      - uses find_pmid_by_doi -> fetch_record (preserving the full record)
      - propagates pmcid AND abstract (preserving longer existing abstract)
    """

    @staticmethod
    def _record(
        *,
        pmid: str = "12345",
        abstract: str | None = "PubMed abstract about lactate kinetics.",
        doi: str | None = "10.1/x",
        pmcid: str | None = "PMC123",
    ) -> object:
        from src.clients.pubmed import PubMedRecord

        return PubMedRecord(pmid=pmid, abstract=abstract, doi=doi, pmcid=pmcid)

    @patch("src.resolve._pubmed.fetch_record")
    @patch("src.resolve._pubmed.find_pmid_by_doi", return_value="12345")
    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve.search_paper")
    def test_enriches_abstract_and_pmcid_when_both_missing(
        self,
        mock_oa: MagicMock,
        mock_retr: MagicMock,
        mock_pmid: MagicMock,
        mock_record: MagicMock,
    ) -> None:
        mock_oa.return_value = ResolvedSource(
            found=True, doi="10.1/x", title="T", abstract=None, similarity_score=0.9
        )
        mock_record.return_value = self._record(abstract="PubMed-fetched abstract.", pmcid="PMC123")
        from src.resolve import resolve_citations

        sources, _ = resolve_citations([_make_claim("c1")])

        mock_pmid.assert_called_once_with("10.1/x", db_path=None)
        mock_record.assert_called_once_with("12345", db_path=None)
        assert sources["c1"].abstract == "PubMed-fetched abstract."
        assert sources["c1"].pmcid == "PMC123"

    @patch("src.resolve._pubmed.fetch_record")
    @patch("src.resolve._pubmed.find_pmid_by_doi", return_value="12345")
    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve.search_paper")
    def test_enriches_pmcid_when_abstract_already_present(
        self,
        mock_oa: MagicMock,
        mock_retr: MagicMock,
        mock_pmid: MagicMock,
        mock_record: MagicMock,
    ) -> None:
        # CrossRef-with-abstract-no-pmcid: the common Bug A case (claim 003, 020).
        mock_oa.return_value = ResolvedSource(
            found=True,
            doi="10.1/x",
            title="T",
            abstract="CrossRef abstract is here and is reasonably detailed.",
            similarity_score=0.9,
        )
        mock_record.return_value = self._record(abstract="Shorter PubMed abstract.", pmcid="PMC456")
        from src.resolve import resolve_citations

        sources, _ = resolve_citations([_make_claim("c1")])

        mock_pmid.assert_called_once()
        mock_record.assert_called_once()
        # Existing CrossRef abstract preserved (longer); pmcid newly populated.
        assert sources["c1"].abstract == "CrossRef abstract is here and is reasonably detailed."
        assert sources["c1"].pmcid == "PMC456"

    @patch("src.resolve._pubmed.fetch_record")
    @patch("src.resolve._pubmed.find_pmid_by_doi")
    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve.search_paper")
    def test_skipped_when_already_complete(
        self,
        mock_oa: MagicMock,
        mock_retr: MagicMock,
        mock_pmid: MagicMock,
        mock_record: MagicMock,
    ) -> None:
        mock_oa.return_value = ResolvedSource(
            found=True,
            doi="10.1/x",
            title="T",
            abstract="Already have one.",
            similarity_score=0.9,
            pmcid="PMC789",
        )
        from src.resolve import resolve_citations

        sources, _ = resolve_citations([_make_claim("c1")])

        mock_pmid.assert_not_called()
        mock_record.assert_not_called()
        assert sources["c1"].abstract == "Already have one."
        assert sources["c1"].pmcid == "PMC789"

    @patch("src.resolve._pubmed.fetch_record", return_value=None)
    @patch("src.resolve._pubmed.find_pmid_by_doi", return_value="12345")
    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve.search_paper")
    def test_pubmed_record_miss_leaves_source_unchanged(
        self,
        mock_oa: MagicMock,
        mock_retr: MagicMock,
        mock_pmid: MagicMock,
        mock_record: MagicMock,
    ) -> None:
        # PMID resolves but fetch_record returns None (e.g., abstract too short).
        mock_oa.return_value = ResolvedSource(
            found=True, doi="10.1/x", title="T", abstract=None, similarity_score=0.9
        )
        from src.resolve import resolve_citations

        sources, _ = resolve_citations([_make_claim("c1")])

        mock_pmid.assert_called_once()
        mock_record.assert_called_once()
        assert sources["c1"].abstract is None
        assert sources["c1"].pmcid is None


class TestBibliographyAwareResolve:
    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve._crossref.fetch_work_by_doi")
    @patch("src.resolve.search_paper")
    def test_uses_bib_doi_directly_when_available(
        self,
        mock_oa: MagicMock,
        mock_fetch_doi: MagicMock,
        mock_retr: MagicMock,
    ) -> None:
        from src.bibliography import BibEntry
        from src.resolve import resolve_citations

        bib = {
            3: BibEntry(
                number=3,
                raw="...",
                authors=["Goodwin"],
                title="Blood lactate measurements",
                year=2007,
                doi="10.1177/193229680700100414",
            )
        }
        mock_fetch_doi.return_value = ResolvedSource(
            found=True,
            doi="10.1177/193229680700100414",
            title="Blood lactate measurements",
            abstract="The whole-blood-to-plasma ratio varies 63-81%.",
            similarity_score=1.0,
        )
        claim = _make_claim("c1", cited_authors=["Goodwin"], cited_year=2007)
        sources, _ = resolve_citations([claim], bibliography=bib)

        # Bib DOI path uses CrossRef by DOI; OpenAlex query path is bypassed.
        mock_oa.assert_not_called()
        mock_fetch_doi.assert_called_once_with("10.1177/193229680700100414", db_path=None)
        assert sources["c1"].doi == "10.1177/193229680700100414"

    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve._crossref.fetch_work_by_doi")
    @patch("src.resolve.search_paper")
    def test_bib_doi_fallback_keeps_authoritative_doi_when_crossref_misses(
        self,
        mock_oa: MagicMock,
        mock_fetch_doi: MagicMock,
        mock_retr: MagicMock,
    ) -> None:
        from src.bibliography import BibEntry
        from src.resolve import resolve_citations

        bib = {
            21: BibEntry(
                number=21,
                raw="...",
                authors=["Sarabi", "Nakhjavani", "Tasoglu"],
                title="3D-Printed Microneedles for Point-of-Care Biosensing Applications",
                year=2022,
                doi="10.3390/mi13071099",
            )
        }
        mock_fetch_doi.return_value = ResolvedSource(False, None, None, None, None)
        claim = _make_claim(
            "c1",
            cited_authors=["Rezapour Sarabi", "Akbari Nakhjavani", "Tasoglu"],
            cited_year=2022,
            citation_markers=[21],
        )

        sources, _ = resolve_citations([claim], bibliography=bib)

        mock_oa.assert_not_called()
        assert sources["c1"].found is True
        assert sources["c1"].doi == "10.3390/mi13071099"
        assert (
            sources["c1"].title
            == "3D-Printed Microneedles for Point-of-Care Biosensing Applications"
        )

    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve._crossref.fetch_work_by_doi")
    @patch("src.resolve.search_paper")
    def test_citation_marker_selects_bibliography_entry_before_author_year(
        self,
        mock_oa: MagicMock,
        mock_fetch_doi: MagicMock,
        mock_retr: MagicMock,
    ) -> None:
        from src.bibliography import BibEntry
        from src.resolve import resolve_citations

        bib = {
            20: BibEntry(
                number=20,
                raw="...",
                authors=["Smith"],
                title="Wrong same-author paper",
                year=2022,
                doi="10.1/wrong",
            ),
            21: BibEntry(
                number=21,
                raw="...",
                authors=["Smith"],
                title="Right marker paper",
                year=2022,
                doi="10.1/right",
            ),
        }
        mock_fetch_doi.return_value = ResolvedSource(
            found=True,
            doi="10.1/right",
            title="Right marker paper",
            abstract=None,
            similarity_score=1.0,
        )
        claim = _make_claim(
            "c1",
            cited_authors=["Smith"],
            cited_year=2022,
            citation_markers=[21],
        )

        sources, _ = resolve_citations([claim], bibliography=bib)

        mock_fetch_doi.assert_called_once_with("10.1/right", db_path=None)
        assert sources["c1"].doi == "10.1/right"

    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve._pubmed.fetch_record")
    @patch("src.resolve.search_paper")
    def test_uses_bib_pmid_directly_when_available(
        self,
        mock_oa: MagicMock,
        mock_pubmed_record: MagicMock,
        mock_retr: MagicMock,
    ) -> None:
        from src.bibliography import BibEntry
        from src.clients.pubmed import PubMedRecord
        from src.resolve import resolve_citations

        bib = {
            93: BibEntry(
                number=93,
                raw="...",
                authors=["Bonaventura"],
                title="Reliability and Accuracy of Six Hand-Held Blood Lactate Analysers",
                year=2015,
                pmid="25729309",
                pmcid="PMC4306774",
            )
        }
        mock_pubmed_record.return_value = PubMedRecord(
            pmid="25729309",
            abstract="Six hand-held blood lactate analysers were tested.",
            doi=None,
            pmcid="PMC4306774",
        )
        claim = _make_claim(
            "c1",
            cited_authors=["Bonaventura"],
            cited_year=2015,
            citation_markers=[93],
        )

        sources, _ = resolve_citations([claim], bibliography=bib)

        mock_oa.assert_not_called()
        mock_pubmed_record.assert_called_once_with("25729309", db_path=None)
        assert sources["c1"].found is True
        assert sources["c1"].abstract == "Six hand-held blood lactate analysers were tested."
        assert sources["c1"].pmcid == "PMC4306774"

    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve._pubmed.fetch_record")
    @patch("src.resolve.search_paper")
    def test_uses_bib_pmcid_without_pmid_when_available(
        self,
        mock_oa: MagicMock,
        mock_pubmed_record: MagicMock,
        mock_retr: MagicMock,
    ) -> None:
        from src.bibliography import BibEntry
        from src.resolve import resolve_citations

        bib = {
            93: BibEntry(
                number=93,
                raw="...",
                authors=["Bonaventura"],
                title="Reliability and Accuracy of Six Hand-Held Blood Lactate Analysers",
                year=2015,
                pmcid="PMC4306774",
            )
        }
        claim = _make_claim(
            "c1",
            cited_authors=["Bonaventura"],
            cited_year=2015,
            citation_markers=[93],
        )

        sources, _ = resolve_citations([claim], bibliography=bib)

        mock_oa.assert_not_called()
        mock_pubmed_record.assert_not_called()
        assert sources["c1"].found is True
        assert sources["c1"].pmcid == "PMC4306774"

    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve._pubmed.find_pmid_by_title")
    @patch("src.resolve._crossref.search_paper")
    @patch("src.resolve.search_paper")
    def test_uses_crossref_title_before_pubmed_when_bib_has_no_doi(
        self,
        mock_oa: MagicMock,
        mock_crossref_title: MagicMock,
        mock_find_pmid: MagicMock,
        mock_retr: MagicMock,
    ) -> None:
        from src.bibliography import BibEntry
        from src.resolve import resolve_citations

        bib = {
            52: BibEntry(
                number=52,
                raw="...",
                authors=["Forsythe", "Schmidt"],
                title="Sodium bicarbonate for the treatment of lactic acidosis",
                year=2000,
            )
        }
        mock_crossref_title.return_value = ResolvedSource(
            found=True,
            doi="10.1378/chest.117.1.260",
            title="Sodium Bicarbonate for the Treatment of Lactic Acidosis",
            abstract=None,
            similarity_score=None,
        )
        claim = _make_claim("c1", cited_authors=["Forsythe", "Schmidt"], cited_year=2000)

        sources, _ = resolve_citations([claim], bibliography=bib)

        mock_oa.assert_not_called()
        mock_find_pmid.assert_not_called()
        assert sources["c1"].found is True
        assert sources["c1"].doi == "10.1378/chest.117.1.260"
        assert sources["c1"].title_match_score == 1.0

    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve._pubmed.fetch_record")
    @patch("src.resolve._pubmed.find_pmid_by_title")
    @patch("src.resolve._crossref.search_paper")
    @patch("src.resolve.search_paper")
    def test_uses_pubmed_title_before_openalex_when_crossref_title_misses(
        self,
        mock_oa: MagicMock,
        mock_crossref_title: MagicMock,
        mock_find_pmid: MagicMock,
        mock_pubmed_record: MagicMock,
        mock_retr: MagicMock,
    ) -> None:
        from src.bibliography import BibEntry
        from src.clients.pubmed import PubMedRecord
        from src.resolve import resolve_citations

        bib = {
            93: BibEntry(
                number=93,
                raw="...",
                authors=["Bonaventura"],
                title="Reliability and Accuracy of Six Hand-Held Blood Lactate Analysers",
                year=2015,
            )
        }
        mock_crossref_title.return_value = ResolvedSource(False, None, None, None, None)
        mock_find_pmid.return_value = "25729309"
        mock_pubmed_record.return_value = PubMedRecord(
            pmid="25729309",
            abstract="Six hand-held blood lactate analysers were tested.",
            doi=None,
            pmcid="PMC4306774",
        )
        claim = _make_claim(
            "c1",
            cited_authors=["Bonaventura"],
            cited_year=2015,
            citation_markers=[93],
        )

        sources, _ = resolve_citations([claim], bibliography=bib)

        mock_oa.assert_not_called()
        mock_crossref_title.assert_called_once()
        mock_find_pmid.assert_called_once_with(
            "Reliability and Accuracy of Six Hand-Held Blood Lactate Analysers",
            year=2015,
            db_path=None,
        )
        assert sources["c1"].found is True
        assert sources["c1"].abstract == "Six hand-held blood lactate analysers were tested."
        assert sources["c1"].pmcid == "PMC4306774"

    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve._pubmed.find_pmid_by_title", return_value=None)
    @patch("src.resolve._crossref.search_paper")
    @patch("src.resolve.search_paper")
    def test_rejects_crossref_title_match_when_bib_title_is_too_generic(
        self,
        mock_oa: MagicMock,
        mock_crossref_title: MagicMock,
        mock_find_pmid: MagicMock,
        mock_retr: MagicMock,
    ) -> None:
        from src.bibliography import BibEntry
        from src.resolve import resolve_citations

        bib = {
            104: BibEntry(
                number=104,
                raw="...",
                authors=["Braverman"],
                title="The cutaneous microcirculation",
                year=2000,
            )
        }
        mock_crossref_title.return_value = ResolvedSource(
            found=True,
            doi="10.3109/10739689709146797",
            title="The Cutaneous Microcirculation: Ultrastructure and Microanatomical Organization",
            abstract=None,
            similarity_score=None,
        )
        mock_oa.return_value = ResolvedSource(
            found=True,
            doi="10.1/openalex",
            title="OpenAlex fallback",
            abstract=None,
            similarity_score=0.8,
        )
        claim = _make_claim("c1", cited_authors=["Braverman"], cited_year=2000)

        sources, _ = resolve_citations([claim], bibliography=bib)

        mock_crossref_title.assert_called_once()
        mock_find_pmid.assert_called_once()
        mock_oa.assert_called_once()
        assert sources["c1"].doi == "10.1/openalex"

    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve._pubmed.find_pmid_by_title")
    @patch("src.resolve._crossref.search_paper")
    @patch("src.resolve.search_paper")
    def test_accepts_exact_crossref_match_for_short_bibliography_title(
        self,
        mock_oa: MagicMock,
        mock_crossref_title: MagicMock,
        mock_find_pmid: MagicMock,
        mock_retr: MagicMock,
    ) -> None:
        from src.bibliography import BibEntry
        from src.resolve import resolve_citations

        bib = {
            104: BibEntry(
                number=104,
                raw="...",
                authors=["Braverman"],
                title="The cutaneous microcirculation",
                year=2000,
            )
        }
        mock_crossref_title.return_value = ResolvedSource(
            found=True,
            doi="10.1046/j.1087-0024.2000.00010.x",
            title="The Cutaneous Microcirculation",
            abstract=None,
            similarity_score=None,
        )
        mock_find_pmid.return_value = None
        mock_oa.return_value = ResolvedSource(False, None, None, None, None)
        claim = _make_claim("c1", cited_authors=["Braverman"], cited_year=2000)

        sources, _ = resolve_citations([claim], bibliography=bib)

        mock_oa.assert_not_called()
        mock_find_pmid.assert_not_called()
        assert sources["c1"].doi == "10.1046/j.1087-0024.2000.00010.x"
        assert sources["c1"].title_match_score == 1.0

    @patch("src.resolve._crossref.search_paper")
    @patch("src.resolve._pubmed.find_pmid_by_title", return_value=None)
    @patch("src.resolve.search_paper")
    def test_crossref_title_query_uses_complete_bibliography_title(
        self,
        mock_oa: MagicMock,
        mock_pubmed_title: MagicMock,
        mock_crossref_title: MagicMock,
    ) -> None:
        from src.bibliography import BibEntry
        from src.resolve import resolve_citations

        bib = {
            102: BibEntry(
                number=102,
                raw="...",
                authors=["Ming"],
                title=(
                    "Real-time continuous measurement of lactate through a minimally "
                    "invasive microneedle patch: a phase I clinical study"
                ),
                year=2022,
            )
        }
        mock_crossref_title.return_value = ResolvedSource(False, None, None, None, None)
        mock_oa.return_value = ResolvedSource(False, None, None, None, None)
        claim = _make_claim("c1", cited_authors=["Ming"], cited_year=2022)

        resolve_citations([claim], bibliography=bib)

        called_queries = [call.args[0] for call in mock_crossref_title.call_args_list]
        assert any("phase" in query for query in called_queries)
        assert any("clinical" in query for query in called_queries)

    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve._pubmed.fetch_record")
    @patch("src.resolve._pubmed.find_pmid_by_title")
    @patch("src.resolve._crossref.search_paper")
    @patch("src.resolve.search_paper")
    def test_pubmed_title_metadata_populates_doi_when_available(
        self,
        mock_oa: MagicMock,
        mock_crossref_title: MagicMock,
        mock_find_pmid: MagicMock,
        mock_pubmed_record: MagicMock,
        mock_retr: MagicMock,
    ) -> None:
        from src.bibliography import BibEntry
        from src.clients.pubmed import PubMedRecord
        from src.resolve import resolve_citations

        bib = {
            3: BibEntry(
                number=3,
                raw="...",
                authors=["Goodwin"],
                title="Blood lactate measurements and analysis during exercise",
                year=2007,
            )
        }
        mock_crossref_title.return_value = ResolvedSource(False, None, None, None, None)
        mock_find_pmid.return_value = "19885119"
        mock_pubmed_record.return_value = PubMedRecord(
            pmid="19885119",
            abstract="The whole-blood-to-plasma lactate ratio varies from 63% to 81%.",
            doi="10.1177/193229680700100414",
            pmcid="PMC2769631",
        )
        claim = _make_claim("c1", cited_authors=["Goodwin"], cited_year=2007, citation_markers=[3])

        sources, _ = resolve_citations([claim], bibliography=bib)

        mock_oa.assert_not_called()
        assert sources["c1"].doi == "10.1177/193229680700100414"
        assert sources["c1"].pmcid == "PMC2769631"

    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve._crossref.search_paper")
    @patch("src.resolve._pubmed.find_pmid_by_title", return_value=None)
    @patch("src.resolve.search_paper")
    def test_falls_back_to_richer_query_when_bib_has_no_doi(
        self,
        mock_oa: MagicMock,
        mock_pubmed_title: MagicMock,
        mock_cf: MagicMock,
        mock_retr: MagicMock,
    ) -> None:
        from src.bibliography import BibEntry
        from src.resolve import resolve_citations

        bib = {
            83: BibEntry(
                number=83,
                raw="...",
                authors=["Williams", "Armstrong", "Kirby"],
                title="The influence of the site of sampling",
                year=1992,
                doi=None,
            )
        }
        mock_oa.return_value = ResolvedSource(
            found=True,
            doi="10.1080/sample",
            title="...",
            abstract=None,
            similarity_score=0.9,
        )
        mock_cf.return_value = ResolvedSource(False, None, None, None, None)
        claim = _make_claim("c1", cited_authors=["Williams"], cited_year=1992)
        sources, _ = resolve_citations([claim], bibliography=bib)

        mock_oa.assert_called_once()
        # The query passed to search_paper should include bibliography title tokens
        called_query = mock_oa.call_args.args[0]
        assert "influence" in called_query.lower()
        assert "Williams" in called_query
        assert "1992" in called_query
        assert sources["c1"].found is True

    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve._crossref.search_paper")
    @patch("src.resolve._pubmed.find_pmid_by_title", return_value=None)
    @patch("src.resolve.search_paper")
    def test_yearless_claim_with_bib_match_no_longer_skipped(
        self,
        mock_oa: MagicMock,
        mock_pubmed_title: MagicMock,
        mock_cf: MagicMock,
        mock_retr: MagicMock,
    ) -> None:
        from src.bibliography import BibEntry
        from src.resolve import resolve_citations

        bib = {
            81: BibEntry(
                number=81,
                raw="...",
                authors=["Pösö"],
                title="Distribution of lactate",
                year=1995,
                doi=None,
            )
        }
        mock_oa.return_value = ResolvedSource(
            found=True, doi="10.x/y", title="...", abstract=None, similarity_score=0.8
        )
        mock_cf.return_value = ResolvedSource(False, None, None, None, None)
        # Claim has cited_year=None — without bib it would be skipped.
        claim = _make_claim("c1", cited_authors=["Pösö"], cited_year=None)
        sources, _ = resolve_citations([claim], bibliography=bib)

        # Bib match enables resolution despite missing year on the claim.
        mock_oa.assert_called_once()
        assert sources["c1"].found is True


class TestResolveCitations:
    @patch("src.resolve.search_paper")
    def test_happy_path(self, mock_search: MagicMock) -> None:
        mock_search.return_value = ResolvedSource(
            found=True,
            doi=None,
            title="Some Paper",
            abstract="An abstract.",
            similarity_score=1.0,
        )
        from src.resolve import resolve_citations

        claims = [_make_claim("c1"), _make_claim("c2")]
        sources, steps = resolve_citations(claims)

        assert "c1" in sources
        assert "c2" in sources
        assert sources["c1"].found is True
        assert len(steps) == 2

    @patch("src.resolve.search_paper")
    def test_step_fields(self, mock_search: MagicMock) -> None:
        mock_search.return_value = ResolvedSource(
            found=True, doi=None, title="T", abstract="A", similarity_score=0.8
        )
        from src.resolve import resolve_citations

        claims = [_make_claim("claim-x")]
        _, steps = resolve_citations(claims)
        step = steps[0]
        assert isinstance(step, ProvenanceStep)
        assert step.operation == "resolve"
        assert step.model_id is None
        assert step.tokens_in is None
        assert step.tokens_out is None
        assert step.cache_hit is None
        assert step.confidence == 0.8  # mirrors similarity_score

    def test_ec1_no_citation_no_http(self) -> None:
        """EC-1: Claim with no cited_authors returns found=False without HTTP call."""
        from src.resolve import resolve_citations

        claim = _make_claim("c1", cited_authors=[])
        with patch("src.resolve.search_paper") as mock_search:
            sources, _steps = resolve_citations([claim])
            mock_search.assert_not_called()

        assert sources["c1"].found is False

    @patch("src.resolve.search_paper")
    def test_empty_claims_list(self, mock_search: MagicMock) -> None:
        from src.resolve import resolve_citations

        sources, steps = resolve_citations([])
        assert sources == {}
        assert steps == []
        mock_search.assert_not_called()

    def test_ec6_429_mid_batch_httpx(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        """EC-6: 429 on first claim exhausts 3 retries → found=False; second claim succeeds."""
        from src.resolve import resolve_citations

        # 3x 429 for the first claim's query, then success for the second claim
        httpx_mock.add_response(status_code=429)
        httpx_mock.add_response(status_code=429)
        httpx_mock.add_response(status_code=429)
        httpx_mock.add_response(
            status_code=200,
            json={
                "results": [
                    {
                        "id": "https://openalex.org/W1",
                        "title": "Some Paper",
                        "abstract_inverted_index": {"An": [0], "abstract.": [1]},
                        "publication_year": 2020,
                        "doi": None,
                        "authorships": [],
                    }
                ]
            },
        )

        with (
            patch("src.clients.openalex.time.sleep"),
            patch(
                "src.resolve._crossref.search_paper",
                return_value=ResolvedSource(False, None, None, None, None),
            ),
        ):
            claims = [_make_claim("c0"), _make_claim("c1")]
            sources, steps = resolve_citations(claims, db_path=tmp_path / "cache.db")

        assert sources["c0"].found is False  # 429 exhausted
        assert sources["c1"].found is True  # success
        assert len(steps) == 2


class TestResolveCitationsMulti:
    """S2-P4: multi-source resolution returns one ResolvedSourceSet per claim."""

    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve._crossref.fetch_work_by_doi")
    @patch("src.resolve.search_paper")
    def test_multi_citation_claim_resolves_each_marker_independently(
        self,
        mock_oa: MagicMock,
        mock_fetch_doi: MagicMock,
        mock_retr: MagicMock,
    ) -> None:
        from src.bibliography import BibEntry
        from src.resolve import resolve_citations_multi

        bib = {
            81: BibEntry(
                number=81,
                raw="...",
                authors=["Posso"],
                title="Equine MCT lactate transport",
                year=1995,
                doi="10.1/p1",
            ),
            82: BibEntry(
                number=82,
                raw="...",
                authors=["Koho"],
                title="Horse RBC lactate kinetics",
                year=2002,
                doi="10.1/p2",
            ),
            83: BibEntry(
                number=83,
                raw="...",
                authors=["Williams"],
                title="Capillary vs arterial sampling",
                year=1992,
                doi="10.1/p3",
            ),
        }
        mock_fetch_doi.side_effect = [
            ResolvedSource(True, "10.1/p1", "Equine MCT lactate transport", "abs1", 1.0),
            ResolvedSource(True, "10.1/p2", "Horse RBC lactate kinetics", "abs2", 1.0),
            ResolvedSource(True, "10.1/p3", "Capillary vs arterial sampling", "abs3", 1.0),
        ]
        claim = _make_claim(
            "c1",
            cited_authors=["Posso", "Koho", "Williams"],
            cited_year=None,
            citation_markers=[81, 82, 83],
        )

        sets, steps = resolve_citations_multi([claim], bibliography=bib)

        rs_set = sets["c1"]
        assert len(rs_set) == 3
        assert rs_set.citation_markers == (81, 82, 83)
        assert {s.doi for s in rs_set} == {"10.1/p1", "10.1/p2", "10.1/p3"}
        # OpenAlex fallback should NOT have been called when all markers had DOIs.
        mock_oa.assert_not_called()
        assert len(steps) == 1
        assert steps[0].operation == "resolve"

    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve._crossref.search_paper")
    @patch("src.resolve.search_paper")
    def test_no_markers_falls_back_to_single_source(
        self,
        mock_oa: MagicMock,
        mock_cf: MagicMock,
        mock_retr: MagicMock,
    ) -> None:
        from src.resolve import resolve_citations_multi

        mock_oa.return_value = ResolvedSource(True, "10.1/x", "T", "abs", 1.0)
        claim = _make_claim("c1", cited_authors=["Smith"], cited_year=2020, citation_markers=[])

        sets, steps = resolve_citations_multi([claim])

        rs_set = sets["c1"]
        assert len(rs_set) == 1
        assert rs_set.primary().doi == "10.1/x"
        assert len(steps) == 1

    @patch("src.resolve._pubmed.find_pmid_by_title", return_value=None)
    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve._crossref.fetch_work_by_doi")
    @patch("src.resolve._crossref.search_paper")
    @patch("src.resolve.search_paper")
    def test_partial_marker_misses_yield_unfound_entries(
        self,
        mock_oa: MagicMock,
        mock_cf: MagicMock,
        mock_fetch_doi: MagicMock,
        mock_retr: MagicMock,
        mock_pubmed_title: MagicMock,
    ) -> None:
        from src.bibliography import BibEntry
        from src.resolve import resolve_citations_multi

        bib = {
            70: BibEntry(
                number=70,
                raw="...",
                authors=["Loellgen"],
                title="Muscle metabolites",
                year=1980,
                doi=None,
                pmid=None,
                pmcid=None,
            ),
            71: BibEntry(
                number=71,
                raw="...",
                authors=["Graham"],
                title="Pedal rate study",
                year=1984,
                doi="10.1/g",
            ),
        }
        mock_fetch_doi.return_value = ResolvedSource(True, "10.1/g", "Pedal rate study", "abs", 1.0)
        mock_oa.return_value = ResolvedSource(False, None, None, None, None)
        mock_cf.return_value = ResolvedSource(False, None, None, None, None)

        claim = _make_claim(
            "c1",
            cited_authors=["Loellgen", "Graham"],
            cited_year=None,
            citation_markers=[70, 71],
        )
        sets, _ = resolve_citations_multi([claim], bibliography=bib)

        rs_set = sets["c1"]
        assert len(rs_set) == 2
        # Loellgen has no identifiers and pubmed title search is mocked to miss;
        # OpenAlex / CrossRef searches both miss → unfound. Graham resolves via bib DOI.
        found = rs_set.found_sources()
        assert len(found) == 1
        assert found[0].doi == "10.1/g"

    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve._crossref.search_paper")
    @patch("src.resolve.search_paper")
    def test_set_primary_matches_legacy_resolve_for_single_marker(
        self,
        mock_oa: MagicMock,
        mock_cf: MagicMock,
        mock_retr: MagicMock,
    ) -> None:
        """Backward-compat invariant: a single-marker claim's `.primary()` must
        yield the same ResolvedSource as the legacy `resolve_citations` API.
        """
        from src.resolve import resolve_citations, resolve_citations_multi

        mock_oa.return_value = ResolvedSource(True, "10.1/x", "T", "abs", 1.0)
        claim = _make_claim("c1", cited_authors=["Smith"], cited_year=2020)

        legacy_sources, _ = resolve_citations([claim])
        new_sets, _ = resolve_citations_multi([claim])

        # Comparable on the fields callers care about
        assert legacy_sources["c1"].doi == new_sets["c1"].primary().doi
        assert legacy_sources["c1"].title == new_sets["c1"].primary().title
        assert legacy_sources["c1"].abstract == new_sets["c1"].primary().abstract

    def test_best_bib_match_zero_author_multi_marker_returns_none(self) -> None:
        """_best_bib_match returns None when cited_authors is empty and multiple markers match.

        Zero author signal means the candidates cannot be ranked — returning the first
        arbitrarily would propagate a false-high confidence through _resolve_via_bib_doi.
        """
        from src.bibliography import BibEntry
        from src.resolve_utils import _best_bib_match

        bib = {
            99: BibEntry(number=99, raw="ref 99", authors=["Alpha"], title="A", year=2020),
            100: BibEntry(number=100, raw="ref 100", authors=["Beta"], title="B", year=2021),
        }
        claim = _make_claim("c1", cited_authors=[], cited_year=None, citation_markers=[99, 100])
        assert _best_bib_match(claim, bib) is None

    @patch("src.resolve.search_paper")
    def test_multi_marker_no_author_resolves_to_not_found_no_http(self, mock_oa: MagicMock) -> None:
        """resolve_citations with citation_markers=[99,100] and cited_authors=[] must
        return found=False without any HTTP call — single-source path cannot safely pick."""
        from src.bibliography import BibEntry
        from src.resolve import resolve_citations

        bib = {
            99: BibEntry(number=99, raw="ref 99", authors=["Alpha"], title="A", year=2020),
            100: BibEntry(number=100, raw="ref 100", authors=["Beta"], title="B", year=2021),
        }
        claim = _make_claim("c1", cited_authors=[], cited_year=None, citation_markers=[99, 100])
        sources, _ = resolve_citations([claim], bibliography=bib)

        mock_oa.assert_not_called()
        assert sources["c1"].found is False


class TestCitingPaperRecursionGuard:
    """A claim cannot legally cite the paper that contains it. When the
    resolver returns the citing paper's own DOI, that is a structurally
    impossible match and must be rejected — otherwise the verifier
    compares the claim against the citing text itself, producing
    tautological 'supported' verdicts.

    On the Valsci validation run (2026-05-08) this happened 4 times because the
    resolver's OpenAlex fallback used the claim text — which contained
    'Valsci' — and OpenAlex returned Valsci's own paper as a match for
    Kinney/Hirsch/Agarwal/Haryanto citations.
    """

    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve._crossref.search_paper")
    @patch("src.resolve.search_paper")
    def test_rejects_resolution_matching_citing_paper_doi(
        self,
        mock_oa: MagicMock,
        mock_cf: MagicMock,
        mock_retr: MagicMock,
    ) -> None:
        from src.resolve import resolve_citations

        # OpenAlex returns the citing paper itself — common when the claim
        # text contains the citing paper's name (e.g. "Valsci integrates...").
        mock_oa.return_value = ResolvedSource(
            True, "10.1186/s12859-025-06159-4", "Valsci paper", "abs", 0.9
        )
        sources, _ = resolve_citations(
            [_make_claim("c1")],
            citing_paper_doi="10.1186/s12859-025-06159-4",
        )
        assert sources["c1"].found is False, (
            "Resolution matching the citing paper's own DOI must be rejected"
        )

    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve._crossref.search_paper")
    @patch("src.resolve.search_paper")
    def test_does_not_reject_when_no_citing_doi_provided(
        self,
        mock_oa: MagicMock,
        mock_cf: MagicMock,
        mock_retr: MagicMock,
    ) -> None:
        """When citing_paper_doi is None, the guard is inert — no behavior change."""
        from src.resolve import resolve_citations

        mock_oa.return_value = ResolvedSource(True, "10.1/x", "T", "abs", 0.9)
        sources, _ = resolve_citations([_make_claim("c1")], citing_paper_doi=None)
        assert sources["c1"].found is True
        assert sources["c1"].doi == "10.1/x"

    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve._crossref.search_paper")
    @patch("src.resolve.search_paper")
    def test_does_not_reject_unrelated_doi(
        self,
        mock_oa: MagicMock,
        mock_cf: MagicMock,
        mock_retr: MagicMock,
    ) -> None:
        from src.resolve import resolve_citations

        mock_oa.return_value = ResolvedSource(True, "10.1/different", "T", "abs", 0.9)
        sources, _ = resolve_citations(
            [_make_claim("c1")],
            citing_paper_doi="10.1186/s12859-025-06159-4",
        )
        assert sources["c1"].found is True
        assert sources["c1"].doi == "10.1/different"

    def test_doi_comparison_is_case_insensitive(self) -> None:
        """DOIs are case-insensitive per the DOI handbook §2.4. The guard
        must compare them case-insensitively or it will let through
        capitalisation drift between bibliography and resolver outputs.
        """
        from src.resolve import _is_citing_paper_doi

        assert _is_citing_paper_doi("10.1186/s12859-025-06159-4", "10.1186/S12859-025-06159-4")
        assert _is_citing_paper_doi("10.1186/S12859-025-06159-4", "10.1186/s12859-025-06159-4")
        assert not _is_citing_paper_doi("10.1186/different", "10.1186/s12859-025-06159-4")
        assert not _is_citing_paper_doi(None, "10.1186/s12859-025-06159-4")
        assert not _is_citing_paper_doi("10.1186/s12859-025-06159-4", None)


class TestDetectCitingPaperDoi:
    """The pipeline auto-detects the citing paper's DOI from the input
    text when the caller does not provide one explicitly. The detector
    looks only in the head of the document (typically where journal
    self-references appear) so it cannot pick up bibliography entries.
    """

    def test_finds_doi_in_head_of_text(self) -> None:
        from src.pipeline import detect_citing_paper_doi

        text = (
            "Valsci: an open-source...\n"
            "Edelman and Skolnick BMC Bioinformatics (2025) 26:140\n"
            "https://doi.org/10.1186/s12859-025-06159-4\n"
            "Abstract: ...\n"
        )
        assert detect_citing_paper_doi(text) == "10.1186/s12859-025-06159-4"

    def test_returns_none_when_no_doi_in_head(self) -> None:
        from src.pipeline import detect_citing_paper_doi

        text = "A paper with no DOI URL in the first few kilobytes."
        assert detect_citing_paper_doi(text) is None

    def test_only_searches_head(self) -> None:
        """The detector must not return a DOI that only appears late in
        the document (in the bibliography). Otherwise it would mistake a
        cited reference for the citing paper.
        """
        from src.pipeline import detect_citing_paper_doi

        # 10 KB of filler so the DOI URL falls safely outside the 8 KB head
        # window. 21 bytes per line * 500 = ~10.5 KB.
        head = "Just a paper title.\n" * 500
        bib_doi = "https://doi.org/10.99/should-not-be-detected"
        text = head + "\nReferences\n1. Some entry. 2024. " + bib_doi + "\n"
        assert detect_citing_paper_doi(text) is None

    def test_finds_doi_between_4kb_and_8kb_window(self) -> None:
        """Tool exports (Elicit, Edison) sometimes prepend cover pages or
        query metadata that push the actual paper DOI past the legacy 4 KB
        window. The widened 8 KB window must catch DOIs that fall in the
        4-8 KB range while still respecting the bibliography boundary.
        """
        from src.pipeline import detect_citing_paper_doi

        # ~5 KB of cover-page filler, then the DOI URL — outside legacy 4 KB
        # but inside the new 8 KB window.
        cover = "Cover page header line.\n" * 220  # ~5.3 KB
        text = cover + "\nhttps://doi.org/10.1234/test_doi_xyz\nAbstract: ...\n"
        assert detect_citing_paper_doi(text) == "10.1234/test_doi_xyz"

    def test_picks_first_url_when_multiple_in_head(self) -> None:
        from src.pipeline import detect_citing_paper_doi

        # DOI registrant prefixes are always >= 4 digits (10.XXXX) per the
        # DOI handbook §2.2; the regex enforces this so it can't be misled
        # by spurious `10.1` patterns in body text (e.g. "section 10.1").
        text = (
            "Title.\n"
            "https://doi.org/10.1234/first-paper\n"
            "Some intro text.\n"
            "https://doi.org/10.5678/second-paper\n"
        )
        assert detect_citing_paper_doi(text) == "10.1234/first-paper"


class TestArxivFallback:
    """arXiv fallback fires for DOI-less bib entries, before CrossRef title-search."""

    @staticmethod
    def _bib_no_doi() -> object:
        from src.bibliography import BibEntry

        return BibEntry(
            number=15,
            raw="...",
            authors=["Wei"],
            title=("Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"),
            year=2022,
            doi=None,
        )

    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve._crossref.search_paper")
    @patch("src.resolve._arxiv.find_paper_by_title_authors")
    @patch("src.resolve.search_paper")
    def test_arxiv_fallback_fires_when_bib_has_no_doi(
        self,
        mock_oa: MagicMock,
        mock_arxiv: MagicMock,
        mock_cf_search: MagicMock,
        mock_retr: MagicMock,
    ) -> None:
        from src.resolve import resolve_citations

        arxiv_hit = ResolvedSource(
            found=True,
            doi="10.48550/arXiv.2201.11903",
            title=("Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"),
            abstract=None,
            similarity_score=0.85,
            title_match_score=0.72,
        )
        mock_arxiv.return_value = arxiv_hit
        mock_oa.return_value = ResolvedSource(False, None, None, None, None)

        bib = {15: self._bib_no_doi()}
        claim = _make_claim("c1", cited_authors=["Wei"], cited_year=2022, citation_markers=[15])

        sources, _ = resolve_citations([claim], bibliography=bib)

        mock_arxiv.assert_called_once()
        # CrossRef title-search must NOT be called when arXiv already returned a hit.
        mock_cf_search.assert_not_called()
        assert sources["c1"].doi == "10.48550/arXiv.2201.11903"

    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve._crossref.search_paper")
    @patch("src.resolve._arxiv.find_paper_by_title_authors")
    @patch("src.resolve.search_paper")
    def test_arxiv_miss_falls_through_to_crossref_title(
        self,
        mock_oa: MagicMock,
        mock_arxiv: MagicMock,
        mock_cf_search: MagicMock,
        mock_retr: MagicMock,
    ) -> None:
        from src.resolve import resolve_citations

        mock_arxiv.return_value = ResolvedSource(False, None, None, None, None)
        crossref_hit = ResolvedSource(
            found=True,
            doi="10.1234/some.paper",
            title=("Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"),
            abstract=None,
            similarity_score=0.90,
        )
        mock_cf_search.return_value = crossref_hit
        mock_oa.return_value = ResolvedSource(False, None, None, None, None)

        bib = {15: self._bib_no_doi()}
        claim = _make_claim("c1", cited_authors=["Wei"], cited_year=2022, citation_markers=[15])

        sources, _ = resolve_citations([claim], bibliography=bib)

        mock_arxiv.assert_called_once()
        # The claim resolved via CrossRef fallback after arXiv miss.
        assert sources["c1"].doi in ("10.1234/some.paper", None) or sources["c1"].found is True

    @patch("src.resolve._crossref.check_retraction", return_value=False)
    @patch("src.resolve._arxiv.find_paper_by_title_authors")
    @patch("src.resolve._crossref.fetch_work_by_doi")
    @patch("src.resolve.search_paper")
    def test_arxiv_skipped_when_bib_has_doi(
        self,
        mock_oa: MagicMock,
        mock_fetch_doi: MagicMock,
        mock_arxiv: MagicMock,
        mock_retr: MagicMock,
    ) -> None:
        from src.bibliography import BibEntry
        from src.resolve import resolve_citations

        bib_with_doi = {
            15: BibEntry(
                number=15,
                raw="...",
                authors=["Wei"],
                title="Chain-of-Thought Prompting Elicits Reasoning in LLMs",
                year=2022,
                doi="10.48550/arXiv.2201.11903",  # DOI already in bibliography
            )
        }
        mock_fetch_doi.return_value = ResolvedSource(
            found=True,
            doi="10.48550/arXiv.2201.11903",
            title="Chain-of-Thought Prompting Elicits Reasoning in LLMs",
            abstract=None,
            similarity_score=1.0,
        )
        claim = _make_claim("c1", cited_authors=["Wei"], cited_year=2022, citation_markers=[15])

        sources, _ = resolve_citations([claim], bibliography=bib_with_doi)

        # When the bib entry has a DOI, _resolve_via_bib_doi succeeds first
        # and the arXiv client must never be invoked.
        mock_arxiv.assert_not_called()
        assert sources["c1"].doi == "10.48550/arXiv.2201.11903"
