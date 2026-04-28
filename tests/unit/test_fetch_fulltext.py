"""Unit tests for src/fetch_fulltext.py — orchestration with sub-clients patched."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from src.fetch_fulltext import fetch_fulltext
from src.models import ResolvedSource


def _src(
    *,
    found: bool = True,
    doi: str | None = "10.1/x",
    oa_url: str | None = None,
    pmcid: str | None = None,
) -> ResolvedSource:
    return ResolvedSource(
        found=found,
        doi=doi,
        title="t",
        abstract="a",
        similarity_score=1.0,
        oa_url=oa_url,
        pmcid=pmcid,
    )


class TestFetchFulltext:
    def test_no_identifiers_short_circuits(self, tmp_path: Path) -> None:
        with (
            patch("src.fetch_fulltext.pdf.download_and_extract") as mock_pdf,
            patch("src.fetch_fulltext.pmc.fetch_fulltext") as mock_pmc,
            patch("src.fetch_fulltext.unpaywall.get_oa_url") as mock_unp,
        ):
            text, method = fetch_fulltext(_src(doi=None), db_path=tmp_path / "c.db")
            assert text is None
            assert method == "abstract_fallback"
            mock_pdf.assert_not_called()
            mock_pmc.assert_not_called()
            mock_unp.assert_not_called()

    def test_oa_url_path(self, tmp_path: Path) -> None:
        with (
            patch(
                "src.fetch_fulltext.pdf.download_and_extract", return_value="full text"
            ) as mock_pdf,
            patch("src.fetch_fulltext.pmc.fetch_fulltext") as mock_pmc,
        ):
            text, method = fetch_fulltext(
                _src(oa_url="https://x/y.pdf"),
                db_path=tmp_path / "c.db",
            )
            assert text == "full text"
            assert method == "oa_url_pdf"
            mock_pdf.assert_called_once()
            mock_pmc.assert_not_called()

    def test_pmc_path(self, tmp_path: Path) -> None:
        with (
            patch("src.fetch_fulltext.pdf.download_and_extract") as mock_pdf,
            patch("src.fetch_fulltext.pmc.fetch_fulltext", return_value="pmc text") as mock_pmc,
            patch("src.fetch_fulltext.unpaywall.get_oa_url") as mock_unp,
        ):
            text, method = fetch_fulltext(
                _src(pmcid="PMC123"),
                db_path=tmp_path / "c.db",
            )
            assert text == "pmc text"
            assert method == "pmc"
            mock_pmc.assert_called_once_with("PMC123", db_path=tmp_path / "c.db")
            mock_pdf.assert_not_called()
            mock_unp.assert_not_called()

    def test_unpaywall_pdf_path(self, tmp_path: Path) -> None:
        with (
            patch(
                "src.fetch_fulltext.pdf.download_and_extract", return_value="up text"
            ) as mock_pdf,
            patch(
                "src.fetch_fulltext.unpaywall.get_oa_url", return_value="https://x/p.pdf"
            ) as mock_unp,
        ):
            text, method = fetch_fulltext(_src(), db_path=tmp_path / "c.db")
            assert text == "up text"
            assert method == "unpaywall_pdf"
            mock_unp.assert_called_once()
            mock_pdf.assert_called_once()

    def test_pmc_fails_falls_through_to_unpaywall(self, tmp_path: Path) -> None:
        with (
            patch(
                "src.fetch_fulltext.pdf.download_and_extract", return_value="up text"
            ) as mock_pdf,
            patch("src.fetch_fulltext.pmc.fetch_fulltext", return_value=None) as mock_pmc,
            patch(
                "src.fetch_fulltext.unpaywall.get_oa_url", return_value="https://x/p.pdf"
            ) as mock_unp,
        ):
            text, method = fetch_fulltext(
                _src(pmcid="PMC123"),
                db_path=tmp_path / "c.db",
            )
            assert text == "up text"
            assert method == "unpaywall_pdf"
            mock_pmc.assert_called_once()
            mock_unp.assert_called_once()
            mock_pdf.assert_called_once()

    def test_all_fail(self, tmp_path: Path) -> None:
        with (
            patch("src.fetch_fulltext.pdf.download_and_extract", return_value=None),
            patch("src.fetch_fulltext.pmc.fetch_fulltext", return_value=None),
            patch("src.fetch_fulltext.unpaywall.get_oa_url", return_value=None),
        ):
            text, method = fetch_fulltext(
                _src(oa_url="https://x/y.pdf", pmcid="PMC1"),
                db_path=tmp_path / "c.db",
            )
            assert text is None
            assert method == "abstract_fallback"
