"""Unit tests for src/fetch_fulltext.py — orchestration with sub-clients patched.

I1 (2026-05-12): tests updated to assert on the structured FetchOutcome
return shape. Two new tests pin the failure-reason recording semantics.
"""

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
            patch("src.fetch_fulltext.europepmc.fetch_oa_url") as mock_epmc,
            patch("src.fetch_fulltext.unpaywall.get_oa_url") as mock_unp,
        ):
            outcome = fetch_fulltext(_src(doi=None), db_path=tmp_path / "c.db")
            assert outcome.text is None
            assert outcome.method == "abstract_fallback"
            mock_pdf.assert_not_called()
            mock_pmc.assert_not_called()
            mock_epmc.assert_not_called()
            mock_unp.assert_not_called()

    def test_oa_url_path(self, tmp_path: Path) -> None:
        with (
            patch(
                "src.fetch_fulltext.pdf.download_and_extract", return_value="full text"
            ) as mock_pdf,
            patch("src.fetch_fulltext.pmc.fetch_fulltext") as mock_pmc,
        ):
            outcome = fetch_fulltext(
                _src(oa_url="https://x/y.pdf"),
                db_path=tmp_path / "c.db",
            )
            assert outcome.text == "full text"
            assert outcome.method == "oa_url_pdf"
            mock_pdf.assert_called_once()
            mock_pmc.assert_not_called()

    def test_pmc_path(self, tmp_path: Path) -> None:
        with (
            patch("src.fetch_fulltext.pdf.download_and_extract") as mock_pdf,
            patch("src.fetch_fulltext.pmc.fetch_fulltext", return_value="pmc text") as mock_pmc,
            patch("src.fetch_fulltext.unpaywall.get_oa_url") as mock_unp,
        ):
            outcome = fetch_fulltext(
                _src(pmcid="PMC123"),
                db_path=tmp_path / "c.db",
            )
            assert outcome.text == "pmc text"
            assert outcome.method == "pmc"
            mock_pmc.assert_called_once_with("PMC123", db_path=tmp_path / "c.db")
            mock_pdf.assert_not_called()
            mock_unp.assert_not_called()

    def test_unpaywall_pdf_path(self, tmp_path: Path) -> None:
        with (
            patch(
                "src.fetch_fulltext.pdf.download_and_extract", return_value="up text"
            ) as mock_pdf,
            patch("src.fetch_fulltext.europepmc.fetch_oa_url", return_value=None) as mock_epmc,
            patch(
                "src.fetch_fulltext.unpaywall.get_oa_url", return_value="https://x/p.pdf"
            ) as mock_unp,
        ):
            outcome = fetch_fulltext(_src(), db_path=tmp_path / "c.db")
            assert outcome.text == "up text"
            assert outcome.method == "unpaywall_pdf"
            mock_epmc.assert_called_once()
            mock_unp.assert_called_once()
            mock_pdf.assert_called_once()

    def test_pmc_fails_falls_through_to_unpaywall(self, tmp_path: Path) -> None:
        with (
            patch(
                "src.fetch_fulltext.pdf.download_and_extract", return_value="up text"
            ) as mock_pdf,
            patch("src.fetch_fulltext.pmc.fetch_fulltext", return_value=None) as mock_pmc,
            patch("src.fetch_fulltext.europepmc.fetch_oa_url", return_value=None) as mock_epmc,
            patch(
                "src.fetch_fulltext.unpaywall.get_oa_url", return_value="https://x/p.pdf"
            ) as mock_unp,
        ):
            outcome = fetch_fulltext(
                _src(pmcid="PMC123"),
                db_path=tmp_path / "c.db",
            )
            assert outcome.text == "up text"
            assert outcome.method == "unpaywall_pdf"
            mock_pmc.assert_called_once()
            mock_epmc.assert_called_once()
            mock_unp.assert_called_once()
            mock_pdf.assert_called_once()

    def test_europepmc_pdf_path_when_pmc_misses(self, tmp_path: Path) -> None:
        """S2-P1: Europe PMC step fires between PMC and Unpaywall when source
        has a DOI, no oa_url, and PMC fulltext lookup fails. Confirmed by the
        OA discovery probe: claim 005 (Raa) and 022 (Ventrelli) both have
        Europe PMC OA URLs that Unpaywall does not always surface.
        """
        with (
            patch(
                "src.fetch_fulltext.pdf.download_and_extract", return_value="epmc text"
            ) as mock_pdf,
            patch(
                "src.fetch_fulltext.europepmc.fetch_oa_url",
                return_value="https://europepmc.org/articles/PMC7437027/pdf",
            ) as mock_epmc,
            patch("src.fetch_fulltext.unpaywall.get_oa_url") as mock_unp,
        ):
            outcome = fetch_fulltext(
                _src(doi="10.1186/s13049-020-00776-z"),
                db_path=tmp_path / "c.db",
            )
            assert outcome.text == "epmc text"
            assert outcome.method == "europepmc_pdf"
            mock_epmc.assert_called_once()
            mock_pdf.assert_called_once_with(
                "https://europepmc.org/articles/PMC7437027/pdf",
                db_path=tmp_path / "c.db",
            )
            # Unpaywall is skipped when Europe PMC succeeds.
            mock_unp.assert_not_called()

    def test_europepmc_failure_falls_through_to_unpaywall(self, tmp_path: Path) -> None:
        """When Europe PMC returns no URL, fall through to Unpaywall.
        Also covers the case where Europe PMC returns a URL but PDF download
        fails (Europe PMC's URL might 403, Unpaywall is the next chance).
        """
        with (
            patch(
                "src.fetch_fulltext.pdf.download_and_extract",
                side_effect=[None, "up text"],  # epmc fails, unpaywall succeeds
            ),
            patch(
                "src.fetch_fulltext.europepmc.fetch_oa_url",
                return_value="https://europepmc.org/dud.pdf",
            ),
            patch(
                "src.fetch_fulltext.unpaywall.get_oa_url",
                return_value="https://publisher.com/p.pdf",
            ) as mock_unp,
        ):
            outcome = fetch_fulltext(_src(), db_path=tmp_path / "c.db")
            assert outcome.text == "up text"
            assert outcome.method == "unpaywall_pdf"
            mock_unp.assert_called_once()

    def test_all_fail(self, tmp_path: Path) -> None:
        with (
            patch("src.fetch_fulltext.pdf.download_and_extract", return_value=None),
            patch("src.fetch_fulltext.pmc.fetch_fulltext", return_value=None),
            patch("src.fetch_fulltext.europepmc.fetch_oa_url", return_value=None),
            patch("src.fetch_fulltext.unpaywall.get_oa_url", return_value=None),
        ):
            outcome = fetch_fulltext(
                _src(oa_url="https://x/y.pdf", pmcid="PMC1"),
                db_path=tmp_path / "c.db",
            )
            assert outcome.text is None
            assert outcome.method == "abstract_fallback"


class TestFetchFulltextOutcomeReasons:
    """I1 (2026-05-12): per-attempt failure-reason recording.

    These tests pin the contract that downstream telemetry depends on —
    each FetchAttempt records WHY the step failed, not just that it did.
    """

    def test_no_identifiers_records_reason(self, tmp_path: Path) -> None:
        """Source with all None identifiers → one attempt, reason='no_identifiers'."""
        outcome = fetch_fulltext(_src(doi=None), db_path=tmp_path / "c.db")
        assert outcome.text is None
        assert outcome.method == "abstract_fallback"
        assert len(outcome.attempts) == 1
        attempt = outcome.attempts[0]
        assert attempt.method == "abstract_fallback"
        assert attempt.success is False
        assert attempt.reason == "no_identifiers"

    def test_attempts_accumulated_in_chain_order(self, tmp_path: Path) -> None:
        """oa_url fails → pmc fails → publisher_html (unknown) skipped → epmc fails → unpaywall succeeds.

        The attempts tuple should record each failure with its specific
        reason, in the order the chain tried them, plus the final success.
        """
        with (
            patch(
                "src.fetch_fulltext.pdf.download_and_extract",
                side_effect=[None, None, "up text"],  # oa_url fails, epmc fails, unpaywall ok
            ),
            patch("src.fetch_fulltext.pmc.fetch_fulltext", return_value=None),
            patch(
                "src.fetch_fulltext.publisher_html.fetch_via_doi",
                return_value=None,
            ),
            patch(
                "src.fetch_fulltext.europepmc.fetch_oa_url",
                return_value="https://europepmc.org/dud.pdf",
            ),
            patch(
                "src.fetch_fulltext.unpaywall.get_oa_url",
                return_value="https://publisher.com/p.pdf",
            ),
        ):
            outcome = fetch_fulltext(
                _src(oa_url="https://x/y.pdf", pmcid="PMC123", doi="10.99/qual"),
                db_path=tmp_path / "c.db",
            )

        assert outcome.text == "up text"
        assert outcome.method == "unpaywall_pdf"

        # Expect 5 attempts: oa_url_pdf (fail), pmc (fail), publisher_html
        # (fail/unknown — doi prefix 10.99 isn't in the known map),
        # europepmc_pdf (fail), unpaywall_pdf (success).
        methods = [a.method for a in outcome.attempts]
        assert methods == [
            "oa_url_pdf",
            "pmc",
            "publisher_html",
            "europepmc_pdf",
            "unpaywall_pdf",
        ]
        # Reasons on failures must be specific.
        assert outcome.attempts[0].reason == "oa_url_pdf_failed"
        assert outcome.attempts[1].reason == "pmc_no_fulltext"
        assert outcome.attempts[2].reason == "publisher_html_unknown"
        assert outcome.attempts[3].reason == "europepmc_pdf_failed"
        # Final success has reason=None.
        assert outcome.attempts[4].success is True
        assert outcome.attempts[4].reason is None
