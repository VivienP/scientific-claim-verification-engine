"""Unit tests for src/clients/europepmc.py — all HTTP mocked via pytest-httpx.

Europe PMC complements NCBI by exposing OA mirrors and abstracts that CrossRef
and PubMed sometimes miss. The probe at eval/e2e/probes/_oa_discovery_probe.py
demonstrated this on claims 005 (Raa BMC OA) and 022 (Ventrelli S2/EuPMC OA).
"""

from __future__ import annotations

from pathlib import Path

from pytest_httpx import HTTPXMock

from src.clients.europepmc import (
    EuropePMCRecord,
    fetch_abstract,
    fetch_oa_url,
    fetch_record,
    find_pmcid_by_doi,
)

_DOI = "10.1186/s13049-020-00776-z"
_API_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def _full_response(
    *,
    pmcid: str | None = "PMC7437027",
    abstract: str = "Capillary lactate is higher than arterial lactate in ICU patients.",
    is_oa: str = "Y",
    pdf_url: str | None = "https://europepmc.org/articles/PMC7437027/pdf",
    html_url: str | None = "https://europepmc.org/article/PMC/PMC7437027",
) -> dict[str, object]:
    fulltext_list: list[dict[str, str]] = []
    if pdf_url:
        fulltext_list.append(
            {
                "documentStyle": "pdf",
                "site": "Europe_PMC",
                "url": pdf_url,
                "availability": "Open access",
                "availabilityCode": "OA",
            }
        )
    if html_url:
        fulltext_list.append(
            {
                "documentStyle": "html",
                "site": "Europe_PMC",
                "url": html_url,
                "availability": "Open access",
                "availabilityCode": "OA",
            }
        )
    return {
        "version": "6.9",
        "hitCount": 1,
        "resultList": {
            "result": [
                {
                    "id": "12345",
                    "source": "MED",
                    "pmid": "12345",
                    "pmcid": pmcid,
                    "doi": _DOI,
                    "title": "Sample title",
                    "abstractText": abstract,
                    "isOpenAccess": is_oa,
                    "fullTextUrlList": {"fullTextUrl": fulltext_list},
                }
            ]
        },
    }


def _empty_response() -> dict[str, object]:
    return {"version": "6.9", "hitCount": 0, "resultList": {"result": []}}


class TestFetchRecord:
    def test_returns_full_data_for_oa_paper(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(json=_full_response())
        record = fetch_record(_DOI, db_path=tmp_path / "epmc.db")

        assert record is not None
        assert record.pmcid == "PMC7437027"
        assert "Capillary lactate" in (record.abstract or "")
        assert record.is_open_access is True
        assert record.pdf_url == "https://europepmc.org/articles/PMC7437027/pdf"
        assert record.html_url == "https://europepmc.org/article/PMC/PMC7437027"

    def test_returns_none_on_empty_result(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(json=_empty_response())
        assert fetch_record("10.invalid/missing", db_path=tmp_path / "epmc.db") is None

    def test_returns_none_on_http_error(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(status_code=503)
        assert fetch_record(_DOI, db_path=tmp_path / "epmc.db") is None

    def test_caches_positive_response(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        db = tmp_path / "epmc.db"
        httpx_mock.add_response(json=_full_response())

        r1 = fetch_record(_DOI, db_path=db)
        r2 = fetch_record(_DOI, db_path=db)

        assert r1 == r2
        # Second call hits the cache; only one HTTP request was made.
        assert len(httpx_mock.get_requests()) == 1

    def test_caches_negative_response(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        db = tmp_path / "epmc.db"
        httpx_mock.add_response(json=_empty_response())

        r1 = fetch_record("10.miss/x", db_path=db)
        r2 = fetch_record("10.miss/x", db_path=db)

        assert r1 is None
        assert r2 is None
        assert len(httpx_mock.get_requests()) == 1

    def test_handles_paper_without_pmcid(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(
            json=_full_response(pmcid=None, pdf_url=None, html_url=None, is_oa="N")
        )
        record = fetch_record(_DOI, db_path=tmp_path / "epmc.db")

        assert record is not None
        assert record.pmcid is None
        assert record.pdf_url is None
        assert record.is_open_access is False


class TestConvenienceAccessors:
    def test_fetch_oa_url_prefers_pdf(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(json=_full_response())
        url = fetch_oa_url(_DOI, db_path=tmp_path / "epmc.db")
        assert url == "https://europepmc.org/articles/PMC7437027/pdf"

    def test_fetch_oa_url_falls_back_to_html(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(json=_full_response(pdf_url=None))
        url = fetch_oa_url(_DOI, db_path=tmp_path / "epmc.db")
        assert url == "https://europepmc.org/article/PMC/PMC7437027"

    def test_fetch_oa_url_returns_none_when_paywall(
        self, httpx_mock: HTTPXMock, tmp_path: Path
    ) -> None:
        httpx_mock.add_response(json=_full_response(pdf_url=None, html_url=None, is_oa="N"))
        assert fetch_oa_url(_DOI, db_path=tmp_path / "epmc.db") is None

    def test_fetch_abstract_returns_text(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(json=_full_response(abstract="Specific assertion here."))
        text = fetch_abstract(_DOI, db_path=tmp_path / "epmc.db")
        assert text == "Specific assertion here."

    def test_find_pmcid_by_doi_returns_pmcid(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(json=_full_response())
        pmcid = find_pmcid_by_doi(_DOI, db_path=tmp_path / "epmc.db")
        assert pmcid == "PMC7437027"

    def test_record_dataclass_is_frozen(self) -> None:
        rec = EuropePMCRecord(
            pmcid="PMC1",
            abstract="abs",
            pdf_url=None,
            html_url=None,
            is_open_access=False,
        )
        try:
            rec.pmcid = "PMC2"  # type: ignore[misc]
        except (AttributeError, TypeError):
            return
        raise AssertionError("EuropePMCRecord must be frozen")
