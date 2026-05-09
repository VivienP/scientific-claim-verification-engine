"""Unit tests for src/copilot/primary_lookup.py — mocked HTTP via pytest-httpx."""

from __future__ import annotations

from pathlib import Path

from pytest_httpx import HTTPXMock

from src.copilot.primary_lookup import find_primary_source_doi
from src.models import ProvenanceStep

_SS_REFS_URL_PREFIX = "https://api.semanticscholar.org/graph/v1/paper/"
_CROSSREF_DOI_URL_PREFIX = "https://api.crossref.org/works/"

_RCT_REFERENCE = {
    "title": "A Randomized Controlled Trial of Drug X",
    "abstract": "Randomized controlled trial of 250 patients with type 2 diabetes.",
    "year": 2019,
    "externalIds": {"DOI": "10.9999/rct.2019"},
}

_REVIEW_REFERENCE = {
    "title": "A Systematic Review of Drug X",
    "abstract": "Systematic review of literature on Drug X efficacy.",
    "year": 2020,
    "externalIds": {"DOI": "10.9999/review.2020"},
}

_SS_RESPONSE_WITH_RCT = {
    "data": [
        {"citedPaper": _RCT_REFERENCE},
        {"citedPaper": _REVIEW_REFERENCE},
    ]
}

_SS_RESPONSE_EMPTY = {"data": []}

_CROSSREF_VERIFIED_RESPONSE = {
    "message": {
        "DOI": "10.9999/rct.2019",
        "title": ["A Randomized Controlled Trial of Drug X"],
    }
}


class TestFindPrimarySourceDoi:
    def test_returns_primary_doi_when_found(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(json=_SS_RESPONSE_WITH_RCT)
        httpx_mock.add_response(json=_CROSSREF_VERIFIED_RESPONSE)

        doi, title, _step = find_primary_source_doi("10.1234/review", db_path=tmp_path / "c.db")

        assert doi == "10.9999/rct.2019"
        assert title == "A Randomized Controlled Trial of Drug X"

    def test_returns_none_when_no_doi_provided(self, tmp_path: Path) -> None:
        doi, title, _step = find_primary_source_doi(None, db_path=tmp_path / "c.db")
        assert doi is None
        assert title is None

    def test_returns_none_when_ss_empty(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(json=_SS_RESPONSE_EMPTY)

        doi, title, _step = find_primary_source_doi("10.1234/review", db_path=tmp_path / "c.db")

        assert doi is None
        assert title is None

    def test_returns_none_when_ss_returns_only_secondary(
        self, httpx_mock: HTTPXMock, tmp_path: Path
    ) -> None:
        ss_response_secondary_only = {"data": [{"citedPaper": _REVIEW_REFERENCE}]}
        httpx_mock.add_response(json=ss_response_secondary_only)

        doi, _title, _step = find_primary_source_doi("10.1234/review", db_path=tmp_path / "c.db")

        assert doi is None

    def test_returns_none_when_crossref_fails(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(json=_SS_RESPONSE_WITH_RCT)
        # CrossRef returns 404 for the candidate DOI
        httpx_mock.add_response(status_code=404)

        doi, _title, _step = find_primary_source_doi("10.1234/review", db_path=tmp_path / "c.db")

        assert doi is None

    def test_returns_none_when_ss_404(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(status_code=404)

        doi, _title, _step = find_primary_source_doi("10.1234/unknown", db_path=tmp_path / "c.db")

        assert doi is None

    def test_returns_none_when_ss_network_error(
        self, httpx_mock: HTTPXMock, tmp_path: Path
    ) -> None:
        import httpx as _httpx

        httpx_mock.add_exception(_httpx.ConnectError("timeout"))

        doi, _title, _step = find_primary_source_doi("10.1234/review", db_path=tmp_path / "c.db")

        assert doi is None

    def test_candidate_without_doi_returns_none(
        self, httpx_mock: HTTPXMock, tmp_path: Path
    ) -> None:
        no_doi_ref = {
            "title": "RCT without DOI",
            "abstract": "Randomized controlled trial of n = 100 patients.",
            "year": 2018,
            "externalIds": {},
        }
        httpx_mock.add_response(json={"data": [{"citedPaper": no_doi_ref}]})

        doi, _title, _step = find_primary_source_doi("10.1234/review", db_path=tmp_path / "c.db")

        assert doi is None


class TestFindPrimarySourceDoiProvenance:
    def test_step_operation(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(json=_SS_RESPONSE_EMPTY)

        _, _, step = find_primary_source_doi("10.1234/x", db_path=tmp_path / "c.db")

        assert isinstance(step, ProvenanceStep)
        assert step.operation == "copilot_primary_lookup"

    def test_step_model_id_is_none(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(json=_SS_RESPONSE_EMPTY)

        _, _, step = find_primary_source_doi("10.1234/x", db_path=tmp_path / "c.db")

        assert step.model_id is None

    def test_step_hashes_are_hex_strings(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(json=_SS_RESPONSE_EMPTY)

        _, _, step = find_primary_source_doi("10.1234/x", db_path=tmp_path / "c.db")

        assert len(step.input_hash) == 64
        assert len(step.output_hash) == 64

    def test_step_tokens_are_none(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(json=_SS_RESPONSE_EMPTY)

        _, _, step = find_primary_source_doi("10.1234/x", db_path=tmp_path / "c.db")

        assert step.tokens_in is None
        assert step.tokens_out is None

    def test_same_input_same_output_hash(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(json=_SS_RESPONSE_EMPTY)
        httpx_mock.add_response(json=_SS_RESPONSE_EMPTY)

        _, _, step1 = find_primary_source_doi("10.1234/x", db_path=tmp_path / "c1.db")
        _, _, step2 = find_primary_source_doi("10.1234/x", db_path=tmp_path / "c2.db")

        assert step1.output_hash == step2.output_hash


class TestYearScoringPreference:
    def test_prefers_year_closest_to_claim(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        ref_2018 = {
            "title": "RCT 2018",
            "abstract": "Randomized controlled trial, n = 50 patients.",
            "year": 2018,
            "externalIds": {"DOI": "10.9999/rct2018"},
        }
        ref_2015 = {
            "title": "RCT 2015",
            "abstract": "Randomized controlled trial, n = 50 patients.",
            "year": 2015,
            "externalIds": {"DOI": "10.9999/rct2015"},
        }
        ss_response = {"data": [{"citedPaper": ref_2018}, {"citedPaper": ref_2015}]}

        crossref_response_2018 = {"message": {"DOI": "10.9999/rct2018", "title": ["RCT 2018"]}}

        httpx_mock.add_response(json=ss_response)
        httpx_mock.add_response(json=crossref_response_2018)

        doi, _, _ = find_primary_source_doi(
            "10.1234/review", claim_year=2019, db_path=tmp_path / "c.db"
        )

        # 2018 is closer to 2019 than 2015 → should pick 2018
        assert doi == "10.9999/rct2018"
