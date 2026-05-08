"""Unit tests for src/clients/crossref.py — all HTTP mocked via pytest-httpx."""

from __future__ import annotations

from pathlib import Path

from pytest_httpx import HTTPXMock

from src.clients.crossref import check_retraction, fetch_work_by_doi, search_paper

_WORKS_URL = "https://api.crossref.org/works"
_DOI = "10.1234/test.2023"

_GOOD_RESPONSE = {
    "message": {
        "items": [
            {
                "DOI": _DOI,
                "title": ["A Great Paper on Things"],
            }
        ]
    }
}

_RETRACTION_RESPONSE = {
    "message": {
        "DOI": _DOI,
        "title": ["A Great Paper on Things"],
        "update-to": [{"type": "retraction", "DOI": "10.1234/retraction.2024"}],
    }
}

_CORRECTION_RESPONSE = {
    "message": {
        "DOI": _DOI,
        "title": ["A Great Paper on Things"],
        "update-to": [{"type": "correction", "DOI": "10.1234/correction.2024"}],
    }
}


class TestSearchPaper:
    def test_happy_path(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(json=_GOOD_RESPONSE)
        result = search_paper("Smith 2023 protein folding", db_path=tmp_path / "c.db")
        assert result.found is True
        assert result.doi == _DOI
        assert result.title == "A Great Paper on Things"
        assert result.abstract is None

    def test_empty_items(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(json={"message": {"items": []}})
        result = search_paper("nobody 1900 nothing", db_path=tmp_path / "c.db")
        assert result.found is False

    def test_network_error(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        import httpx as _httpx

        httpx_mock.add_exception(_httpx.ConnectError("refused"))
        result = search_paper("query", db_path=tmp_path / "c.db")
        assert result.found is False

    def test_429_retry_then_success(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(status_code=429)
        httpx_mock.add_response(json=_GOOD_RESPONSE)
        result = search_paper("Smith 2023", db_path=tmp_path / "c.db")
        assert result.found is True

    def test_cache_hit_skips_http(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        db = tmp_path / "c.db"
        httpx_mock.add_response(json=_GOOD_RESPONSE)
        r1 = search_paper("Smith 2023", db_path=db)
        r2 = search_paper("Smith 2023", db_path=db)
        assert r1.doi == r2.doi
        # Only one HTTP call was registered
        assert len(httpx_mock.get_requests()) == 1

    def test_doi_stripped(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        resp = {"message": {"items": [{"DOI": "https://doi.org/10.5678/foo", "title": ["T"]}]}}
        httpx_mock.add_response(json=resp)
        result = search_paper("query", db_path=tmp_path / "c.db")
        assert result.doi == "10.5678/foo"

    def test_prefers_journal_article_over_preprint_candidate(
        self, httpx_mock: HTTPXMock, tmp_path: Path
    ) -> None:
        httpx_mock.add_response(
            json={
                "message": {
                    "items": [
                        {
                            "DOI": "10.1101/preprint",
                            "type": "posted-content",
                            "title": [
                                "Real-time Continuous Measurement of Lactate through a "
                                "Minimally-invasive Microneedle Biosensor: a Phase I "
                                "Clinical Study"
                            ],
                        },
                        {
                            "DOI": "10.1136/bmjinnov-2021-000864",
                            "type": "journal-article",
                            "title": [
                                "Real-time continuous measurement of lactate through a "
                                "minimally invasive microneedle patch: a phase I "
                                "clinical study"
                            ],
                        },
                    ]
                }
            }
        )

        result = search_paper(
            "Real-time continuous measurement of lactate through a minimally invasive "
            "microneedle patch: a phase I clinical study Ming 2022",
            db_path=tmp_path / "c.db",
        )

        assert result.doi == "10.1136/bmjinnov-2021-000864"
        assert "rows=5" in str(httpx_mock.get_requests()[0].url)

    def test_does_not_prefer_short_title_when_long_title_has_more_overlap(
        self, httpx_mock: HTTPXMock, tmp_path: Path
    ) -> None:
        """Bug C (S1-P1-C): the asymmetric score `|q & t| / |t|` overweighted
        short titles whose tokens were all in the query. Cause of the
        wrong-pick on claim 005 (Collange short title beat Raa long title).
        Jaccard `|q & t| / |q | t|` is symmetric and rewards absolute overlap.

        Calibration: short title has ALL its content tokens in the query
        (asymmetric 2/2 = 1.0); long title has more total overlap but adds
        out-of-query tokens (asymmetric 4/8 = 0.5). Under asymmetric, short
        wins (bug). Under Jaccard, long wins (fix).
        """
        query = "capillary lactate sampling depth dermal Raa 2020 ICU"
        short_title_all_in_query = "Capillary lactate"  # 2 tokens, both in query
        # 8 content tokens, 4 in query.
        long_title_more_overlap = (
            "Comparison of capillary and arterial lactate sampling ICU shock patients"
        )
        httpx_mock.add_response(
            json={
                "message": {
                    "items": [
                        {
                            "DOI": "10.bad/short",
                            "type": "journal-article",
                            "title": [short_title_all_in_query],
                        },
                        {
                            "DOI": "10.good/long",
                            "type": "journal-article",
                            "title": [long_title_more_overlap],
                        },
                    ]
                }
            }
        )

        result = search_paper(query, db_path=tmp_path / "c.db")

        assert result.doi == "10.good/long"

    def test_zero_overlap_yields_zero_score_without_error(
        self, httpx_mock: HTTPXMock, tmp_path: Path
    ) -> None:
        """Edge case: a candidate whose title shares no content tokens with
        the query must score 0.0 deterministically (no zero-division).
        """
        httpx_mock.add_response(
            json={
                "message": {
                    "items": [
                        {
                            "DOI": "10.match/yes",
                            "type": "journal-article",
                            "title": ["Lactate kinetics during exercise"],
                        },
                        {
                            "DOI": "10.unrelated/no",
                            "type": "journal-article",
                            "title": ["Cosmological inflation"],
                        },
                    ]
                }
            }
        )

        result = search_paper("lactate kinetics exercise", db_path=tmp_path / "c.db")

        assert result.doi == "10.match/yes"


class TestMultiSignalScore:
    """S4b-4: title-only Jaccard could not distinguish candidates whose
    titles overlap similarly with the query. Author + year signals break
    the tie when the query carries them."""

    def test_author_match_breaks_title_overlap_tie(
        self, httpx_mock: HTTPXMock, tmp_path: Path
    ) -> None:
        """Two candidates with identical title-Jaccard against the query;
        only one matches the cited author surname. The author-matching
        candidate must win.
        """
        # Both titles share these tokens with the query: "lactate", "ISF".
        # Title Jaccard alone is identical for both. The query carries
        # "Smith 2020"; only the first item lists Smith as an author.
        query = "Smith 2020 lactate ISF"
        httpx_mock.add_response(
            json={
                "message": {
                    "items": [
                        {
                            "DOI": "10.right/smith",
                            "type": "journal-article",
                            "title": ["Lactate in ISF"],
                            "author": [{"family": "Smith", "given": "J."}],
                            "issued": {"date-parts": [[2020]]},
                        },
                        {
                            "DOI": "10.wrong/jones",
                            "type": "journal-article",
                            "title": ["Lactate in ISF"],
                            "author": [{"family": "Jones", "given": "B."}],
                            "issued": {"date-parts": [[2018]]},
                        },
                    ]
                }
            }
        )
        result = search_paper(query, db_path=tmp_path / "c.db")
        assert result.doi == "10.right/smith"

    def test_year_match_breaks_remaining_tie_when_authors_equal(
        self, httpx_mock: HTTPXMock, tmp_path: Path
    ) -> None:
        """Both candidates match the query author equally; only one matches
        the year. The matching-year candidate wins."""
        query = "Smith 2020 lactate ISF"
        httpx_mock.add_response(
            json={
                "message": {
                    "items": [
                        {
                            "DOI": "10.wrong/year",
                            "type": "journal-article",
                            "title": ["Lactate in ISF"],
                            "author": [{"family": "Smith", "given": "J."}],
                            "issued": {"date-parts": [[2010]]},
                        },
                        {
                            "DOI": "10.right/year",
                            "type": "journal-article",
                            "title": ["Lactate in ISF"],
                            "author": [{"family": "Smith", "given": "J."}],
                            "issued": {"date-parts": [[2020]]},
                        },
                    ]
                }
            }
        )
        result = search_paper(query, db_path=tmp_path / "c.db")
        assert result.doi == "10.right/year"

    def test_no_author_signal_falls_back_to_title_jaccard(
        self, httpx_mock: HTTPXMock, tmp_path: Path
    ) -> None:
        """When the query carries no recognisable author, the score
        reduces to title Jaccard (* 0.5) plus type bonus — preserving
        the pre-S4b-4 ordering."""
        # Query has no surname.
        query = "lactate kinetics exercise"
        httpx_mock.add_response(
            json={
                "message": {
                    "items": [
                        {
                            "DOI": "10.match/yes",
                            "type": "journal-article",
                            "title": ["Lactate kinetics during exercise"],
                            "author": [{"family": "Adams", "given": "X."}],
                        },
                        {
                            "DOI": "10.unrelated/no",
                            "type": "journal-article",
                            "title": ["Cosmological inflation"],
                            "author": [{"family": "Bell", "given": "Y."}],
                        },
                    ]
                }
            }
        )
        result = search_paper(query, db_path=tmp_path / "c.db")
        assert result.doi == "10.match/yes"


class TestFetchWorkByDoi:
    def test_fetches_exact_doi_endpoint(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(
            json={
                "message": {
                    "DOI": "10.3390/mi13071099",
                    "title": ["3D-Printed Microneedles for Point-of-Care Biosensing Applications"],
                }
            }
        )

        result = fetch_work_by_doi("10.3390/mi13071099", db_path=tmp_path / "c.db")

        assert result.found is True
        assert result.doi == "10.3390/mi13071099"
        assert result.title == "3D-Printed Microneedles for Point-of-Care Biosensing Applications"
        assert "/works/10.3390%2Fmi13071099" in str(httpx_mock.get_requests()[0].url)

    def test_404_returns_not_found(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(status_code=404)

        result = fetch_work_by_doi("10.404/missing", db_path=tmp_path / "c.db")

        assert result.found is False


class TestCheckRetraction:
    def _url(self) -> str:
        import urllib.parse

        return f"{_WORKS_URL}/{urllib.parse.quote(_DOI, safe='')}"

    def test_retraction_detected(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(json=_RETRACTION_RESPONSE)
        assert check_retraction(_DOI, db_path=tmp_path / "c.db") is True

    def test_correction_not_retraction(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(json=_CORRECTION_RESPONSE)
        assert check_retraction(_DOI, db_path=tmp_path / "c.db") is False

    def test_network_error_returns_false(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        import httpx as _httpx

        httpx_mock.add_exception(_httpx.ConnectError("refused"))
        assert check_retraction(_DOI, db_path=tmp_path / "c.db") is False

    def test_cache_hit_skips_http(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        db = tmp_path / "c.db"
        httpx_mock.add_response(json=_RETRACTION_RESPONSE)
        r1 = check_retraction(_DOI, db_path=db)
        r2 = check_retraction(_DOI, db_path=db)
        assert r1 is True
        assert r2 is True
        assert len(httpx_mock.get_requests()) == 1
