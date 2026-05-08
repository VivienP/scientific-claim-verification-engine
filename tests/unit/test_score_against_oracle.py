"""Unit tests for scripts/score_against_oracle.py — pure scoring logic, no IO."""

from __future__ import annotations

import sys
from pathlib import Path

# `scripts/` is not a package; load by path so the test does not depend on
# PYTHONPATH being configured to include it.
_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "score_against_oracle.py"
sys.path.insert(0, str(_SCRIPT.parent))

from score_against_oracle import score_run  # noqa: E402


def _claim(
    claim_id: str = "c1",
    *,
    cited_authors: list[str] | None = None,
    cited_year: int | None = None,
    resolved_doi: str | None = None,
) -> dict:
    return {
        "claim_id": claim_id,
        "cited_authors": cited_authors or [],
        "cited_year": cited_year,
        "source": {"doi": resolved_doi},
    }


def _oracle_entry(
    *,
    cited_authors_first: str,
    cited_year: int | None,
    expected_doi: str | None,
) -> dict:
    return {
        "cited_authors_first": cited_authors_first,
        "cited_year": cited_year,
        "expected_doi": expected_doi,
    }


class TestCorrectSourceRate:
    def test_all_correct_yields_one(self) -> None:
        report = {
            "claims": [
                _claim("c1", cited_authors=["Smith"], cited_year=2020, resolved_doi="10.1/x"),
                _claim("c2", cited_authors=["Jones"], cited_year=2021, resolved_doi="10.2/y"),
            ],
        }
        oracle = {
            "claims": [
                _oracle_entry(cited_authors_first="Smith", cited_year=2020, expected_doi="10.1/x"),
                _oracle_entry(cited_authors_first="Jones", cited_year=2021, expected_doi="10.2/y"),
            ],
        }
        sc = score_run(report, oracle)
        assert sc.correct_source_rate == 1.0
        assert sc.citation_found_rate == 1.0
        assert sc.n_external == 2

    def test_one_wrong_resolution_lowers_correct_rate(self) -> None:
        report = {
            "claims": [
                _claim("c1", cited_authors=["Smith"], cited_year=2020, resolved_doi="10.1/x"),
                _claim("c2", cited_authors=["Jones"], cited_year=2021, resolved_doi="10.2/WRONG"),
            ],
        }
        oracle = {
            "claims": [
                _oracle_entry(cited_authors_first="Smith", cited_year=2020, expected_doi="10.1/x"),
                _oracle_entry(cited_authors_first="Jones", cited_year=2021, expected_doi="10.2/y"),
            ],
        }
        sc = score_run(report, oracle)
        # Both claims resolved to *some* DOI → citation_found_rate = 100%
        assert sc.citation_found_rate == 1.0
        # But only one was correct → correct_source_rate = 50%
        assert sc.correct_source_rate == 0.5

    def test_self_cite_claims_excluded_from_denominator(self) -> None:
        """Claims with no cited_authors describe internal results and should
        not penalize correct_source_rate — they are correctly classified
        as not_addressed by the pipeline.
        """
        report = {
            "claims": [
                _claim("c1", cited_authors=["Smith"], cited_year=2020, resolved_doi="10.1/x"),
                _claim("c2", cited_authors=[], cited_year=None, resolved_doi=None),
                _claim("c3", cited_authors=[], cited_year=None, resolved_doi=None),
            ],
        }
        oracle = {
            "claims": [
                _oracle_entry(cited_authors_first="Smith", cited_year=2020, expected_doi="10.1/x"),
            ],
        }
        sc = score_run(report, oracle)
        assert sc.n_self_cite == 2
        assert sc.n_external == 1
        assert sc.correct_source_rate == 1.0

    def test_doi_comparison_case_insensitive(self) -> None:
        """DOIs are case-insensitive per the DOI handbook §2.4."""
        report = {
            "claims": [
                _claim("c1", cited_authors=["Smith"], cited_year=2020, resolved_doi="10.1/X"),
            ],
        }
        oracle = {
            "claims": [
                _oracle_entry(cited_authors_first="Smith", cited_year=2020, expected_doi="10.1/x"),
            ],
        }
        sc = score_run(report, oracle)
        assert sc.correct_source_rate == 1.0

    def test_unmapped_external_claim_counted_as_wrong(self) -> None:
        """A claim that cites externally but has no oracle entry must NOT
        be counted as correct (the oracle is incomplete; we cannot vouch
        for it). Otherwise oracle-coverage gaps would inflate the score.
        """
        report = {
            "claims": [
                _claim(
                    "c1",
                    cited_authors=["Unknown"],
                    cited_year=2020,
                    resolved_doi="10.1/something",
                ),
            ],
        }
        oracle = {"claims": []}
        sc = score_run(report, oracle)
        assert sc.n_external == 1
        assert sc.correct_source_rate == 0.0

    def test_resolved_to_no_doi_is_not_correct(self) -> None:
        report = {
            "claims": [
                _claim("c1", cited_authors=["Smith"], cited_year=2020, resolved_doi=None),
            ],
        }
        oracle = {
            "claims": [
                _oracle_entry(cited_authors_first="Smith", cited_year=2020, expected_doi="10.1/x"),
            ],
        }
        sc = score_run(report, oracle)
        assert sc.citation_found_rate == 0.0
        assert sc.correct_source_rate == 0.0
