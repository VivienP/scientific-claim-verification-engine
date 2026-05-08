"""Unit tests for src/clients/arxiv.py — all HTTP mocked via pytest-httpx.

arXiv is the primary authority for ML/AI preprints. This client is inserted
before CrossRef title-search in the bib-fallback chain so papers like
Wei 2022 (Chain-of-Thought) resolve to arXiv:2201.11903 instead of a
mis-matched journal record.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from pytest_httpx import HTTPXMock

from src.clients.arxiv import find_paper_by_title_authors

_ATOM_NS = "http://www.w3.org/2005/Atom"

# ---------------------------------------------------------------------------
# XML fixtures
# ---------------------------------------------------------------------------


def _feed_xml(entries: list[dict[str, object]]) -> str:
    """Build an Atom feed XML string from a list of entry dicts."""
    entry_blocks = []
    for e in entries:
        authors_xml = "".join(
            f"<author xmlns='{_ATOM_NS}'><name>{a}</name></author>"
            for a in (e.get("authors") or [])
        )
        published = f"{e['year']}-01-01T00:00:00Z" if e.get("year") else ""
        entry_blocks.append(
            f"""<entry xmlns="{_ATOM_NS}">
  <id>http://arxiv.org/abs/{e["arxiv_id"]}v1</id>
  <title>{e["title"]}</title>
  {authors_xml}
  <published>{published}</published>
</entry>"""
        )
    entries_xml = "\n".join(entry_blocks)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="{_ATOM_NS}">
  <title type="html">arXiv Query</title>
  {entries_xml}
</feed>"""


def _wei_feed_xml() -> str:
    return _feed_xml(
        [
            {
                "arxiv_id": "2201.11903",
                "title": ("Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"),
                "authors": ["Jason Wei", "Xuezhi Wang"],
                "year": 2022,
            }
        ]
    )


def _empty_feed_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="{_ATOM_NS}">
  <title type="html">arXiv Query</title>
</feed>"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFindPaperByTitleAuthors:
    def test_finds_correct_paper_by_title(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(text=_wei_feed_xml())

        result = find_paper_by_title_authors(
            "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
            ["Wei"],
            2022,
            db_path=tmp_path / "arxiv.db",
        )

        assert result.found is True
        assert result.doi == "10.48550/arXiv.2201.11903"
        assert result.title is not None
        assert "Chain-of-Thought" in result.title

    def test_returns_not_found_on_no_results(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(text=_empty_feed_xml())

        result = find_paper_by_title_authors(
            "Some paper title here",
            ["Author"],
            2020,
            db_path=tmp_path / "arxiv.db",
        )

        assert result.found is False
        assert result.doi is None

    def test_low_score_candidates_rejected(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        # A paper with completely unrelated title and author should not pass
        # the minimum composite score threshold of 0.4.
        unrelated_feed = _feed_xml(
            [
                {
                    "arxiv_id": "9999.99999",
                    "title": "Quantum Computing Applications in Chemistry",
                    "authors": ["Alice Zhao"],
                    "year": 2018,
                }
            ]
        )
        httpx_mock.add_response(text=unrelated_feed)

        result = find_paper_by_title_authors(
            "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
            ["Wei"],
            2022,
            db_path=tmp_path / "arxiv.db",
        )

        assert result.found is False

    def test_score_uses_multi_signal_blend(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        """Author overlap should dominate over pure title similarity.

        Entry A: author=Wei (strong author match, zero title overlap)
        Entry B: author=Jones (no author match, strong title overlap)
        Query: title="Chain-of-Thought Prompting", authors=["Wei"], year=2022

        With 50% title / 30% author / 15% year weighting, A wins because
        author match (0.3 * 1.0) outweighs title overlap advantage of B.
        We verify that the same margin is produced by _candidate_score directly.
        """
        two_entry_feed = _feed_xml(
            [
                {
                    "arxiv_id": "2201.00001",
                    "title": "Reasoning Survey Overview",  # zero title overlap
                    "authors": ["Jason Wei"],  # strong author match
                    "year": 2022,
                },
                {
                    "arxiv_id": "2201.00002",
                    "title": "Chain-of-Thought Prompting Methods",  # good title overlap
                    "authors": ["Bob Jones"],  # no author match
                    "year": 2022,
                },
            ]
        )
        httpx_mock.add_response(text=two_entry_feed)

        result = find_paper_by_title_authors(
            "Chain-of-Thought Prompting",
            ["Wei"],
            2022,
            db_path=tmp_path / "arxiv.db",
        )

        # Entry A (Wei) should win due to author overlap outweighing title
        assert result.found is True
        assert result.doi == "10.48550/arXiv.2201.00001"

        # Verify that _candidate_score agrees with our winner selection
        from src.clients.crossref import _candidate_score  # type: ignore[attr-defined]

        scoring_query = "Chain-of-Thought Prompting Wei 2022"
        item_a = {
            "title": ["Reasoning Survey Overview"],
            "author": [{"family": "wei"}],
            "issued": {"date-parts": [[2022]]},
            "DOI": "10.48550/arXiv.2201.00001",
        }
        item_b = {
            "title": ["Chain-of-Thought Prompting Methods"],
            "author": [{"family": "jones"}],
            "issued": {"date-parts": [[2022]]},
            "DOI": "10.48550/arXiv.2201.00002",
        }
        score_a, _, _ = _candidate_score(scoring_query, item_a, 0)
        score_b, _, _ = _candidate_score(scoring_query, item_b, 1)

        assert score_a > score_b, (
            f"_candidate_score says A ({score_a:.3f}) should beat B ({score_b:.3f})"
        )

    def test_caches_search_response(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        db = tmp_path / "arxiv.db"
        httpx_mock.add_response(text=_wei_feed_xml())

        r1 = find_paper_by_title_authors(
            "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
            ["Wei"],
            2022,
            db_path=db,
        )
        r2 = find_paper_by_title_authors(
            "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
            ["Wei"],
            2022,
            db_path=db,
        )

        assert r1 == r2
        assert r1.found is True
        # Second call must be served from cache — only one HTTP request.
        assert len(httpx_mock.get_requests()) == 1

    @patch("src.clients.arxiv.time.sleep")
    def test_retries_on_429(
        self, mock_sleep: MagicMock, httpx_mock: HTTPXMock, tmp_path: Path
    ) -> None:
        httpx_mock.add_response(status_code=429)
        httpx_mock.add_response(text=_wei_feed_xml())

        result = find_paper_by_title_authors(
            "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
            ["Wei"],
            2022,
            db_path=tmp_path / "arxiv.db",
        )

        assert result.found is True
        assert result.doi == "10.48550/arXiv.2201.11903"
        mock_sleep.assert_called_once()
