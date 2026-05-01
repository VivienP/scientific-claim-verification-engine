"""Unit tests for src/bm25_selector.py — deterministic BM25 passage selection."""

from __future__ import annotations

from src.bm25_selector import select_passages
from src.models import PaperChunk


def _chunk(text: str, idx: int = 0) -> PaperChunk:
    return PaperChunk(
        doi="10.1/x",
        section="other",
        text=text,
        char_start=idx * 100,
        char_end=idx * 100 + len(text),
    )


class TestSelectPassages:
    def test_returns_top_3_by_default(self) -> None:
        chunks = [
            _chunk("apples and oranges grow on trees in fruit orchards near valleys", 0),
            _chunk("the cat sat on a mat reading the newspaper this morning quietly", 1),
            _chunk("apples are red and apples are crunchy and apples taste delicious", 2),
            _chunk("dogs run fast across the green field chasing balls and squirrels", 3),
            _chunk("apples grow on apple trees in autumn season every year reliably", 4),
        ]
        result = select_passages("apples", chunks, top_k=3)
        assert len(result) == 3
        # All top-3 should mention apples (the highest-scoring)
        for chunk in result:
            assert "apple" in chunk.text.lower()

    def test_term_match_ranks_higher(self) -> None:
        chunks = [
            _chunk("the weather is nice today and we are going outside for a walk", 0),
            _chunk("dogs and cats both make wonderful companions for many families", 1),
            _chunk("the field of mathematics has many branches including topology", 2),
            _chunk("VEGF protein expression was reduced in treated cells significantly", 3),
            _chunk("today the sun is shining and the temperature is mild outside", 4),
        ]
        result = select_passages("VEGF protein expression", chunks, top_k=1)
        assert len(result) == 1
        assert "VEGF" in result[0].text

    def test_empty_chunks_returns_empty(self) -> None:
        assert select_passages("anything", [], top_k=3) == []

    def test_fewer_chunks_than_top_k(self) -> None:
        chunks = [_chunk("only chunk here with some content for the test", 0)]
        result = select_passages("chunk", chunks, top_k=3)
        assert result == chunks

    def test_zero_scores_returns_empty(self) -> None:
        chunks = [
            _chunk("alpha beta gamma delta epsilon zeta eta theta iota kappa", 0),
            _chunk("lambda mu nu xi omicron pi rho sigma tau upsilon phi chi", 1),
            _chunk("psi omega aleph bet gimel dalet he waw zayin het tet yod", 2),
        ]
        # query has zero overlap with any chunk
        result = select_passages("xxxx yyyy zzzz", chunks, top_k=2)
        assert result == []

    def test_deterministic(self) -> None:
        chunks = [
            _chunk("apples and pears are tasty fruits that grow in temperate orchards", 0),
            _chunk("oranges and lemons are citrus fruits with vitamin C and acid", 1),
            _chunk("apples make delicious pies when baked with cinnamon sugar", 2),
        ]
        a = select_passages("apples", chunks, top_k=2)
        b = select_passages("apples", chunks, top_k=2)
        assert a == b

    def test_top_k_respected(self) -> None:
        chunks = [
            _chunk(f"chunk number {i} about the topic of interest here", i) for i in range(10)
        ]
        result = select_passages("topic interest", chunks, top_k=5)
        assert len(result) == 5
