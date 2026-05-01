"""Unit tests for src/bm25_selector.py — deterministic BM25 passage selection."""

from __future__ import annotations

import structlog
from structlog.testing import capture_logs

from src.bm25_selector import _ENCODER, select_passages
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


class TestTokenBudget:
    def test_token_budget_respected_when_all_small(self) -> None:
        """All chunks small → top_k still acts, total tokens stay under budget."""
        chunks = [
            _chunk(f"apples grow in orchards across many regions of the world {i}", i)
            for i in range(5)
        ]
        result = select_passages("apples orchards", chunks, top_k=3, max_total_tokens=6000)
        assert len(result) == 3
        total = sum(len(_ENCODER.encode(c.text)) for c in result)
        assert total < 6000

    def test_budget_stops_at_overshoot(self) -> None:
        """3 medium chunks where 1+2 fit, but adding 3rd overshoots → only 2 returned."""
        # Each chunk ~145 tokens (133 + 12 wrapper). Budget 350:
        # ch0 → running=145; ch1 → 290 (fits); ch2 → 435 > 350 → break.
        sentence = "TREM2 expression in microglia regulates inflammation in disease. "
        chunks = [_chunk((sentence * 10) + f"marker_{i}", i) for i in range(3)]
        result = select_passages(
            "TREM2 microglia inflammatory", chunks, top_k=3, max_total_tokens=350
        )
        assert len(result) == 2

    def test_oversized_single_chunk_truncated_at_sentence_boundary(self) -> None:
        """Single chunk far over budget → truncated at sentence boundary, no mid-word cut."""
        sentence = "TREM2 modulates microglial activation in Alzheimer's models. "
        big_text = sentence * 200  # ~3200 tokens
        original = PaperChunk(
            doi="10.1/big",
            section="results",
            text=big_text,
            char_start=5000,
            char_end=5000 + len(big_text),
        )
        max_tokens = 200
        result = select_passages(
            "TREM2 microglia", [original], top_k=3, max_total_tokens=max_tokens
        )
        assert len(result) == 1
        truncated = result[0]
        # Shorter than original
        assert len(truncated.text) < len(big_text)
        # Ends at a sentence boundary
        assert truncated.text.rstrip().endswith((".", "!", "?"))
        # No mid-word cut: the last word of truncated text must appear whole in original
        last_word = truncated.text.rstrip().rstrip(".!?").split()[-1]
        assert last_word in big_text
        # char_start/char_end preserved (Flaw 2 — they reference the source paper)
        assert truncated.char_start == original.char_start
        assert truncated.char_end == original.char_end
        # Tokenizer-based, not char-based — encoded text must fit the budget
        assert len(_ENCODER.encode(truncated.text)) <= max_tokens

    def test_warning_logged_on_truncation(self) -> None:
        """Truncation must emit a structlog warning with chunk metadata."""
        # structlog needs to be configured for capture_logs to see entries
        structlog.configure(
            processors=[structlog.testing.LogCapture()],
            wrapper_class=structlog.make_filtering_bound_logger(0),
            cache_logger_on_first_use=False,
        )
        sentence = "TREM2 modulates microglial activation in models of neurodegeneration. "
        big_text = sentence * 200
        chunk = PaperChunk(
            doi="10.1/big",
            section="results",
            text=big_text,
            char_start=5000,
            char_end=5000 + len(big_text),
        )
        with capture_logs() as logs:
            select_passages("TREM2 microglia", [chunk], top_k=3, max_total_tokens=200)
        truncation_events = [e for e in logs if e.get("event") == "chunk_truncated_to_fit"]
        assert len(truncation_events) == 1
        evt = truncation_events[0]
        assert evt["doi"] == "10.1/big"
        assert evt["section"] == "results"
        assert evt["char_start"] == 5000
        assert evt["char_end"] == 5000 + len(big_text)
        assert evt["original_chars"] == len(big_text)
        assert evt["truncated_chars"] < evt["original_chars"]
        assert evt["original_tokens"] > evt["truncated_tokens"]
        assert evt["budget"] == 200

    def test_top_k_hard_ceiling(self) -> None:
        """top_k caps result size even when budget allows more."""
        chunks = [
            _chunk(f"TREM2 microglia regulation pathway number {i} described here", i)
            for i in range(10)
        ]
        result = select_passages("TREM2 microglia", chunks, top_k=3, max_total_tokens=100_000)
        assert len(result) == 3

    def test_early_return_path_respects_budget(self) -> None:
        """Regression test: with chunks <= top_k, budget must still apply.

        Pre-fix this hit the `if len(chunks) <= top_k: return list(chunks)` early
        return and bypassed any budget check — Edison TREM2's exact failure mode.
        """
        sentence = "TREM2 expression in microglia regulates inflammatory pathways. "
        big_text = sentence * 300
        small = _chunk("TREM2 brief mention.", 1)
        chunks = [
            PaperChunk(
                doi="10.1/big",
                section="results",
                text=big_text,
                char_start=0,
                char_end=len(big_text),
            ),
            small,
        ]
        # top_k=3, only 2 chunks → pre-fix would return both unchanged.
        # Post-fix: budget enforced, big chunk truncated, then small chunk added if it fits.
        result = select_passages("TREM2 microglia", chunks, top_k=3, max_total_tokens=300)
        # First chunk must be the big one (highest BM25 score), and it must have been truncated
        assert len(result) >= 1
        assert len(result[0].text) < len(big_text)

    def test_deterministic_with_budget(self) -> None:
        """Same inputs (including budget kwarg) → identical output across calls."""
        chunks = [
            _chunk("TREM2 microglia activation pathway A described here in detail", 0),
            _chunk("TREM2 deficient mice show altered microglial morphology consistently", 1),
            _chunk("unrelated discussion of cardiovascular outcomes in older adults", 2),
        ]
        a = select_passages("TREM2 microglia", chunks, top_k=2, max_total_tokens=500)
        b = select_passages("TREM2 microglia", chunks, top_k=2, max_total_tokens=500)
        assert a == b

    def test_char_offsets_preserved_on_truncation(self) -> None:
        """char_start/char_end reference the source paper — not modified by truncation."""
        sentence = "TREM2 modulates microglial response. "
        big_text = sentence * 300
        original = PaperChunk(
            doi="10.1/preserve",
            section="methods",
            text=big_text,
            char_start=1000,
            char_end=9000,  # deliberately != char_start + len(text)
        )
        result = select_passages("TREM2 microglial", [original], top_k=1, max_total_tokens=150)
        assert len(result) == 1
        assert result[0].char_start == 1000
        assert result[0].char_end == 9000
