"""Unit tests for src/chunker.py — deterministic IMRAD chunking."""

from __future__ import annotations

from src.chunker import chunk_paper

_FULL_PAPER = """Abstract
This paper studies foo and bar in the context of baz, with implications across many fields.

Introduction
This is a longer introduction section discussing background, motivation, and prior work in detail.

Methods
We used standard procedures for cell culture, RNA extraction, and statistical analysis throughout.

Results
We observed a significant increase in expression with our experimental treatment compared to control.

Discussion
Our findings suggest a novel mechanism by which X regulates Y in this biological system."""


class TestChunker:
    def test_full_imrad(self) -> None:
        chunks = chunk_paper("10.1/x", _FULL_PAPER)
        sections = [c.section for c in chunks]
        assert "introduction" in sections
        assert "methods" in sections
        assert "results" in sections
        assert "discussion" in sections

    def test_section_order_preserved(self) -> None:
        chunks = chunk_paper("10.1/x", _FULL_PAPER)
        starts = [c.char_start for c in chunks]
        assert starts == sorted(starts)

    def test_no_headers_returns_single_other(self) -> None:
        text = "Just a single paragraph of running text with no section headers at all to be found here."
        chunks = chunk_paper("10.1/x", text)
        assert len(chunks) == 1
        assert chunks[0].section == "other"

    def test_empty_text_returns_empty(self) -> None:
        assert chunk_paper("10.1/x", "") == []
        assert chunk_paper("10.1/x", "   \n\n  ") == []

    def test_materials_and_methods(self) -> None:
        text = "Materials and Methods\n" + ("This describes the procedures used. " * 10)
        chunks = chunk_paper("10.1/x", text)
        assert any(c.section == "methods" for c in chunks)

    def test_conclusions_maps_to_discussion(self) -> None:
        text = (
            "Introduction\n"
            + ("Some intro text describing the problem in detail. " * 5)
            + "\nConclusions\n"
            + ("Wrap-up text describing what we learned. " * 5)
        )
        chunks = chunk_paper("10.1/x", text)
        sections = [c.section for c in chunks]
        assert "introduction" in sections
        assert "discussion" in sections

    def test_numbered_introduction(self) -> None:
        text = "1. Introduction\n" + ("Text body with substantial content for the chunker. " * 5)
        chunks = chunk_paper("10.1/x", text)
        assert any(c.section == "introduction" for c in chunks)

    def test_char_offsets_slice_back(self) -> None:
        chunks = chunk_paper("10.1/x", _FULL_PAPER)
        for chunk in chunks:
            assert _FULL_PAPER[chunk.char_start : chunk.char_end] == chunk.text

    def test_doi_propagated(self) -> None:
        chunks = chunk_paper("10.42/test", _FULL_PAPER)
        assert all(c.doi == "10.42/test" for c in chunks)

    def test_short_chunks_filtered(self) -> None:
        # Sections with < 50 chars between headers are dropped
        text = "Methods\nshort\nResults\n" + ("Long content. " * 20)
        chunks = chunk_paper("10.1/x", text)
        # "short" alone (5 chars) under methods should be filtered
        for c in chunks:
            assert len(c.text) >= 50

    def test_deterministic(self) -> None:
        a = chunk_paper("10.1/x", _FULL_PAPER)
        b = chunk_paper("10.1/x", _FULL_PAPER)
        assert a == b

    def test_text_before_first_header_is_other(self) -> None:
        text = (
            "Some preamble text without any section header for the first portion of the paper. " * 3
            + "\nIntroduction\n"
            + ("Real introduction body with enough content to pass the filter. " * 5)
        )
        chunks = chunk_paper("10.1/x", text)
        sections = [c.section for c in chunks]
        assert sections[0] == "other"
        assert "introduction" in sections
