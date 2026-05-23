"""Negative-control suite pinning `extract_claims()` rejection behavior.

Asserts that the extractor returns zero claims on non-claim text that
legitimately appears inside scientific publications (References sections,
Methods-only paragraphs, headings, captions, Acknowledgments/Funding) and
exactly two claims on a calibration positive control.

Each fixture's mocked LLM response is a real one-off Anthropic API capture,
committed under `tests/unit/fixtures/extract_negative/anthropic_responses/`.
Tests parametrized over fixtures whose canned response does not yet exist
will skip — see commit 0.3 of the Phase 0 plan in
`reports/you-are-claude-code-bubbly-firefly.md`.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from anthropic.types import TextBlock

from src.extract import extract_claims

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "extract_negative"
RESPONSE_DIR = FIXTURE_DIR / "anthropic_responses"

NEGATIVE_CONTROLS: list[tuple[str, int]] = [
    ("references_section_only", 0),
    ("methods_only", 0),
    ("figure_table_captions", 0),
    ("headings_only", 0),
    ("acknowledgments_funding", 0),
    ("results_paragraph_positive_control", 2),
]


def _text_block(text: str) -> TextBlock:
    return TextBlock(type="text", text=text)


def _wire_mock_anthropic(mock_anthropic_cls: MagicMock, response_text: str) -> None:
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_response = MagicMock()
    mock_response.content = [_text_block(response_text)]
    mock_response.usage.input_tokens = 0
    mock_response.usage.output_tokens = 0
    mock_response.usage.cache_read_input_tokens = 0
    mock_response.usage.cache_creation_input_tokens = 0
    mock_stream_ctx = MagicMock()
    mock_stream_ctx.__enter__.return_value = mock_stream_ctx
    mock_stream_ctx.get_final_message.return_value = mock_response
    mock_client.messages.stream.return_value = mock_stream_ctx


@pytest.mark.parametrize(
    ("fixture_name", "expected_claim_count"),
    NEGATIVE_CONTROLS,
    ids=[name for name, _ in NEGATIVE_CONTROLS],
)
@patch("src.extract.anthropic.Anthropic")
def test_extract_claims_negative_control(
    mock_anthropic_cls: MagicMock,
    fixture_name: str,
    expected_claim_count: int,
) -> None:
    response_path = RESPONSE_DIR / f"{fixture_name}.json"
    if not response_path.exists():
        pytest.skip(
            f"Canned Anthropic response not yet recorded for {fixture_name}. "
            "See commit 0.3 of reports/you-are-claude-code-bubbly-firefly.md."
        )

    fixture_text = (FIXTURE_DIR / f"{fixture_name}.txt").read_text(encoding="utf-8")
    canned_response = response_path.read_text(encoding="utf-8")
    _wire_mock_anthropic(mock_anthropic_cls, canned_response)

    claims, _step = extract_claims(fixture_text)

    actual = len(claims)
    direction = "over-extraction" if actual > expected_claim_count else "under-extraction"
    assert actual == expected_claim_count, (
        f"{fixture_name}: expected {expected_claim_count} claims, got {actual} "
        f"({direction} regression)"
    )
