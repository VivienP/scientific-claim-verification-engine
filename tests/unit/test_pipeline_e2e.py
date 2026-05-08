"""End-to-end integration tests for src/pipeline.run_pipeline.

These tests exercise the *full* pipeline — extraction → bibliography →
multi-source resolve → fetch → chunk → BM25 select → verifier dispatch —
with mocking only at the two true I/O boundaries:

  * anthropic.Anthropic (LLM SDK)
  * httpx (CrossRef, OpenAlex, PubMed, Europe PMC, Unpaywall, PDF)

Everything in between (parser regex, content tokens, JSON marshalling,
ProvenanceStep emission, routing decision tree, multi-source aggregation)
runs unmocked. Because the LLM is the dominant cost driver and the only
non-determinism source, this gives us deterministic, offline, repeatable
end-to-end coverage that can run in pre-commit without network or credit
spend.

These complement (not replace) the per-function unit tests in
tests/unit/. A passing unit suite + a passing end-to-end suite together
mean: the routing decision tree is correct AND the integration of the
verifier with the resolver / fetcher / report builder produces a coherent
artifact.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
from pytest_httpx import HTTPXMock

from src.models import Claim
from src.pipeline import PipelineConfig, run_pipeline
from src.verify_protocol import assert_verifier_steps_valid

# A tiny but realistic input. The bibliography parser needs the
# "References" header + numbered entries to populate the bib map.
INPUT_TEXT = """\
Lactate concentration in interstitial fluid (ISF) tracks blood lactate
with a small lag of ~5 minutes [1].

References

[1] Smith J, Jones A. Lactate in ISF. Journal of Whatever (2020).
DOI: 10.1234/lactate.2020
"""


def _llm_extract_response() -> str:
    return json.dumps(
        {
            "claims": [
                {
                    "claim_id": "x1",
                    "claim_text": "Lactate concentration in ISF tracks blood lactate "
                    "with a small lag of ~5 minutes",
                    "cited_authors": ["Smith", "Jones"],
                    "cited_year": 2020,
                    "claim_type": "factual_qualitative",
                }
            ]
        }
    )


def _llm_verify_response() -> str:
    return json.dumps(
        {
            "status": "supported",
            "explanation": "The abstract states a 5-minute lag.",
            "confidence": 0.9,
        }
    )


def _make_mock_message(text: str) -> MagicMock:
    """Mimic anthropic.types.Message shape used by src.verify / src.extract.

    Uses real anthropic.types.TextBlock instances because the verifier code
    routes on ``isinstance(first_block, TextBlock)`` — a MagicMock would
    silently fail that check and yield an empty response_text.
    """
    from anthropic.types import TextBlock

    block = TextBlock(citations=None, text=text, type="text")
    msg = MagicMock()
    msg.content = [block]
    msg.usage = MagicMock(
        input_tokens=200,
        output_tokens=80,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=200,
    )
    return msg


def _llm_router(prompt_marker: str) -> MagicMock:
    """Pick extract vs. verify based on a sentinel in the user message."""
    if "<text>" in prompt_marker:
        return _make_mock_message(_llm_extract_response())
    return _make_mock_message(_llm_verify_response())


class TestRunPipelineEndToEnd:
    """The pipeline should produce a coherent artifact for a synthetic input."""

    def _setup_http_mocks(self, httpx_mock: HTTPXMock) -> None:
        """Catch-all callback: route every URL to the right canned response.

        pytest_httpx routes by exact URL; an integration test that does not
        know the resolver's exact query string would fail. Using a callback
        gives us URL-prefix routing without scripting every variant.
        """

        def respond(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "/works/10.1234%2Flactate.2020" in url:
                return httpx.Response(
                    200,
                    json={
                        "message": {
                            "DOI": "10.1234/lactate.2020",
                            "title": ["Lactate in ISF"],
                            "abstract": "<jats:p>Lactate in ISF tracks blood lactate "
                            "with a ~5 minute lag.</jats:p>",
                        }
                    },
                )
            if "api.openalex.org/works" in url:
                return httpx.Response(200, json={"results": []})
            if "api.crossref.org/works" in url:
                return httpx.Response(200, json={"message": {"items": []}})
            if "eutils.ncbi.nlm.nih.gov" in url or "europepmc.org" in url:
                return httpx.Response(200, json={"esearchresult": {"idlist": []}})
            if "api.unpaywall.org" in url:
                return httpx.Response(200, json={"best_oa_location": None})
            return httpx.Response(404, json={})

        httpx_mock.add_callback(respond, is_reusable=True)

    def _setup_llm_mock(self, mock_create: MagicMock) -> None:
        def respond(**kwargs: object) -> MagicMock:
            messages: Iterable[Any] = kwargs.get("messages", [])  # type: ignore[assignment]
            user_text = ""
            for m in messages:
                if isinstance(m, dict) and m.get("role") == "user":
                    content = m.get("content", "")
                    user_text += content if isinstance(content, str) else ""
            return _llm_router(user_text)

        mock_create.side_effect = respond

    @patch("anthropic.Anthropic")
    def test_pipeline_produces_one_claim_verification_with_valid_steps(
        self, mock_anthropic_cls: MagicMock, httpx_mock: HTTPXMock
    ) -> None:
        self._setup_http_mocks(httpx_mock)
        client_instance = MagicMock()
        self._setup_llm_mock(client_instance.messages.create)
        mock_anthropic_cls.return_value = client_instance

        cvs, _ = run_pipeline(
            INPUT_TEXT,
            config=PipelineConfig(api_key="sk-test"),
        )

        assert len(cvs) == 1
        cv = cvs[0]
        assert cv.claim.claim_text.startswith("Lactate concentration")
        # The verifier must yield one of the four valid statuses regardless
        # of which resolver path won (DOI, search fallback, or unresolved
        # leading to citing-context fallback).
        assert cv.result.status in {
            "supported",
            "partially_supported",
            "unsupported",
            "not_addressed",
        }
        # Verifier-emitted steps must respect the ProvenanceStep contract.
        assert_verifier_steps_valid(
            cv.claim, [s for s in cv.steps if s.operation in {"verify", "aggregate"}]
        )

    @patch("anthropic.Anthropic")
    def test_pipeline_emits_extract_resolve_fetch_verify_step_chain(
        self, mock_anthropic_cls: MagicMock, httpx_mock: HTTPXMock
    ) -> None:
        self._setup_http_mocks(httpx_mock)
        client_instance = MagicMock()
        self._setup_llm_mock(client_instance.messages.create)
        mock_anthropic_cls.return_value = client_instance

        _, all_steps = run_pipeline(
            INPUT_TEXT,
            config=PipelineConfig(api_key="sk-test"),
        )

        ops = [s.operation for s in all_steps]
        # Every step type that pipeline.run_pipeline guarantees emits at least once.
        assert "extract" in ops
        assert "resolve" in ops
        assert "fetch_fulltext" in ops
        assert "verify" in ops

    @patch("anthropic.Anthropic")
    def test_pipeline_skips_extract_when_pre_extracted_claims_provided(
        self, mock_anthropic_cls: MagicMock, httpx_mock: HTTPXMock
    ) -> None:
        self._setup_http_mocks(httpx_mock)
        client_instance = MagicMock()
        # If extract were called, this would route to _llm_router with <text>
        # and return an "extract" payload. We give it a verify-shaped response
        # so any unexpected extract call would raise on JSON parse.
        client_instance.messages.create.side_effect = lambda **kw: _make_mock_message(
            _llm_verify_response()
        )
        mock_anthropic_cls.return_value = client_instance

        pre_claim = Claim(
            claim_id="pre1",
            claim_text="Lactate concentration in ISF tracks blood lactate.",
            cited_authors=["Smith", "Jones"],
            cited_year=2020,
            claim_type="factual_qualitative",
        )
        cvs, all_steps = run_pipeline(
            INPUT_TEXT,
            config=PipelineConfig(api_key="sk-test"),
            pre_extracted_claims=[pre_claim],
        )
        assert len(cvs) == 1
        ops = [s.operation for s in all_steps]
        # The pre-extracted path must NOT have emitted an extract step.
        assert "extract" not in ops
        assert "resolve" in ops
        assert "verify" in ops
