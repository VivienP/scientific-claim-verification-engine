"""Integration test pinning the user-caught NEJM Goodwin 2022 case (Track D5).

This test makes a real HTTPS request to https://www.nejm.org and asserts
that the publisher_html fallback successfully retrieves the article body
containing the exact figure that the original silent-failure bug missed:
"The incidence of sustained response at week 12 was 20% in the 25-mg group."

Gated behind `--run-integration` per tests/integration/conftest.py and
`.claude/rules/offline-tests.md`. Skipped in the default pytest run so
pre-commit stays offline.

Why this exists: on 2026-05-11 a user flagged that our verifier emitted
status="unsupported", confidence=0.75 on Elicit's claim about Goodwin's
sustained-response figure. The full text supports the claim verbatim,
but the fetch chain at the time only tried PDF endpoints and gave up
when Semantic Scholar's `oa_url` pointed at NEJM's paywalled PDF
endpoint. Track D1 added `publisher_html` to the fetch chain. This
test pins that fix: if NEJM ever changes their HTML structure or the
fetch chain regresses, this test breaks and surfaces the regression.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.fetch_fulltext import fetch_fulltext
from src.models import ResolvedSource

_GOODWIN_DOI = "10.1056/NEJMoa2206443"
_GOODWIN_TITLE = "Single-Dose Psilocybin for a Treatment-Resistant Episode of Major Depression"


def _goodwin_source() -> ResolvedSource:
    """Reconstruct the Goodwin ResolvedSource as Semantic Scholar resolved it
    in `reports/runs/elicit_psilocybin_rerun_860b1ae5/report.json` — including
    the paywalled `oa_url` that the original bug surfaced.
    """
    return ResolvedSource(
        found=True,
        doi=_GOODWIN_DOI,
        title=_GOODWIN_TITLE,
        abstract=(
            "In this phase 2 double-blind trial, we randomly assigned adults "
            "with treatment-resistant depression to receive a single dose of "
            "psilocybin at 25 mg, 10 mg, or 1 mg (control)..."
        ),
        similarity_score=0.95,
        oa_url="https://www.nejm.org/doi/pdf/10.1056/NEJMoa2206443?articleTools=true",
        pmcid=None,
    )


@pytest.mark.xfail(
    strict=False,
    reason=(
        "NEJM serves Cloudflare bot-challenge interstitials and returns 403 "
        "to programmatic HTTP clients (including ours, regardless of User-Agent). "
        "Their robots.txt explicitly disallows GPTBot/CCBot/PerplexityBot/"
        "SemanticScholarBot and Cloudflare blocks the rest at the edge. "
        "Bypassing this would require a headless browser (Playwright) or "
        "paid NEJM API access — both out of scope. "
        "Track A's `unverifiable` safety net handles this case correctly: "
        "for NEJM papers without OA, the verifier emits "
        "(status='unverifiable', confidence=None) instead of the previous "
        "silent confident-`unsupported` failure. See regression entry "
        "elicit_psilocybin__ae1ff864 — resolved structurally by Track A, "
        "even though Track D's HTML fetch cannot retrieve NEJM directly."
    ),
)
def test_goodwin_nejm_2022_fetches_fulltext_via_publisher_html(
    tmp_path: Path,
) -> None:
    """The Goodwin NEJM 2022 paper is fetchable via the publisher_html step.

    Currently EXPECTED TO FAIL due to NEJM's Cloudflare bot protection
    (see xfail decorator above). When NEJM eventually serves us — via API
    auth, a headless-browser fetcher, or any other route — this test
    will start passing and the xfail will surface (strict=False keeps
    CI green either way).

    Asserts what success would look like: fetch chain returns non-None
    text via the publisher_html method, containing the exact 20% figure
    that the original verifier could not see at abstract depth.
    """
    db_path = tmp_path / "fetch_cache.db"
    source = _goodwin_source()

    outcome = fetch_fulltext(source, db_path=db_path)
    text, method = outcome.text, outcome.method

    # The chain successfully retrieved full text.
    assert text is not None, (
        f"fetch_fulltext returned None for Goodwin {_GOODWIN_DOI}. "
        f"method={method!r}. Check publisher_html fetcher or NEJM HTML structure."
    )

    # Via the publisher_html route (not oa_url_pdf, which is paywalled).
    assert method == "publisher_html", (
        f"Expected method='publisher_html' for Goodwin, got {method!r}. "
        f"The oa_url path may have changed, or chain order may have regressed."
    )

    # Contains the marquee Results-section figure that originally failed.
    # NEJM HTML structure is the surface that may change — if this assertion
    # ever breaks, inspect the extracted text and update the BodyTextExtractor
    # CAPTURE_TAGS or check whether NEJM is now serving SPA-rendered content.
    text_lower = text.lower()
    assert "20%" in text_lower, (
        "Extracted text missing the '20%' figure. NEJM HTML may have changed."
    )
    assert "sustained response" in text_lower, (
        "Extracted text missing 'sustained response' phrase. NEJM HTML may have changed."
    )
    assert "week 12" in text_lower or "12 weeks" in text_lower or "12-week" in text_lower, (
        "Extracted text missing the week-12 timepoint. NEJM HTML may have changed."
    )

    # Length sanity check — full NEJM article body is typically >15kb.
    assert len(text) > 5000, (
        f"Extracted text length {len(text)} is suspiciously short. "
        "publisher_html may have triggered the _MIN_TEXT_LENGTH gate, "
        "or NEJM is now serving a paywall page."
    )
