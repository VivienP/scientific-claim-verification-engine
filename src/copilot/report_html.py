"""Render an EnrichedVerification list to a self-contained HTML report.

The report is a single ``copilot_report.html`` file written to ``run_dir/``.
No external CDN, no JavaScript framework — all CSS inline, ~50 lines of
vanilla JS for the HITL Accept/Reject flow that exports a ``review_session.json``
sidecar via browser download.

Safety: Jinja2 autoescape is enabled for HTML, so claim text / rationale
content extracted by the LLM cannot inject ``<script>`` tags into the page.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from jinja2 import Environment, FileSystemLoader

from src.copilot.models import EnrichedVerification

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_TEMPLATE_NAME = "copilot_report.html.j2"

# Map V1 verdict status → CSS class + display label.
_VERDICT_DISPLAY: dict[str, tuple[str, str]] = {
    "supported": ("supported", "Supported"),
    "partially_supported": ("partial", "Partial"),
    "unsupported": ("unsupported", "Unsupported"),
    "not_addressed": ("not-addressed", "Not addressed"),
    "error": ("error", "Error"),
}

_RETRIEVAL_DISPLAY: dict[str, tuple[str, str]] = {
    "fulltext": ("fulltext", "Fulltext"),
    "abstract": ("abstract", "Abstract"),
    "title_only": ("title-only", "Title only"),
    "citing_paper_context": ("citing", "Citing context"),
    "fulltext_unavailable": ("unknown", "Unavailable"),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_copilot_report(
    run_dir: Path,
    enriched: list[EnrichedVerification],
    *,
    run_id: str | None = None,
    runtime_seconds: float | None = None,
    total_cost_usd: float | None = None,
) -> Path:
    """Render the copilot HTML report and write it to ``run_dir/copilot_report.html``.

    Returns the path to the written HTML file. Never raises on rendering;
    caller is responsible for ensuring ``run_dir`` exists.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    if run_id is None:
        run_id = run_dir.name

    context = _build_context(
        enriched,
        run_id=run_id,
        runtime_seconds=runtime_seconds,
        total_cost_usd=total_cost_usd,
    )

    # autoescape=True (always on) — required because the template extension
    # is .html.j2 and select_autoescape(["html"]) would not match it.
    # All LLM-extracted text (claim_text, rationale) flows into this template,
    # so unconditional escaping is the safe default.
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(_TEMPLATE_NAME)
    html = template.render(**context)

    output_path = run_dir / "copilot_report.html"
    output_path.write_text(html, encoding="utf-8")

    logger.info(
        "copilot_report_written",
        run_id=run_id,
        path=str(output_path),
        n_claims=context["n_claims"],
        n_with_fix=context["n_with_fix"],
    )

    return output_path


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------


def _build_context(
    enriched: list[EnrichedVerification],
    *,
    run_id: str,
    runtime_seconds: float | None,
    total_cost_usd: float | None,
) -> dict[str, Any]:
    n_claims = len(enriched)

    # Verdict counts
    counts = {"supported": 0, "partially_supported": 0, "unsupported": 0, "not_addressed": 0}
    for ev in enriched:
        status = ev.base.result.status
        if status in counts:
            counts[status] += 1

    n_with_fix = sum(1 for ev in enriched if ev.copilot.recommended_fix is not None)

    pct_total = sum(counts.values())
    pct_supported = _safe_pct(counts["supported"], pct_total)
    pct_partial = _safe_pct(counts["partially_supported"], pct_total)
    pct_unsupported = _safe_pct(counts["unsupported"], pct_total)
    pct_not_addressed = _safe_pct(counts["not_addressed"], pct_total)

    cost = total_cost_usd if total_cost_usd is not None else 0.0
    runtime = runtime_seconds if runtime_seconds is not None else 0.0
    cost_per_claim = (cost / n_claims) if n_claims > 0 else 0.0

    mode_str = enriched[0].mode.value if enriched else "—"

    return {
        "run_id": run_id,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "mode": mode_str,
        "n_claims": n_claims,
        "n_supported": counts["supported"],
        "n_partial": counts["partially_supported"],
        "n_unsupported": counts["unsupported"],
        "n_not_addressed": counts["not_addressed"],
        "n_with_fix": n_with_fix,
        "pct_bar_total": pct_total,
        "pct_supported": pct_supported,
        "pct_partial": pct_partial,
        "pct_unsupported": pct_unsupported,
        "pct_not_addressed": pct_not_addressed,
        "total_cost_usd": cost,
        "runtime_seconds": int(runtime),
        "cost_per_claim": cost_per_claim,
        "claims": [_build_claim_item(ev) for ev in enriched],
    }


def _build_claim_item(ev: EnrichedVerification) -> dict[str, Any]:
    cv = ev.base
    verdict_css, verdict_label = _VERDICT_DISPLAY.get(
        cv.result.status, ("error", cv.result.status.title())
    )
    retrieval_css, retrieval_label = _RETRIEVAL_DISPLAY.get(
        cv.result.retrieval_status, ("unknown", cv.result.retrieval_status)
    )

    fix = ev.copilot.recommended_fix
    fix_dict: dict[str, Any] | None = None
    if fix is not None:
        fix_dict = {
            "action": fix.action,
            "regulatory_risk_level": fix.regulatory_risk_level,
            "suggested_doi": fix.suggested_doi,
            "suggested_doi_title": fix.suggested_doi_title,
            "reworded_claim": fix.reworded_claim,
            "confidence": fix.confidence,
        }

    return {
        "claim_id": cv.claim.claim_id,
        "claim_text": cv.claim.claim_text,
        "verdict_css": verdict_css,
        "verdict_label": verdict_label,
        "rationale": ev.copilot.verdict_rationale,
        "source_doi": cv.source.doi,
        "source_title": cv.source.title,
        "retrieval_css": retrieval_css,
        "retrieval_label": retrieval_label,
        "is_primary_source": ev.copilot.is_primary_source,
        "study_design": ev.copilot.study_design,
        "risk_of_bias": ev.copilot.risk_of_bias,
        "primary_source_doi": ev.copilot.primary_source_doi,
        "fix": fix_dict,
        "provenance_steps": ev.copilot_steps,
    }


def _safe_pct(num: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return round(100.0 * num / denom, 1)
