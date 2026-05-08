"""CLI that runs the end-to-end pipeline and writes baseline metrics.

This is a thin wrapper around `eval.e2e.measurement` (pure logic) and the
existing pipeline modules (extract, resolve, fetch_fulltext, verify). Library
logic is split out so unit tests can run offline without Anthropic credentials.

Usage:
    python scripts/measure_e2e_recall.py \
        --paper eval/e2e/reference_paper_v1.json \
        --output eval/e2e/results/baseline_pre_fixes.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import structlog
from dotenv import load_dotenv

from eval.e2e.measurement import Metrics, align_claims, compute_metrics
from eval.e2e.schema import ReferencePaper, load_reference_paper
from src.bibliography import parse_bibliography
from src.extract import extract_claims
from src.models import Claim, ResolvedSource, VerificationResult
from src.pipeline import PipelineConfig, verify_one_claim
from src.report import _compute_cost
from src.resolve import resolve_citations_multi

load_dotenv()

logger: structlog.BoundLogger = structlog.get_logger(__name__)


def _run_pipeline(
    text: str, *, max_cost_usd: float
) -> tuple[
    list[Claim],
    dict[str, ResolvedSource],
    dict[str, VerificationResult],
    float,
]:
    """Run extract -> resolve -> verify on `text` and return all artifacts.

    Aborts via sys.exit(1) if cumulative cost exceeds `max_cost_usd`.
    """
    all_steps = []

    claims, extract_step = extract_claims(text)
    all_steps.append(extract_step)
    logger.info("e2e_extract_complete", n_claims=len(claims))

    bibliography = parse_bibliography(text)
    source_sets, resolve_steps = resolve_citations_multi(claims, bibliography=bibliography)
    all_steps.extend(resolve_steps)
    sources: dict[str, ResolvedSource] = {
        cid: rs_set.primary() for cid, rs_set in source_sets.items()
    }
    logger.info(
        "e2e_resolve_complete",
        n_sources=len(sources),
        n_bibliography_entries=len(bibliography),
    )

    cumulative_cost = _compute_cost(all_steps)
    if cumulative_cost > max_cost_usd:
        logger.error(
            "cost_limit_exceeded_pre_verify",
            cumulative_cost=cumulative_cost,
            max_cost_usd=max_cost_usd,
        )
        sys.exit(1)

    config = PipelineConfig()
    verifications: dict[str, VerificationResult] = {}
    for claim in claims:
        cv = verify_one_claim(
            claim, source_sets[claim.claim_id], citing_paper_text=text, config=config
        )
        all_steps.extend(cv.steps)
        verifications[claim.claim_id] = cv.result

        cumulative_cost = _compute_cost(all_steps)
        if cumulative_cost > max_cost_usd:
            logger.error(
                "cost_limit_exceeded_during_verify",
                cumulative_cost=cumulative_cost,
                max_cost_usd=max_cost_usd,
            )
            sys.exit(1)

    return claims, sources, verifications, cumulative_cost


def run_measurement(paper_path: Path, output_path: Path, *, max_cost_usd: float = 5.0) -> Metrics:
    """Top-level orchestration: load -> pipeline -> align -> compute -> save."""
    paper: ReferencePaper = load_reference_paper(paper_path)

    source_text_path = Path(paper.source_text_path)
    if not source_text_path.exists():
        logger.error(
            "source_text_missing",
            source_text_path=str(source_text_path),
            hint="Export the manuscript to plain text and place it at this path.",
        )
        sys.exit(1)
    text = source_text_path.read_text(encoding="utf-8")

    logger.info(
        "e2e_measurement_start",
        paper_title=paper.paper_title,
        n_gt_claims=len(paper.claims),
        source_chars=len(text),
    )

    extracted, sources, verifications, total_cost = _run_pipeline(text, max_cost_usd=max_cost_usd)

    matches = align_claims(paper.claims, extracted)
    metrics = compute_metrics(paper.claims, extracted, sources, verifications, matches)

    output = {
        "run_id": str(uuid.uuid4()),
        "timestamp": time.time(),
        "paper_path": str(paper_path),
        "paper_title": paper.paper_title,
        "metrics": {
            "extraction_recall": metrics.extraction_recall,
            "extraction_precision": metrics.extraction_precision,
            "resolution_accuracy": metrics.resolution_accuracy,
            "e2e_coverage_useful": metrics.e2e_coverage_useful,
            "not_addressed_unknown_cause": metrics.not_addressed_unknown_cause,
        },
        "counts": metrics.counts,
        "matches": [asdict(m) for m in matches],
        "total_cost_usd": total_cost,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info(
        "e2e_measurement_complete",
        extraction_recall=metrics.extraction_recall,
        extraction_precision=metrics.extraction_precision,
        resolution_accuracy=metrics.resolution_accuracy,
        e2e_coverage_useful=metrics.e2e_coverage_useful,
        total_cost_usd=total_cost,
        output_path=str(output_path),
    )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure end-to-end pipeline metrics against an annotated reference paper."
    )
    parser.add_argument(
        "--paper",
        type=Path,
        required=True,
        help="Path to a reference_paper_*.json annotation file (see eval/e2e/schema.py)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write metrics JSON (e.g. eval/e2e/results/baseline_pre_fixes.json)",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=5.0,
        dest="max_cost",
        help="Maximum USD cost before aborting (default $5.00)",
    )
    args = parser.parse_args()
    run_measurement(args.paper, args.output, max_cost_usd=args.max_cost)


if __name__ == "__main__":
    main()
