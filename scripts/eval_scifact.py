"""Evaluate the verification pipeline on a SciFact split and report F1 metrics."""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Literal

import structlog

from src.clients._cache import default_db_path, prune_expired
from src.models import Claim, ResolvedSource
from src.report import _compute_cost
from src.verify import verify_claim

logger: structlog.BoundLogger = structlog.get_logger(__name__)

MODEL_ID = "claude-sonnet-4-6"

# Label mapping from SciFact ground truth to pipeline VerificationStatus
_LABEL_MAP: dict[str, str] = {
    "SUPPORT": "supported",
    "CONTRADICT": "unsupported",
}


def _load_corpus(corpus_path: Path) -> dict[str, dict[str, object]]:
    """Load SciFact corpus.jsonl as a dict keyed by paper_id (str)."""
    corpus: dict[str, dict[str, object]] = {}
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            paper = json.loads(line)
            # abstract is list[str] — join into single string
            abstract_parts: list[str] = paper.get("abstract", []) or []
            paper["abstract_text"] = " ".join(abstract_parts)
            corpus[str(paper["paper_id"])] = paper
    return corpus


def _load_claims(claims_path: Path) -> list[dict[str, object]]:
    """Load SciFact claims.jsonl."""
    claims = []
    with open(claims_path, encoding="utf-8") as f:
        for line in f:
            claims.append(json.loads(line))
    return claims


def _compute_metrics(predictions: list[str], ground_truth: list[str]) -> dict[str, float]:
    """Compute per-class precision/recall/F1 + macro-F1."""
    classes = ["supported", "unsupported", "not_addressed"]
    per_class: dict[str, dict[str, float]] = {}

    for cls in classes:
        tp = sum(1 for p, g in zip(predictions, ground_truth, strict=True) if p == cls and g == cls)
        fp = sum(1 for p, g in zip(predictions, ground_truth, strict=True) if p == cls and g != cls)
        fn = sum(1 for p, g in zip(predictions, ground_truth, strict=True) if p != cls and g == cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1}

    # Macro-F1
    macro_f1 = sum(per_class[cls]["f1"] for cls in classes) / len(classes)

    # Micro: overall precision/recall/F1
    tp_total = sum(1 for p, g in zip(predictions, ground_truth, strict=True) if p == g)
    precision = tp_total / len(predictions) if predictions else 0.0
    recall = tp_total / len(ground_truth) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "per_class": per_class,  # type: ignore[dict-item]
    }


def run_eval(
    split: Literal["dev"],
    corpus_path: Path,
    claims_path: Path,
    output_path: Path,
    *,
    limit: int | None = 50,
    max_cost_usd: float = 5.0,
    model_id: str = MODEL_ID,
) -> dict[str, float]:
    """Run pipeline on a SciFact split and return metrics dict.

    Refuses split="test" with structlog.error + sys.exit(1).
    Tracks cumulative cost via ProvenanceStep tokens; aborts with sys.exit(1) if exceeded.
    Default limit=50 (smoke test). Pass limit=None for full ~300-claim run.
    Label mapping: SUPPORT→supported, CONTRADICT→unsupported, {}→not_addressed.
    partially_supported counted as supported for F1.
    """
    if split == "test":  # type: ignore[comparison-overlap]  # runtime guard; type narrows to Literal["dev"] but guard is defensive
        logger.error("benchmark_isolation_violation", split=split)
        sys.exit(1)

    prune_expired(default_db_path())

    corpus = _load_corpus(corpus_path)
    all_claims = _load_claims(claims_path)

    if limit is not None:
        all_claims = all_claims[:limit]

    predictions: list[str] = []
    ground_truth: list[str] = []
    all_steps = []
    cumulative_cost = 0.0
    n_claims = 0

    for raw_claim in all_claims:
        claim_text = str(raw_claim["claim"])
        evidence: dict[str, object] = raw_claim.get("evidence", {}) or {}  # type: ignore[assignment]

        if evidence:
            # Use first evidence entry
            paper_id = next(iter(evidence))
            evidence_entry = evidence[paper_id]
            label_raw = evidence_entry.get("label", "") if isinstance(evidence_entry, dict) else ""
            gt_status = _LABEL_MAP.get(str(label_raw), "not_addressed")
            paper = corpus.get(str(paper_id), {})
            abstract_text = str(paper.get("abstract_text", "")) if paper else ""
            source = ResolvedSource(
                found=bool(abstract_text),
                doi=None,
                title=str(paper.get("title", "")) if paper else None,
                abstract=abstract_text or None,
                similarity_score=1.0,
            )
        else:
            gt_status = "not_addressed"
            source = ResolvedSource(
                found=False, doi=None, title=None, abstract=None, similarity_score=None
            )

        claim = Claim(
            claim_id=str(uuid.uuid4()),
            claim_text=claim_text,
            cited_authors=[],
            cited_year=None,
            claim_type="factual_qualitative",
        )

        result, step = verify_claim(claim, source, model_id=model_id)
        all_steps.append(step)
        n_claims += 1

        cumulative_cost = _compute_cost(all_steps)
        if cumulative_cost > max_cost_usd:
            logger.error(
                "cost_limit_exceeded",
                cumulative_cost=cumulative_cost,
                max_cost_usd=max_cost_usd,
                n_claims_processed=n_claims,
            )
            sys.exit(1)

        # Map partially_supported → supported for eval
        pred_status = result.status
        if pred_status == "partially_supported":
            pred_status = "supported"

        predictions.append(pred_status)
        ground_truth.append(gt_status)

    metrics = _compute_metrics(predictions, ground_truth)

    n_supported_gt = sum(1 for g in ground_truth if g == "supported")
    n_unsupported_gt = sum(1 for g in ground_truth if g == "unsupported")
    n_not_addressed_gt = sum(1 for g in ground_truth if g == "not_addressed")

    output_data = {
        "split": split,
        "limit": limit,
        "model_id": model_id,
        "timestamp": time.time(),
        "metrics": metrics,
        "n_claims": n_claims,
        "n_supported_gt": n_supported_gt,
        "n_unsupported_gt": n_unsupported_gt,
        "n_not_addressed_gt": n_not_addressed_gt,
        "total_cost_usd": cumulative_cost,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info(
        "eval_complete",
        split=split,
        n_claims=n_claims,
        f1=metrics["f1"],
        macro_f1=metrics["macro_f1"],
        total_cost_usd=cumulative_cost,
        output_path=str(output_path),
    )

    return {k: v for k, v in metrics.items() if isinstance(v, float)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the verification pipeline on SciFact.")
    parser.add_argument(
        "--split",
        choices=["dev"],
        default="dev",
        help="Which split to evaluate (only 'dev' allowed — test split is locked).",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("eval/scifact/corpus.jsonl"),
        help="Path to SciFact corpus.jsonl",
    )
    parser.add_argument(
        "--claims",
        type=Path,
        default=Path("eval/scifact/claims.jsonl"),
        help="Path to SciFact claims.jsonl",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max number of claims to evaluate (default 50). Pass 0 for unlimited.",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=5.0,
        dest="max_cost",
        help="Maximum USD cost before aborting (default $5.00).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        dest="output",
        help="Output JSON path. Defaults to eval/results/{split}_{timestamp}.json",
    )

    args = parser.parse_args()

    limit: int | None = args.limit if args.limit > 0 else None
    output_path: Path = args.output or Path(f"eval/results/{args.split}_{int(time.time())}.json")

    run_eval(
        split=args.split,
        corpus_path=args.corpus,
        claims_path=args.claims,
        output_path=output_path,
        limit=limit,
        max_cost_usd=args.max_cost,
    )


if __name__ == "__main__":
    main()
