"""Direct LLM baseline: verify claims with naive single-prompt (no pipeline).

Same 50 claims + corpus abstracts as the smoke eval run.
Used to establish that the pipeline's structured verification adds value over
a naive "paste everything into one message" approach.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import anthropic
import structlog

logger: structlog.BoundLogger = structlog.get_logger(__name__)

MODEL_ID = "claude-sonnet-4-6"

_LABEL_MAP: dict[str, str] = {
    "SUPPORT": "supported",
    "CONTRADICT": "unsupported",
}

_VALID_STATUSES = {"supported", "unsupported", "not_addressed"}

_NAIVE_PROMPT = """\
Given the following scientific claim and abstract, determine whether the abstract \
supports, contradicts, or does not address the claim.

Claim: {claim}

Abstract: {abstract}

Reply with a JSON object: {{"status": "supported|unsupported|not_addressed", "confidence": 0.0-1.0}}
"""


def _load_corpus(corpus_path: Path) -> dict[str, dict[str, object]]:
    corpus: dict[str, dict[str, object]] = {}
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            paper = json.loads(line)
            abstract_parts: list[str] = paper.get("abstract", []) or []
            paper["abstract_text"] = " ".join(abstract_parts)
            corpus[str(paper["doc_id"])] = paper
    return corpus


def _load_claims(claims_path: Path) -> list[dict[str, object]]:
    claims = []
    with open(claims_path, encoding="utf-8") as f:
        for line in f:
            claims.append(json.loads(line))
    return claims


def _compute_metrics(predictions: list[str], ground_truth: list[str]) -> dict[str, Any]:
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
    macro_f1 = sum(per_class[cls]["f1"] for cls in classes) / len(classes)
    tp_total = sum(1 for p, g in zip(predictions, ground_truth, strict=True) if p == g)
    precision = tp_total / len(predictions) if predictions else 0.0
    recall = tp_total / len(ground_truth) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }


def run_direct_baseline(
    corpus_path: Path,
    claims_path: Path,
    output_path: Path,
    *,
    limit: int = 50,
) -> dict[str, float]:
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    corpus = _load_corpus(corpus_path)
    all_claims = _load_claims(claims_path)[:limit]

    predictions: list[str] = []
    ground_truth: list[str] = []
    total_tokens_in = 0
    total_tokens_out = 0

    for raw_claim in all_claims:
        claim_text = str(raw_claim["claim"])
        evidence: dict[str, object] = raw_claim.get("evidence", {}) or {}  # type: ignore[assignment]

        if evidence:
            paper_id = next(iter(evidence))
            evidence_entry = evidence[paper_id]
            label_raw = (
                evidence_entry[0].get("label", "")
                if isinstance(evidence_entry, list) and evidence_entry
                else ""
            )
            gt_status = _LABEL_MAP.get(str(label_raw), "not_addressed")
            paper = corpus.get(str(paper_id), {})
            abstract_text = str(paper.get("abstract_text", "")) if paper else ""
        else:
            gt_status = "not_addressed"
            abstract_text = ""

        if not abstract_text:
            predictions.append("not_addressed")
            ground_truth.append(gt_status)
            continue

        prompt = _NAIVE_PROMPT.format(claim=claim_text, abstract=abstract_text)
        response = client.messages.create(
            model=MODEL_ID,
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
        )
        total_tokens_in += response.usage.input_tokens
        total_tokens_out += response.usage.output_tokens

        from anthropic.types import TextBlock as _TextBlock

        first_block = response.content[0] if response.content else None
        response_text = first_block.text if isinstance(first_block, _TextBlock) else ""

        # Strip fences
        stripped = response_text.strip()
        if stripped.startswith("```"):
            nl = stripped.find("\n")
            if nl != -1:
                stripped = stripped[nl + 1 :]
            if stripped.endswith("```"):
                stripped = stripped[: stripped.rfind("```")].rstrip()

        try:
            parsed = json.loads(stripped)
            status = str(parsed.get("status", "not_addressed"))
            if status not in _VALID_STATUSES:
                status = "not_addressed"
        except (json.JSONDecodeError, KeyError, TypeError):
            status = "not_addressed"

        predictions.append(status)
        ground_truth.append(gt_status)
        logger.info("direct_baseline_call", status=status, gt=gt_status)

    metrics = _compute_metrics(predictions, ground_truth)

    # Cost (same constants as report.py)
    cost = total_tokens_in * 3e-6 + total_tokens_out * 15e-6

    output_data = {
        "approach": "direct_llm_naive",
        "model_id": MODEL_ID,
        "limit": limit,
        "timestamp": time.time(),
        "metrics": metrics,
        "n_claims": len(predictions),
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
        "total_cost_usd": cost,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info(
        "direct_baseline_complete",
        f1=metrics["f1"],
        macro_f1=metrics["macro_f1"],
        cost=cost,
        output_path=str(output_path),
    )

    return {k: v for k, v in metrics.items() if isinstance(v, float)}


if __name__ == "__main__":
    metrics = run_direct_baseline(
        corpus_path=Path("eval/scifact/corpus.jsonl"),
        claims_path=Path("eval/scifact/claims_dev.jsonl"),
        output_path=Path("eval/results/direct_baseline.json"),
        limit=50,
    )
    print(f"Direct baseline F1: {metrics['f1']:.4f}")
    sys.exit(0)
