#!/usr/bin/env python
"""End-to-end pipeline demo using a real Crow/Falcon agent output."""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

# Check for API key before any imports that might fail obscurely
if not os.environ.get("ANTHROPIC_API_KEY"):
    print("Error: ANTHROPIC_API_KEY environment variable not set.")  # noqa: T201 — intentional CLI output
    sys.exit(1)

from src.extract import extract_claims
from src.report import build_report
from src.resolve import resolve_citations
from src.verify import verify_claim


def main() -> None:
    sample_path = Path(__file__).parent / "inputs" / "crow_sample.txt"
    text = sample_path.read_text(encoding="utf-8")
    report_id = str(uuid.uuid4())
    all_steps = []

    claims, extract_step = extract_claims(text)
    all_steps.append(extract_step)
    print(f"Extracted {len(claims)} claims.")  # noqa: T201 — intentional CLI output

    sources, resolve_steps = resolve_citations(claims)
    all_steps.extend(resolve_steps)

    results = {}
    for claim in claims:
        result, verify_step = verify_claim(claim, sources[claim.claim_id])
        results[claim.claim_id] = result
        all_steps.append(verify_step)

    run_dir = build_report(report_id, text, claims, sources, results, all_steps)
    print(f"Report written to: {run_dir}")  # noqa: T201 — intentional CLI output


if __name__ == "__main__":
    main()
