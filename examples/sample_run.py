#!/usr/bin/env python
"""End-to-end pipeline demo using a real Edison Scientific Literature agent output."""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

from src.extract import extract_claims
from src.report import build_report
from src.resolve import resolve_citations
from src.verify import verify_claim


def main() -> None:
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    sample_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1
        else Path(__file__).parent / "inputs" / "crow_sample.txt"
    )
    text = sample_path.read_text(encoding="utf-8")
    report_id = str(uuid.uuid4())
    all_steps = []

    claims, extract_step = extract_claims(text)
    all_steps.append(extract_step)
    print(f"Extracted {len(claims)} claims.")

    sources, resolve_steps = resolve_citations(claims)
    all_steps.extend(resolve_steps)

    results = {}
    for claim in claims:
        result, verify_step = verify_claim(claim, sources[claim.claim_id])
        results[claim.claim_id] = result
        all_steps.append(verify_step)

    run_dir = build_report(report_id, text, claims, sources, results, all_steps)
    print(f"Report written to: {run_dir}")

    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    status = report["summary"]["verifiability_status"]
    if status != "verifiable":
        print(
            "\nWARNING: This text contains few or no resolvable citations. "
            "The verification engine cannot assess claims that do not point to "
            "specific sources."
        )


if __name__ == "__main__":
    main()
