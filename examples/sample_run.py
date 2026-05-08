#!/usr/bin/env python
"""End-to-end pipeline demo using a real Edison Scientific Literature agent output.

Default input: benchmarks/real_outputs/edison_trem2/input.txt (TREM2 microglia, ~21 claims).
Pass any path as argv[1] to override.

Expected output (with cache warm, ~$1.25, ~3-4 min):
  Extracted 21 claims.
  Report written to: reports/runs/{uuid}/
  Full-text retrieval methods: abstract_fallback=7, oa_url_pdf=8, pmc_xml=6
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from src.pipeline import PipelineConfig, run_pipeline
from src.report import build_report


def main() -> None:
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    sample_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).parent.parent
        / "benchmarks"
        / "real_outputs"
        / "edison_trem2"
        / "input.txt"
    )
    text = sample_path.read_text(encoding="utf-8")
    report_id = str(uuid.uuid4())

    config = PipelineConfig(api_key=os.environ["ANTHROPIC_API_KEY"])
    verifications, all_steps = run_pipeline(text, config=config)
    print(f"Extracted {len(verifications)} claims.")

    claims = [cv.claim for cv in verifications]
    sources = {cv.claim.claim_id: cv.source for cv in verifications}
    results = {cv.claim.claim_id: cv.result for cv in verifications}
    fetch_method_counts = Counter(cv.fetch_method for cv in verifications)

    run_dir = build_report(report_id, text, claims, sources, results, all_steps)
    print(f"Report written to: {run_dir}")
    print(
        "Full-text retrieval methods: "
        + ", ".join(f"{m}={n}" for m, n in sorted(fetch_method_counts.items()))
    )

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
