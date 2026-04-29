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
from pathlib import Path

from dotenv import load_dotenv

from src.bm25_selector import select_passages
from src.chunker import chunk_paper
from src.extract import extract_claims
from src.fetch_fulltext import fetch_fulltext
from src.report import build_report
from src.resolve import resolve_citations
from src.verify import verify_claim, verify_claim_fulltext_with_numeric


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
    all_steps = []

    claims, extract_step = extract_claims(text)
    all_steps.append(extract_step)
    print(f"Extracted {len(claims)} claims.")

    sources, resolve_steps = resolve_citations(claims)
    all_steps.extend(resolve_steps)

    results = {}
    fulltext_methods: dict[str, str] = {}
    for claim in claims:
        source = sources[claim.claim_id]
        fulltext, method = fetch_fulltext(source)
        fulltext_methods[claim.claim_id] = method

        if fulltext is not None:
            chunks = chunk_paper(source.doi or claim.claim_id, fulltext)
            passages = select_passages(claim.claim_text, chunks, top_k=3)
            result, verify_steps = verify_claim_fulltext_with_numeric(claim, source, passages)
            all_steps.extend(verify_steps)
        else:
            result, verify_step = verify_claim(claim, source)
            all_steps.append(verify_step)

        results[claim.claim_id] = result

    run_dir = build_report(report_id, text, claims, sources, results, all_steps)
    print(f"Report written to: {run_dir}")
    print(
        "Full-text retrieval methods: "
        + ", ".join(
            f"{m}={list(fulltext_methods.values()).count(m)}"
            for m in sorted(set(fulltext_methods.values()))
        )
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
