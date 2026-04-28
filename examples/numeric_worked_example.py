#!/usr/bin/env python
"""End-to-end worked example: numeric engine on the Nguyen 2020 ARM claim.

Demonstrates that the deterministic OR/CI consistency check verifies a published
statistic from a peer-reviewed paper as internally consistent. This is a positive
demonstration: the engine confirms a real published claim is mathematically
self-consistent, not a flagged inconsistency.

Source claim (from Edison Scientific Literature output, Nguyen 2020 paper):
    "ARM were 77.5% in A+T- vs 7.8% in A-T- (OR 40.53, 95% CI 23.58-73.71)"
Source paper: doi:10.1007/s00401-020-02200-3

Expected output:
    Claim: ARM were 77.5%...OR 40.53, 95% CI 23.58-73.71
    Numeric engine verdict: CONSISTENT
    Detail: OR/CI internally consistent: 23.58 <= 40.53 <= 73.71, ratio 3.13.

Runs in <30s with prompt caching warm. Cost ~$0.01 per run.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from src.numeric.engine import run_numeric_check

CLAIM_TEXT = (
    "ARM were 77.5% of microglia in A+T- vs 7.8% in A-T- "
    "(OR 40.53, 95% CI 23.58-73.71)"
)
SOURCE_DOI = "10.1007/s00401-020-02200-3"
SOURCE_CITATION = (
    "Nguyen et al. 2020, Acta Neuropathologica — APOE and TREM2 regulate "
    "amyloid responsive microglia in Alzheimer's disease."
)


def main() -> None:
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    print("=" * 72)
    print("Numeric Engine Worked Example — Nguyen 2020 ARM claim")
    print("=" * 72)
    print()
    print("Claim:")
    print(f"  {CLAIM_TEXT}")
    print()
    print(f"Source: {SOURCE_CITATION}")
    print(f"  DOI: {SOURCE_DOI}")
    print()
    print("Running numeric engine (extract + OR/CI consistency check)...")
    print()

    result, steps = run_numeric_check(CLAIM_TEXT, claim_id="nguyen2020-arm")

    if result is None:
        print("Engine: no OR/CI triple extracted from claim.")
        print(f"  Steps run: {len(steps)}")
        sys.exit(2)

    verdict = "CONSISTENT" if result.consistent else "INCONSISTENT"
    print(f"Numeric engine verdict: {verdict}")
    print(f"Check type:             {result.check_type}")
    print(f"Detail:                 {result.explanation}")
    print()
    print("Extracted assertions:")
    for a in result.extracted:
        print(f"  - role={a.role:<10s} value={a.value:>10.4f} unit={a.unit!s:<6s} "
              f"raw={a.raw_text!r}")
    print()
    print(f"Provenance steps emitted: {len(steps)}")
    for s in steps:
        print(f"  - {s.operation:<20s} model_id={s.model_id} "
              f"tokens_in={s.tokens_in} tokens_out={s.tokens_out}")

    print()
    print("Interpretation:")
    print("  Nguyen et al. 2020 reported a published OR with 95% CI from snRNA-seq")
    print("  data on amyloid-responsive microglia. The deterministic engine verifies")
    print("  the reported OR lies within its CI and the CI ratio is plausible — i.e.,")
    print("  the published statistic is internally self-consistent. No LLM is involved")
    print("  in the comparison itself; the same input always produces the same verdict.")


if __name__ == "__main__":
    main()
