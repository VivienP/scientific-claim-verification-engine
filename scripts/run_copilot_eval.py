"""CLI entrypoint: score a copilot-enriched run against the gold annotations.

Usage:
  python scripts/run_copilot_eval.py \
      --enriched reports/runs/<run_id>/enriched.json \
      --gold eval/e2e/reference_paper_v1_verdicts.json \
      --output eval/results/copilot_eval_<run_id>.json

Exits non-zero if the Phase B gate fails (precision < 0.80, hallucination > 0,
or fix-present-rate < 0.60). This is the offline regression bar for any change
to ``src/copilot/`` modules.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.copilot.auto_eval import evaluate, report_to_dict


def main() -> int:
    parser = argparse.ArgumentParser(description="Score copilot output vs gold annotations.")
    parser.add_argument(
        "--enriched",
        type=Path,
        required=True,
        help="Path to a JSON file with serialised EnrichedVerification list.",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        required=True,
        help="Path to the gold annotation file (e.g. lactate_isf_gold.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=False,
        help="If provided, write the eval report JSON here.",
    )
    parser.add_argument(
        "--enforce-gate",
        action="store_true",
        help="Exit 1 if Phase B gate fails (default: report only).",
    )
    args = parser.parse_args()

    # The expected on-disk format is what build_serialised_enriched() produces
    # in examples/copilot_run.py — reconstructing dataclasses from JSON is
    # delegated to the importer below. For now this CLI is a stub that the
    # E2E run script will populate; auto_eval.py is the library.
    print("Loading enriched run from", args.enriched, file=sys.stderr)
    print("Loading gold from", args.gold, file=sys.stderr)

    # Concrete deserialisation of EnrichedVerification is implemented by
    # examples/copilot_run.py once Day B-8 lands. For now this CLI emits a
    # template invocation and exits 0 if both files exist.
    if not args.enriched.exists():
        print(f"error: enriched file not found: {args.enriched}", file=sys.stderr)
        return 2
    if not args.gold.exists():
        print(f"error: gold file not found: {args.gold}", file=sys.stderr)
        return 2

    raw = json.loads(args.enriched.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        print(
            "error: --enriched must contain a JSON list of EnrichedVerification dicts. "
            "Use examples/copilot_run.py to produce one.",
            file=sys.stderr,
        )
        return 2

    # Lazy import — only load if we have a real run to score.
    from examples.copilot_run import deserialize_enriched

    enriched = deserialize_enriched(raw)
    report = evaluate(enriched, args.gold)
    payload = report_to_dict(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote eval report to {args.output}", file=sys.stderr)

    print(json.dumps(payload, indent=2))

    if args.enforce_gate and not report.passes_phase_b_gate:
        print("PHASE B GATE FAILED", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
