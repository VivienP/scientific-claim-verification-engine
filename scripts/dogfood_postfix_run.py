"""Run the post-fix pipeline (Tracks A + D + F + G + I) against any captured
dogfood input under benchmarks/real_outputs/{name}/input.txt and diff it
against the pre-fix baseline at the same location.

Usage:
    python -m scripts.dogfood_postfix_run elicit_psilocybin
    python -m scripts.dogfood_postfix_run answerthis_lactate

Outputs:
    reports/runs/{name}_postfix_{ts}/  — report.json + provenance.jsonl
                                         + fetch_traces.jsonl

KPI checks printed at the end:
    - True rule violations (numeric claim + confident verdict + abstract-only)
      per no-confident-verdict-without-evidence.md. Distinct from the looser
      "supported|unsupported + abstract_only" pattern, which also catches the
      design's intentional pass-through case (qualitative claims that the
      abstract directly supports verbatim).
    - unverifiable_by_reason  (where did new "unverifiable" verdicts come from?)
    - fetch_failures_by_reason (which fetch steps failed on which publishers?)
    - Verdict-flip diff vs the pre-fix baseline (claim-text-matched).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from src.numeric.heuristics import _claim_has_specific_numeric
from src.pipeline import PipelineConfig, run_pipeline
from src.report import build_report

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
INSUFFICIENT_EVIDENCE = {"abstract_only", "title_only", "citing_paper_context", "no_evidence"}


def _run(benchmark: str) -> int:
    input_path = ROOT / "benchmarks" / "real_outputs" / benchmark / "input.txt"
    prefix_report = ROOT / "benchmarks" / "real_outputs" / benchmark / "report.json"
    if not input_path.exists():
        print(f"FATAL: input not found at {input_path}")
        return 1
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("FATAL: ANTHROPIC_API_KEY missing")
        return 1

    text = input_path.read_text(encoding="utf-8")
    run_id = f"{benchmark}_postfix_{int(time.time())}"
    print(f"Starting post-fix pipeline run: {run_id}")
    print(f"  Input: {input_path} ({len(text)} chars)")

    t0 = time.perf_counter()
    config = PipelineConfig(api_key=os.environ["ANTHROPIC_API_KEY"])
    verifications, steps = run_pipeline(text, config=config)
    elapsed = time.perf_counter() - t0
    print(f"  Pipeline complete in {elapsed:.1f}s ({len(verifications)} claims)")

    fetch_outcomes = {
        cv.claim.claim_id: cv.fetch_outcome for cv in verifications if cv.fetch_outcome is not None
    }
    run_dir = build_report(
        str(uuid.uuid4()),
        text,
        claims=[v.claim for v in verifications],
        sources={v.claim.claim_id: v.source for v in verifications},
        results={v.claim.claim_id: v.result for v in verifications},
        provenance_steps=steps,
        fetch_outcomes=fetch_outcomes,
    )
    target_dir = run_dir.parent / run_id
    if target_dir.exists():
        target_dir = run_dir.parent / f"{run_id}_{uuid.uuid4().hex[:6]}"
    run_dir.rename(target_dir)
    print(f"  Report written: {target_dir}")

    new_report = json.loads((target_dir / "report.json").read_text(encoding="utf-8"))
    new_summary = new_report["summary"]
    new_claims = new_report["claims"]

    print("\n" + "=" * 72)
    print(f"KPI BREAKDOWN - {benchmark}")
    print("=" * 72)

    if prefix_report.exists():
        old_report = json.loads(prefix_report.read_text(encoding="utf-8"))
        old_summary = old_report["summary"]
        old_claims = old_report["claims"]
        print("Verdict distribution             pre-fix -> post-fix")
        for k in (
            "supported",
            "partially_supported",
            "unsupported",
            "not_addressed",
            "unverifiable",
        ):
            o = old_summary.get(k) or 0
            n = new_summary.get(k) or 0
            print(f"  {k:24s}     {o:3d} -> {n:3d}  ({n - o:+d})")
    else:
        print("(no pre-fix baseline at " + str(prefix_report) + ")")
        old_claims = []

    # Rule-compliant violation check: BOTH conditions must hold per
    # .claude/rules/no-confident-verdict-without-evidence.md
    true_violations = []
    for c in new_claims:
        v = c["verification"]
        if (
            v["status"] in ("supported", "unsupported")
            and v.get("evidence_quality") in INSUFFICIENT_EVIDENCE
            and _claim_has_specific_numeric(c["claim_text"])
        ):
            true_violations.append(c)
    print(f"\nRule violations (numeric+confident+abstract): {len(true_violations)}")
    for c in true_violations:
        print(f"  VIOLATION: {c['claim_text'][:120]}")

    print(f"\nunverifiable_by_reason:    {new_summary.get('unverifiable_by_reason', {})}")
    print(f"fetch_attempts_by_method:  {new_summary.get('fetch_attempts_by_method', {})}")
    print(f"fetch_failures_by_reason:  {new_summary.get('fetch_failures_by_reason', {})}")

    cost = new_summary.get("total_cost_usd", 0)
    print(f"\nTotal cost:               ${cost:.4f}")
    print(f"Citation found rate:      {new_summary.get('citation_found_rate', 0):.1%}")
    print(f"Fulltext success rate:    {new_summary.get('fulltext_success_rate', 0):.1%}")

    # Pre-fix silent-failure count (numeric+confident+abstract on the OLD report).
    if old_claims:
        old_silent = sum(
            1
            for c in old_claims
            if c["verification"]["status"] in ("supported", "unsupported")
            and c["verification"].get("evidence_quality") == "abstract_only"
            and _claim_has_specific_numeric(c["claim_text"])
        )
        n_old = len(old_claims)
        n_new = new_summary["total_claims"]
        n_viol = len(true_violations)
        print(
            f"\nKPI - silent-failure count (numeric+confident+abstract):\n"
            f"  pre-fix:  {old_silent}/{n_old} ({100 * old_silent / n_old:.1f}%)\n"
            f"  post-fix: {n_viol}/{n_new} ({100 * n_viol / n_new:.1f}%)"
        )
        # Verdict flips (claim-text-matched).
        old_by_text = {c["claim_text"]: c["verification"]["status"] for c in old_claims}
        flips: Counter[tuple[str, str]] = Counter()
        unchanged = 0
        for c in new_claims:
            old_v = old_by_text.get(c["claim_text"])
            new_v = c["verification"]["status"]
            if old_v is None:
                continue
            if old_v == new_v:
                unchanged += 1
            else:
                flips[(old_v, new_v)] += 1
        print(f"\nVerdict flips (text-matched; unchanged={unchanged}):")
        for (o, n), cnt in flips.most_common():
            print(f"  {o:22s} -> {n:22s} {cnt}")

    return 0 if not true_violations else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", help="Subdir under benchmarks/real_outputs/")
    args = parser.parse_args()
    return _run(args.benchmark)


if __name__ == "__main__":
    sys.exit(main())
