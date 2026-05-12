"""Replay captured Elicit-psilocybin verdicts through ``safe_verification_result``.

Measures the KPI movement attributable to the deterministic helper alone,
isolated from the prompt-side `unsupported`-vs-`not_addressed` split.
The prompt-side effect cannot be measured by replay — it requires a fresh LLM call.

Usage:
    python -m scripts.replay_psilocybin_kpi reports/runs/elicit_psilocybin_rerun_860b1ae5/report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.models import safe_verification_result


def replay(report_path: Path) -> dict[str, Any]:
    """Walk every claim, replay the original LLM output through the helper, and count."""
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    claims = raw["claims"]

    pre_supported_abstract_only = 0
    pre_unsupported_abstract_only = 0
    pre_confident_total = 0
    post_unverifiable = 0
    post_still_confident = 0
    qualitative_survivors: list[dict[str, Any]] = []  # not caught by numeric helper

    for entry in claims:
        v = entry["verification"]
        eq = v.get("evidence_quality")
        status = v.get("status")
        conf = v.get("confidence")
        if eq != "abstract_only":
            continue
        if status not in ("supported", "unsupported"):
            continue
        if conf is None:
            continue

        pre_confident_total += 1
        if status == "supported":
            pre_supported_abstract_only += 1
        elif status == "unsupported":
            pre_unsupported_abstract_only += 1

        # Replay through the new helper. claim_text triggers the numeric regex.
        replayed = safe_verification_result(
            status=status,
            confidence=float(conf),
            evidence_quality="abstract_only",
            claim_text=entry["claim_text"],
            explanation=v.get("explanation", ""),
            unverifiable_reason="numeric_claim_abstract_only",
        )
        if replayed.status == "unverifiable":
            post_unverifiable += 1
        else:
            post_still_confident += 1
            qualitative_survivors.append(
                {
                    "claim_id": entry["claim_id"],
                    "claim_text_preview": entry["claim_text"][:120],
                    "original_status": status,
                    "post_helper_status": replayed.status,
                }
            )

    return {
        "report": str(report_path),
        "total_claims_in_report": len(claims),
        "pre_fix_confident_on_abstract_only": pre_confident_total,
        "pre_fix_supported_abstract_only": pre_supported_abstract_only,
        "pre_fix_unsupported_abstract_only": pre_unsupported_abstract_only,
        "post_helper_downgraded_to_unverifiable": post_unverifiable,
        "post_helper_still_confident": post_still_confident,
        "qualitative_survivors_awaiting_track_g": qualitative_survivors,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    result = replay(args.report)
    print(json.dumps(result, indent=2))
