"""Throwaway: simulate proposed _find_or_ci_triple logic against existing Elicit checks.

Validates risk R1 (no regression on the 19 currently-correct checks) and
risk-free demonstration of Bug A/B fixes on the 4 known false positives.

Run:  python scripts/_sim_numeric_fix.py

This script is a sanity check only and is not part of the test suite.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

RATIO_KEYWORDS_LONG = (
    "odds ratio",
    "hazard ratio",
    "risk ratio",
    "relative risk",
    "incidence rate ratio",
    "rate ratio",
)
RATIO_KEYWORDS_SHORT = {"or", "hr", "rr", "rrr", "ahr", "ihr", "shr", "irr"}
WORD_RE = re.compile(r"[A-Za-z]+")


def has_ratio_keyword(text: str) -> bool:
    t = text.lower()
    if any(k in t for k in RATIO_KEYWORDS_LONG):
        return True
    return bool({w.lower() for w in WORD_RE.findall(text)} & RATIO_KEYWORDS_SHORT)


def is_ratio_primary(a: dict) -> bool:
    if a["role"] != "primary":
        return False
    if a.get("unit") == "%" or "%" in a["raw_text"]:
        return False
    if has_ratio_keyword(a["raw_text"]):
        return True
    if has_ratio_keyword(a["context"]):
        ctx_l = a["context"].lower()
        return not ("reduction" in ctx_l or "change" in ctx_l or "difference" in ctx_l)
    return False


def find_triple(assertions: list[dict]) -> tuple[float, float, float] | None:
    primary_idx = next(
        (i for i, a in enumerate(assertions) if is_ratio_primary(a)), None
    )
    if primary_idx is None:
        return None
    primary = assertions[primary_idx]
    p_raw_l = primary["raw_text"].lower()

    next_primary_idx = next(
        (j for j in range(primary_idx + 1, len(assertions)) if assertions[j]["role"] == "primary"),
        len(assertions),
    )
    window = assertions[primary_idx + 1 : next_primary_idx]

    strong_lows = [
        a for a in assertions if a["role"] == "ci_low" and p_raw_l in a["context"].lower()
    ]
    strong_highs = [
        a for a in assertions if a["role"] == "ci_high" and p_raw_l in a["context"].lower()
    ]
    if strong_lows and strong_highs:
        return (primary["value"], strong_lows[0]["value"], strong_highs[0]["value"])

    win_lows = [a for a in window if a["role"] == "ci_low"]
    win_highs = [a for a in window if a["role"] == "ci_high"]
    if win_lows and win_highs:
        return (primary["value"], win_lows[0]["value"], win_highs[0]["value"])

    return None


def replay_old_triple(ext: list[dict]) -> tuple[float, float, float] | None:
    old_idx: int | None = None
    for i, a in enumerate(ext):
        if a["role"] == "primary":
            ctx_l = a["context"].lower()
            raw_l = a["raw_text"].lower()
            if "odds ratio" in ctx_l or "or " in raw_l or raw_l.startswith("or"):
                old_idx = i
                break
    if old_idx is None:
        for i, a in enumerate(ext):
            if a["role"] == "primary" and a.get("unit") is None:
                old_idx = i
                break
    if old_idx is None:
        return None
    old_v = ext[old_idx]["value"]
    old_lo = next((a["value"] for a in ext[old_idx:] if a["role"] == "ci_low"), None)
    old_hi = next((a["value"] for a in ext[old_idx:] if a["role"] == "ci_high"), None)
    if old_lo is None or old_hi is None:
        return None
    return (old_v, old_lo, old_hi)


def main() -> int:
    runs = ["elicit_glp1_mace", "elicit_io_nsclc_gaps", "elicit_psilocybin"]
    regs: list[tuple[str, str, object, object, str]] = []
    kills: list[tuple[str, str, object, str]] = []
    ok = 0

    for run in runs:
        rep = json.loads(
            pathlib.Path(f"benchmarks/real_outputs/{run}/report.json").read_text(encoding="utf-8")
        )
        for c in rep["claims"]:
            v = c.get("verification") or {}
            nc = v.get("numeric_check")
            if nc is None or nc["check_type"] != "or_ci_consistency":
                continue
            ext = nc["extracted"]
            old_triple = replay_old_triple(ext)
            new_triple = find_triple(ext)
            old_consistent = nc["consistent"]
            if old_consistent:
                if new_triple == old_triple:
                    ok += 1
                else:
                    regs.append((run, c["claim_id"], old_triple, new_triple, c["claim_text"][:140]))
            else:
                if new_triple is None:
                    kills.append((run, c["claim_id"], old_triple, c["claim_text"][:120]))
                else:
                    regs.append((run, c["claim_id"], "FP-old", new_triple, c["claim_text"][:140]))

    print(f"=== Currently-correct preserved: {ok}/19 ===")
    print(f"=== Currently-FP killed: {len(kills)}/4 ===")
    for d in kills:
        print(f"  [OK] {d[0]} [{d[1][:8]}] killed FP {d[2]}")
    print()
    print("=== Disagreements (must be empty) ===")
    if not regs:
        print("  [OK] none")
        return 0
    for d in regs:
        print(f"  [FAIL] {d[0]} [{d[1][:8]}] OLD={d[2]} NEW={d[3]}")
        print(f"     {d[4]}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
