#!/usr/bin/env python
"""Display SciFact benchmark comparison table from eval result JSON files.

Usage:
    python scripts/show_metrics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

# ---------------------------------------------------------------------------
# Paths (relative to project root — run from project root)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
_PIPELINE_PATH = _ROOT / "eval" / "results" / "baseline_phase0.json"
_BASELINE_PATH = _ROOT / "eval" / "results" / "direct_baseline.json"

# ---------------------------------------------------------------------------
# ANSI
# ---------------------------------------------------------------------------
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_RED = "\033[31m"

_WIDTH = 68

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _pct(val: float) -> str:
    return f"{val:.3f}"


def _delta(pipeline: float, baseline: float) -> str:
    d = pipeline - baseline
    sign = "+" if d >= 0 else ""
    colour = _GREEN if d >= 0 else _RED
    plain = f"{sign}{d:.3f}"
    return f"{colour}{plain:>8s}{_RESET}"


def _row(
    label: str,
    pv: float,
    bv: float,
    *,
    show_delta: bool = True,
) -> None:
    delta = _delta(pv, bv) if show_delta else ""
    print(f"  {label:24s}  {_pct(pv):>10s}  {_pct(bv):>14s}  {delta}")


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def show(pipeline: dict[str, Any], baseline: dict[str, Any]) -> None:
    pm: dict[str, Any] = pipeline["metrics"]
    bm: dict[str, Any] = baseline["metrics"]
    pn: int = pipeline["n_claims"]
    bn: int = baseline["n_claims"]
    pc: float = pipeline["total_cost_usd"]
    bc: float = baseline["total_cost_usd"]
    model: str = pipeline["model_id"]

    border = "═" * _WIDTH
    thin = "─" * _WIDTH

    print(f"\n{_BOLD}{border}{_RESET}")
    print(f" {_BOLD}SciFact Dev Benchmark  —  {model}{_RESET}")
    print(thin)
    header = f"  {'':24s}  {'Pipeline':>10s}  {'Direct Claude':>14s}  {'Δ':>8s}"
    print(f"{_BOLD}{header}{_RESET}")
    print(thin)

    _row(f"F1 macro  (n={pn})", float(pm["macro_f1"]), float(bm["macro_f1"]))
    _row(f"Precision (n={pn})", float(pm["precision"]), float(bm["precision"]))
    _row(f"Recall    (n={pn})", float(pm["recall"]), float(bm["recall"]))

    print(f"\n  {_BOLD}Per-class F1:{_RESET}")
    p_pc: dict[str, Any] = pm["per_class"]
    b_pc: dict[str, Any] = bm["per_class"]
    for cls, label in (
        ("supported", "  • Supported"),
        ("unsupported", "  • Unsupported"),
        ("not_addressed", "  • Not addressed"),
    ):
        _row(label, float(p_pc[cls]["f1"]), float(b_pc[cls]["f1"]))

    print(f"\n{thin}")
    print(f"  {'Run cost':24s}  ${pc:>9.2f}  ${bc:>13.2f}")
    print(
        f"  {_DIM}* Baseline run on {bn} claims;"
        f" pipeline on {pn} claims.{_RESET}"
    )
    print(f"{_BOLD}{border}{_RESET}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    for path in (_PIPELINE_PATH, _BASELINE_PATH):
        if not path.exists():
            print(f"Error: {path} not found.", file=sys.stderr)
            sys.exit(1)

    pipeline = _load(_PIPELINE_PATH)
    baseline = _load(_BASELINE_PATH)
    show(pipeline, baseline)


if __name__ == "__main__":
    main()
