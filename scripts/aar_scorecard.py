#!/usr/bin/env python
"""Compute the AAR scorecard for one or more pipeline run directories.

Usage:

    python scripts/aar_scorecard.py reports/runs/<report_id>
    python scripts/aar_scorecard.py reports/runs/<id1> reports/runs/<id2> --json
    python scripts/aar_scorecard.py --all   # every run under reports/runs/

The script writes nothing by default; pass --output to persist a JSON
snapshot. Exit code is 0 unless --fail-below is used and any metric
falls below the threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aar import AARScorecard, compute_aar_for_run, render_scorecard_markdown  # noqa: E402

DEFAULT_RUNS_DIR = PROJECT_ROOT / "reports" / "runs"


def discover_run_dirs(run_paths: list[Path], use_all: bool) -> list[Path]:
    """Return the list of run directories to score.

    --all expands to every direct child of reports/runs/ that contains a
    report.json. Explicit paths are passed through.
    """
    if use_all:
        if not DEFAULT_RUNS_DIR.exists():
            return []
        return sorted(p for p in DEFAULT_RUNS_DIR.iterdir() if (p / "report.json").exists())
    return run_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="*", type=Path)
    parser.add_argument("--all", action="store_true", dest="use_all")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of markdown")
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the scorecard(s) to this path (JSON list)",
    )
    parser.add_argument(
        "--fail-below",
        type=float,
        default=None,
        metavar="THRESHOLD",
        help="Exit with code 1 if any of {pcov, psnd, ctran} falls below this fraction",
    )
    args = parser.parse_args()

    run_dirs = discover_run_dirs(args.run_dirs, args.use_all)
    if not run_dirs:
        print("No run directories provided. Pass paths or --all.", file=sys.stderr)
        return 2

    cards: list[tuple[Path, AARScorecard]] = []
    for run_dir in run_dirs:
        try:
            card = compute_aar_for_run(run_dir)
        except FileNotFoundError as exc:
            print(f"SKIP {run_dir}: {exc}", file=sys.stderr)
            continue
        cards.append((run_dir, card))

    if args.json or args.output:
        payload = [{"run_dir": str(p), **asdict(c)} for p, c in cards]
        text = json.dumps(payload, indent=2)
        if args.output:
            args.output.write_text(text, encoding="utf-8")
            print(f"Wrote {len(cards)} scorecards to {args.output}")
        if args.json:
            print(text)
    else:
        for run_dir, card in cards:
            print(f"## {run_dir.name}\n")
            print(render_scorecard_markdown(card))

    if args.fail_below is not None:
        for run_dir, card in cards:
            for metric_name in ("pcov", "psnd", "ctran"):
                value = getattr(card, metric_name)
                if value < args.fail_below:
                    print(
                        f"FAIL {run_dir.name}: {metric_name}={value:.2%} "
                        f"< threshold {args.fail_below:.2%}",
                        file=sys.stderr,
                    )
                    return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
