"""Lock the SciFact dev baseline against accidental regressions (S1-P0).

Reads two SciFact eval JSON files (the format produced by `scripts/eval_scifact.py`)
and verifies that the candidate's f1 and macro_f1 are within tolerance of the baseline.
Exits 0 on pass, 1 on regression.

CLI usage::

    python scripts/check_scifact_regression.py \
        eval/results/baseline_phase0.json \
        eval/results/post_s1.json

The defaults match the project rule (no-regression.md): F1 must not drop more than
1 % vs the recorded baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    return {
        "f1": float(metrics["f1"]),
        "macro_f1": float(metrics["macro_f1"]),
    }


def check_regression(
    baseline_path: Path,
    candidate_path: Path,
    *,
    f1_tolerance: float = 0.01,
    macro_f1_tolerance: float = 0.01,
) -> tuple[bool, str]:
    """Return (passes, reason). Passes when both metrics are within tolerance.

    A drop of exactly `tolerance` is considered passing; only a drop strictly
    greater than tolerance fails the check. Improvements always pass.
    """
    baseline = _load_metrics(baseline_path)
    candidate = _load_metrics(candidate_path)

    f1_drop = baseline["f1"] - candidate["f1"]
    macro_drop = baseline["macro_f1"] - candidate["macro_f1"]

    if f1_drop > f1_tolerance:
        return (
            False,
            f"f1 regression: baseline={baseline['f1']:.3f} candidate={candidate['f1']:.3f} "
            f"drop={f1_drop:.3f} > tolerance={f1_tolerance:.3f}",
        )
    if macro_drop > macro_f1_tolerance:
        return (
            False,
            f"macro_f1 regression: baseline={baseline['macro_f1']:.3f} "
            f"candidate={candidate['macro_f1']:.3f} drop={macro_drop:.3f} "
            f"> tolerance={macro_f1_tolerance:.3f}",
        )
    return (
        True,
        f"within tolerance: f1 {baseline['f1']:.3f}->{candidate['f1']:.3f} "
        f"macro_f1 {baseline['macro_f1']:.3f}->{candidate['macro_f1']:.3f}",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Path to baseline SciFact eval JSON")
    parser.add_argument("candidate", type=Path, help="Path to candidate SciFact eval JSON")
    parser.add_argument("--f1-tolerance", type=float, default=0.01)
    parser.add_argument("--macro-f1-tolerance", type=float, default=0.01)
    args = parser.parse_args(argv)

    passes, reason = check_regression(
        args.baseline,
        args.candidate,
        f1_tolerance=args.f1_tolerance,
        macro_f1_tolerance=args.macro_f1_tolerance,
    )
    print(reason)
    return 0 if passes else 1


if __name__ == "__main__":
    sys.exit(main())
