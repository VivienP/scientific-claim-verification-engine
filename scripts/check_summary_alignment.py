"""Verify that README.md real-output benchmark table matches SUMMARY.md.

Parses the markdown tables in both files and compares the numeric values for:
  claims, supported, partially_supported, unsupported, not_addressed
for each tool row and the Total row.

Exit 0 on agreement (silent). Exit 1 on any mismatch (prints a diff).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Partial label strings to identify each tool row (case-insensitive substring match)
TOOL_LABELS = ["Edison", "Sakana", "AnswerThis", "Total"]

# Column names to validate (must appear as headers in both tables, case-insensitive, _ == space)
CHECK_COLS = ["claims", "supported", "partially_supported", "unsupported", "not_addressed"]


def _normalize(s: str) -> str:
    return s.lower().replace("_", " ").replace("*", "").strip()


def _parse_table(text: str, after_marker: str) -> dict[str, dict[str, str]]:
    """Return {row_label: {col_header: cell_value}} for the first markdown table after marker."""
    idx = text.find(after_marker)
    if idx == -1:
        raise ValueError(f"Marker not found: {after_marker!r}")
    chunk = text[idx:]

    lines = [ln for ln in chunk.splitlines() if re.match(r"\s*\|", ln)]
    if len(lines) < 3:
        raise ValueError(f"No table (≥3 lines) found after {after_marker!r}")

    headers = [_normalize(h) for h in lines[0].split("|") if h.strip()]
    rows: dict[str, dict[str, str]] = {}
    for line in lines[2:]:
        cells = [c.strip().strip("*").strip() for c in line.split("|") if c.strip()]
        if not cells:
            continue
        label = cells[0]
        rows[label] = dict(zip(headers, cells[1:], strict=False))
    return rows


def _find_row(label: str, rows: dict[str, dict[str, str]]) -> dict[str, str] | None:
    for key, val in rows.items():
        if label.lower() in _normalize(key):
            return val
    return None


def _find_col(col: str, row: dict[str, str]) -> str | None:
    target = _normalize(col)
    for k, v in row.items():
        if _normalize(k) == target:
            return v
    return None


def _to_int(val: str) -> int:
    return int(re.sub(r"[^\d]", "", val))


def main() -> int:
    summary_path = ROOT / "benchmarks" / "real_outputs" / "SUMMARY.md"
    readme_path = ROOT / "README.md"

    try:
        summary_text = summary_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"ERROR: {summary_path} not found", file=sys.stderr)
        return 1

    try:
        readme_text = readme_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"ERROR: {readme_path} not found", file=sys.stderr)
        return 1

    try:
        summary_rows = _parse_table(summary_text, "# Real-Tool Benchmark Summary")
    except ValueError as e:
        print(f"ERROR parsing SUMMARY.md: {e}", file=sys.stderr)
        return 1

    try:
        readme_rows = _parse_table(readme_text, "## Real-Output Benchmark")
    except ValueError as e:
        print(f"ERROR parsing README.md: {e}", file=sys.stderr)
        return 1

    mismatches: list[str] = []

    for label in TOOL_LABELS:
        s_row = _find_row(label, summary_rows)
        r_row = _find_row(label, readme_rows)

        if s_row is None:
            mismatches.append(f"SUMMARY: no row matching '{label}'")
            continue
        if r_row is None:
            mismatches.append(f"README: no row matching '{label}'")
            continue

        for col in CHECK_COLS:
            s_val = _find_col(col, s_row)
            r_val = _find_col(col, r_row)

            if s_val is None or r_val is None:
                # column absent in one file — skip silently
                continue

            try:
                s_int = _to_int(s_val)
                r_int = _to_int(r_val)
            except ValueError:
                continue

            if s_int != r_int:
                mismatches.append(
                    f"tool={label!r} col={col!r}: SUMMARY={s_int} README={r_int}"
                )

    if mismatches:
        print("ALIGNMENT FAILURE — mismatches between SUMMARY.md and README.md:")
        for m in mismatches:
            print(f"  {m}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
