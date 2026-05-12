"""Verify that README.md real-output benchmark table matches README.md.

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
        rows[label] = dict(zip(headers[1:], cells[1:], strict=False))
    return rows


def _find_row(label: str, rows: dict[str, dict[str, str]]) -> dict[str, str] | None:
    for key, val in rows.items():
        if label.lower() in _normalize(key):
            return val
    return None


def _find_row_key(label: str, rows: dict[str, dict[str, str]]) -> str | None:
    for key in rows:
        if label.lower() in _normalize(key):
            return key
    return None


def _find_col(col: str, row: dict[str, str]) -> str | None:
    target = _normalize(col)
    for k, v in row.items():
        if _normalize(k) == target:
            return v
    return None


def _to_int(val: str) -> int:
    return int(re.sub(r"[^\d]", "", val))


def _to_percent_tenths(val: str) -> int:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", val)
    if match is None:
        raise ValueError(f"percentage not found: {val!r}")
    return round(float(match.group(1)) * 10)


def _check_compact_track_record(
    summary_rows: dict[str, dict[str, str]], readme_text: str
) -> list[str]:
    """Validate the compact README Track Record row against SUMMARY total.

    The README's Track Record section carries a single row for real-output benchmarks
    (total claim count + citation-found rate) rather than the full table.
    This check validates those public numbers against README.md.
    """
    mismatches: list[str] = []
    total = _find_row("Total", summary_rows)
    if total is None:
        return ["SUMMARY: no Total row"]

    try:
        track_rows = _parse_table(readme_text, "## Track Record")
    except ValueError as exc:
        return [f"ERROR parsing README.md compact Track Record: {exc}"]

    readme_key = _find_row_key("Real AI-for-science tools", track_rows)
    if readme_key is None:
        return ["README: no Track Record row matching 'Real AI-for-science tools'"]
    readme_row = track_rows[readme_key]

    summary_claims_raw = _find_col("claims", total)
    summary_cfr_raw = _find_col("citation_found_rate", total)
    readme_result = _find_col("result", readme_row) or ""

    if summary_claims_raw is None:
        mismatches.append("SUMMARY: Total row missing 'claims'")
    else:
        summary_claims = _to_int(summary_claims_raw)
        readme_claims = _to_int(readme_key)
        if summary_claims != readme_claims:
            mismatches.append(
                f"compact README claims mismatch: SUMMARY={summary_claims} README={readme_claims}"
            )

    if summary_cfr_raw is None:
        mismatches.append("SUMMARY: Total row missing 'citation_found_rate'")
    else:
        summary_cfr = _to_percent_tenths(summary_cfr_raw)
        readme_cfr = _to_percent_tenths(readme_result)
        if summary_cfr != readme_cfr:
            mismatches.append(
                "compact README citation_found_rate mismatch: "
                f"SUMMARY={summary_cfr_raw} README={readme_result}"
            )

    return mismatches


def main() -> int:
    summary_path = ROOT / "benchmarks" / "real_outputs" / "README.md"
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
        print(f"ERROR parsing README.md: {e}", file=sys.stderr)
        return 1

    readme_rows: dict[str, dict[str, str]] | None = None
    try:
        readme_rows = _parse_table(readme_text, "## Real-Output Benchmark")
    except ValueError:
        compact_mismatches = _check_compact_track_record(summary_rows, readme_text)
        if compact_mismatches:
            print("ALIGNMENT FAILURE — mismatches between README.md and README.md:")
            for m in compact_mismatches:
                print(f"  {m}")
            return 1
        return 0

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
                mismatches.append(f"tool={label!r} col={col!r}: SUMMARY={s_int} README={r_int}")

    if mismatches:
        print("ALIGNMENT FAILURE — mismatches between README.md and README.md:")
        for m in mismatches:
            print(f"  {m}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
