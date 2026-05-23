"""Run extract_claims() against a single text file and dump full output.

Establishes empirical extraction baselines on arbitrary scientific-publication
inputs (Elicit lit reviews, AnswerThis exports, paper text). Output is written
to reports/audits/<label>/ as three files:
  - claims.json: serialized list of Claim records
  - provenance.jsonl: the extraction ProvenanceStep, one JSON object per line
  - summary.md: human-readable summary (counts, types, sample claims)

The output directory is under reports/, which is gitignored, so per-run
artifacts do not pollute the repository.

Usage:
    python scripts/audit_extractor_baseline.py <input.txt> [--label <slug>]

If --label is omitted, derives from the input filename stem.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import fitz  # type: ignore[import-untyped]
import structlog
from dotenv import load_dotenv

from src.extract import extract_claims
from src.models import Claim, ProvenanceStep

load_dotenv()

logger: structlog.BoundLogger = structlog.get_logger(__name__)

OUTPUT_ROOT = Path("reports/audits")


def _slugify(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_").lower()


def _load_input(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        with fitz.open(path) as doc:
            return "".join(page.get_text() for page in doc)
    return path.read_text(encoding="utf-8")


def _write_outputs(
    out_dir: Path, text: str, claims: list[Claim], step: ProvenanceStep
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "claims.json").write_text(
        json.dumps([dataclasses.asdict(c) for c in claims], indent=2),
        encoding="utf-8",
    )
    (out_dir / "provenance.jsonl").write_text(
        json.dumps(dataclasses.asdict(step)) + "\n",
        encoding="utf-8",
    )

    types_counter: dict[str, int] = {}
    for c in claims:
        types_counter[c.claim_type] = types_counter.get(c.claim_type, 0) + 1

    lines: list[str] = [
        f"# Extractor baseline: {out_dir.name}",
        "",
        f"- Input chars: {len(text)}",
        f"- Claims extracted: {len(claims)}",
        f"- Tokens in: {step.tokens_in}",
        f"- Tokens out: {step.tokens_out}",
        f"- Cache hit: {step.cache_hit}",
        f"- Model: {step.model_id}",
        "",
        "## Claim types",
        "",
    ]
    for claim_type, count in sorted(types_counter.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {claim_type}: {count}")
    lines.extend(["", "## Sample claims (first 10)", ""])
    for i, c in enumerate(claims[:10], 1):
        markers = (
            f" [{','.join(map(str, c.citation_markers))}]" if c.citation_markers else ""
        )
        authors = ", ".join(c.cited_authors) if c.cited_authors else "—"
        year = str(c.cited_year) if c.cited_year is not None else "—"
        lines.append(f"{i}. **[{c.claim_type}]** ({authors} {year}){markers}")
        lines.append(f"   > {c.claim_text}")
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a .txt or .pdf file with scientific publication content",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Output directory name under reports/audits/ (default: input stem)",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=4096,
        help="Cap on LLM output tokens (default: 4096). Raise for dense lit reviews "
        "to reduce truncation-driven under-extraction.",
    )
    args = parser.parse_args()

    if not args.input_path.exists():
        print(f"error: input file not found: {args.input_path}", file=sys.stderr)
        sys.exit(1)

    text = _load_input(args.input_path)
    label = args.label or _slugify(args.input_path.stem)
    out_dir = OUTPUT_ROOT / label

    print(
        f"Extracting from {args.input_path} ({len(text)} chars, "
        f"max_output_tokens={args.max_output_tokens})"
    )
    claims, step = extract_claims(text, max_output_tokens=args.max_output_tokens)
    print(
        f"  -> {len(claims)} claims extracted "
        f"({step.tokens_in} tokens in, {step.tokens_out} out, cache_hit={step.cache_hit})"
    )

    _write_outputs(out_dir, text, claims, step)
    print(f"Wrote outputs to {out_dir}/")


if __name__ == "__main__":
    main()
