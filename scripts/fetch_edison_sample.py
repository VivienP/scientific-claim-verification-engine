#!/usr/bin/env python
"""Fetch a Literature agent response from Edison Scientific and save as sample input.

The saved file can then be fed directly into the verification pipeline:

    python scripts/fetch_edison_sample.py "What is the role of TREM2 in Alzheimer's?"
    python examples/sample_run.py examples/inputs/edison_trem2.txt
    python scripts/show_report.py

Requirements:
    pip install edison-client          # or: pip install -e ".[edison]"
    EDISON_API_KEY environment variable (from platform.edisonscientific.com/profile)
    ANTHROPIC_API_KEY environment variable (for the verification pipeline)

Usage:
    python scripts/fetch_edison_sample.py QUERY [--slug SLUG] [--high]

    QUERY     Scientific question to send to the Literature agent
    --slug    Short identifier for the output filename (default: derived from query)
    --high    Use LITERATURE_HIGH (deeper reasoning, slower, costs more)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

try:
    from edison_client import EdisonClient, JobNames
except ImportError:
    print(
        "Error: edison-client is not installed.\n"
        "  pip install edison-client\n"
        "  or: pip install -e \".[edison]\"",
        file=sys.stderr,
    )
    sys.exit(1)

_OUTPUT_DIR = Path(__file__).parent.parent / "examples" / "inputs"


def _slugify(text: str) -> str:
    """Convert a query string to a short safe filename slug (max 40 chars)."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower())
    slug = slug.strip("_")
    return slug[:40].rstrip("_") or "query"


def fetch(query: str, slug: str, *, high: bool = False) -> Path:
    api_key = os.environ.get("EDISON_API_KEY")
    if not api_key:
        print("Error: EDISON_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    job_name = JobNames.LITERATURE_HIGH if high else JobNames.LITERATURE
    client = EdisonClient(api_key=api_key)

    print(f"Querying Edison Scientific Literature agent{'(High)' if high else ''}…")
    print(f"  Query: {query}\n")

    (task_response,) = client.run_tasks_until_done({"name": job_name, "query": query})

    if not getattr(task_response, "has_successful_answer", True):
        print("Warning: Edison returned no successful answer for this query.", file=sys.stderr)

    # formatted_answer includes inline citations like (Smith et al., 2023)
    # which is what extract_claims() is designed to parse.
    # Use getattr because the return type is a union and not all members
    # expose these fields in their type stubs.
    text: str = (
        getattr(task_response, "formatted_answer", None)
        or getattr(task_response, "answer", None)
        or ""
    )
    if not text:
        print("Error: Empty response from Edison.", file=sys.stderr)
        sys.exit(1)

    output_path = _OUTPUT_DIR / f"edison_{slug}.txt"
    output_path.write_text(text, encoding="utf-8")

    print(f"Saved to: {output_path}")
    print(f"Length  : {len(text)} chars")
    print(f"\nPreview :\n{text[:400]}…\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch an Edison Scientific Literature response and save as pipeline input."
    )
    parser.add_argument("query", help="Scientific question to send to the Literature agent")
    parser.add_argument(
        "--slug",
        default=None,
        help="Short identifier for the output filename (default: derived from query)",
    )
    parser.add_argument(
        "--high",
        action="store_true",
        help="Use LITERATURE_HIGH (deeper reasoning, slower)",
    )
    args = parser.parse_args()

    slug = args.slug or _slugify(args.query)
    fetch(args.query, slug, high=args.high)


if __name__ == "__main__":
    main()
