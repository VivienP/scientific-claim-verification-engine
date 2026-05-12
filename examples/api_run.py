"""End-to-end client example for the Phase C lite API.

Demonstrates the async-jobs + polling flow:

    POST /verify       -> {job_id, poll_url}
    GET /jobs/{id}     -> poll until status in {completed, failed}
    GET /runs/{id}/copilot_report.html  (download HTML)

Usage:
    # 1) Start the server in another terminal:
    export VERIFIER_API_KEY=local-dev-key
    uvicorn src.api.app:app --host 127.0.0.1 --port 8000

    # 2) Run this client:
    python -m examples.api_run --input path/to/text.txt --output reports/api/

The script polls every ~5s and downloads the HTML report when complete.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description="Lite API client example.")
    parser.add_argument("--api", default="http://127.0.0.1:8000", help="Base URL of the API.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the input .txt file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/api"),
        help="Local directory to save the downloaded HTML report.",
    )
    parser.add_argument(
        "--mode",
        choices=["v1", "copilot"],
        default="copilot",
        help="Pipeline mode.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between job-status polls.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Max seconds to wait for the job to complete.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("VERIFIER_API_KEY")
    if not api_key:
        print(
            "error: VERIFIER_API_KEY env var is required (must match the server's key).",
            file=sys.stderr,
        )
        return 2

    if not args.input.exists():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 2

    text = args.input.read_text(encoding="utf-8")
    headers = {"X-API-Key": api_key}

    with httpx.Client(base_url=args.api, headers=headers, timeout=30.0) as client:
        # 1) Submit
        r = client.post("/verify", json={"text": text, "mode": args.mode})
        r.raise_for_status()
        submitted = r.json()
        job_id = submitted["job_id"]
        print(f"Submitted job: {job_id}", file=sys.stderr)

        # 2) Poll
        deadline = time.time() + args.timeout
        last_status = ""
        while time.time() < deadline:
            r = client.get(f"/jobs/{job_id}")
            r.raise_for_status()
            body = r.json()
            status = body["status"]
            if status != last_status:
                print(f"  status: {status}", file=sys.stderr)
                last_status = status
            if status == "completed":
                break
            if status == "failed":
                print(f"job failed: {body.get('error')}", file=sys.stderr)
                return 1
            time.sleep(args.poll_interval)
        else:
            print("error: timed out waiting for completion", file=sys.stderr)
            return 1

        # 3) Download HTML if copilot mode produced one
        result = body.get("result") or {}
        html_url_path = result.get("report_html_url")
        if html_url_path:
            args.output.mkdir(parents=True, exist_ok=True)
            html_resp = client.get(html_url_path)
            html_resp.raise_for_status()
            out_html = args.output / f"{body['run_id']}.html"
            out_html.write_text(html_resp.text, encoding="utf-8")
            print(f"Wrote HTML report: {out_html}", file=sys.stderr)

        # Pretty-print the result envelope to stdout for piping.
        import json as _json

        print(_json.dumps(body, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
