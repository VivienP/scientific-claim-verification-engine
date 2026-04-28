"""Fetch a real FutureHouse Crow output and save under benchmarks/real_outputs/.

Writes:
  benchmarks/real_outputs/futurehouse_crow/input.txt
  benchmarks/real_outputs/futurehouse_crow/meta.json

The pipeline run on this input is performed by a separate step (Stage 1B post-fetch)
to keep the fetch isolated from pipeline failures and rate-limit handling.
"""
import json
import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from futurehouse_client import FutureHouseClient, JobNames
from futurehouse_client.models.app import TaskRequest

load_dotenv()

QUESTION = "What is the role of TREM2 in Alzheimer's disease microglia?"


def main() -> None:
    api_key = os.environ["FUTUREHOUSE_API_KEY"]
    client = FutureHouseClient(api_key=api_key)

    print(f"Submitting query to Crow: {QUESTION!r}")
    print("This may take 2-3 minutes...")

    response = client.run_tasks_until_done(
        TaskRequest(name=JobNames.CROW, query=QUESTION)
    )

    output_dir = Path("benchmarks/real_outputs/futurehouse_crow")
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = output_dir / "input.txt"
    meta_path = output_dir / "meta.json"

    answer = response.formatted_answer or response.answer or ""
    input_path.write_text(answer, encoding="utf-8")

    meta = {
        "tool": "FutureHouse Crow",
        "agent_or_model": "crow-v1",
        "prompt_used": QUESTION,
        "fetch_date": str(date.today()),
        "source_url_if_any": None,
        "has_successful_answer": bool(response.has_successful_answer),
        "license_note": "Output used for verification benchmarking, see FutureHouse ToS",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nSaved {len(answer)} chars to {input_path}")
    print(f"Saved metadata to {meta_path}")
    print(f"Has successful answer: {response.has_successful_answer}")
    print("\nNext: run the Phase 1 pipeline on this input to produce report.json.")


if __name__ == "__main__":
    main()
