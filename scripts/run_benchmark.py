"""Runs the frozen 200-case benchmark through the live governed RAG API."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx


def citations_are_valid(response: dict) -> bool:
    """Rechecks that returned citation IDs and quote spans belong to retrieved evidence."""

    if response.get("status") == "abstained":
        return not response.get("citations")
    by_id = {chunk["chunk_id"]: chunk["text"] for chunk in response.get("retrieved", [])}
    citations = response.get("citations", [])
    return bool(citations) and all(
        citation.get("chunk_id") in by_id and citation.get("quote") in by_id[citation["chunk_id"]]
        for citation in citations
    )


def main() -> None:
    """Writes evidence-bearing API result records, optionally resuming an interrupted run."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, help="Run only the first N cases for a local preflight.")
    parser.add_argument("--resume", action="store_true", help="Append only cases not already recorded in --output.")
    parser.add_argument("--delay-seconds", type=float, default=0.0, help="Pause between API calls to reduce provider throttling.")
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line]
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be at least 1.")
        cases = cases[:args.limit]
    if args.delay_seconds < 0:
        raise SystemExit("--delay-seconds must be zero or greater.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed_ids: set[str] = set()
    if args.resume and args.output.exists():
        completed_ids = {
            json.loads(line)["id"] for line in args.output.read_text(encoding="utf-8").splitlines() if line
        }
    pending_cases = [case for case in cases if case["id"] not in completed_ids]
    mode = "a" if args.resume else "w"
    # Local cross-encoder startup and a remote generator retry can exceed a short client
    # timeout even while the governed API is still healthy. Keep the API timeout explicit and
    # retry a single idempotent request before classifying it as an operational failure.
    with httpx.Client(timeout=300) as client, args.output.open(mode, encoding="utf-8") as output:
        readiness = client.get(f"{args.api_url.rstrip('/')}/readyz")
        if readiness.status_code != 200:
            raise SystemExit(
                f"Benchmark aborted: API dependencies are not ready ({readiness.status_code}): "
                f"{readiness.text}"
            )
        for position, case in enumerate(pending_cases, start=1):
            try:
                payload = {"question": case["question"], "trace_id": f"benchmark-{case['id']}"}
                try:
                    reply = client.post(f"{args.api_url.rstrip('/')}/v1/query", json=payload)
                except httpx.TimeoutException:
                    # The trace ID makes duplicate delivery observable; retry only once so a
                    # hung provider cannot turn one benchmark case into unbounded traffic.
                    reply = client.post(f"{args.api_url.rstrip('/')}/v1/query", json=payload)
                reply.raise_for_status()
                response = reply.json()
                record = {
                    "id": case["id"], "question": case["question"], "reference_answer": case["reference_answer"],
                    "expected_abstention": case["expected_abstention"], "answer": response["answer"],
                    "retrieved_contexts": [chunk["text"] for chunk in response.get("retrieved", [])],
                    "retrieved_chunk_ids": [chunk["chunk_id"] for chunk in response.get("retrieved", [])],
                    "gold_chunk_ids": case["gold_chunk_ids"],
                    "citation_valid": citations_are_valid(response), "actual_abstention": response["status"] == "abstained",
                    "failure_stage": (
                        "provider" if "generation provider was unavailable" in response["answer"]
                        else "citation_validation" if response["status"] == "abstained" else "generation"
                    ),
                }
                # A response that violates the expected refusal decision or deterministic
                # citation contract is a benchmark failure even before Ragas grades content.
                record["failed"] = (
                    not record["citation_valid"]
                    or record["actual_abstention"] != record["expected_abstention"]
                )
            except httpx.HTTPError as error:
                record = {
                    "id": case["id"], "question": case["question"], "reference_answer": case["reference_answer"],
                    "expected_abstention": case["expected_abstention"], "answer": "", "retrieved_contexts": [],
                    "retrieved_chunk_ids": [], "gold_chunk_ids": case["gold_chunk_ids"],
                    "citation_valid": False, "actual_abstention": True, "failure_stage": "provider", "error": str(error),
                    "failed": True,
                }
            output.write(json.dumps(record) + "\n")
            output.flush()
            print({"completed": position, "total": len(pending_cases), "id": case["id"]}, flush=True)
            if args.delay_seconds and position < len(pending_cases):
                # Sequential pacing avoids turning a free or shared provider route into a
                # benchmark of rate-limit behavior rather than retrieval and grounding quality.
                time.sleep(args.delay_seconds)


if __name__ == "__main__":
    main()
