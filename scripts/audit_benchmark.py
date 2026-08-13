"""Audits benchmark traceability before an expensive retrieval or Ragas run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def audit(dataset: Path) -> dict:
    """Checks that every answerable scenario names its canonical frozen source target."""

    records = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line]
    errors: list[dict] = []
    for record in records:
        if record["expected_abstention"]:
            if record["gold_chunk_ids"] or record["source_ids"]:
                errors.append({"id": record["id"], "reason": "abstention case has source evidence"})
            continue
        titles = record.get("source_titles", [])
        if not titles:
            errors.append({"id": record["id"], "reason": "missing canonical source_titles"})
        elif any(title not in record["question"] for title in titles):
            errors.append({"id": record["id"], "reason": "question omits canonical source title"})
        if not record["gold_chunk_ids"] or not record["evidence_quotes"]:
            errors.append({"id": record["id"], "reason": "missing gold chunk or evidence quote"})
    return {
        "dataset": str(dataset),
        "case_count": len(records),
        "answerable_cases": sum(not item["expected_abstention"] for item in records),
        "errors": errors,
        "status": "valid" if len(records) == 200 and not errors else "invalid",
    }


def main() -> None:
    """Prints JSON suitable for a CI artifact and exits nonzero on an invalid benchmark."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.dataset)
    print(json.dumps(result, indent=2))
    if result["status"] != "valid":
        raise SystemExit("Benchmark audit failed.")


if __name__ == "__main__":
    main()
