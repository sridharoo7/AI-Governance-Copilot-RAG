"""Audits a prospective semantic-chunk release before embeddings are created."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import yaml

from rag_copilot.ingestion import extract_pages, semantic_chunks


def main() -> None:
    """Builds deterministic chunk-quality diagnostics from the frozen local PDF corpus."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = yaml.safe_load(Path("data/corpus/expanded_manifest.yaml").read_text(encoding="utf-8"))
    chunks = []
    for source in manifest["sources"]:
        for page in extract_pages(source["source_id"], Path(source["local_path"])):
            chunks.extend(semantic_chunks(page))
    word_counts = [len(chunk["text"].split()) for chunk in chunks]
    fragment_count = sum(not chunk["text"].endswith((".", "!", "?")) for chunk in chunks)
    report = {
        "release_id": args.release_id,
        "chunk_count": len(chunks),
        "word_count": {
            "min": min(word_counts), "median": statistics.median(word_counts),
            "p95": sorted(word_counts)[int(0.95 * len(word_counts)) - 1], "max": max(word_counts),
        },
        "fragment_count": fragment_count,
        "unsafe_text_count": sum("\ue000" in chunk["text"] or "\ufffd" in chunk["text"] for chunk in chunks),
        "sectioned_chunk_count": sum(bool(chunk.get("section")) for chunk in chunks),
        "status": "pass" if not fragment_count and not any(count > 360 for count in word_counts) else "review",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
