"""Indexes the approved frozen corpus into Weaviate using local BGE embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import weaviate
import yaml
from rag_copilot.embeddings import OllamaEmbedder
from rag_copilot.ingestion import extract_pages, index_weaviate, semantic_chunks
from rag_copilot.settings import settings


def main() -> None:
    """Parses every approved local PDF and indexes page-provenance chunks in Weaviate."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", required=True, help="New immutable corpus-release identifier to index.")
    parser.add_argument("--confirm-release", required=True)
    parser.add_argument(
        "--replace-existing-release",
        action="store_true",
        help="Delete and rebuild only the confirmed corpus release after reviewing the target count.",
    )
    args = parser.parse_args(); config = settings()
    if args.confirm_release != args.release_id:
        raise SystemExit("Release confirmation must equal --release-id.")
    expanded = yaml.safe_load(Path("data/corpus/expanded_manifest.yaml").read_text())
    mapping = {Path(item["local_path"]): item for item in expanded["sources"]}
    chunks = []
    for path, source in mapping.items():
        for page in extract_pages(source["source_id"], path):
            for chunk in semantic_chunks(page):
                # Carry the publisher URL forward exactly as recorded in the frozen manifest.
                # This makes every returned citation independently auditable by the user.
                chunk.update({"title": source["title"], "source_url": source["url"]})
                chunks.append(chunk)
    client = weaviate.connect_to_local(host=config.weaviate_url.replace("http://", "").split(":")[0])
    try:
        # Use the same local Ollama model for all corpus vectors. Future projects can reuse
        # this running service while maintaining their own Weaviate collection and release ID.
        embedder = OllamaEmbedder(config.ollama_base_url, config.ollama_embedding_model)
        indexed = index_weaviate(
            client,
            chunks,
            args.release_id,
            embedder,
            replace_existing_release=args.replace_existing_release,
        )
        print({"indexed": indexed, "release": args.release_id, "source_chunk_count": len(chunks)})
    finally:
        client.close()


if __name__ == "__main__":
    main()
