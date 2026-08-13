"""Deletes one explicitly confirmed, superseded Weaviate corpus release."""

from __future__ import annotations

import argparse

import weaviate
from weaviate.classes.query import Filter

from rag_copilot.ingestion import _delete_release_in_confirmed_rounds
from rag_copilot.settings import settings


def main() -> None:
    """Deletes only the reviewed release ID after count and typed confirmation checks."""

    parser = argparse.ArgumentParser(description="Remove one obsolete immutable corpus release.")
    parser.add_argument("--release-id", required=True, help="Exact release identifier to remove.")
    parser.add_argument("--confirm-release", required=True, help="Must exactly repeat --release-id.")
    parser.add_argument("--expected-count", type=int, required=True, help="Expected object count before deletion.")
    args = parser.parse_args()
    if args.confirm_release != args.release_id:
        raise SystemExit("Release confirmation must exactly equal --release-id.")

    config = settings()
    host = config.weaviate_url.replace("http://", "").split(":")[0]
    client = weaviate.connect_to_local(host=host)
    try:
        collection = client.collections.use("GovernanceChunk")
        release_filter = Filter.by_property("corpus_release_id").equal(args.release_id)
        count = collection.aggregate.over_all(total_count=True, filters=release_filter).total_count or 0
        # A count mismatch stops the operation before mutation. This protects a release from
        # accidental deletion when an operator copied an incorrect identifier or count.
        if count != args.expected_count:
            raise SystemExit(f"Refusing deletion: found {count} objects; expected {args.expected_count}.")
        _delete_release_in_confirmed_rounds(collection, release_filter, args.release_id)
        remaining = collection.aggregate.over_all(total_count=True, filters=release_filter).total_count or 0
        if remaining:
            raise RuntimeError(f"Deletion incomplete: {remaining} objects remain for {args.release_id!r}.")
        print({"deleted": count, "release": args.release_id, "remaining": remaining})
    finally:
        client.close()


if __name__ == "__main__":
    main()
