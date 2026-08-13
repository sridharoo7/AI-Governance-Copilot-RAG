"""Weaviate implementation of native dense-plus-BM25 hybrid retrieval."""

from __future__ import annotations

from collections.abc import Sequence

from .schemas import ChunkMetadata, EvidenceChunk


class WeaviateHybridRetriever:
    """Queries Weaviate using relative-score hybrid fusion and explainable scores."""

    def __init__(self, client, collection_name: str, corpus_release_id: str, embedder, alpha: float = 0.55) -> None:
        """Accepts an initialized Weaviate v4 client to keep connection ownership external."""

        self.client = client
        self.collection_name = collection_name
        self.corpus_release_id = corpus_release_id
        self.embedder = embedder
        self.alpha = alpha

    async def retrieve(
        self, question: str, limit: int = 30, source_ids: Sequence[str] | None = None
    ) -> list[EvidenceChunk]:
        """Runs native hybrid retrieval inside the active release and optional named sources."""

        from weaviate.classes.query import Filter, MetadataQuery

        collection = self.client.collections.use(self.collection_name)
        # Because this collection stores externally supplied Ollama vectors, Weaviate cannot
        # vectorize the query itself. Providing the matching query vector activates the dense
        # retrieval leg while `query` continues to drive the BM25 leg of hybrid search.
        query_vector = self.embedder.encode_one(question)
        release_filter = Filter.by_property("corpus_release_id").equal(self.corpus_release_id)
        if source_ids:
            source_filter = Filter.by_property("source_id").equal(source_ids[0])
            for source_id in source_ids[1:]:
                source_filter = source_filter | Filter.by_property("source_id").equal(source_id)
            active_filter = release_filter & source_filter
        else:
            active_filter = release_filter
        result = collection.query.hybrid(
            query=question,
            vector=query_vector,
            alpha=self.alpha,
            limit=limit,
            filters=active_filter,
            return_metadata=MetadataQuery(score=True, explain_score=True),
        )
        return [
            EvidenceChunk(
                chunk_id=item.properties["chunk_id"], text=item.properties["text"],
                metadata=ChunkMetadata(
                    source_id=item.properties["source_id"], title=item.properties["title"],
                    source_url=item.properties["source_url"], corpus_release_id=item.properties["corpus_release_id"],
                    page=item.properties.get("page"), section=item.properties.get("section"),
                    parent_chunk_id=item.properties["parent_chunk_id"],
                ), hybrid_score=float(item.metadata.score or 0),
            ) for item in result.objects
        ]
