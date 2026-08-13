"""Hybrid retrieval and reranking ports with deterministic local fallbacks for testing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .schemas import EvidenceChunk


class HybridRetriever(Protocol):
    """Returns the configured dense-plus-BM25 candidate pool."""

    async def retrieve(
        self, question: str, limit: int = 30, source_ids: Sequence[str] | None = None
    ) -> list[EvidenceChunk]: ...


class Reranker(Protocol):
    """Reorders the candidate pool using a query-document cross encoder."""

    async def rerank(self, question: str, candidates: list[EvidenceChunk], limit: int = 6) -> list[EvidenceChunk]: ...


class InMemoryHybridRetriever:
    """Small deterministic retrieval implementation used by tests and the offline demo."""

    def __init__(self, chunks: list[EvidenceChunk] | None = None) -> None:
        """Accepts pre-indexed chunks so no external service is required in unit tests."""

        self.chunks = chunks or []

    async def retrieve(
        self, question: str, limit: int = 30, source_ids: Sequence[str] | None = None
    ) -> list[EvidenceChunk]:
        """Scores lexical overlap as a transparent local substitute for Weaviate hybrid search."""

        tokens = {token.lower().strip(".,?!") for token in question.split()}
        ranked = []
        for chunk in self.chunks:
            if source_ids and chunk.metadata.source_id not in source_ids:
                continue
            overlap = sum(token in chunk.text.lower() for token in tokens) / max(len(tokens), 1)
            ranked.append(chunk.model_copy(update={"hybrid_score": overlap}))
        return sorted(ranked, key=lambda item: item.hybrid_score, reverse=True)[:limit]


class ScoreReranker:
    """Test-safe reranker that preserves retrieval score ordering until a model is configured."""

    async def rerank(
        self, question: str, candidates: list[EvidenceChunk], limit: int = 6
    ) -> list[EvidenceChunk]:
        """Copies hybrid scores into rerank scores for deterministic local execution."""

        reranked = [chunk.model_copy(update={"rerank_score": chunk.hybrid_score}) for chunk in candidates]
        return sorted(reranked, key=lambda item: item.rerank_score, reverse=True)[:limit]


class SentenceTransformersReranker:
    """Local cross-encoder fallback used when managed Cohere reranking is unavailable."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        """Defers model loading until startup so imports remain lightweight for unit tests."""

        self.model_name = model_name
        self._model = None

    @staticmethod
    def _device() -> str:
        """Selects Apple Silicon MPS when PyTorch exposes it, otherwise keeps CPU portability."""

        import torch

        # MPS uses the Mac GPU without changing any retrieval semantics. CPU remains the safe
        # fallback for Intel Macs, CI runners, or environments where PyTorch lacks MPS support.
        return "mps" if torch.backends.mps.is_available() else "cpu"

    async def rerank(self, question: str, candidates: list[EvidenceChunk], limit: int = 6) -> list[EvidenceChunk]:
        """Scores query-document pairs with a cross encoder and retains only final evidence."""

        from sentence_transformers import CrossEncoder

        if self._model is None:
            self._model = CrossEncoder(self.model_name, device=self._device())
        scores = self._model.predict([(question, item.text) for item in candidates])
        ranked = [item.model_copy(update={"rerank_score": float(score)}) for item, score in zip(candidates, scores, strict=True)]
        return sorted(ranked, key=lambda item: item.rerank_score, reverse=True)[:limit]


class CohereReranker:
    """Managed Cohere reranker with an explicit local cross-encoder fallback."""

    def __init__(self, api_key: str, fallback: Reranker) -> None:
        """Stores the credential only in memory and delegates outages to the local reranker."""

        self.api_key, self.fallback = api_key, fallback

    async def rerank(self, question: str, candidates: list[EvidenceChunk], limit: int = 6) -> list[EvidenceChunk]:
        """Calls Cohere's rerank endpoint, preserving candidate provenance and safe fallback behavior."""

        if not self.api_key:
            return await self.fallback.rerank(question, candidates, limit)
        import httpx
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post("https://api.cohere.com/v2/rerank", headers={"Authorization": f"Bearer {self.api_key}"}, json={"model": "rerank-v3.5", "query": question, "documents": [item.text for item in candidates], "top_n": limit})
                response.raise_for_status()
            return [candidates[item["index"]].model_copy(update={"rerank_score": float(item["relevance_score"])}) for item in response.json()["results"]]
        except httpx.HTTPError:
            return await self.fallback.rerank(question, candidates, limit)
