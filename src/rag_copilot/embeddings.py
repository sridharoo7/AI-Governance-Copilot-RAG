"""Ollama embedding client shared by ingestion and query-time hybrid retrieval."""

from __future__ import annotations

import httpx


class OllamaEmbedder:
    """Generates normalized vectors using the local Ollama embedding API."""

    def __init__(self, base_url: str, model: str) -> None:
        """Stores the local endpoint and a single versioned embedding model identity."""

        self.base_url = base_url.rstrip("/")
        self.model = model

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Embeds a batch so index-time vectors use one consistent model and dimension."""

        if not texts:
            return []
        # Ollama's /api/embed endpoint returns L2-normalized vectors. The caller supplies
        # the identical model for document and query vectors, preventing vector-space drift.
        response = httpx.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts, "truncate": False},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        vectors = payload["embeddings"]
        if len(vectors) != len(texts):
            raise ValueError("Ollama returned a different embedding count than the input batch.")
        return [[float(value) for value in vector] for vector in vectors]

    def encode_one(self, text: str) -> list[float]:
        """Embeds one user query for the dense leg of Weaviate hybrid retrieval."""

        return self.encode([text])[0]
