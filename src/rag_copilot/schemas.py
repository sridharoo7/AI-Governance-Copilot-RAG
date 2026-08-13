"""Public API and workflow data contracts for the governed RAG service."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class ChunkMetadata(BaseModel):
    """Preserves source provenance required to render and validate citations."""

    source_id: str
    title: str
    source_url: HttpUrl
    corpus_release_id: str
    page: int | None = None
    section: str | None = None
    parent_chunk_id: str


class EvidenceChunk(BaseModel):
    """A retrieved child chunk together with its retrieval diagnostics."""

    chunk_id: str
    text: str
    metadata: ChunkMetadata
    hybrid_score: float = 0.0
    rerank_score: float = 0.0


class ClaimCitation(BaseModel):
    """Binds an atomic claim to an exact quote in a retrieved evidence chunk."""

    chunk_id: str
    quote: str = Field(min_length=1)


class AtomicClaim(BaseModel):
    """A factual statement that must be independently grounded before release."""

    text: str = Field(min_length=1)
    citations: list[ClaimCitation] = Field(min_length=1)


class GroundedAnswer(BaseModel):
    """Structured model output before the deterministic citation gate runs."""

    answer: str
    claims: list[AtomicClaim] = Field(default_factory=list)
    abstained: bool
    abstention_reason: str | None = None


class QueryRequest(BaseModel):
    """Client query with optional trace ID supplied by an upstream gateway."""

    question: str = Field(min_length=3, max_length=2000)
    trace_id: str | None = None


class CitationView(BaseModel):
    """Citation payload displayed by the API and user interface."""

    chunk_id: str
    quote: str
    title: str
    url: HttpUrl
    page: int | None
    section: str | None


class QueryResponse(BaseModel):
    """Final answer that has passed the citation gate or was safely abstained."""

    status: Literal["answered", "abstained"]
    answer: str
    citations: list[CitationView] = Field(default_factory=list)
    trace_id: str
    corpus_release_id: str
    retrieved: list[EvidenceChunk] = Field(default_factory=list)
    generated_at: datetime


class IngestRequest(BaseModel):
    """A source registration request; fetching is deliberately a separate worker step."""

    source_id: str
    source_url: HttpUrl
    title: str
    license: str
    version: str

