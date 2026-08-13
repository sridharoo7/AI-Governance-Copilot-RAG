"""FastAPI application exposing governed query, ingestion, health, and trace endpoints."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException
import httpx

from .embeddings import OllamaEmbedder
from .graph import answer_question, build_rag_graph
from .providers import build_provider
from .retrieval import CohereReranker, SentenceTransformersReranker
from .schemas import IngestRequest, QueryRequest, QueryResponse
from .settings import rag_config, settings
from .weaviate_adapter import WeaviateHybridRetriever

app = FastAPI(title="Evidence-Grounded AI Governance Copilot", version="0.1.0")
logger = logging.getLogger(__name__)
_settings = settings()
# This in-process store is intentionally temporary. It makes every API response traceable during
# local development; production replaces it with a durable LangGraph checkpoint/trace backend.
_trace_store: dict[str, QueryResponse] = {}


def _production_graph():
    """Connects the API to the active Weaviate release and managed/local reranker chain."""

    from urllib.parse import urlparse

    import weaviate

    parsed = urlparse(_settings.weaviate_url)
    grpc = urlparse(f"//{_settings.weaviate_grpc_url}")
    client = weaviate.connect_to_custom(http_host=parsed.hostname or "weaviate", http_port=parsed.port or 8080,
        http_secure=parsed.scheme == "https", grpc_host=grpc.hostname or "weaviate", grpc_port=grpc.port or 50051,
        grpc_secure=False)
    config = rag_config()
    # One shared embedding client guarantees the query vector matches the corpus-vector space.
    embedder = OllamaEmbedder(_settings.ollama_base_url, _settings.ollama_embedding_model)
    retriever = WeaviateHybridRetriever(client, config["retrieval"]["collection"], _settings.corpus_release_id, embedder, config["retrieval"]["hybrid_alpha"])
    local = SentenceTransformersReranker(config["models"]["reranker_fallback"])
    reranker = CohereReranker(_settings.cohere_api_key, local)
    return build_rag_graph(retriever, reranker, build_provider(_settings), _settings)


_graph = _production_graph()


@app.get("/health")
async def health() -> dict[str, str]:
    """Reports the active corpus release without exposing secrets or provider internals."""

    return {"status": "ok", "corpus_release_id": _settings.corpus_release_id}


@app.get("/healthz")
@app.get("/readyz")
async def probe() -> dict[str, str]:
    """Reports readiness only when the local embedding dependency can accept queries."""

    # Retrieval cannot work without the BGE-M3 Ollama endpoint. A plain process-level health
    # response previously allowed a benchmark to turn an Ollama outage into hundreds of false
    # RAG abstentions, so readiness explicitly exercises the dependency before test traffic.
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{_settings.ollama_base_url.rstrip('/')}/api/tags")
            response.raise_for_status()
    except httpx.HTTPError as error:
        raise HTTPException(status_code=503, detail="Ollama embedding service is unavailable.") from error
    return await health()


@app.post("/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Runs the citation-enforced retrieval graph for one user question."""

    trace_id = request.trace_id or str(uuid.uuid4())
    try:
        # Persist only the governed response object after the graph has completed every safety gate.
        response = await answer_question(_graph, request.question, trace_id)
    except Exception:
        # An unexpected adapter or model-library defect is operationally distinct from an
        # evidence abstention, but governed clients must never receive an uncaught HTTP 500.
        logger.exception("rag_unhandled_workflow_failure trace_id=%s", trace_id)
        response = QueryResponse(
            status="abstained",
            answer="I cannot answer from the approved corpus because the governed workflow was unavailable.",
            trace_id=trace_id,
            corpus_release_id=_settings.corpus_release_id,
            generated_at=datetime.now(UTC),
        )
    _trace_store[response.trace_id] = response
    return response


@app.post("/v1/ingest", status_code=202)
async def ingest(request: IngestRequest) -> dict[str, str]:
    """Accepts an auditable source registration for a later approval-controlled worker run."""

    return {"status": "accepted", "source_id": request.source_id, "release": _settings.corpus_release_id}


@app.get("/v1/traces/{trace_id}")
async def trace(trace_id: str) -> dict[str, str]:
    """Reserves the trace API; production tracing is attached to LangGraph checkpoints."""

    if not trace_id:
        raise HTTPException(status_code=400, detail="trace_id is required")
    # Return metadata only; the raw question and source passages are deliberately not exposed here.
    response = _trace_store.get(trace_id)
    if response is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return {"trace_id": trace_id, "status": response.status, "corpus_release_id": response.corpus_release_id}
