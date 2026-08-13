# Architecture defaults and trade-offs

## Selected defaults

| Decision | Default | Why it is selected |
|---|---|---|
| Orchestration | LangGraph | The answer path has fixed safety gates, explicit branches, retries, and auditable state. LangGraph keeps this control flow inspectable and replayable. |
| Integrations | LangChain Core | Use only for model and retriever abstractions; it does not own safety-critical workflow routing. |
| Vector database | Weaviate | Native BM25 plus dense hybrid retrieval, relative-score fusion, explainable scores, source filters, and reranking integration reduce custom search infrastructure. |
| Embeddings | Ollama `bge-m3` | Local reusable BGE service for both document and query vectors; 1,024-dimensional vectors, long-input support, and no embedding API key. |
| Reranking | Cohere Rerank primary; local BGE reranker fallback | Cohere provides managed cross-encoder quality; local fallback supports a reproducible demo and provider outage path. |
| LLM route | OmniRoute OpenAI-compatible adapter | Keeps the application provider-neutral and allows routes/fallbacks to be configured without code changes. |
| Evaluation | Ragas + deterministic citation checks | Ragas assesses faithfulness while code verifies IDs, exact quote spans, and evidence-set membership. |

## Why not LangChain-only?

LangChain provides useful integrations, but a governed RAG answer needs a visible state machine: retrieve, rerank, gate evidence, generate, validate citations, and abstain. LangGraph provides those discrete nodes and durable state. A free-form agent loop could skip or weaken a mandatory safety gate.

## Why Weaviate instead of ChromaDB?

Weaviate is the production default because it natively combines vector search with BM25, can return score explanations, and supports reranking on hybrid results. ChromaDB is convenient for a local dense-vector prototype but needs a separate BM25 index and application-owned fusion/ranking path to meet this project's hybrid requirement. The retrieval port keeps a future ChromaDB development adapter possible without changing APIs.

## Why an OmniRoute adapter, not OmniRoute coupling?

OmniRoute is treated as an OpenAI-compatible gateway behind a small provider interface. Endpoint, credentials, model route, retry policy, and timeout are external configuration. A direct-provider adapter remains available so a gateway issue cannot silently remove the service's ability to abstain safely.

## Quality-gate policy

The gate requires faithfulness >= 0.95, citation validity = 1.00, answer correctness >= 0.85, abstention F1 >= 0.90, and no approved-baseline regression greater than 0.02. A failed run must classify the fault before changing retrieval settings, prompts, corpus, or baseline.
