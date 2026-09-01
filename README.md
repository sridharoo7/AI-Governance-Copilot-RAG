# Evidence-Grounded AI Governance Copilot-RAG

An English-only, evidence-grounded Retrieval-Augmented Generation (RAG) copilot for AI governance and agentic-security guidance. It answers only from an approved, frozen NIST/OWASP corpus, attaches source-excerpt citations to factual claims, and abstains when the retrieved evidence is insufficient.

> **Scope:** Engineering decision support, not legal advice. The initial corpus and benchmark are English only.

## What this project demonstrates

- **Governed retrieval:** BGE-M3 dense retrieval plus BM25 keyword retrieval in Weaviate, followed by cross-encoder reranking.
- **Evidence-first answering:** the generator receives at most six reranked evidence chunks and must return structured atomic claims with chunk IDs and exact quotes.
- **Hard grounding controls:** citation IDs, quote spans, retrieved-set membership, evidence coverage, and answer scope are validated before an answer is released.
- **Safe abstention:** insufficient evidence, malformed model output, and provider outages do not result in unsupported answers.
- **Reproducibility:** source provenance, prompts, retrieval settings, corpus release IDs, model routes, and quality thresholds are version controlled.
- **Measured quality:** a 200-case source-grounded benchmark uses Ragas faithfulness and correctness scoring plus deterministic citation and abstention metrics.

## R2 evaluated release

The active evaluated release is `governance-security-expanded-2026-07-30-r2`.

| Metric | R2 result | Release gate |
|---|---:|---:|
| Ragas faithfulness | **0.9543** | >= 0.95 |
| Citation validity | **1.0000** | 1.00 |
| Ragas answer correctness | **0.8982** | >= 0.85 |
| Abstention F1 | **0.9302** | >= 0.90 |

R2 contains 4,782 audited semantic chunks. The chunk audit found zero sentence fragments and zero unsafe OCR/control-template chunks. See the [final project report](docs/final-project-report.md) and [R2 metrics](data/evaluation/runs/full-200-r2-ragas-metrics.json) for the evidence and diagnostics.

## Architecture

```text
Question
  -> query normalization / source-anchor detection
  -> parallel hybrid retrieval: BGE-M3 dense + BM25 keyword search
  -> Weaviate relative-score fusion (30 candidates)
  -> Cohere rerank or local BAAI cross-encoder fallback
  -> top 6 evidence chunks
  -> evidence-sufficiency decision
  -> structured answer: atomic claims + chunk IDs + exact quotes
  -> deterministic citation and claim validation
  -> cited answer or governed abstention
```

The FastAPI service executes this path as an explicit LangGraph workflow. LangChain is used for integrations only; LangGraph makes the state, branches, retries, and abstention path inspectable and testable. For detailed decisions and trade-offs, read [architecture defaults](docs/architecture-defaults.md).

## Technology stack

| Concern | Implementation |
|---|---|
| API | FastAPI, Pydantic, OpenAPI at `/docs` |
| Workflow | LangGraph |
| Vector database | Weaviate 1.28 with explicit vectors and hybrid BM25/dense retrieval |
| Embeddings | `bge-m3` served locally by Ollama |
| Reranking | Cohere `rerank-v3.5`, with `BAAI/bge-reranker-v2-m3` local fallback |
| Generation | OpenAI-compatible adapter for OmniRoute, Gemini, OpenRouter, or a direct local endpoint |
| Evaluation | Ragas with a pinned Gemini judge, deterministic citation validation, and checkpoint/resume support |
| UI | Next.js |
| Local infrastructure | Docker Compose for Weaviate; Dockerfiles for API and UI |

## Prerequisites

- Docker Desktop/Engine.
- Python **3.11** or newer.
- Node.js 20+ for the Next.js UI.
- [Ollama](https://ollama.com/) running locally for embeddings.
- A generation route: Gemini, OpenRouter, OmniRoute, or an OpenAI-compatible local endpoint.
- A Google Gemini API key for release-grade Ragas scoring.

Install the local models once:

```bash
ollama pull bge-m3
ollama pull qwen3:8b # only when using the direct local generation route
ollama serve
```

The local cross-encoder downloads on first use. On Apple Silicon it uses MPS when PyTorch exposes it; otherwise it falls back to CPU.

## Secure configuration

Create a local environment file. It is ignored by Git; never commit API keys.

```bash
cp .env.example .env
```

For a host-run API, set this baseline in `.env`:

```dotenv
WEAVIATE_URL=http://localhost:8080
WEAVIATE_GRPC_URL=localhost:50051
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=bge-m3
CORPUS_RELEASE_ID=governance-security-expanded-2026-07-30-r2
```

Select **one** primary generation route.

### Gemini generation and Ragas judge

```dotenv
RAG_GENERATION_PROVIDER=gemini
GOOGLE_API_KEY=replace-with-your-key
GEMINI_GENERATION_MODEL=gemini-3.1-flash-lite
RAGAS_JUDGE_PROVIDER=gemini_openai_compatible
RAGAS_JUDGE_MODEL=gemini-3.1-flash-lite
```

### OpenRouter generation

```dotenv
RAG_GENERATION_PROVIDER=openrouter
OPENROUTER_API_KEY=replace-with-your-key
OPENROUTER_MODEL=your-pinned-openrouter-model-id
OPENROUTER_SUPPORTS_STRUCTURED_OUTPUTS=true
```

### OmniRoute generation with a direct fallback

```dotenv
OMNIROUTE_BASE_URL=https://your-omniroute-host/v1
OMNIROUTE_API_KEY=replace-with-your-key
RAG_MODEL=your-pinned-model-id
DIRECT_LLM_BASE_URL=http://localhost:11434/v1
DIRECT_LLM_API_KEY=ollama
```

The OpenRouter path intentionally has no local fallback so benchmark/release behavior remains attributable to the pinned route. Secrets belong in `.env` or a production secret manager, never in source, benchmark files, or logs.

## Local setup and first run

### 1. Install Python dependencies

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

### 2. Start Weaviate

```bash
docker compose up -d weaviate
docker compose ps
```

### 3. Verify the frozen corpus

The expanded corpus manifest lists ten official NIST PDFs, totaling 2,677 pages. Verify source/page/checksum expectations before indexing:

```bash
PYTHONPATH=src python scripts/validate_expanded_corpus.py
```

### 4. Index R2

If R2 is not already indexed, run the approval-controlled ingest command. It writes only the release named in both flags.

```bash
PYTHONPATH=src python scripts/ingest_release.py \
  --release-id governance-security-expanded-2026-07-30-r2 \
  --confirm-release governance-security-expanded-2026-07-30-r2
```

Audit the resulting chunk release:

```bash
PYTHONPATH=src python scripts/audit_chunk_release.py \
  --release-id governance-security-expanded-2026-07-30-r2
```

### 5. Run the API

```bash
PYTHONPATH=src uvicorn rag_copilot.api:app --host 127.0.0.1 --port 8000
```

In a separate terminal, verify readiness and open the OpenAPI UI:

```bash
curl http://127.0.0.1:8000/readyz
open http://127.0.0.1:8000/docs
```

`/readyz` checks Ollama because embeddings are required for every governed query. Do not run a benchmark while it returns a non-200 response.

### 6. Ask a grounded question

```bash
curl --request POST http://127.0.0.1:8000/v1/query \
  --header 'Content-Type: application/json' \
  --data '{"question":"According to the approved corpus, what is the purpose of the NIST AI Risk Management Framework?"}'
```

An `answered` response contains citations, retrieved chunk diagnostics, a trace ID, and the active corpus release. An `abstained` response is expected when the corpus cannot support every claim.

### 7. Run the web UI (optional)

```bash
cd web
npm install
npm run dev
```

Open `http://localhost:3000`. The UI shows answers, citation excerpts, retrieval/rerank diagnostics, corpus release, and the insufficient-evidence state.

## API contract

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Process health and active corpus release |
| `GET` | `/healthz`, `/readyz` | Readiness, including the Ollama embedding dependency |
| `POST` | `/v1/query` | Run the governed RAG workflow |
| `POST` | `/v1/ingest` | Register a candidate source for an approval-controlled worker run |
| `GET` | `/v1/traces/{trace_id}` | Return non-sensitive local trace metadata |

`POST /v1/query` accepts:

```json
{"question": "Your question", "trace_id": "optional-client-trace-id"}
```

Every response carries `trace_id` and `corpus_release_id`. Citations include title, URL, page/section when known, exact quote, and chunk ID.

## Corpus and release management

- Keep source URLs, versions, licenses, local locations, page counts, and checksums in `data/corpus/expanded_manifest.yaml`.
- Treat every corpus release as immutable. Index a new identifier rather than overwriting an approved release.
- Run the chunk audit and the full benchmark before promotion.
- Use `CORPUS_RELEASE_ID` to isolate runtime retrieval to one reviewed release.
- Remove an obsolete vector release only after the replacement passes. This affects Weaviate objects only, not source PDFs or historical metrics.

```bash
PYTHONPATH=src python scripts/delete_corpus_release.py \
  --release-id RELEASE_TO_REMOVE \
  --confirm-release RELEASE_TO_REMOVE \
  --expected-count VERIFIED_OBJECT_COUNT
```

## Testing and quality gate

### Fast local checks

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python -m rag_copilot.evaluate \
  --dataset data/evaluation/releases/benchmark_200_r2.jsonl \
  --validate-only
```

### Live benchmark preflight

```bash
PYTHONPATH=src python scripts/run_benchmark.py \
  --dataset data/evaluation/releases/benchmark_200_r2.jsonl \
  --output data/evaluation/runs/preflight-10.jsonl \
  --limit 10
```

### Full 200-case benchmark and Ragas evaluation

```bash
PYTHONPATH=src python scripts/run_benchmark.py \
  --dataset data/evaluation/releases/benchmark_200_r2.jsonl \
  --output data/evaluation/runs/full-200-r2.jsonl

PYTHONPATH=src caffeinate -dimsu python -m rag_copilot.evaluate \
  --dataset data/evaluation/releases/benchmark_200_r2.jsonl \
  --responses data/evaluation/runs/full-200-r2.jsonl \
  --checkpoint data/evaluation/runs/full-200-r2-ragas-checkpoint.json \
  --report data/evaluation/runs/full-200-r2-ragas-metrics.json \
  --baseline data/evaluation/approved_baseline.json \
  --score-delay-seconds 12
```

`caffeinate` is optional and macOS-specific; it prevents laptop sleep during a long evaluation. The checkpoint resumes completed per-case judge scores after a Gemini free-tier quota pause. Do not treat an API or Ollama outage as a quality result—restore readiness and rerun the affected benchmark output.

The benchmark has 200 paragraph-based scenarios: 110 direct factual, 45 multi-source/comparative, 25 exact-term retrieval stress, and 20 unanswerable/adversarial cases. It contains 180 answerable cases; expected refusals are assessed through abstention F1 rather than being counted as unfaithful answers.

## Repository map

```text
src/rag_copilot/       FastAPI service, LangGraph workflow, retrieval, citations, providers, evaluation
config/                Versioned prompts and RAG/runtime quality configuration
data/corpus/           Frozen corpus manifests and source material
data/evaluation/       Benchmark datasets, approved baseline, response/metric outputs
scripts/               Corpus validation, ingest, audit, benchmark, PDF/deck, and deletion tools
tests/                 Unit and behavior tests for grounding, ingestion, providers, and evaluation
web/                   Next.js evidence-exploration interface
docs/                  Architecture decisions, runbook, benchmark guide, and final report
planning/              Kanban documentation and Airtable-importable CSV backlog
artifacts/             Benchmark PDF, chunk audit, and leadership slide deck
```

## Documentation and artifacts

- [Architecture defaults and trade-offs](docs/architecture-defaults.md)
- [Operations and corpus-release runbook](docs/operations.md)
- [R2 benchmark, human-readable](docs/releases/benchmark_200_r2.md)
- [Final engineering report](docs/final-project-report.md)
- [Airtable-importable backlog](planning/airtable_backlog.csv)
- [Leadership PowerPoint](artifacts/slides/evidence_grounded_rag_leadership_deck.pptx)
- [Rendered benchmark PDF](artifacts/benchmark/ai_governance_rag_benchmark_200.pdf)

## Security and operational notes

- `.env`, private keys, service-account files, virtual environments, and local runtime state are ignored by Git. Keep `.env.example` sanitized.
- The API returns a governed abstention rather than an uncaught `500` when a workflow component fails.
- The local trace store is development-only and stores response metadata in process memory. Use a durable, access-controlled trace/checkpoint store before production deployment.
- Docker Compose is for local infrastructure. Add authentication, authorization, secret management, TLS, durable tracing, retention controls, rate limits, and load testing before external deployment.
- Corpus updates require manual review, a new immutable release ID, full ingestion checks, and a complete benchmark run before promotion.

## License and source responsibility

Review the license and reuse terms recorded for every approved source before redistributing corpus content. Preserve source URLs, version information, and citations in downstream use.
