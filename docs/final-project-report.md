# Evidence-Grounded AI Governance Copilot-RAG — Final Engineering Report

**Release evaluated:** `governance-security-expanded-2026-07-30-r2`  
**Status:** Quality-gated release candidate — all defined evaluation thresholds pass  
**Scope:** English-only decision-support RAG for AI governance and agentic-security guidance; it is not legal advice.

## Executive summary (ELI10)

Imagine a librarian who is allowed to answer only from a locked shelf of trusted books. The librarian first finds pages using both **meaning** and **exact words**, asks a second specialist to put the best pages first, and then answers only when those pages prove the answer. Every statement must point back to the exact page excerpt. If the shelf does not support an answer, the librarian says “I do not have enough evidence” instead of guessing.

That is this project. We built and evaluated a governed RAG system over frozen NIST and OWASP material. The first version worked but had poor-quality PDF fragments mixed into its search index. We treated that as an engineering incident: measured it, isolated its causes, built a clean R2 index without destroying R1, reran the full 200-scenario evaluation, and promoted R2 only after it passed every gate.

## 1. The problem we designed for

Generative models are useful at explaining complex governance and security material, but a normal chatbot has two unacceptable failure modes for this use case:

1. It can state a plausible but unsupported fact.
2. It can cite a document that is related to the topic but does not actually support the statement.

The system therefore has a more important goal than sounding fluent: **make every factual answer auditable and refuse unsupported answers**. The corpus is frozen and versioned so that a future answer can be reproduced against the same evidence, configuration, model route, and evaluation dataset.

## 2. Delivered scope

The repository contains the complete local implementation and supporting delivery artifacts.

| Area | Delivered capability | Primary location |
|---|---|---|
| Governed API | FastAPI query, ingestion, evaluation, health/readiness, and trace-aware responses | `src/rag_copilot/api.py` |
| Orchestration | Explicit LangGraph workflow with retrieval, reranking, sufficiency, generation, validation, and abstention branches | `src/rag_copilot/graph.py` |
| Corpus ingestion | PDF extraction, text normalization, sentence-safe chunks, provenance metadata, BGE embeddings, and Weaviate indexing | `src/rag_copilot/ingestion.py` |
| Retrieval | Weaviate dense + BM25 hybrid retrieval and source/release filters | `src/rag_copilot/weaviate_adapter.py` |
| Reranking | Cohere option with local `BAAI/bge-reranker-v2-m3` fallback | `src/rag_copilot/retrieval.py` |
| Grounding | Structured claims, exact chunk/span checks, lexical support checks, and abstention | `src/rag_copilot/citations.py` |
| Provider routing | OpenRouter/OmniRoute-compatible adapter with retry and failure handling | `src/rag_copilot/providers.py` |
| Evaluation | 200-case benchmark runner, Ragas judge integration, resumable checkpoints, and CI thresholds | `src/rag_copilot/evaluate.py`, `scripts/run_benchmark.py` |
| User experience | Next.js chat with citations, excerpts, and retrieval diagnostics | `web/` |
| Operations | Docker Compose, prompt/config versioning, GitHub Actions, runbook, and Kanban/CSV backlog | `docker-compose.yml`, `config/`, `.github/workflows/quality.yml`, `docs/operations.md`, `planning/` |
| Leadership collateral | Benchmark PDF and technical-leadership PowerPoint | `artifacts/benchmark/`, `artifacts/slides/` |

## 3. Architecture strategy

```text
Question
  -> normalize and identify source anchors
  -> retrieve in parallel: BGE-M3 semantic search + BM25 keyword search
  -> hybrid fusion in Weaviate (relative-score fusion)
  -> cross-encoder rerank
  -> select at most six evidence chunks
  -> decide whether evidence is sufficient
  -> structured answer with atomic claims, chunk IDs, and quote spans
  -> deterministic citation checks + independent entailment verification
  -> answer with citations OR governed abstention
```

### Architecture decisions and rationale

- **LangGraph rather than an unconstrained agent loop:** the path is a controlled state machine. Every branch is inspectable, retryable, traceable, and testable. A governed RAG response should not depend on an agent deciding to skip validation.
- **Weaviate rather than a vector-only store:** it natively combines dense and BM25 retrieval, supports relative-score fusion, and preserves one operational retrieval service instead of requiring a separate BM25 index and custom merge logic.
- **Hybrid retrieval rather than semantic-only retrieval:** semantic search catches paraphrases such as “model risk controls”; BM25 catches exact normative terms, control IDs, acronyms, and legal/security vocabulary. Governance queries need both.
- **Cross-encoder reranking:** initial retrieval optimizes recall; reranking reads the question and each candidate together to optimize precision. We retrieve 30 candidates and expose only the best six to generation, which reduces irrelevant context and citation drift.
- **Evidence before prose:** citations are not decorative links appended after answer generation. Claims must name retrieved chunk IDs and quote spans. The answer is rejected if an ID is unknown, a span is invalid, a citation was not retrieved, or the claim is unsupported.
- **Release isolation:** every chunk has `corpus_release_id`, source, URL, page, section, parent ID, and child ID. Queries only see the active release. This makes corpus promotion and rollback deliberate operations.
- **Configuration as a first-class artifact:** prompts live in `config/prompts.yaml`; retrieval, chunking, model, and quality thresholds live in `config/rag.yaml`; secrets remain in `.env` and are not committed.

## 4. Corpus and ingestion design

### Frozen evidence set

The corpus is a reviewed, English-only NIST/OWASP release. Its manifest records source URL, local path, provenance, checksums/version data, and corpus release identifier. The data is frozen because changing the evidence changes the meaning of an evaluation score.

### R2 chunk construction

R2 is not a blind “split every N characters” pipeline. The ingestion procedure is:

1. Extract each PDF page while retaining source/page provenance.
2. Normalize Unicode and remove soft hyphens and repeated page headers/footers.
3. Identify sentence boundaries.
4. Accumulate whole sentences into a target of **220 words**.
5. Preserve a **35-word, whole-sentence overlap** between consecutive chunks.
6. Reject index poison: chunks below 35 words, above 360 words, incomplete sentence endings, replacement/private-use characters, and control-template text such as `[select from:]` or `Assessment Objective:`.
7. Store source, URL, page, section, parent ID, child ID, text, and release ID with each vector.

The result is a semantic child chunk that contains a complete thought, not a dangling line from a PDF table. Page and section provenance make its citation reviewable.

### R2 chunk audit

The independent chunk audit recorded:

| Audit check | R2 result |
|---|---:|
| Indexed chunks | 4,782 |
| Minimum / median / P95 / maximum words | 35 / 199 / 220 / 357 |
| Sentence fragments | 0 |
| Unsafe OCR/control text chunks | 0 |
| Chunks with section provenance | 4,782 |
| Audit status | Pass |

The audit is stored in `artifacts/chunking/governance-security-expanded-2026-07-30-r2-audit.json`.

## 5. Strict grounding and abstention policy

The answer contract is intentionally stricter than “the LLM included a URL.” For every factual atomic claim, the generated structure must provide one or more chunk IDs and an exact source quote span. The validator rejects:

- malformed response structures;
- invented or unknown chunk IDs;
- chunks that were not retrieved for that request;
- quote spans outside the cited chunk;
- claims without adequate lexical/evidence support;
- contradictory support; and
- public-answer text that contains factual material outside validated claims.

When validation fails, the workflow gets one controlled revision attempt. If support remains insufficient, it returns a standard abstention with the trace ID and corpus-release ID. This is deliberate behavior, not a generic server failure.

## 6. Benchmark and quality strategy

The benchmark contains **exactly 200 paragraph-based, source-grounded scenarios**:

| Category | Cases | What it checks |
|---|---:|---|
| Direct factual | 110 | Correct recovery and explanation of a supported fact |
| Multi-source/comparative | 45 | Combining evidence without blending or misattributing sources |
| Exact-term/retrieval stress | 25 | Acronyms, normative wording, and near-neighbor passages |
| Unanswerable/adversarial | 20 | Correct refusal when the frozen corpus cannot prove the requested claim |

There are 160 visible development cases and 40 CI-only cases to reduce benchmark overfitting. Each case carries source-level reference material and expected response behavior. The R2 dataset is `data/evaluation/releases/benchmark_200_r2.jsonl`; the human-readable version is `docs/releases/benchmark_200_r2.md`.

### Measured gates

The pipeline fails its build if any of the following are violated:

| Metric | Gate | Why it matters |
|---|---:|---|
| Ragas faithfulness | >= 0.95 | Are answer claims supported by the retrieved context? |
| Citation validity | 1.00 | Is every rendered citation structurally and deterministically valid? |
| Answer correctness | >= 0.85 | Does the answer meet the benchmark reference answer? |
| Abstention F1 | >= 0.90 | Does it refuse unsupported questions without refusing valid ones? |
| Regression | no metric decline > 0.02 | Does a change silently degrade a previously approved release? |

The evaluator is checkpointed. A Gemini free-tier quota interruption does not discard completed scores; the next run resumes from the stored metric/case checkpoint. Judge provider/model are recorded in the results so score comparisons remain meaningful.

## 7. What went wrong in R1, and how we investigated it

We did not call R1 “done” merely because it answered sample questions. The full benchmark exposed real quality problems.

### R1 baseline versus later R1 hardening

| Metric | Early R1 | R1 after red-team fixes | Required gate |
|---|---:|---:|---:|
| Faithfulness | 0.9218 | 0.9405 | 0.95 |
| Citation validity | 1.0000 | 1.0000 | 1.00 |
| Answer correctness | 0.4654 | 0.8087 | 0.85 |
| Abstention F1 | 0.9524 | 0.8511 | 0.90 |

Early operational issues also surfaced: API `500` errors during initial retrieval, Weaviate replacement deletion timing, OpenRouter HTTP 429/503 responses, malformed structured generation, local Ollama outages, and Gemini quota exhaustion. These were treated separately from model quality; an infrastructure outage must never be recorded as a meaningful “the RAG abstained” result.

### Root-cause analysis

The key quality root cause was not “the LLM needs a better prompt.” Raw PDF text produced sentence fragments, running headers/footers, OCR artifacts, and control/template material. Those fragments were semantically weak vectors and BM25 noise. They displaced useful passages in candidate retrieval, increased false abstentions, and made it harder for the answer generator to ground complete claims.

Other contributing causes were:

- exact-source benchmark scenarios needed deterministic source-scoped retrieval rather than relying only on broad corpus similarity;
- provider rate limits and service outages needed explicit governed handling and health checks;
- a configuration/documentation mismatch described 900/450-token chunks while code was operating with a much smaller split policy; and
- an invalid evaluation run mixed an Ollama outage into the response file. It was discarded rather than scored.

## 8. Corrective actions

We used a non-destructive engineering sequence rather than overwriting the original release:

1. Preserved R1 for comparison and rollback.
2. Added PDF normalization and header/footer removal.
3. Rebuilt chunking around complete sentences and full-sentence overlap.
4. Added deterministic unsafe-fragment filters and a chunk audit command.
5. Added section provenance to every R2 child chunk.
6. Added explicit source-anchor paths for benchmark cases where source/title/passage constraints are part of the question.
7. Reserved reranked evidence across requested sources before selecting the final six chunks.
8. Added readiness checks for Ollama before a benchmark starts, so a disconnected embedding service fails fast instead of poisoning results.
9. Added robust OpenRouter response parsing, retry/failure classification, and a governed provider-failure response.
10. Corrected the versioned runtime defaults to R2’s actual chunk settings and promoted R2 as the active corpus release.
11. Added release-scoped deletion with exact typed confirmation and expected object-count protection for safe lifecycle management.

## 9. Final measured result: R2

The valid clean R2 run used 200 cases and the pinned Gemini OpenAI-compatible judge `gemini-3.1-flash-lite`.

| Metric | R2 result | Gate | Outcome |
|---|---:|---:|---|
| Faithfulness | **0.9543** | >= 0.95 | Pass |
| Citation validity | **1.0000** | 1.00 | Pass |
| Answer correctness | **0.8982** | >= 0.85 | Pass |
| Abstention F1 | **0.9302** | >= 0.90 | Pass |

Operationally, the clean R2 run returned 177 answers and 23 abstentions. Twenty abstentions were the intentionally unanswerable/adversarial cases. Three were false abstentions. This produces the reported abstention F1 and is substantially better than the seven false abstentions measured in the hardened R1 run.

The final metrics and diagnostic classification are available at `data/evaluation/runs/full-200-r2-ragas-metrics.json`. The result is a quality-gate pass, not an estimate.

## 10. Release lifecycle and current state

R2 is now the configured active release in `.env`, `config/rag.yaml`, runtime defaults, Docker Compose defaults, and benchmark PDF metadata. R1 was kept while R2 was measured so that improvement claims were evidence-based. After R2 passed every gate, the superseded R1 Weaviate vectors were deleted using a count-checked, release-scoped command. Frozen source files and historical benchmark/metric artifacts remain on disk for auditability; deleting vectors does not erase the historical engineering record.

There are normal production follow-ups, not blockers to the evaluated R2 release:

- connect/import the CSV backlog into Airtable once credentials and base access are supplied;
- run the existing GitHub Actions workflow in the target repository and protect the main branch with the quality-gate result;
- schedule repeat evaluation when the corpus, embedding model, reranker, generator, or judge changes;
- conduct authentication, authorization, secret management, retention, and load testing before exposing the API outside a trusted environment; and
- promote future corpus changes only as new immutable releases followed by the full benchmark.
