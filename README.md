# Evidence-Grounded AI Governance Copilot-RAG

An English-only, evidence-grounded RAG system for AI governance and agentic-security guidance. It uses Weaviate hybrid retrieval, reranking, structured citations, and a hard abstention gate when the evidence does not support an answer.

## Quick start

```bash
cp .env.example .env
docker compose up --build
```

The API is available at `http://localhost:8000/docs`; the web UI is at `http://localhost:3000`.

## Quality gate

```bash
python -m rag_copilot.evaluate --dataset data/evaluation/benchmark_200.jsonl
pytest
```

## Expanded corpus

The qualified expanded corpus contains ten official NIST publications, each at least 90 pages (2,677 pages total). Verify the frozen release before indexing it:

```bash
python scripts/validate_expanded_corpus.py
python scripts/ingest_release.py \
  --release-id governance-security-expanded-2026-07-30-r2 \
  --confirm-release governance-security-expanded-2026-07-30-r2
```

See [architecture defaults](docs/architecture-defaults.md), [operations](docs/operations.md), and [the Airtable-importable backlog](planning/airtable_backlog.csv).
