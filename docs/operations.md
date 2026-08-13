# Operations and corpus-release runbook

1. Register candidate sources with URL, license, version, and ownership.
2. Download into a staging area, compute SHA-256, extract page/section metadata, and run parsing checks.
3. Produce parent/child semantic chunks and index them under a new corpus release ID; never overwrite an approved release.
4. Run the 200-case evaluator and review retrieval, citation, answer, and abstention diagnostics.
5. Approve the release and baseline together; record prompt/config/model versions.
6. On quality regression, classify the failure as ingestion, chunking, retrieval recall, reranking precision, citation validation, generation, or evaluator/config drift. Add a focused regression test before promoting a fix.

The expanded corpus is intentionally evaluated by source family and risk domain. More pages increase retrieval ambiguity and operational cost; they do not independently prove quality. Every expanded release must pass hash/page-count checks, retrieval recall by source family, citation precision, and the full answer/abstention quality gate.

Secrets must remain in environment variables or secret managers. Never log full prompts, source text, credentials, or PII without explicit data-handling approval.
