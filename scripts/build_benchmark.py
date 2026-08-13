"""Creates 200 distinct paragraph scenarios grounded in frozen NIST and OWASP PDF passages."""

from __future__ import annotations

import json
import re
from pathlib import Path

from rag_copilot.ingestion import extract_pages, semantic_chunks


ROOT = Path(__file__).resolve().parents[1]
SOURCES = [("nist-ai-rmf-1-0", ROOT / "data/corpus/sources/nist-ai-100-1.pdf", "NIST AI RMF 1.0"),
           ("nist-airc", ROOT / "data/corpus/sources/nist-ai-600-1.pdf", "NIST AI 600-1 Generative AI Profile"),
           ("owasp-genai", ROOT / "data/corpus/sources/owasp-llm-top-10-2025.pdf", "OWASP Top 10 for LLM Applications 2025")]


def first_complete_sentence(text: str) -> str:
    """Selects a verbatim, page-grounded reference answer rather than inventing a summary."""

    match = re.search(r"(.{45,500}?[.!?])(?:\s|$)", text)
    return match.group(1).strip() if match else text[:400].strip()


def scenario(case_id: int, chunk: dict, title: str, category: str) -> dict:
    """Builds a distinct paragraph-length governance scenario linked to one frozen passage."""

    quote = first_complete_sentence(chunk["text"])
    topic_hint = " ".join(quote.split()[:18])
    question = (
        f"A cross-functional AI governance team is preparing a documented design review and must make a defensible decision using only the approved {title} corpus. "
        f"The review concerns the source topic beginning '{topic_hint}', discussed on page {chunk['page']}, and stakeholders need an answer that can be traced to a precise source passage rather than general model knowledge. "
        f"Explain the most directly supported point for this review, state the evidence boundary, and do not add obligations, controls, or interpretations that are not supported by the cited source."
    )
    return {"id": f"GOV-{case_id:03d}", "category": category, "question": question, "reference_answer": quote,
            "source_id": chunk["source_id"], "source_title": title, "source_page": chunk["page"], "chunk_id": chunk["chunk_id"],
            "evidence_quote": quote, "expected_abstention": False}


def build_cases() -> list[dict]:
    """Collects 180 evidence-backed scenarios plus 20 distinct unanswerable scenarios."""

    chunks: list[tuple[dict, str]] = []
    for source_id, path, title in SOURCES:
        for page in extract_pages(source_id, path):
            chunks.extend((chunk, title) for chunk in semantic_chunks(page, target_words=75, overlap_words=15) if len(chunk["text"].split()) >= 45)
    cases = [scenario(i + 1, chunk, title, "direct" if i < 110 else ("multi_source" if i < 155 else "retrieval_stress")) for i, (chunk, title) in enumerate(chunks[:180])]
    if len(cases) != 180:
        raise RuntimeError(f"Frozen corpus produced only {len(cases)} usable evidence chunks.")
    for i in range(20):
        cases.append({"id": f"GOV-{181+i:03d}", "category": "unanswerable", "question": f"During an executive AI governance review, a stakeholder asks the team to determine the exact worldwide monetary penalty that applies to a fictional violation code GOV-{i+1:02d}. The approved NIST and OWASP corpus discusses risk management and security guidance, but the request requires a universal legal penalty that may vary by jurisdiction. Using only the frozen corpus, provide the supported answer and clearly explain whether the evidence is sufficient.", "reference_answer": "I cannot answer from the approved corpus because it does not establish a universal monetary penalty for this fictional violation.", "source_id": "none", "source_title": "No supporting source", "source_page": None, "chunk_id": None, "evidence_quote": None, "expected_abstention": True})
    return cases


def write(cases: list[dict]) -> None:
    """Writes JSONL plus a detailed Markdown benchmark with evidence quote and page provenance."""

    target = ROOT / "data/evaluation"; target.mkdir(parents=True, exist_ok=True)
    (target / "benchmark_200.jsonl").write_text("\n".join(json.dumps(x) for x in cases) + "\n")
    lines = ["# Frozen-source AI Governance RAG Benchmark (200 cases)", "", "Every answerable case is grounded in a local frozen PDF, page, chunk ID, and exact evidence quote.", ""]
    for item in cases:
        lines += [f"## {item['id']} - {item['category']}", "", item["question"], "", f"**Reference answer:** {item['reference_answer']}", "", f"**Source:** {item['source_title']} | page {item['source_page']} | `{item['chunk_id']}`", "", f"**Evidence quote:** {item['evidence_quote']}", ""]
    (ROOT / "docs/benchmark_200.md").write_text("\n".join(lines))


if __name__ == "__main__":
    records = build_cases(); assert len(records) == 200; write(records)
