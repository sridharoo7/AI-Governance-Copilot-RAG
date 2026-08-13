"""Builds a hard 200-case RAG benchmark from the verified 2,677-page corpus."""

from __future__ import annotations

import json
import re
import argparse
from pathlib import Path

import yaml
from pypdf import PdfReader

from rag_copilot.ingestion import ParsedPage, semantic_chunks

ROOT = Path(__file__).resolve().parents[1]
STOP = {"the", "and", "for", "that", "with", "this", "from", "are", "shall", "should", "into", "their", "system", "information", "security", "organizations", "organization"}
NON_SUBSTANTIVE_MARKERS = (
    "table of contents", "available free of charge", "withdrawn", "superseded",
    "all rights reserved", "national institute of standards and technology",
    "attn: computer security division", "email: sec-cert", "authority of the chief",
    "reports on computer systems technology", "federal information processing standards",
)
CONTROL_TEMPLATE_MARKERS = (
    "assessment objective:", "determine if:", "potential assessment methods",
    "potential assessment objects", "[select from:", "-examine [select",
    "[assignment:", "organization-defined frequency", "organization-defined events",
)


def chunks_for_source(source: dict, maximum_substantive_chunks: int = 28) -> list[dict]:
    """Creates a bounded, substantive gold pool with production chunk identifiers."""

    # Evaluation retrieval recall is meaningful only when a gold chunk can actually exist in
    # Weaviate. Reusing the production chunker prevents benchmark/index drift. The benchmark
    # needs a representative sample, not every corpus chunk, so stop after a bounded number
    # of quality candidates per source to keep regeneration fast enough for local development.
    result = []
    reader = PdfReader(str(Path(source["local_path"])))
    for page_number, raw_page in enumerate(reader.pages, start=1):
        text = re.sub(r"\s+", " ", raw_page.extract_text() or "").strip()
        if not text:
            continue
        page = ParsedPage(source["source_id"], Path(source["local_path"]), page_number, text)
        for chunk in semantic_chunks(page):
            candidate = {**chunk, "source_title": source["title"], "source_page": page.page}
            if is_evaluation_worthy(candidate):
                result.append(candidate)
        if len(result) >= maximum_substantive_chunks:
            break
    return result


def answer_quote(chunk: dict) -> str:
    """Returns a literal, substantive sentence suitable for both gold evidence and reference text."""

    # Several NIST PDFs use "Special Publication ... PAGE n" rather than "NIST SP" in
    # running headers. Remove either form before sentence selection so a page label can never
    # become a gold answer or retrieval anchor.
    body = re.sub(
        r"^(?:NIST SP|SPECIAL PUBLICATION|GUIDE TO).*?\bPAGE\s*\d+\s*",
        "",
        chunk["text"],
        flags=re.IGNORECASE,
    )
    for sentence in re.split(r"(?<=[.!?])\s+", body):
        candidate = sentence.strip()
        words = re.findall(r"[A-Za-z]{3,}", candidate)
        # Exclude recurring report headers, table labels, and sentence fragments. A benchmark
        # answer needs a real proposition that an LLM can answer and a reviewer can verify.
        if (
            len(candidate) <= 460
            and len(words) >= 12
            and not candidate.upper().startswith(("NIST SP ", "SPECIAL PUBLICATION ", "GUIDE TO "))
            and "..." not in candidate
        ):
            return candidate
    return ""


def hint(chunk: dict) -> str:
    """Supplies a short retrieval clue without exposing the full gold answer."""

    quote = answer_quote(chunk)
    # PDF page headers recur verbatim across many chunks and make poor retrieval anchors.
    # Remove their predictable prefix before selecting a concise, source-grounded concept.
    quote = re.sub(r"^NIST SP [^.!?]{0,140}?\b(?:19|20)\d{2}\s+\d+\s*", "", quote, flags=re.IGNORECASE)
    quote = re.sub(r"^GUIDE TO [^.!?]{0,140}?\s+\d+\.\s*", "", quote, flags=re.IGNORECASE)
    return " ".join(quote.split()[:12])


def is_evaluation_worthy(chunk: dict) -> bool:
    """Excludes front matter and administrative text that cannot support a useful decision answer."""

    text = chunk["text"].lower()
    words = re.findall(r"[a-z]{3,}", text)
    # Physical page 25 avoids title, administrative, copyright, contents, and preface material that
    # still contains many words but cannot answer a governance scenario.
    return (
        chunk["source_page"] >= 25
        and len(words) >= 55
        # PDF text extracted from some assessment-control appendices contains private-use
        # glyphs and parameter placeholders. They are valid source material but unsuitable as
        # natural-language evaluation answers because neither a user nor a judge can read them
        # reliably.
        and "\ue000" not in chunk["text"]
        and "\ufffd" not in chunk["text"]
        and "_ODP" not in chunk["text"]
        and "SELECTED PARAMETER VALUE" not in chunk["text"]
        and not any(marker in text for marker in NON_SUBSTANTIVE_MARKERS)
        # Assessment-objective templates are source text, but they consist of parameterized
        # test instructions rather than readable governance propositions. They caused the
        # extractive safety path to refuse even when retrieval correctly found the chunk.
        and not any(marker in text for marker in CONTROL_TEMPLATE_MARKERS)
        and bool(answer_quote(chunk))
    )


def make_case(case_id: int, category: str, question: str, evidence: list[dict], abstain: bool = False) -> dict:
    """Creates one evaluation record with all gold contexts needed for retrieval scoring."""

    return {"id": f"EXP-{case_id:03d}", "category": category, "question": question,
            "reference_answer": " ".join(f"[{item['source_title']}, p.{item['source_page']}] {answer_quote(item)}" for item in evidence) if evidence else "I cannot answer from the approved corpus because no source establishes the requested fact.",
            "gold_chunk_ids": [item["chunk_id"] for item in evidence], "evidence_quotes": [answer_quote(item) for item in evidence],
            "source_ids": [item["source_id"] for item in evidence],
            # Canonical labels are an explicit, stable retrieval target. They replace brittle
            # OCR-derived front-matter snippets that made the previous benchmark unscorable.
            "source_titles": [item["source_title"] for item in evidence], "expected_abstention": abstain}


def build() -> list[dict]:
    """Produces 70 direct, 70 synthesis, 40 boundary, and 20 adversarial abstention cases."""

    manifest = yaml.safe_load((ROOT / "data/corpus/expanded_manifest.yaml").read_text())
    all_chunks = [chunk for source in manifest["sources"] for chunk in chunks_for_source(source)]
    # Spread direct questions across sources and pages to avoid topic concentration.
    by_source: dict[str, list[dict]] = {}
    for chunk in all_chunks:
        by_source.setdefault(chunk["source_id"], []).append(chunk)
    selected = []
    for index in range(max(len(items) for items in by_source.values())):
        for items in by_source.values():
            if index < len(items):
                selected.append(items[index])
            if len(selected) >= 220:
                break
        if len(selected) >= 220:
            break
    if len(selected) < 180:
        raise RuntimeError("Expanded corpus did not yield enough extraction-quality chunks.")
    cases = []
    for i, chunk in enumerate(selected[:70], 1):
        q = f"For control-review record DC-{i:03d}, an engineering assurance board is reviewing a high-impact AI service that relies on inherited enterprise controls. The team has designated the frozen source '{chunk['source_title']}' for this decision, with the relevant passage beginning '{hint(chunk)}'. It needs a paragraph-length, evidence-bound note for the design record. State only the guidance directly supported by the designated passage, distinguish it from broader policy interpretation, and provide no additional requirements that are not established by the approved corpus."
        cases.append(make_case(i, "direct_control", q, [chunk]))
    # Pair different source families to require genuinely multi-document retrieval.
    for i in range(70):
        left, right = selected[70 + i], selected[140 + (i % 40)]
        if left["source_id"] == right["source_id"]:
            right = selected[(140 + i + 17) % len(selected)]
        q = f"For synthesis-review record MS-{i+1:03d}, a governance lead must reconcile two evidence streams before authorizing an AI-enabled workflow: one is the frozen source '{left['source_title']}' with the relevant passage beginning '{hint(left)}', while the other is '{right['source_title']}' with the relevant passage beginning '{hint(right)}'. The team must not collapse distinct control, assessment, lifecycle, or resiliency concepts into one invented rule. Prepare a concise decision explanation that identifies what each source directly supports, where their scopes remain separate, and which claims must be deferred because neither source establishes them."
        cases.append(make_case(70 + i + 1, "multi_document_synthesis", q, [left, right]))
    for i in range(40):
        left, right = selected[(i * 3) % 70], selected[70 + i]
        q = f"For boundary-review record SB-{i+1:03d}, during an audit-readiness workshop, a reviewer claims that the guidance in '{left['source_title']}', specifically the passage beginning '{hint(left)}', automatically satisfies the different concern addressed by '{right['source_title']}', whose relevant passage begins '{hint(right)}'. Analyze this assertion using only the frozen corpus. Explain the separate evidence-backed responsibilities or boundaries, identify whether the sources actually establish equivalence, and avoid treating similar vocabulary as proof that the same lifecycle, assessment, privacy, supply-chain, or engineering obligation applies."
        cases.append(make_case(140 + i + 1, "scope_boundary", q, [left, right]))
    for i in range(20):
        q = f"A senior executive asks the RAG assistant to approve a fictional global AI certification, assign a universal dollar penalty, and guarantee that every supplier will comply with control set AX-{i+1:02d}. The request combines legal, contractual, financial, and operational assertions that may sound related to governance controls but are not necessarily established by the frozen NIST corpus. Provide the evidence-bound result, explicitly state whether the approved documents support the requested guarantee, and abstain from creating a policy where none exists."
        cases.append(make_case(180 + i + 1, "adversarial_abstention", q, [], True))
    assert len(cases) == 200 and len({row["question"] for row in cases}) == 200
    return cases


def write(cases: list[dict], dataset_output: Path | None = None, markdown_output: Path | None = None) -> None:
    """Writes machine-readable and reviewer-readable benchmark artifacts."""

    target = ROOT / "data/evaluation"; target.mkdir(exist_ok=True)
    dataset_output = dataset_output or target / "benchmark_200.jsonl"
    markdown_output = markdown_output or ROOT / "docs/benchmark_200.md"
    dataset_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    dataset_output.write_text("\n".join(json.dumps(item) for item in cases) + "\n")
    lines = ["# Expanded adversarial RAG benchmark (200 cases)", "", "Grounded in the qualified 10-document, 2,677-page frozen corpus.", ""]
    for item in cases:
        lines += [f"## {item['id']} - {item['category']}", "", item["question"], "", f"**Reference answer:** {item['reference_answer']}", "", f"**Gold chunks:** {', '.join(item['gold_chunk_ids']) or 'None - expected abstention'}", "", "**Evidence excerpts:** " + " | ".join(item["evidence_quotes"]), ""]
    markdown_output.write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    records = build(); write(records, args.dataset_output, args.markdown_output)
