"""Frozen-PDF parsing, semantic chunking, embedding, and Weaviate indexing pipeline."""

from __future__ import annotations

import hashlib
import re
import time
import uuid
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class ParsedPage:
    """One extracted PDF page with traceable source and page identity."""

    source_id: str
    path: Path
    page: int
    text: str


def sha256(path: Path) -> str:
    """Returns the immutable content hash stored in the approved release manifest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_pages(source_id: str, path: Path) -> list[ParsedPage]:
    """Extracts normalized text per page so citations can always return a PDF page number."""

    pages = []
    for number, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = normalize_pdf_text(page.extract_text() or "")
        if text:
            pages.append(ParsedPage(source_id, path, number, text))
    return pages


def normalize_pdf_text(raw_text: str) -> str:
    """Normalizes PDF extraction noise without inventing source content.

    Frozen PDFs contain recurring line-level page headers and Unicode compatibility forms.
    Removing those deterministic layout artifacts before chunking prevents a heading from being
    embedded as if it were a governance claim.  Replacement and private-use glyphs remain
    visible to downstream quality gates instead of being silently guessed or repaired.
    """

    text = unicodedata.normalize("NFKC", raw_text).replace("\u00ad", "")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    meaningful = [line for line in lines if line and not _is_running_header_or_footer(line)]
    return re.sub(r"\s+", " ", " ".join(meaningful)).strip()


def _is_running_header_or_footer(line: str) -> bool:
    """Identifies repeated publisher/page boilerplate rather than substantive text."""

    normalized = line.casefold()
    return bool(
        re.fullmatch(r"(?:chapter|appendix)\s+[a-z0-9 .-]+\s+page\s+\d+", normalized)
        or re.fullmatch(r"page\s+\d+", normalized)
        or "this publication is available free of charge" in normalized
    )


def semantic_chunks(page: ParsedPage, target_words: int = 220, overlap_words: int = 35) -> list[dict]:
    """Creates r2 semantic child chunks using complete sentences and structural provenance.

    The target is deliberately 220 words: large enough to retain a complete governance rule
    with its qualification, but small enough for dense retrieval and cross-encoder precision.
    Overlap copies trailing *sentences*, never an arbitrary tail of words, preventing a child
    chunk from beginning with a misleading fragment such as "for assessors based on".
    """

    sentences = _sentences(page.text)
    section = _section_label(page.text)
    chunks: list[dict] = []
    current: list[str] = []
    words = 0
    for sentence in sentences:
        count = len(sentence.split())
        if current and words + count > target_words:
            _append_child(chunks, page, current, section)
            current = _trailing_complete_sentences(current, overlap_words)
            words = sum(len(item.split()) for item in current)
        current.append(sentence)
        words += count
    if current and len(" ".join(current).split()) >= 35:
        _append_child(chunks, page, current, section)
    return chunks


def _sentences(text: str) -> list[str]:
    """Splits normalized text into complete sentence units while retaining dense PDF clauses."""

    candidates = re.split(r"(?<=[.!?])\s+(?=(?:[A-Z0-9•]|\())", text)
    # A final PDF extraction tail often ends mid-sentence at a page boundary. It has no
    # trustworthy semantic boundary, so exclude it instead of indexing a fragment that could
    # later become an apparently cited but incomplete public answer.
    return [
        candidate.strip() for candidate in candidates
        if len(candidate.split()) >= 4 and candidate.strip().endswith((".", "!", "?"))
    ]


def _section_label(text: str) -> str | None:
    """Extracts a conservative numeric or all-caps section marker for citation context."""

    match = re.search(r"\b(?:\d+(?:\.\d+){0,3}|[A-Z][A-Z -]{5,})\b", text[:400])
    return match.group(0).strip() if match else None


def _trailing_complete_sentences(sentences: list[str], overlap_words: int) -> list[str]:
    """Retains the smallest whole-sentence suffix meeting the configured overlap budget."""

    selected: list[str] = []
    count = 0
    for sentence in reversed(sentences):
        selected.insert(0, sentence)
        count += len(sentence.split())
        if count >= overlap_words:
            break
    return selected


def _append_child(chunks: list[dict], page: ParsedPage, sentences: list[str], section: str | None) -> None:
    """Adds one provenance-complete child only when it is readable enough for retrieval."""

    text = " ".join(sentences).strip()
    if not _is_indexable_child(text):
        return
    digest = hashlib.sha1(f"{page.source_id}:{page.page}:{text}".encode()).hexdigest()[:12]
    chunks.append({
        "chunk_id": f"{page.source_id}-p{page.page}-{digest}", "source_id": page.source_id,
        "page": page.page, "text": text,
        "parent_chunk_id": f"{page.source_id}-p{page.page}-{section or 'page'}",
        "section": section,
    })


def _is_indexable_child(text: str) -> bool:
    """Rejects OCR/control-template children that would poison semantic and BM25 retrieval."""

    lowered = text.casefold()
    return (
        len(text.split()) >= 35
        and len(text.split()) <= 360
        and text.endswith((".", "!", "?"))
        and "\ue000" not in text and "\ufffd" not in text
        and lowered.count("[select from:") == 0
        and lowered.count("assessment objective:") == 0
    )


def index_weaviate(
    client,
    chunks: list[dict],
    release_id: str,
    embedder,
    embedding_batch_size: int = 48,
    replace_existing_release: bool = False,
) -> int:
    """Embeds a release once, rejecting duplicate or partial release state by default."""

    ensure_collection(client)
    collection = client.collections.use("GovernanceChunk")
    from weaviate.classes.query import Filter

    release_filter = Filter.by_property("corpus_release_id").equal(release_id)
    existing = collection.aggregate.over_all(total_count=True, filters=release_filter).total_count or 0
    if existing:
        if existing == len(chunks) and not replace_existing_release:
            # A matching release is immutable and already indexed; reusing it prevents a
            # rerun from silently biasing retrieval with duplicated source passages.
            return 0
        if not replace_existing_release:
            raise RuntimeError(
                f"Release {release_id!r} has {existing} objects; expected {len(chunks)}. "
                "Refusing a non-idempotent ingest. Review the release, then explicitly replace it."
            )
        # This scoped deletion is intentional and only runs with the explicit replacement
        # flag. It cannot affect another corpus release stored in the same collection.
        _delete_release_in_confirmed_rounds(collection, release_filter, release_id)
    # Keep Ollama requests bounded. Sending the complete multi-thousand-page corpus in one HTTP
    # request would make a local Mac service less stable and complicate failure recovery.
    from weaviate.collections.classes.data import DataObject

    indexed = 0
    for start in range(0, len(chunks), embedding_batch_size):
        chunk_batch = chunks[start : start + embedding_batch_size]
        # The query path uses this same embedding model, maintaining one vector space.
        vectors = embedder.encode([item["text"] for item in chunk_batch])
        objects = [
            DataObject(
                properties={**chunk, "corpus_release_id": release_id},
                # Stable IDs make a transport retry incapable of creating a second copy.
                uuid=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{release_id}:{chunk['chunk_id']}")),
                vector=vector,
            )
            for chunk, vector in zip(chunk_batch, vectors, strict=True)
        ]
        result = collection.data.insert_many(objects)
        if result.has_errors:
            raise RuntimeError(f"Weaviate rejected {len(result.errors)} objects in batch starting at {start}.")
        indexed += len(objects)
        if indexed % (embedding_batch_size * 10) == 0 or indexed == len(chunks):
            # Progress appears in the supervised rebuild log without exposing source text.
            print({"indexed": indexed, "total": len(chunks), "release": release_id}, flush=True)
    return indexed


def _delete_release_in_confirmed_rounds(
    collection, release_filter, release_id: str, timeout_seconds: int = 90
) -> None:
    """Deletes every matching object despite Weaviate's 10,000-object delete-many limit."""

    # Weaviate batch deletion caps one call at 10,000 matches. Count before every round and
    # require a visible decrease before continuing, so a faulty filter cannot spin forever.
    for _round in range(20):
        before = collection.aggregate.over_all(total_count=True, filters=release_filter).total_count or 0
        if before == 0:
            return
        collection.data.delete_many(where=release_filter)
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            remaining = collection.aggregate.over_all(total_count=True, filters=release_filter).total_count or 0
            if remaining == 0:
                return
            if remaining < before:
                break
            time.sleep(0.5)
        else:
            raise RuntimeError(
                f"Timed out waiting for release {release_id!r} deletion to reduce from {before} objects."
            )
    raise RuntimeError(f"Deletion of release {release_id!r} exceeded the permitted 20 rounds.")


def ensure_collection(client) -> None:
    """Creates the explicit-vector collection and provenance fields exactly once."""

    if client.collections.exists("GovernanceChunk"):
        collection = client.collections.use("GovernanceChunk")
        # r1 predates section provenance. Add the non-destructive property in place so r2
        # objects can carry it while r1 objects and their rollback path remain untouched.
        if hasattr(collection, "config"):
            from weaviate.classes.config import DataType, Property

            existing_names = {property.name for property in collection.config.get().properties}
            if "section" not in existing_names:
                collection.config.add_property(
                    Property(name="section", data_type=DataType.TEXT, index_searchable=True)
                )
        return
    from weaviate.classes.config import Configure, DataType, Property

    client.collections.create(
        name="GovernanceChunk",
        vector_config=Configure.Vectors.self_provided(),
        properties=[
            Property(name="chunk_id", data_type=DataType.TEXT, index_filterable=True, index_searchable=True),
            Property(name="text", data_type=DataType.TEXT, index_searchable=True),
            Property(name="source_id", data_type=DataType.TEXT, index_filterable=True, index_searchable=True),
            Property(name="title", data_type=DataType.TEXT, index_searchable=True),
            Property(name="source_url", data_type=DataType.TEXT),
            Property(name="page", data_type=DataType.INT, index_filterable=True),
            Property(name="section", data_type=DataType.TEXT, index_searchable=True),
            Property(name="parent_chunk_id", data_type=DataType.TEXT, index_filterable=True),
            Property(name="corpus_release_id", data_type=DataType.TEXT, index_filterable=True),
        ],
    )
