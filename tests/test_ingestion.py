"""Tests release-idempotency decisions without a live Weaviate service."""

import pytest
from pathlib import Path

from rag_copilot.ingestion import ParsedPage, index_weaviate, normalize_pdf_text, semantic_chunks


class _Aggregate:
    """Supplies only the count interface used by the ingest preflight."""

    def __init__(self, total_count: int) -> None:
        """Stores the synthetic release object count."""

        self.total_count = total_count


class _Collection:
    """Represents a pre-existing release and fails if indexing is attempted unexpectedly."""

    def __init__(self, total_count: int) -> None:
        """Builds aggregate and batch surfaces required by the function under test."""

        self.aggregate = self
        self.total_count = total_count

    def over_all(self, **_kwargs) -> _Aggregate:
        """Reports the configured number of release objects."""

        return _Aggregate(self.total_count)


class _Client:
    """Provides the collection lookup used after collection existence is verified."""

    def __init__(self, total_count: int) -> None:
        """Creates a minimal collection namespace."""

        self.collections = self
        self.collection = _Collection(total_count)

    def exists(self, _name: str) -> bool:
        """Marks the governed collection as already present."""

        return True

    def use(self, _name: str) -> _Collection:
        """Returns the synthetic pre-existing collection."""

        return self.collection


def test_complete_release_is_skipped_without_embedding() -> None:
    """Prevents an accidental rerun from creating a second copy of every chunk."""

    chunks = [{"chunk_id": "one"}, {"chunk_id": "two"}]
    assert index_weaviate(_Client(total_count=2), chunks, "release-1", embedder=None) == 0


def test_partial_or_duplicate_release_requires_explicit_replacement() -> None:
    """Blocks unsafe append behavior whenever the target release count is inconsistent."""

    with pytest.raises(RuntimeError, match="Refusing a non-idempotent ingest"):
        index_weaviate(_Client(total_count=4), [{"chunk_id": "one"}, {"chunk_id": "two"}], "release-1", embedder=None)


def test_r2_chunking_overlaps_only_complete_sentences() -> None:
    """Prevents the raw-word overlap that previously created leading sentence fragments."""

    text = " ".join([
        "Section 1.1 establishes the governance objective for this controlled service.",
        "Organizations document accountable roles before making deployment decisions.",
        "The risk owner reviews evidence before authorizing continued operation.",
        "Independent assessment results are retained to support the final decision.",
        "Continuous monitoring records material changes to the environment of operation.",
        "The organization communicates residual risk to relevant stakeholders.",
        "Documented evidence supports repeatable governance decisions across the lifecycle.",
        "Reviewers preserve the resulting rationale for independent audit and oversight.",
    ])
    page = ParsedPage("source", Path("source.pdf"), 1, text)
    chunks = semantic_chunks(page, target_words=40, overlap_words=8)
    assert len(chunks) >= 2
    assert all(chunk["text"].endswith(".") for chunk in chunks)
    assert any(chunks[1]["text"].startswith(sentence) for sentence in text.split(". "))
    assert all("section" in chunk for chunk in chunks)


def test_normalization_removes_known_layout_boilerplate_without_hiding_ocr() -> None:
    """Keeps source defects observable so they can be rejected before vector indexing."""

    raw = "PAGE 12\nThis publication is available free of charge\nA complete governance statement.\nBad \ue000 glyph."
    normalized = normalize_pdf_text(raw)
    assert "available free of charge" not in normalized
    assert "A complete governance statement." in normalized
    assert "\ue000" in normalized
