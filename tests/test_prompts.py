"""Tests that versioned prompt templates can be safely rendered at runtime."""

from rag_copilot.graph import _render_prompt
from rag_copilot.schemas import ChunkMetadata, EvidenceChunk


def test_grounded_prompt_renders_literal_json_schema() -> None:
    """Ensures JSON examples do not get mistaken for Python format placeholders."""

    evidence = [EvidenceChunk(
        chunk_id="chunk-1",
        text="Approved source excerpt.",
        metadata=ChunkMetadata(
            source_id="source-1",
            title="Source",
            source_url="https://example.org",
            corpus_release_id="release-1",
            parent_chunk_id="parent-1",
        ),
    )]
    rendered = _render_prompt("What does this excerpt say?", evidence)
    assert '"answer":"string"' in rendered
    assert "[chunk-1] Approved source excerpt." in rendered
