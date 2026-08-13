"""Tests scenario-to-retrieval query normalization without model or database services."""

from rag_copilot.graph import (
    designated_passage_queries,
    extractive_anchor_fallback,
    normalize_retrieval_query,
    source_ids_named_in_question,
)
from rag_copilot.schemas import ChunkMetadata, EvidenceChunk


def test_uses_quoted_nist_source_concept_for_retrieval() -> None:
    """Prevents long governance scenario instructions from diluting an explicit title query."""

    question = (
        "For control-review record DC-003, provide a decision note for the source concept "
        "beginning 'NIST Special Publication 800-37 Revision 2 Risk Management Framework for'."
    )
    assert normalize_retrieval_query(question) == "NIST Special Publication 800-37 Revision 2 Risk Management Framework for"


def test_preserves_a_normal_question_without_source_anchor() -> None:
    """Keeps ordinary user questions intact when no trusted title anchor is supplied."""

    question = "How should organizations use automation in the RMF?"
    assert normalize_retrieval_query(question) == question


def test_uses_a_meaningful_title_without_a_publisher_prefix() -> None:
    """Keeps valid titles such as SP 800-86 discoverable without a NIST prefix."""

    question = (
        "For a review, use the source concept beginning 'Guide to Integrating Forensic "
        "Techniques into Incident Response Recommendations of'."
    )
    assert normalize_retrieval_query(question) == "Guide to Integrating Forensic Techniques into Incident Response Recommendations of"


def test_rejects_ocr_fragment_as_a_retrieval_anchor() -> None:
    """Prevents spaced individual letters in extracted PDF front matter from hijacking search."""

    question = "Review the source concept beginning 'I N F O R M A T I O N S E C U R I T Y'."
    assert normalize_retrieval_query(question) == question


def test_maps_named_canonical_title_to_source_scope() -> None:
    """Ensures a named approved publication cannot be diluted by the rest of the corpus."""

    question = "Compare 'Risk Management Framework for Information Systems and Organizations' guidance."
    assert source_ids_named_in_question(question) == ["nist-sp-800-37r2"]


def test_pairs_each_designated_passage_with_its_source() -> None:
    """Ensures a comparison cannot retrieve only the more lexically prominent source."""

    question = (
        "one is the frozen source 'Risk Management Framework for Information Systems and Organizations' "
        "with the relevant passage beginning 'Monitor the system and the associated controls on an ongoing basis', "
        "while the other is 'Guide to Integrating Forensic Techniques into Incident Response' "
        "with the relevant passage beginning 'In some cases, there are so many possible data sources'."
    )
    assert designated_passage_queries(question) == [
        ("Monitor the system and the associated controls on an ongoing basis", ["nist-sp-800-37r2"]),
        ("In some cases, there are so many possible data sources", ["nist-sp-800-86"]),
    ]


def test_pairs_boundary_review_passages_with_each_canonical_source() -> None:
    """Supports the different prose shape used by boundary-review scenarios."""

    question = (
        "guidance in 'Risk Management Framework for Information Systems and Organizations', "
        "specifically the passage beginning 'Monitor controls continuously and document changes to the system', "
        "automatically satisfies the concern addressed by 'Guide to Integrating Forensic Techniques into Incident Response', "
        "whose relevant passage begins 'In some cases there are many possible data sources to acquire'."
    )
    assert designated_passage_queries(question) == [
        ("Monitor controls continuously and document changes to the system", ["nist-sp-800-37r2"]),
        ("In some cases there are many possible data sources to acquire", ["nist-sp-800-86"]),
    ]


def test_pairs_two_passages_from_the_same_source() -> None:
    """Keeps two evidence targets distinct even when both are in one publication."""

    question = (
        "guidance in 'Security and Privacy Controls for Information Systems and Organizations', "
        "specifically the passage beginning 'Security controls are safeguards used by organizations', "
        "and the concern addressed by 'Security and Privacy Controls for Information Systems and Organizations', "
        "whose relevant passage begins 'Privacy controls are administrative technical and physical safeguards'."
    )
    assert designated_passage_queries(question) == [
        ("Security controls are safeguards used by organizations", ["nist-sp-800-53r5"]),
        ("Privacy controls are administrative technical and physical safeguards", ["nist-sp-800-53r5"]),
    ]


def test_extractive_fallback_returns_an_exact_anchored_sentence() -> None:
    """Ensures a citation-model format failure can use already retrieved explicit evidence safely."""

    chunk = EvidenceChunk(
        chunk_id="chunk-1",
        text="The designated representative carries out assigned functions but cannot accept risk for the system.",
        metadata=ChunkMetadata(
            source_id="source", title="Source", source_url="https://example.com/source.pdf",
            corpus_release_id="release", parent_chunk_id="parent",
        ),
    )
    answer = extractive_anchor_fallback(
        "Use the passage beginning 'designated representative carries out assigned functions'.", [chunk]
    )
    assert answer is not None
    assert answer.answer.startswith("The designated")
    assert answer.claims[0].citations[0].quote in chunk.text
