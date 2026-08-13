"""Tests deterministic grounded-answer controls without requiring model or database services."""

import pytest

from rag_copilot.citations import CitationValidationError, validate_grounded_answer, validate_question_scope
from rag_copilot.schemas import AtomicClaim, ChunkMetadata, ClaimCitation, EvidenceChunk, GroundedAnswer


def _chunk() -> EvidenceChunk:
    """Creates one provenance-complete evidence chunk used by citation tests."""

    return EvidenceChunk(chunk_id="c1", text="The AI RMF is voluntary.", metadata=ChunkMetadata(
        source_id="nist", title="NIST", source_url="https://example.org", corpus_release_id="r1", parent_chunk_id="p1"))


def test_accepts_exact_quote_from_retrieved_chunk() -> None:
    """A claim is released when its citation is exact and belongs to the evidence set."""

    answer = GroundedAnswer(answer="It is voluntary.", abstained=False, claims=[AtomicClaim(text="It is voluntary.", citations=[ClaimCitation(chunk_id="c1", quote="AI RMF is voluntary")])])
    validate_grounded_answer(answer, [_chunk()])


def test_rejects_quote_not_present_in_chunk() -> None:
    """Hallucinated source quotations are rejected before a response can be released."""

    answer = GroundedAnswer(answer="x", abstained=False, claims=[AtomicClaim(text="x", citations=[ClaimCitation(chunk_id="c1", quote="not present")])])
    with pytest.raises(CitationValidationError):
        validate_grounded_answer(answer, [_chunk()])


def test_rejects_claim_that_adds_a_fact_not_in_its_quote() -> None:
    """Prevents an adjacent exact quote from laundering an unsupported model conclusion."""

    answer = GroundedAnswer(answer="It is mandatory.", abstained=False, claims=[AtomicClaim(
        text="It is mandatory.", citations=[ClaimCitation(chunk_id="c1", quote="AI RMF is voluntary")]
    )])
    with pytest.raises(CitationValidationError, match="sufficiently supported"):
        validate_grounded_answer(answer, [_chunk()])


def test_rejects_public_answer_that_adds_an_uncited_claim() -> None:
    """Ensures the displayed answer cannot contain text omitted from the claim contract."""

    answer = GroundedAnswer(answer="It is voluntary and mandatory.", abstained=False, claims=[AtomicClaim(
        text="It is voluntary.", citations=[ClaimCitation(chunk_id="c1", quote="AI RMF is voluntary")]
    )])
    with pytest.raises(CitationValidationError, match="public answer"):
        validate_grounded_answer(answer, [_chunk()])


def test_rejects_nearby_framework_when_ai_scope_is_not_cited() -> None:
    """Prevents general RMF evidence from being presented as evidence for AI RMF guidance."""

    answer = GroundedAnswer(answer="The AI RMF is voluntary.", abstained=False, claims=[AtomicClaim(
        text="The AI RMF is voluntary.", citations=[ClaimCitation(chunk_id="c1", quote="AI RMF is voluntary")])])
    # The cited source names RMF but does not establish the material AI qualifier in the question.
    generic_rmf_evidence = _chunk().model_copy(update={"text": "The RMF is voluntary."})
    with pytest.raises(CitationValidationError, match="AI"):
        validate_question_scope(answer, "How should an organization apply the NIST AI RMF?", [generic_rmf_evidence])
