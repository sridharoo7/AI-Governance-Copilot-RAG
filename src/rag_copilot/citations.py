"""Deterministic controls that prevent unsupported citations from reaching users."""

from __future__ import annotations

import re

from .schemas import EvidenceChunk, GroundedAnswer


class CitationValidationError(ValueError):
    """Raised when generated claims do not map precisely to retrieved evidence."""


def validate_grounded_answer(answer: GroundedAnswer, evidence: list[EvidenceChunk]) -> None:
    """Rejects uncited claims, unavailable chunks, and claims not backed by their quotes."""

    if answer.abstained:
        if answer.claims:
            raise CitationValidationError("An abstention must not contain factual claims.")
        return
    if not answer.claims:
        raise CitationValidationError("A non-abstaining answer must contain cited claims.")
    # The lookup is the trust boundary: models may cite only chunks returned by this query.
    by_id = {chunk.chunk_id: chunk for chunk in evidence}
    for claim in answer.claims:
        support_quotes: list[str] = []
        for citation in claim.citations:
            chunk = by_id.get(citation.chunk_id)
            if chunk is None:
                raise CitationValidationError(f"Citation references unavailable chunk {citation.chunk_id}.")
            # Exact substring matching prevents a model from inventing a plausible paraphrase as a quote.
            if citation.quote not in chunk.text:
                raise CitationValidationError(
                    f"Citation quote is not an exact span of chunk {citation.chunk_id}."
                )
            support_quotes.append(citation.quote)
        if not _claim_is_lexically_entailed(claim.text, support_quotes):
            raise CitationValidationError("A cited claim is not sufficiently supported by its quote span.")
    if not _claim_is_lexically_entailed(answer.answer, [claim.text for claim in answer.claims]):
        raise CitationValidationError("The public answer contains content outside its validated claims.")


def _claim_is_lexically_entailed(claim: str, quotes: list[str]) -> bool:
    """Applies a deterministic minimum entailment check before optional model review.

    Exact citation spans alone do not prove a model's adjacent paraphrase.  Requiring every
    substantive claim token to occur in its supporting quote is conservative, transparent,
    and prevents a fluent but unsupported conclusion from escaping when an NLI service is
    unavailable.  Function words are intentionally ignored.
    """

    ignored = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or", "that", "the", "to", "with"}
    claim_terms = {
        token.casefold() for token in re.findall(r"[A-Za-z0-9]{3,}", claim)
        if token.casefold() not in ignored
    }
    quote_terms = {
        token.casefold() for token in re.findall(r"[A-Za-z0-9]{3,}", " ".join(quotes))
        if token.casefold() not in ignored
    }
    return bool(claim_terms) and claim_terms <= quote_terms


def validate_question_scope(answer: GroundedAnswer, question: str, evidence: list[EvidenceChunk]) -> None:
    """Rejects answers whose cited evidence omits material acronym scope from the question."""

    if answer.abstained:
        return
    by_id = {chunk.chunk_id: chunk for chunk in evidence}
    cited_text = " ".join(
        by_id[citation.chunk_id].text
        for claim in answer.claims
        for citation in claim.citations
        if citation.chunk_id in by_id
    )
    # Uppercase acronyms are high-signal scope anchors in governance questions (for example,
    # AI, RMF, NIST, OWASP). Requiring them in cited evidence prevents nearby-framework
    # substitutions such as answering an AI RMF question with a general security RMF passage.
    answer_text = " ".join([answer.answer, *(claim.text for claim in answer.claims)])
    question_acronyms = set(re.findall(r"\b[A-Z][A-Z0-9-]{1,15}\b", question))
    answer_acronyms = set(re.findall(r"\b[A-Z][A-Z0-9-]{1,15}\b", answer_text))
    # Scenario labels can appear only in a question (for example, DC-001). They are not
    # source-scope requirements unless the model repeats them as a factual assertion.
    required_acronyms = question_acronyms & answer_acronyms
    missing = sorted(
        acronym for acronym in required_acronyms if not re.search(rf"\b{re.escape(acronym)}\b", cited_text)
    )
    if missing:
        raise CitationValidationError(
            f"Cited evidence does not support question scope term(s): {', '.join(missing)}."
        )


def evidence_is_sufficient(evidence: list[EvidenceChunk], minimum_score: float = 0.15) -> bool:
    """Applies a conservative retrieval threshold before generation is attempted."""

    return bool(evidence) and evidence[0].rerank_score >= minimum_score
