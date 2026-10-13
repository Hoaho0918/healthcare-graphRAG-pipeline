"""Extractive, deterministic answer formatter over hybrid retrieval results.

There is no LLM in this repository and none is added here. An answer is built
by quoting retrieved source text verbatim under numbered markers. The only
generated sentences are fixed templates that describe where the evidence came
from; nothing asserts a clinical conclusion, diagnosis or instruction.

Educational retrieval prototype. Not clinical decision support.
"""

from src.answering.models import ABSTENTION_TEXT, Citation, GroundedAnswer
from src.retrieval.models import HybridRetrievalResponse, RetrievalResult

DEFAULT_MAX_CITATIONS = 5
DEFAULT_QUOTE_LIMIT = 400
DEFAULT_MINIMUM_VECTOR_SCORE = 0.0

GRAPH_UNAVAILABLE_WARNING = (
    "Reviewed graph evidence was unavailable, so this answer rests on vector "
    "retrieval only."
)
GRAPH_DISABLED_WARNING = (
    "Graph retrieval was disabled, so this answer rests on vector retrieval only."
)
NO_GRAPH_EVIDENCE_WARNING = (
    "No human-reviewed graph evidence matched this question; every citation is "
    "an unreviewed vector match."
)


def _condense(text: str, limit: int) -> str:
    condensed = " ".join(text.split())

    if len(condensed) <= limit:
        return condensed

    return condensed[:limit].rstrip() + "..."


def _is_usable(result: RetrievalResult, minimum_vector_score: float) -> bool:
    if not result.evidence_text.strip():
        return False

    if result.retrieval_source == "vector" and result.score < minimum_vector_score:
        return False

    return True


def _to_citation(result: RetrievalResult, marker_number: int, quote_limit: int) -> Citation:
    return Citation(
        marker=f"[{marker_number}]",
        rank=result.rank,
        retrieval_source=result.retrieval_source,
        score=result.score,
        score_type=result.score_type,
        document_id=result.document_id,
        document_title=result.document_title,
        source_filename=result.source_filename,
        page_number=result.page_number,
        chunk_id=result.chunk_id,
        source_url=result.source_url,
        evidence_text=_condense(result.evidence_text, quote_limit),
        relationship_type=result.relationship_type,
        review_status=result.review_status,
        confidence=result.confidence,
    )


def _evidence_descriptor(citation: Citation, result: RetrievalResult) -> str:
    if citation.retrieval_source == "graph":
        relation = (
            f"{result.source_entity} -[{citation.relationship_type}]-> {result.target_entity}"
        )

        return (
            f"Reviewed graph evidence: {relation} "
            f"(review: {citation.review_status}, confidence: {citation.confidence})"
        )

    return f"Vector match (cosine similarity {citation.score})"


def _compose_answer_text(
    question: str,
    citations: list[Citation],
    results: list[RetrievalResult],
) -> str:
    lines = [
        f'Evidence retrieved from the indexed clinical guidelines for: "{question}"',
        "",
    ]

    for citation, result in zip(citations, results):
        lines.append(f"{citation.marker} {_evidence_descriptor(citation, result)}")
        lines.append(f'    "{citation.evidence_text}"')
        lines.append(f"    - {citation.provenance_line()}")
        lines.append("")

    lines.append(
        "The passages above are quoted from the cited sources. This layer draws no "
        "clinical conclusion and gives no diagnosis or treatment instruction."
    )

    return "\n".join(lines)


def _collect_warnings(
    retrieval: HybridRetrievalResponse,
    citations: list[Citation],
    skipped: int,
    max_citations: int,
) -> list[str]:
    warnings: list[str] = []

    if retrieval.graph_status == "unavailable":
        warnings.append(GRAPH_UNAVAILABLE_WARNING)

        if retrieval.graph_warning:
            warnings.append(retrieval.graph_warning)
    elif retrieval.graph_status == "disabled":
        warnings.append(GRAPH_DISABLED_WARNING)
    elif not any(c.retrieval_source == "graph" for c in citations) and citations:
        warnings.append(NO_GRAPH_EVIDENCE_WARNING)

    if skipped:
        warnings.append(
            f"{skipped} further retrieved passage(s) were not cited "
            f"(max_citations={max_citations})."
        )

    return warnings


def build_answer(
    question: str,
    retrieval: HybridRetrievalResponse,
    max_citations: int = DEFAULT_MAX_CITATIONS,
    quote_limit: int = DEFAULT_QUOTE_LIMIT,
    minimum_vector_score: float = DEFAULT_MINIMUM_VECTOR_SCORE,
) -> GroundedAnswer:
    """Format retrieval results into a grounded, citation-only answer.

    Results arrive already ranked and deduplicated by HybridRetriever; their
    order is preserved so the output is deterministic.
    """
    usable = [
        result
        for result in retrieval.results
        if _is_usable(result, minimum_vector_score)
    ]

    cited_results = usable[:max_citations]
    skipped = len(usable) - len(cited_results)

    citations = [
        _to_citation(result, number, quote_limit)
        for number, result in enumerate(cited_results, start=1)
    ]

    warnings = _collect_warnings(retrieval, citations, skipped, max_citations)

    if not citations:
        return GroundedAnswer(
            question=question,
            answer_text=ABSTENTION_TEXT,
            citations=[],
            insufficient_evidence=True,
            graph_status=retrieval.graph_status,
            graph_warning=retrieval.graph_warning,
            warnings=warnings,
        )

    return GroundedAnswer(
        question=question,
        answer_text=_compose_answer_text(question, citations, cited_results),
        citations=citations,
        insufficient_evidence=False,
        graph_status=retrieval.graph_status,
        graph_warning=retrieval.graph_warning,
        warnings=warnings,
    )


def print_answer(answer: GroundedAnswer) -> None:
    print()
    print(answer.answer_text)

    if answer.warnings:
        print("\nWarnings:")

        for warning in answer.warnings:
            print(f"  - {warning}")

    print(
        f"\ncitations={answer.citation_count} "
        f"(graph={answer.graph_citations}, vector={answer.vector_citations})  "
        f"insufficient_evidence={answer.insufficient_evidence}  "
        f"graph_status={answer.graph_status}"
    )
    print(f"\n{answer.disclaimer}")
