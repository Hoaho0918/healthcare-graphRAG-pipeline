"""Tests for the evidence-grounded answer layer.

Retrieval responses are constructed directly, so these tests need no database,
no embedding model and no network.
"""

from src.answering.answer_builder import (
    GRAPH_DISABLED_WARNING,
    GRAPH_UNAVAILABLE_WARNING,
    NO_GRAPH_EVIDENCE_WARNING,
    build_answer,
)
from src.answering.models import ABSTENTION_TEXT, DISCLAIMER
from src.retrieval.models import HybridRetrievalResponse, RetrievalResult

QUESTION = "What treats type 2 diabetes?"


def vector_result(rank: int = 1, score: float = 0.61, **overrides) -> RetrievalResult:
    payload = {
        "rank": rank,
        "retrieval_source": "vector",
        "score": score,
        "score_type": "cosine_similarity",
        "document_id": "nice_type2_adults_001",
        "document_title": "Type 2 diabetes in adults: management",
        "topic": "type_2_diabetes_management",
        "domain": "diabetes",
        "source_filename": "type-2-diabetes-in-adults-management.pdf",
        "page_number": 71,
        "chunk_id": "nice_type2_adults_001_p71_c0",
        "source_url": "",
        "evidence_text": "When starting first-line therapy with metformin and other medicines.",
    }
    payload.update(overrides)
    return RetrievalResult(**payload)


def graph_result(rank: int = 1, **overrides) -> RetrievalResult:
    payload = {
        "rank": rank,
        "retrieval_source": "graph",
        "score": 1.0,
        "score_type": "review_confidence",
        "document_id": "nice_tirzepatide_001",
        "document_title": "Tirzepatide for treating type 2 diabetes",
        "topic": "type_2_diabetes_treatment",
        "domain": "diabetes",
        "source_filename": "tirzepatide-for-treating-type-2-diabetes.pdf",
        "page_number": 4,
        "chunk_id": "nice_tirzepatide_001_p4_c0",
        "source_url": "",
        "evidence_text": "Tirzepatide is recommended for treating type 2 diabetes.",
        "relationship_type": "TREATS",
        "review_status": "approved",
        "confidence": "high",
        "source_entity": "Type 2 diabetes",
        "target_entity": "Tirzepatide",
    }
    payload.update(overrides)
    return RetrievalResult(**payload)


def response(results, **overrides) -> HybridRetrievalResponse:
    payload = {
        "query": QUESTION,
        "results": results,
        "graph_status": "ok",
        "graph_warning": None,
        "vector_hits": sum(1 for r in results if r.retrieval_source == "vector"),
        "graph_hits": sum(1 for r in results if r.retrieval_source == "graph"),
    }
    payload.update(overrides)
    return HybridRetrievalResponse(**payload)


# --- vector evidence -------------------------------------------------------


def test_answer_from_vector_evidence() -> None:
    answer = build_answer(QUESTION, response([vector_result()]))

    assert answer.insufficient_evidence is False
    assert answer.citation_count == 1
    assert answer.vector_citations == 1
    assert answer.graph_citations == 0
    assert "first-line therapy with metformin" in answer.answer_text
    assert answer.citations[0].marker == "[1]"


def test_vector_only_answer_warns_that_nothing_was_reviewed() -> None:
    answer = build_answer(QUESTION, response([vector_result()]))

    assert NO_GRAPH_EVIDENCE_WARNING in answer.warnings


def test_vector_citation_has_no_graph_only_fields() -> None:
    citation = build_answer(QUESTION, response([vector_result()])).citations[0]

    assert citation.relationship_type is None
    assert citation.review_status is None
    assert citation.score_type == "cosine_similarity"


# --- graph evidence --------------------------------------------------------


def test_answer_from_reviewed_graph_evidence() -> None:
    answer = build_answer(QUESTION, response([graph_result()]))

    assert answer.insufficient_evidence is False
    assert answer.graph_citations == 1
    assert "Reviewed graph evidence" in answer.answer_text
    assert "Type 2 diabetes -[TREATS]-> Tirzepatide" in answer.answer_text


def test_graph_citation_carries_relationship_and_review_status() -> None:
    citation = build_answer(QUESTION, response([graph_result()])).citations[0]

    assert citation.relationship_type == "TREATS"
    assert citation.review_status == "approved"
    assert citation.confidence == "high"
    assert citation.score_type == "review_confidence"


def test_graph_evidence_answer_has_no_vector_only_warning() -> None:
    answer = build_answer(QUESTION, response([graph_result()]))

    assert NO_GRAPH_EVIDENCE_WARNING not in answer.warnings


# --- mixed evidence --------------------------------------------------------


def test_mixed_evidence_cites_both_sources_in_retrieval_order() -> None:
    answer = build_answer(
        QUESTION, response([graph_result(rank=1), vector_result(rank=2)])
    )

    assert [c.retrieval_source for c in answer.citations] == ["graph", "vector"]
    assert [c.marker for c in answer.citations] == ["[1]", "[2]"]
    assert answer.graph_citations == 1
    assert answer.vector_citations == 1


def test_markers_are_renumbered_independently_of_retrieval_rank() -> None:
    answer = build_answer(
        QUESTION, response([graph_result(rank=4), vector_result(rank=9)])
    )

    assert [c.marker for c in answer.citations] == ["[1]", "[2]"]
    assert [c.rank for c in answer.citations] == [4, 9]


def test_uncited_results_are_reported_not_silently_dropped() -> None:
    results = [graph_result(rank=1)] + [
        vector_result(rank=n, chunk_id=f"chunk_{n}") for n in range(2, 6)
    ]

    answer = build_answer(QUESTION, response(results), max_citations=2)

    assert answer.citation_count == 2
    assert any("3 further retrieved passage(s) were not cited" in w for w in answer.warnings)


# --- citations and provenance ----------------------------------------------


def test_every_citation_preserves_full_provenance() -> None:
    answer = build_answer(QUESTION, response([graph_result(), vector_result(rank=2)]))

    for citation in answer.citations:
        assert citation.document_id
        assert citation.source_filename
        assert citation.chunk_id
        assert citation.page_number >= 1
        assert citation.evidence_text


def test_provenance_line_includes_file_page_chunk_and_source() -> None:
    line = build_answer(QUESTION, response([graph_result()])).citations[0].provenance_line()

    assert "tirzepatide-for-treating-type-2-diabetes.pdf, page 4" in line
    assert "chunk nice_tirzepatide_001_p4_c0" in line
    assert "retrieved by graph" in line
    assert "relation TREATS" in line
    assert "review approved" in line


def test_provenance_appears_in_the_answer_text() -> None:
    text = build_answer(QUESTION, response([vector_result()])).answer_text

    assert "type-2-diabetes-in-adults-management.pdf, page 71" in text
    assert "nice_type2_adults_001_p71_c0" in text


def test_evidence_is_quoted_verbatim_up_to_the_quote_limit() -> None:
    long_text = "Offer metformin. " * 60
    answer = build_answer(
        QUESTION, response([vector_result(evidence_text=long_text)]), quote_limit=50
    )

    quote = answer.citations[0].evidence_text

    assert quote.endswith("...")
    assert len(quote) <= 53
    assert long_text.strip().startswith(quote[:40])


def test_answer_states_it_draws_no_clinical_conclusion() -> None:
    answer = build_answer(QUESTION, response([vector_result()]))

    assert "draws no clinical conclusion" in answer.answer_text
    assert answer.disclaimer == DISCLAIMER


# --- abstention ------------------------------------------------------------


def test_no_results_produces_abstention() -> None:
    answer = build_answer(QUESTION, response([]))

    assert answer.insufficient_evidence is True
    assert answer.answer_text == ABSTENTION_TEXT
    assert answer.citations == []


def test_blank_evidence_text_is_not_usable() -> None:
    answer = build_answer(QUESTION, response([vector_result(evidence_text="   ")]))

    assert answer.insufficient_evidence is True
    assert answer.answer_text == ABSTENTION_TEXT


def test_minimum_vector_score_can_force_abstention() -> None:
    answer = build_answer(
        QUESTION, response([vector_result(score=0.05)]), minimum_vector_score=0.3
    )

    assert answer.insufficient_evidence is True


def test_minimum_vector_score_does_not_filter_graph_evidence() -> None:
    answer = build_answer(
        QUESTION, response([graph_result(score=0.5)]), minimum_vector_score=0.9
    )

    assert answer.insufficient_evidence is False
    assert answer.graph_citations == 1


# --- retrieval status carried through --------------------------------------


def test_neo4j_unavailable_warning_is_carried_into_the_answer() -> None:
    retrieval = response(
        [vector_result()],
        graph_status="unavailable",
        graph_warning="Neo4j unavailable (ServiceUnavailable: refused)",
        graph_hits=0,
    )

    answer = build_answer(QUESTION, retrieval)

    assert answer.graph_status == "unavailable"
    assert answer.graph_warning == "Neo4j unavailable (ServiceUnavailable: refused)"
    assert GRAPH_UNAVAILABLE_WARNING in answer.warnings
    assert "Neo4j unavailable (ServiceUnavailable: refused)" in answer.warnings
    assert answer.insufficient_evidence is False
    assert answer.citation_count == 1


def test_neo4j_unavailable_with_no_vector_hits_still_abstains_cleanly() -> None:
    retrieval = response([], graph_status="unavailable", graph_warning="down")
    answer = build_answer(QUESTION, retrieval)

    assert answer.insufficient_evidence is True
    assert answer.answer_text == ABSTENTION_TEXT
    assert GRAPH_UNAVAILABLE_WARNING in answer.warnings


def test_disabled_graph_is_reported_distinctly_from_unavailable() -> None:
    retrieval = response([vector_result()], graph_status="disabled")
    answer = build_answer(QUESTION, retrieval)

    assert GRAPH_DISABLED_WARNING in answer.warnings
    assert GRAPH_UNAVAILABLE_WARNING not in answer.warnings


# --- determinism -----------------------------------------------------------


def test_repeated_builds_are_byte_identical() -> None:
    retrieval = response([graph_result(rank=1), vector_result(rank=2)])

    first = build_answer(QUESTION, retrieval).model_dump_json()
    second = build_answer(QUESTION, retrieval).model_dump_json()

    assert first == second


def test_citation_order_follows_retrieval_order_exactly() -> None:
    results = [
        graph_result(rank=1, chunk_id="a"),
        graph_result(rank=2, chunk_id="b"),
        vector_result(rank=3, chunk_id="c"),
    ]

    answer = build_answer(QUESTION, response(results))

    assert [c.chunk_id for c in answer.citations] == ["a", "b", "c"]


def test_no_randomness_in_abstention_text() -> None:
    first = build_answer(QUESTION, response([])).answer_text
    second = build_answer(QUESTION, response([])).answer_text

    assert first == second == ABSTENTION_TEXT
