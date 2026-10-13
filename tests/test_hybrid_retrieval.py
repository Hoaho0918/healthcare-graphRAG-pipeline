"""Tests for hybrid vector + graph retrieval.

No database and no embedding model: the vector store and the Neo4j client are
both injected as fakes, so these tests exercise merge, provenance,
deduplication, ordering and fallback deterministically.
"""

import pytest

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.models import RetrievalResult, confidence_score

CHUNK_A = {
    "chunk_id": "nice_type2_adults_001_p71_c0",
    "document_id": "nice_type2_adults_001",
    "document_title": "Type 2 diabetes in adults: management",
    "organization": "NICE",
    "domain": "diabetes",
    "topic": "type_2_diabetes_management",
    "source_filename": "type-2-diabetes-in-adults-management.pdf",
    "source_url": "",
    "page_number": 71,
    "chunk_index": 0,
    "text": "When starting first-line therapy with metformin and other medicines...",
}

CHUNK_B = {
    **CHUNK_A,
    "chunk_id": "nice_type2_adults_001_p36_c0",
    "page_number": 36,
    "text": "cardiovascular risk factors that are more commonly associated...",
}

CHUNK_C = {
    **CHUNK_A,
    "chunk_id": "nice_tirzepatide_001_p4_c0",
    "document_id": "nice_tirzepatide_001",
    "document_title": "Tirzepatide for treating type 2 diabetes",
    "topic": "type_2_diabetes_treatment",
    "source_filename": "tirzepatide-for-treating-type-2-diabetes.pdf",
    "page_number": 4,
    "text": "Tirzepatide is recommended for treating type 2 diabetes...",
}

ALL_CHUNKS = [CHUNK_A, CHUNK_B, CHUNK_C]


class FakeVectorStore:
    """Mimics LocalVectorStore's search contract."""

    def __init__(self, hits: list[dict] | None = None) -> None:
        self.metadata = ALL_CHUNKS
        self._hits = hits if hits is not None else [
            {**CHUNK_A, "score": 0.61},
            {**CHUNK_C, "score": 0.44},
        ]
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        self.calls.append((query, top_k))
        return self._hits[:top_k]


def graph_row(chunk: dict, **overrides) -> dict:
    row = {
        "condition": "Type 2 diabetes",
        "treatment": "Metformin",
        "recommendation": None,
        "technology": None,
        "chunk_id": chunk["chunk_id"],
        "document_id": chunk["document_id"],
        "source_filename": chunk["source_filename"],
        "page_number": chunk["page_number"],
        "source_url": "",
        "evidence_text": "first-line therapy with metformin",
        "confidence": "high",
        "review_status": "approved",
    }
    row.update(overrides)
    return row


class FakeGraphClient:
    """Mimics Neo4jClient for the two read shapes the retriever uses."""

    def __init__(self, conditions=("type 2 diabetes",), rows_by_relation=None) -> None:
        self.conditions = conditions
        self.rows_by_relation = rows_by_relation or {"TREATS": [graph_row(CHUNK_A)]}

    def __call__(self) -> "FakeGraphClient":
        return self

    def __enter__(self) -> "FakeGraphClient":
        return self

    def __exit__(self, *exception_details: object) -> None:
        return None

    def run_read(self, query: str, **parameters: object) -> list[dict]:
        if "Entity:Condition)\nRETURN" in query or "condition.name_normalized AS" in query:
            return [{"name": name.title(), "name_normalized": name} for name in self.conditions]

        for relation, rows in self.rows_by_relation.items():
            if f":{relation}]" in query:
                return rows

        return []


class UnavailableGraphClient:
    def __call__(self) -> "UnavailableGraphClient":
        raise ConnectionRefusedError("Could not connect to bolt://localhost:7687")


def build(vector_store=None, graph_client=None, graph_enabled=True) -> HybridRetriever:
    return HybridRetriever(
        vector_store or FakeVectorStore(),
        graph_client_factory=graph_client or FakeGraphClient(),
        graph_enabled=graph_enabled,
    )


# --- merge -----------------------------------------------------------------


def test_merge_returns_both_sources() -> None:
    response = build().retrieve("treatments for type 2 diabetes", top_k=5)

    sources = {result.retrieval_source for result in response.results}

    assert sources == {"graph", "vector"}
    assert response.graph_status == "ok"
    assert response.vector_hits == 2
    assert response.graph_hits == 1


def test_matched_conditions_are_reported() -> None:
    response = build().retrieve("what treats type 2 diabetes?", top_k=5)

    assert response.matched_conditions == ["type 2 diabetes"]


def test_condition_not_mentioned_yields_no_graph_hits() -> None:
    response = build().retrieve("how is pregnancy care organised?", top_k=5)

    assert response.graph_hits == 0
    assert response.matched_conditions == []
    assert all(result.retrieval_source == "vector" for result in response.results)


def test_vector_store_is_called_with_the_requested_top_k() -> None:
    store = FakeVectorStore()
    build(vector_store=store).retrieve("type 2 diabetes", top_k=1)

    assert store.calls == [("type 2 diabetes", 1)]


# --- provenance ------------------------------------------------------------


def test_graph_results_carry_relationship_and_review_status() -> None:
    response = build().retrieve("type 2 diabetes", top_k=5)
    graph_results = [r for r in response.results if r.retrieval_source == "graph"]

    assert graph_results
    for result in graph_results:
        assert result.relationship_type == "TREATS"
        assert result.review_status == "approved"
        assert result.confidence == "high"
        assert result.target_entity == "Metformin"


def test_vector_results_have_no_graph_only_fields() -> None:
    response = build().retrieve("type 2 diabetes", top_k=5)

    for result in response.results:
        if result.retrieval_source == "vector":
            assert result.relationship_type is None
            assert result.review_status is None


def test_every_result_has_full_source_provenance() -> None:
    response = build().retrieve("type 2 diabetes", top_k=5)

    assert response.results

    for result in response.results:
        assert result.document_id
        assert result.source_filename
        assert result.chunk_id
        assert result.page_number >= 1
        assert result.evidence_text


def test_graph_results_are_enriched_with_document_title_from_chunk_metadata() -> None:
    response = build().retrieve("type 2 diabetes", top_k=5)
    graph_result = next(r for r in response.results if r.retrieval_source == "graph")

    assert graph_result.document_title == "Type 2 diabetes in adults: management"
    assert graph_result.topic == "type_2_diabetes_management"


# --- deduplication ---------------------------------------------------------


def test_chunk_found_by_both_sources_is_merged_into_one_result() -> None:
    response = build().retrieve("type 2 diabetes", top_k=5)

    chunk_ids = [result.chunk_id for result in response.results]

    assert chunk_ids.count(CHUNK_A["chunk_id"]) == 1
    assert response.duplicates_merged == 1


def test_merged_result_keeps_the_richer_graph_provenance_and_the_vector_score() -> None:
    response = build().retrieve("type 2 diabetes", top_k=5)
    merged = next(r for r in response.results if r.chunk_id == CHUNK_A["chunk_id"])

    assert merged.retrieval_source == "graph"
    assert merged.relationship_type == "TREATS"
    assert merged.vector_score == 0.61
    assert merged.also_found_by == ["vector"]


def test_identical_graph_rows_collapse() -> None:
    client = FakeGraphClient(rows_by_relation={"TREATS": [graph_row(CHUNK_A), graph_row(CHUNK_A)]})
    response = build(graph_client=client).retrieve("type 2 diabetes", top_k=5)

    assert len([r for r in response.results if r.chunk_id == CHUNK_A["chunk_id"]]) == 1


def test_same_chunk_with_different_relations_is_kept_separately() -> None:
    client = FakeGraphClient(
        rows_by_relation={
            "TREATS": [graph_row(CHUNK_A)],
            "USES_TECHNOLOGY": [graph_row(CHUNK_A, technology="Insulin pump")],
        }
    )
    response = build(graph_client=client).retrieve("type 2 diabetes", top_k=5)

    relations = {
        result.relationship_type
        for result in response.results
        if result.chunk_id == CHUNK_A["chunk_id"]
    }

    assert relations == {"TREATS", "USES_TECHNOLOGY"}


def test_vector_only_chunks_survive_deduplication() -> None:
    response = build().retrieve("type 2 diabetes", top_k=5)

    assert CHUNK_C["chunk_id"] in {result.chunk_id for result in response.results}


# --- deterministic ordering ------------------------------------------------


def test_graph_results_rank_above_vector_results() -> None:
    response = build().retrieve("type 2 diabetes", top_k=5)
    sources = [result.retrieval_source for result in response.results]

    assert sources == sorted(sources, key=lambda source: 0 if source == "graph" else 1)


def test_ranks_are_sequential_from_one() -> None:
    response = build().retrieve("type 2 diabetes", top_k=5)

    assert [result.rank for result in response.results] == list(
        range(1, len(response.results) + 1)
    )


def test_vector_results_are_ordered_by_descending_score() -> None:
    response = build().retrieve("type 2 diabetes", top_k=5)
    scores = [r.score for r in response.results if r.retrieval_source == "vector"]

    assert scores == sorted(scores, reverse=True)


def test_repeated_retrieval_is_byte_identical() -> None:
    retriever = build()

    first = retriever.retrieve("type 2 diabetes", top_k=5).model_dump_json()
    second = retriever.retrieve("type 2 diabetes", top_k=5).model_dump_json()

    assert first == second


def test_equal_scores_break_ties_deterministically_by_chunk_id() -> None:
    store = FakeVectorStore(
        hits=[{**CHUNK_C, "score": 0.5}, {**CHUNK_B, "score": 0.5}]
    )
    client = FakeGraphClient(conditions=())

    ordered = build(vector_store=store, graph_client=client).retrieve("x", top_k=5)

    assert [r.chunk_id for r in ordered.results] == sorted(
        [CHUNK_B["chunk_id"], CHUNK_C["chunk_id"]]
    )


def test_confidence_scores_are_ordered_high_medium_low() -> None:
    assert confidence_score("high") > confidence_score("medium") > confidence_score("low")
    assert confidence_score(None) < confidence_score("low")


# --- Neo4j fallback --------------------------------------------------------


def test_unavailable_neo4j_returns_vector_results_with_a_warning() -> None:
    response = build(graph_client=UnavailableGraphClient()).retrieve(
        "type 2 diabetes", top_k=5
    )

    assert response.graph_status == "unavailable"
    assert "Neo4j unavailable" in response.graph_warning
    assert response.graph_hits == 0
    assert len(response.results) == 2
    assert all(result.retrieval_source == "vector" for result in response.results)


def test_unavailable_neo4j_does_not_raise() -> None:
    retriever = build(graph_client=UnavailableGraphClient())

    retriever.retrieve("anything at all", top_k=3)


def test_graph_can_be_disabled_explicitly() -> None:
    response = build(graph_enabled=False).retrieve("type 2 diabetes", top_k=5)

    assert response.graph_status == "disabled"
    assert response.graph_hits == 0
    assert response.graph_available is False


def test_empty_vector_hits_and_unavailable_graph_returns_empty_not_error() -> None:
    response = build(
        vector_store=FakeVectorStore(hits=[]), graph_client=UnavailableGraphClient()
    ).retrieve("type 2 diabetes", top_k=5)

    assert response.results == []
    assert response.graph_status == "unavailable"


# --- schema ----------------------------------------------------------------


def test_result_rejects_an_unknown_retrieval_source() -> None:
    with pytest.raises(ValueError):
        RetrievalResult(
            rank=1,
            retrieval_source="semantic",
            score=0.5,
            score_type="cosine_similarity",
            document_id="d",
            source_filename="f.pdf",
            page_number=1,
            chunk_id="c",
            evidence_text="t",
        )
