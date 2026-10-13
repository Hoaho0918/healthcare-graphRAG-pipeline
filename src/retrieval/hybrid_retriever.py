"""Hybrid retrieval: existing local vector search merged with reviewed graph evidence.

Reuses `LocalVectorStore` as-is. No second vector store, no LLM, no answer
generation, no text-to-Cypher: graph hits come only from the three fixed,
allow-listed queries in `src.graph.graph_queries`, triggered by matching known
Condition names against the query string.

If Neo4j is unreachable the vector results are returned normally with
`graph_status="unavailable"` and a warning. Nothing raises.

Educational retrieval prototype. Not clinical decision support.
"""

import argparse
import json

from src.embeddings.local_embedder import LocalEmbedder
from src.embeddings.local_vector_store import LocalVectorStore
from src.embeddings.search_local_index import ensure_current_metadata_schema
from src.graph.graph_queries import (
    recommendations_for_condition,
    technology_for_condition,
    treatments_for_condition,
)
from src.graph.neo4j_client import Neo4jClient
from src.graph.triplet_schema import normalise_name
from src.retrieval.models import (
    HybridRetrievalResponse,
    RetrievalResult,
    confidence_score,
)

LIST_CONDITIONS = """
MATCH (condition:Entity:Condition)
RETURN condition.name AS name, condition.name_normalized AS name_normalized
ORDER BY name_normalized
"""

# Each fixed query, with the relation it traverses and the column holding the
# entity on the far side. Nothing here is derived from user input.
GRAPH_QUERIES = (
    ("TREATS", treatments_for_condition, "treatment"),
    ("APPLIES_TO", recommendations_for_condition, "recommendation"),
    ("USES_TECHNOLOGY", technology_for_condition, "technology"),
)


class HybridRetriever:
    """Merges vector chunk hits and evidence-backed graph hits.

    `vector_store` must expose `.search(query, top_k)` and `.metadata`.
    `graph_client_factory` is injectable so tests need no database.
    """

    def __init__(
        self,
        vector_store: LocalVectorStore,
        graph_client_factory=Neo4jClient.fast_fail,
        graph_enabled: bool = True,
    ) -> None:
        self.vector_store = vector_store
        self.graph_client_factory = graph_client_factory
        self.graph_enabled = graph_enabled
        self.chunks_by_id = {
            chunk["chunk_id"]: chunk for chunk in (vector_store.metadata or [])
        }

    def retrieve(self, query: str, top_k: int = 5) -> HybridRetrievalResponse:
        vector_results = self._vector_results(query, top_k)
        graph_results, status, warning, matched = self._graph_results(query)

        merged, duplicates = self._merge(vector_results, graph_results)

        return HybridRetrievalResponse(
            query=query,
            results=self._rank(merged),
            graph_status=status,
            graph_warning=warning,
            vector_hits=len(vector_results),
            graph_hits=len(graph_results),
            duplicates_merged=duplicates,
            matched_conditions=matched,
        )

    def _vector_results(self, query: str, top_k: int) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                rank=1,  # replaced by _rank
                retrieval_source="vector",
                score=round(float(hit["score"]), 4),
                score_type="cosine_similarity",
                document_id=hit["document_id"],
                document_title=hit.get("document_title", ""),
                topic=hit.get("topic", ""),
                domain=hit.get("domain", ""),
                source_filename=hit["source_filename"],
                page_number=hit["page_number"],
                chunk_id=hit["chunk_id"],
                source_url=hit.get("source_url", ""),
                evidence_text=hit["text"],
            )
            for hit in self.vector_store.search(query=query, top_k=top_k)
        ]

    def _graph_results(
        self, query: str
    ) -> tuple[list[RetrievalResult], str, str | None, list[str]]:
        if not self.graph_enabled:
            return [], "disabled", "Graph retrieval disabled by caller.", []

        try:
            with self.graph_client_factory() as client:
                matched = self._matching_conditions(client, query)
                rows = self._collect_graph_rows(client, matched)
        except Exception as error:
            return (
                [],
                "unavailable",
                (
                    f"Neo4j unavailable ({type(error).__name__}: {error}). "
                    "Returning vector results only. Start it with: docker compose up -d"
                ),
                [],
            )

        return [self._graph_result(row) for row in rows], "ok", None, matched

    def _matching_conditions(self, client, query: str) -> list[str]:
        """Condition names that literally appear in the query. No Cypher generation."""
        normalised_query = normalise_name(query)

        return [
            row["name_normalized"]
            for row in client.run_read(LIST_CONDITIONS)
            if row["name_normalized"] and row["name_normalized"] in normalised_query
        ]

    def _collect_graph_rows(self, client, conditions: list[str]) -> list[dict]:
        rows: list[dict] = []

        for condition in conditions:
            for relationship_type, query_function, entity_column in GRAPH_QUERIES:
                for row in query_function(client, condition):
                    rows.append(
                        {
                            **row,
                            "relationship_type": relationship_type,
                            "target_entity": row.get(entity_column),
                        }
                    )

        return rows

    def _graph_result(self, row: dict) -> RetrievalResult:
        chunk = self.chunks_by_id.get(row["chunk_id"], {})

        return RetrievalResult(
            rank=1,  # replaced by _rank
            retrieval_source="graph",
            score=confidence_score(row.get("confidence")),
            score_type="review_confidence",
            document_id=row["document_id"],
            document_title=chunk.get("document_title", ""),
            topic=chunk.get("topic", ""),
            domain=chunk.get("domain", ""),
            source_filename=row["source_filename"],
            page_number=row["page_number"],
            chunk_id=row["chunk_id"],
            source_url=row.get("source_url", "") or "",
            evidence_text=row["evidence_text"],
            relationship_type=row["relationship_type"],
            review_status=row.get("review_status"),
            confidence=row.get("confidence"),
            source_entity=row.get("condition"),
            target_entity=row.get("target_entity"),
        )

    @staticmethod
    def _merge(
        vector_results: list[RetrievalResult],
        graph_results: list[RetrievalResult],
    ) -> tuple[list[RetrievalResult], int]:
        """Collapse equivalent evidence, keeping the richest provenance.

        Graph rows win over vector rows for the same chunk, because they carry
        relationship_type, review_status and confidence. The vector score is
        preserved on the surviving row rather than discarded.
        """
        graph_by_key: dict[tuple[str, str | None], RetrievalResult] = {}
        duplicates = 0

        for result in graph_results:
            key = (result.chunk_id, result.relationship_type)

            if key in graph_by_key:
                duplicates += 1
                continue

            graph_by_key[key] = result

        vector_scores = {result.chunk_id: result.score for result in vector_results}
        graph_chunk_ids = {result.chunk_id for result in graph_by_key.values()}

        for result in graph_by_key.values():
            if result.chunk_id in vector_scores:
                result.vector_score = vector_scores[result.chunk_id]
                result.also_found_by = ["vector"]

        surviving_vector: list[RetrievalResult] = []

        for result in vector_results:
            if result.chunk_id in graph_chunk_ids:
                duplicates += 1
                continue

            surviving_vector.append(result)

        return list(graph_by_key.values()) + surviving_vector, duplicates

    @staticmethod
    def _rank(results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Deterministic order: reviewed graph evidence first, then vector by score.

        Every sort key is total, so equal scores never reorder between runs.
        """
        ordered = sorted(
            results,
            key=lambda result: (
                0 if result.retrieval_source == "graph" else 1,
                -result.score,
                result.chunk_id,
                result.relationship_type or "",
                result.target_entity or "",
            ),
        )

        for rank, result in enumerate(ordered, start=1):
            result.rank = rank

        return ordered


def build_default_retriever(graph_enabled: bool = True) -> HybridRetriever:
    vector_store = LocalVectorStore(LocalEmbedder())
    vector_store.load()
    ensure_current_metadata_schema(vector_store.metadata)

    return HybridRetriever(vector_store, graph_enabled=graph_enabled)


def print_response(response: HybridRetrievalResponse, preview: int = 260) -> None:
    print(f"\nQuery: {response.query}")
    print(
        f"vector hits={response.vector_hits}  graph hits={response.graph_hits}  "
        f"duplicates merged={response.duplicates_merged}  graph={response.graph_status}"
    )

    if response.matched_conditions:
        print(f"conditions matched in query: {', '.join(response.matched_conditions)}")

    if response.graph_warning:
        print(f"\nWARNING: {response.graph_warning}")

    print()

    for result in response.results:
        text = " ".join(result.evidence_text.split())
        text = text if len(text) <= preview else text[:preview].rstrip() + "..."

        print(f"[{result.rank}] source={result.retrieval_source}  "
              f"{result.score_type}={result.score}")

        if result.retrieval_source == "graph":
            print(
                f"    relation    : {result.source_entity} -[{result.relationship_type}]-> "
                f"{result.target_entity}"
            )
            print(f"    review      : {result.review_status}  confidence: {result.confidence}")

            if result.vector_score is not None:
                print(f"    also found by vector search, cosine={result.vector_score}")

        print(f"    document    : {result.document_title or result.document_id}")
        print(f"    source      : {result.source_filename}, page {result.page_number}")
        print(f"    chunk_id    : {result.chunk_id}")
        print(f"    evidence    : {text}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hybrid vector + reviewed-graph retrieval over the local corpus."
    )
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-graph", action="store_true", help="Vector retrieval only.")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()

    retriever = build_default_retriever(graph_enabled=not args.no_graph)
    response = retriever.retrieve(args.question, top_k=args.top_k)

    if args.json:
        print(response.model_dump_json(indent=2))
    else:
        print_response(response)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
