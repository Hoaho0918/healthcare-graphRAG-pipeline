"""Load manually reviewed clinical triplets into Neo4j.

Validation runs before anything touches the database: an unsupported label,
relation type or direction aborts the whole load. Node labels and relation
types are interpolated into Cypher only after being checked against the
closed allow-lists in triplet_schema, never from free text.
"""

import argparse
from collections import defaultdict
from pathlib import Path

from src.graph.neo4j_client import Neo4jClient, Neo4jConfigError
from src.graph.schema import apply_constraints
from src.graph.triplet_schema import (
    ALLOWED_DIRECTIONS,
    TRIPLETS_PATH,
    ReviewedTriplet,
    TripletValidationError,
    load_triplets,
)

MERGE_TEMPLATE = """
UNWIND $rows AS row
MERGE (source:Entity {{key: row.source_key}})
  ON CREATE SET source.name = row.source_name,
                source.name_normalized = row.source_name_normalized,
                source.entity_type = row.source_label
SET source:{source_label}
MERGE (target:Entity {{key: row.target_key}})
  ON CREATE SET target.name = row.target_name,
                target.name_normalized = row.target_name_normalized,
                target.entity_type = row.target_label
SET target:{target_label}
MERGE (source)-[relation:{relation} {{chunk_id: row.chunk_id, triplet_id: row.triplet_id}}]->(target)
SET relation.document_id = row.document_id,
    relation.source_filename = row.source_filename,
    relation.page_number = row.page_number,
    relation.source_url = row.source_url,
    relation.evidence_text = row.evidence_text,
    relation.confidence = row.confidence,
    relation.review_status = row.review_status
RETURN count(relation) AS merged
"""

LINK_TO_CHUNK = """
UNWIND $rows AS row
MATCH (source:Entity {key: row.source_key})-[relation]->(target:Entity {key: row.target_key})
WHERE relation.triplet_id = row.triplet_id
MATCH (chunk:Chunk {chunk_id: row.chunk_id})
RETURN count(chunk) AS linked
"""


def triplet_row(triplet: ReviewedTriplet) -> dict:
    return {
        "triplet_id": triplet.triplet_id,
        "source_key": triplet.source.key,
        "source_name": triplet.source.name,
        "source_name_normalized": triplet.source.name_normalized,
        "source_label": triplet.source.label,
        "target_key": triplet.target.key,
        "target_name": triplet.target.name,
        "target_name_normalized": triplet.target.name_normalized,
        "target_label": triplet.target.label,
        "chunk_id": triplet.chunk_id,
        "document_id": triplet.document_id,
        "source_filename": triplet.source_filename,
        "page_number": triplet.page_number,
        "source_url": triplet.source_url,
        "evidence_text": triplet.evidence_text,
        "confidence": triplet.confidence,
        "review_status": triplet.review_status,
    }


def group_by_shape(triplets: list[ReviewedTriplet]) -> dict[tuple[str, str, str], list[dict]]:
    """Group rows by (source label, relation, target label) so each MERGE is fixed."""
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)

    for triplet in triplets:
        shape = (triplet.source.label, triplet.relation, triplet.target.label)

        if shape not in ALLOWED_DIRECTIONS:
            raise TripletValidationError(f"unsupported direction {shape}")

        grouped[shape].append(triplet_row(triplet))

    return grouped


def load_into_graph(client: Neo4jClient, triplets: list[ReviewedTriplet]) -> int:
    merged = 0

    for (source_label, relation, target_label), rows in sorted(group_by_shape(triplets).items()):
        query = MERGE_TEMPLATE.format(
            source_label=source_label,
            relation=relation,
            target_label=target_label,
        )

        result = client.run_write(query, rows=rows)
        count = result[0]["merged"] if result else 0
        merged += count

        print(f"  {source_label} -[{relation}]-> {target_label}: {count}")

    return merged


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load reviewed clinical triplets into the local Neo4j graph."
    )
    parser.add_argument("--triplets", type=Path, default=TRIPLETS_PATH)

    args = parser.parse_args()

    try:
        triplets = load_triplets(args.triplets)
    except TripletValidationError as error:
        print(f"VALIDATION FAILED: {error}")
        print("Nothing was written to the graph.")
        return 2

    print(f"Validated {len(triplets)} reviewed triplets from {args.triplets}\n")

    try:
        with Neo4jClient() as client:
            apply_constraints(client)

            print("Merging relations by shape:")
            merged = load_into_graph(client, triplets)

            linked = client.run_read(
                LINK_TO_CHUNK, rows=[triplet_row(triplet) for triplet in triplets]
            )

            print(f"\nRelations merged     : {merged}")
            print(
                "Triplets whose chunk exists in the graph: "
                f"{linked[0]['linked'] if linked else 0} of {len(triplets)}"
            )

            counts = client.node_counts()
            print(f"Entity nodes         : {counts.get('entities', 0)}")

    except Neo4jConfigError as error:
        print(f"Configuration error: {error}")
        return 2
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}")
        print("Is Neo4j running? Start it with: docker compose up -d")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
