"""Three fixed, allow-listed graph queries.

No text-to-Cypher. Every query below is a constant; only the condition name is
passed as a parameter. Every row returned carries full provenance.

Educational retrieval prototype. Not clinical decision support.
"""

import argparse
import json

from src.graph.neo4j_client import Neo4jClient, Neo4jConfigError
from src.graph.triplet_schema import normalise_name

PROVENANCE_RETURN = """
       relation.chunk_id AS chunk_id,
       relation.document_id AS document_id,
       relation.source_filename AS source_filename,
       relation.page_number AS page_number,
       relation.source_url AS source_url,
       relation.evidence_text AS evidence_text,
       relation.confidence AS confidence,
       relation.review_status AS review_status
"""

TREATMENTS_FOR_CONDITION = f"""
MATCH (treatment:Entity)-[relation:TREATS]->(condition:Entity:Condition)
WHERE condition.name_normalized = $condition
RETURN treatment.name AS treatment,
       treatment.entity_type AS treatment_type,
       condition.name AS condition,
{PROVENANCE_RETURN}
ORDER BY treatment, chunk_id
"""

RECOMMENDATIONS_FOR_CONDITION = f"""
MATCH (recommendation:Entity:Recommendation)-[relation:APPLIES_TO]->(condition:Entity:Condition)
WHERE condition.name_normalized = $condition
RETURN recommendation.name AS recommendation,
       condition.name AS condition,
{PROVENANCE_RETURN}
ORDER BY recommendation, chunk_id
"""

TECHNOLOGY_FOR_CONDITION = f"""
MATCH (condition:Entity:Condition)-[relation:USES_TECHNOLOGY]->(technology:Entity:Technology)
WHERE condition.name_normalized = $condition
RETURN technology.name AS technology,
       condition.name AS condition,
{PROVENANCE_RETURN}
ORDER BY technology, chunk_id
"""


def treatments_for_condition(client: Neo4jClient, condition_name: str) -> list[dict]:
    return client.run_read(
        TREATMENTS_FOR_CONDITION, condition=normalise_name(condition_name)
    )


def recommendations_for_condition(client: Neo4jClient, condition_name: str) -> list[dict]:
    return client.run_read(
        RECOMMENDATIONS_FOR_CONDITION, condition=normalise_name(condition_name)
    )


def technology_for_condition(client: Neo4jClient, condition_name: str) -> list[dict]:
    return client.run_read(
        TECHNOLOGY_FOR_CONDITION, condition=normalise_name(condition_name)
    )


QUERIES = {
    "treatments": treatments_for_condition,
    "recommendations": recommendations_for_condition,
    "technology": technology_for_condition,
}


def print_rows(rows: list[dict]) -> None:
    if not rows:
        print("No results. Check the condition name against the reviewed triplets.")
        return

    for index, row in enumerate(rows, start=1):
        headline = row.get("treatment") or row.get("recommendation") or row.get("technology")

        print(f"[{index}] {headline}")
        print(f"    condition   : {row['condition']}")
        print(f"    confidence  : {row['confidence']}  review: {row['review_status']}")
        print(f"    source      : {row['source_filename']}, page {row['page_number']}")
        print(f"    document_id : {row['document_id']}")
        print(f"    chunk_id    : {row['chunk_id']}")
        print(f"    source_url  : {row['source_url'] or '(empty in manifest)'}")
        print(f"    evidence    : {row['evidence_text']}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a fixed, safe graph query.")
    parser.add_argument("query", choices=sorted(QUERIES))
    parser.add_argument("condition", help="Condition name, for example 'type 2 diabetes'")
    parser.add_argument("--json", action="store_true", help="Print raw JSON rows.")

    args = parser.parse_args()

    try:
        with Neo4jClient() as client:
            rows = QUERIES[args.query](client, args.condition)
    except Neo4jConfigError as error:
        print(f"Configuration error: {error}")
        return 2
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}")
        print("Is Neo4j running? Start it with: docker compose up -d")
        return 1

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print(f"\n{args.query} for '{args.condition}': {len(rows)} result(s)\n")
        print_rows(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
