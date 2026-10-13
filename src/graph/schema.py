"""Idempotent graph constraints."""

from src.graph.neo4j_client import Neo4jClient

CONSTRAINTS = (
    (
        "document_id_unique",
        "CREATE CONSTRAINT document_id_unique IF NOT EXISTS "
        "FOR (d:Document) REQUIRE d.document_id IS UNIQUE",
    ),
    (
        "chunk_id_unique",
        "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
        "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
    ),
    (
        "entity_key_unique",
        "CREATE CONSTRAINT entity_key_unique IF NOT EXISTS "
        "FOR (e:Entity) REQUIRE e.key IS UNIQUE",
    ),
)


def apply_constraints(client: Neo4jClient) -> list[str]:
    """Create every constraint. Safe to run repeatedly."""
    applied: list[str] = []

    for name, statement in CONSTRAINTS:
        client.run_write(statement)
        applied.append(name)

    return applied


def main() -> None:
    with Neo4jClient() as client:
        for name in apply_constraints(client):
            print(f"constraint ready: {name}")

    print(f"\n{len(CONSTRAINTS)} constraints applied.")


if __name__ == "__main__":
    main()
