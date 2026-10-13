"""Load document and chunk provenance into Neo4j.

Reads data/processed/document_chunks.json and creates Document and Chunk nodes
with FROM_DOCUMENT and NEXT_CHUNK relationships. All Cypher is parameterised
and uses MERGE, so reruns do not duplicate anything.
"""

import argparse
import json
from itertools import pairwise
from pathlib import Path

from src.graph.neo4j_client import Neo4jClient, Neo4jConfigError
from src.graph.schema import apply_constraints

CHUNKS_PATH = Path("data/processed/document_chunks.json")
BATCH_SIZE = 200

MERGE_DOCUMENTS = """
UNWIND $rows AS row
MERGE (document:Document {document_id: row.document_id})
SET document.title = row.title,
    document.organization = row.organization,
    document.domain = row.domain,
    document.topic = row.topic,
    document.source_url = row.source_url,
    document.source_filename = row.source_filename
RETURN count(document) AS merged
"""

MERGE_CHUNKS = """
UNWIND $rows AS row
MERGE (chunk:Chunk {chunk_id: row.chunk_id})
SET chunk.text = row.text,
    chunk.page_number = row.page_number,
    chunk.chunk_index = row.chunk_index,
    chunk.source_filename = row.source_filename,
    chunk.document_id = row.document_id
WITH chunk, row
MATCH (document:Document {document_id: row.document_id})
MERGE (chunk)-[:FROM_DOCUMENT]->(document)
RETURN count(chunk) AS merged
"""

MERGE_NEXT_CHUNK = """
UNWIND $rows AS row
MATCH (earlier:Chunk {chunk_id: row.from_chunk_id})
MATCH (later:Chunk {chunk_id: row.to_chunk_id})
MERGE (earlier)-[:NEXT_CHUNK]->(later)
RETURN count(*) AS merged
"""


def load_chunks(chunks_path: Path) -> list[dict]:
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"{chunks_path} not found. Run: python -m src.ingestion.batch_ingest"
        )

    with chunks_path.open(encoding="utf-8") as file:
        return json.load(file)


def build_document_rows(chunks: list[dict]) -> list[dict]:
    documents: dict[str, dict] = {}

    for chunk in chunks:
        documents.setdefault(
            chunk["document_id"],
            {
                "document_id": chunk["document_id"],
                "title": chunk["document_title"],
                "organization": chunk["organization"],
                "domain": chunk["domain"],
                "topic": chunk["topic"],
                "source_url": chunk["source_url"],
                "source_filename": chunk["source_filename"],
            },
        )

    return [documents[key] for key in sorted(documents)]


def build_chunk_rows(chunks: list[dict]) -> list[dict]:
    return [
        {
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "text": chunk["text"],
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"],
            "source_filename": chunk["source_filename"],
        }
        for chunk in chunks
    ]


def build_next_chunk_rows(chunks: list[dict]) -> list[dict]:
    """Adjacent chunks within the same document, ordered by page then index."""
    by_document: dict[str, list[dict]] = {}

    for chunk in chunks:
        by_document.setdefault(chunk["document_id"], []).append(chunk)

    rows: list[dict] = []

    for document_id in sorted(by_document):
        ordered = sorted(
            by_document[document_id],
            key=lambda chunk: (chunk["page_number"], chunk["chunk_index"]),
        )

        rows.extend(
            {"from_chunk_id": earlier["chunk_id"], "to_chunk_id": later["chunk_id"]}
            for earlier, later in pairwise(ordered)
        )

    return rows


def run_batched(client: Neo4jClient, query: str, rows: list[dict]) -> int:
    merged = 0

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        result = client.run_write(query, rows=batch)
        merged += result[0]["merged"] if result else 0

    return merged


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load document and chunk provenance into Neo4j."
    )
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH)

    args = parser.parse_args()

    try:
        chunks = load_chunks(args.chunks)
    except FileNotFoundError as error:
        print(f"ERROR: {error}")
        return 2

    document_rows = build_document_rows(chunks)
    chunk_rows = build_chunk_rows(chunks)
    next_chunk_rows = build_next_chunk_rows(chunks)

    print(f"Chunks file          : {args.chunks}")
    print(f"Documents to merge   : {len(document_rows)}")
    print(f"Chunks to merge      : {len(chunk_rows)}")
    print(f"NEXT_CHUNK links     : {len(next_chunk_rows)}")

    try:
        with Neo4jClient() as client:
            apply_constraints(client)
            print("\nConstraints applied.")

            run_batched(client, MERGE_DOCUMENTS, document_rows)
            print(f"Documents merged     : {len(document_rows)}")

            run_batched(client, MERGE_CHUNKS, chunk_rows)
            print(f"Chunks merged        : {len(chunk_rows)}")

            run_batched(client, MERGE_NEXT_CHUNK, next_chunk_rows)
            print(f"NEXT_CHUNK merged    : {len(next_chunk_rows)}")

            counts = client.node_counts()

            print("\nGraph now contains:")

            for key in ("documents", "chunks", "entities", "from_document", "next_chunk"):
                print(f"  {key:15}: {counts.get(key, 0)}")

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
