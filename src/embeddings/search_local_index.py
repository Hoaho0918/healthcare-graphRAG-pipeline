"""Semantic search over the local clinical guideline vector index.

Educational retrieval prototype. Not clinical decision support.
"""

import argparse

from src.embeddings.local_embedder import LocalEmbedder
from src.embeddings.local_vector_store import LocalVectorStore

PREVIEW_CHARACTER_LIMIT = 300

REQUIRED_RESULT_FIELDS = (
    "chunk_id",
    "document_id",
    "document_title",
    "domain",
    "topic",
    "source_filename",
    "page_number",
    "text",
)


def ensure_current_metadata_schema(metadata: list[dict]) -> None:
    """Fail clearly when the index on disk predates the current chunk schema."""
    if not metadata:
        raise SystemExit(
            "The vector index is empty. Rebuild it with:\n"
            "  python -m src.embeddings.build_local_index"
        )

    missing = [field for field in REQUIRED_RESULT_FIELDS if field not in metadata[0]]

    if missing:
        raise SystemExit(
            "The vector index was built from an older chunk schema and is missing: "
            f"{', '.join(missing)}.\nRebuild it with:\n"
            "  python -m src.ingestion.batch_ingest\n"
            "  python -m src.embeddings.build_local_index"
        )


def format_preview(text: str, limit: int = PREVIEW_CHARACTER_LIMIT) -> str:
    preview = " ".join(text.split())

    if len(preview) <= limit:
        return preview

    return preview[:limit].rstrip() + "..."


def print_results(results: list[dict], preview_limit: int) -> None:
    print("\nTop matching clinical chunks:\n")

    for rank, result in enumerate(results, start=1):
        print(f"Rank            : {rank}")
        print(f"Similarity score: {result['score']}")
        print(f"Document        : {result['document_title']}")
        print(f"Domain          : {result['domain']}")
        print(f"Topic           : {result['topic']}")
        print(f"Source file     : {result['source_filename']}")
        print(f"Page            : {result['page_number']}")
        print(f"Chunk ID        : {result['chunk_id']}")
        print(f"Preview         : {format_preview(result['text'], preview_limit)}")
        print("-" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search the local clinical guideline vector index."
    )

    parser.add_argument(
        "question",
        type=str,
        help="Natural-language clinical question to search for.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return. Default: 5.",
    )

    parser.add_argument(
        "--preview-chars",
        type=int,
        default=PREVIEW_CHARACTER_LIMIT,
        help=f"Characters of chunk text to show. Default: {PREVIEW_CHARACTER_LIMIT}.",
    )

    args = parser.parse_args()

    embedder = LocalEmbedder()
    vector_store = LocalVectorStore(embedder)

    vector_store.load()
    ensure_current_metadata_schema(vector_store.metadata)

    results = vector_store.search(
        query=args.question,
        top_k=args.top_k,
    )

    print(f"\nQuestion: {args.question}")
    print_results(results, args.preview_chars)


if __name__ == "__main__":
    main()
