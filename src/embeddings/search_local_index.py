import argparse

from src.embeddings.local_embedder import LocalEmbedder
from src.embeddings.local_vector_store import LocalVectorStore


def print_results(results: list[dict]) -> None:
    print("\nTop matching clinical chunks:\n")

    for rank, result in enumerate(results, start=1):
        preview = result["text"][:700].replace("\n", " ")

        print(f"Rank: {rank}")
        print(f"Similarity score: {result['score']}")
        print(f"Document: {result['document_title']}")
        print(f"Topic: {result['topic']}")
        print(f"Source: {result['source_filename']}, page {result['page_number']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Text: {preview}...")
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

    args = parser.parse_args()

    embedder = LocalEmbedder()
    vector_store = LocalVectorStore(embedder)

    vector_store.load()

    results = vector_store.search(
        query=args.question,
        top_k=args.top_k,
    )

    print(f"\nQuestion: {args.question}")
    print_results(results)


if __name__ == "__main__":
    main()
