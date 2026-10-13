"""Run the five cross-document retrieval checks in one pass.

Loads the embedding model once instead of once per query. This reports what
was retrieved and how many distinct documents each query reached. It does NOT
judge correctness: relevance is for a human to review.

Educational retrieval prototype. Not clinical decision support.
"""

import argparse

from src.embeddings.local_embedder import LocalEmbedder
from src.embeddings.local_vector_store import LocalVectorStore
from src.embeddings.search_local_index import (
    ensure_current_metadata_schema,
    format_preview,
)

TEST_QUERIES = (
    "How is cardiovascular risk addressed in adults with type 1 diabetes?",
    "What guidance concerns diabetes management during pregnancy?",
    "What information relates to preventing diabetic foot problems?",
    "What are hybrid closed-loop systems used for?",
    "What type 2 diabetes prevention approaches are described?",
)

PREVIEW_LIMIT = 220


def run_query(vector_store: LocalVectorStore, question: str, top_k: int) -> None:
    results = vector_store.search(query=question, top_k=top_k)

    print("=" * 110)
    print(f"QUERY: {question}")
    print("=" * 110)

    for rank, result in enumerate(results, start=1):
        print(f"  [{rank}] score={result['score']:<8} {result['document_title']}")
        print(
            f"      domain={result['domain']}  topic={result['topic']}  "
            f"file={result['source_filename']}  page={result['page_number']}"
        )
        print(f"      chunk_id={result['chunk_id']}")
        print(f"      {format_preview(result['text'], PREVIEW_LIMIT)}")
        print()

    distinct_documents = {result["document_id"] for result in results}
    print(f"  Distinct documents in top {top_k}: {len(distinct_documents)}")
    print(f"  Documents reached: {', '.join(sorted(distinct_documents))}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the standard cross-document retrieval checks."
    )

    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    embedder = LocalEmbedder()
    vector_store = LocalVectorStore(embedder)

    vector_store.load()
    ensure_current_metadata_schema(vector_store.metadata)

    print(f"Index loaded: {len(vector_store.metadata)} chunks")
    print(
        "Distinct documents indexed: "
        f"{len({item['document_id'] for item in vector_store.metadata})}\n"
    )

    for question in TEST_QUERIES:
        run_query(vector_store, question, args.top_k)

    print("Review these results manually. No accuracy claim is made here.")


if __name__ == "__main__":
    main()
