from src.embeddings.local_embedder import LocalEmbedder
from src.embeddings.local_vector_store import LocalVectorStore, load_chunks


def main() -> None:
    chunks = load_chunks()

    print(f"Loaded {len(chunks)} clinical chunks.")

    embedder = LocalEmbedder()
    vector_store = LocalVectorStore(embedder)

    vector_store.build_index(chunks)
    vector_store.save()

    print("\nLocal vector index saved successfully:")
    print("  data/processed/vector_index.npz")
    print("  data/processed/vector_metadata.json")


if __name__ == "__main__":
    main()
