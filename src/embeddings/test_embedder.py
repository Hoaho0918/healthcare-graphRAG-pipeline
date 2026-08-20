from src.embeddings.local_embedder import LocalEmbedder


def main() -> None:
    embedder = LocalEmbedder()

    text = "Metformin may be used to manage type 2 diabetes."
    embedding = embedder.embed_query(text)

    print(f"Embedding dimensions: {len(embedding)}")
    print(f"First five values: {embedding[:5]}")


if __name__ == "__main__":
    main()
