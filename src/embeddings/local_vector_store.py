import json
from pathlib import Path

import numpy as np

from src.embeddings.local_embedder import LocalEmbedder


CHUNKS_PATH = Path("data/processed/document_chunks.json")
INDEX_PATH = Path("data/processed/vector_index.npz")
METADATA_PATH = Path("data/processed/vector_metadata.json")


class LocalVectorStore:
    def __init__(self, embedder: LocalEmbedder) -> None:
        self.embedder = embedder
        self.embeddings: np.ndarray | None = None
        self.metadata: list[dict] = []

    def build_index(self, chunks: list[dict]) -> None:
        texts = [chunk["text"] for chunk in chunks]

        vectors = self.embedder.embed_texts(texts)

        self.embeddings = np.array(vectors, dtype=np.float32)
        self.metadata = chunks

    def save(self) -> None:
        if self.embeddings is None:
            raise ValueError("Build the index before saving it.")

        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            INDEX_PATH,
            embeddings=self.embeddings,
        )

        with METADATA_PATH.open("w", encoding="utf-8") as file:
            json.dump(
                self.metadata,
                file,
                ensure_ascii=False,
                indent=2,
            )

    def load(self) -> None:
        index_data = np.load(INDEX_PATH)

        self.embeddings = index_data["embeddings"]

        with METADATA_PATH.open(encoding="utf-8") as file:
            self.metadata = json.load(file)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.embeddings is None:
            raise ValueError("Load or build the index before searching.")

        query_vector = np.array(
            self.embedder.embed_query(query),
            dtype=np.float32,
        )

        scores = self.embeddings @ query_vector
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[dict] = []

        for index in top_indices:
            result = self.metadata[index].copy()
            result["score"] = round(float(scores[index]), 4)
            results.append(result)

        return results


def load_chunks() -> list[dict]:
    with CHUNKS_PATH.open(encoding="utf-8") as file:
        return json.load(file)
