import json
from pathlib import Path

from src.ingestion.pdf_parser import parse_pdf


PDF_PATH = Path("data/raw/sample_guideline.pdf")
OUTPUT_PATH = Path("data/processed/document_chunks.json")


def main() -> None:
    chunks = parse_pdf(PDF_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            [chunk.model_dump() for chunk in chunks],
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Created {len(chunks)} chunks.")
    print(f"Saved output to: {OUTPUT_PATH}")

    if chunks:
        print("\nFirst chunk:")
        print(chunks[0].model_dump_json(indent=2))


if __name__ == "__main__":
    main()
