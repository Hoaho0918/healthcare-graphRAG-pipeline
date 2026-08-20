import re
from pathlib import Path
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.models.document_chunk import DocumentChunk


MINIMUM_CHUNK_LENGTH = 200

UNWANTED_PHRASES = [
    "© nice",
    "all rights reserved",
    "subject to notice of rights",
    "page ",
    "contents",
    "your responsibility",
]


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)

    text = re.sub(
        r"Type 2 diabetes in adults: management \(NG28\)",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"© NICE \d{4}\. All rights reserved\.?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"Page \d+ of \d+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


def is_useful_chunk(text: str) -> bool:
    normalized_text = text.lower().strip()

    if len(normalized_text) < MINIMUM_CHUNK_LENGTH:
        return False

    if any(phrase in normalized_text for phrase in UNWANTED_PHRASES):
        return False

    dotted_leader_count = normalized_text.count("...")

    if dotted_leader_count >= 3:
        return False

    return True


def parse_pdf(pdf_path: str | Path) -> list[DocumentChunk]:
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1_800,
        chunk_overlap=250,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[DocumentChunk] = []

    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        page_text = clean_text(raw_text)

        if not page_text:
            continue

        page_chunks = splitter.split_text(page_text)

        for chunk_index, chunk_text in enumerate(page_chunks):
            if not is_useful_chunk(chunk_text):
                continue

            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid4()),
                    source_file=pdf_path.name,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    text=chunk_text,
                )
            )

    return chunks
