import re
from pathlib import Path
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.models.document_chunk import DocumentChunk


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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

        for chunk_index, text in enumerate(page_chunks):
            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid4()),
                    source_file=pdf_path.name,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    text=text,
                )
            )

    return chunks
