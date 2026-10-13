"""Extract, clean and chunk text from NICE clinical guideline PDFs.

Cleaning policy (deliberate and conservative):

* Layout boilerplate is removed during cleaning, BEFORE chunking: running
  headers, copyright/notice-of-rights footers, page-number lines and the
  standard NICE "Your responsibility" disclaimer page.
* Clinical wording is never rewritten. Lines are either dropped whole as
  boilerplate or kept verbatim; no paraphrasing or substitution happens
  inside retained sentences.
* `is_useful_chunk` only rejects chunks that are empty, very short, dominated
  by table-of-contents dot leaders, or still mostly boilerplate after
  cleaning. A chunk is never discarded merely because a footer phrase
  survived somewhere inside it.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.ingestion.manifest_loader import ManifestEntry
from src.models.document_chunk import DocumentChunk

CHUNK_SIZE = 1_800
CHUNK_OVERLAP = 250
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

MINIMUM_CHUNK_LENGTH = 80
MINIMUM_CHUNK_WORDS = 10
MAXIMUM_BOILERPLATE_RATIO = 0.5
MAXIMUM_TRAILING_NOISE_LINES = 4
MINIMUM_DISCLAIMER_MARKERS = 3
MINIMUM_WRAPPED_HEADER_LENGTH = 25

NICE_DOCUMENT_CODE = r"(?:NG|TA|PH|CG|QS|IPG|MTG|DG|ESNM|ES|HST)\d+"

# Dropped wherever they appear on a page. Each matches a complete layout line.
BOILERPLATE_LINE_PATTERNS = [
    re.compile(r"^©\s*NICE\s*\d{4}", re.IGNORECASE),
    re.compile(r"all rights reserved", re.IGNORECASE),
    re.compile(r"subject to notice of rights", re.IGNORECASE),
    re.compile(r"conditions#notice-of-rights", re.IGNORECASE),
    re.compile(r"^https?://www\.nice\.org\.uk/terms-and", re.IGNORECASE),
    re.compile(r"^page\s+\d+(\s+of(\s+\d+)?)?$", re.IGNORECASE),
    re.compile(r"^of\s+\d+$", re.IGNORECASE),
]

# Dropped only from the end of a page, where pypdf strands footer fragments.
TRAILING_NOISE_PATTERNS = [
    re.compile(r"^\d{1,4}$"),
    re.compile(rf"^\(?{NICE_DOCUMENT_CODE}\)?[.,]?$", re.IGNORECASE),
]

# Used as a backstop ratio check on chunks that survive cleaning.
RESIDUAL_BOILERPLATE_PATTERNS = [
    re.compile(r"©\s*NICE\s*\d{4}[^.]{0,160}\.", re.IGNORECASE),
    re.compile(r"subject to notice of rights[^.]{0,160}\.?", re.IGNORECASE),
    re.compile(r"all rights reserved\.?", re.IGNORECASE),
    re.compile(r"page \d+ of \d+", re.IGNORECASE),
    re.compile(r"www\.nice\.org\.uk/\S*", re.IGNORECASE),
]

DISCLAIMER_MARKERS = (
    "your responsibility",
    "represent the view of nice",
    "yellow card scheme",
    "unlawful discrimination",
    "it is not mandatory to apply the recommendations",
    "environmental impact",
)

DOT_LEADER_PATTERN = re.compile(r"\.{4,}")
# Only whitespace and separator punctuation. Cannot match letters or digits,
# so clinical wording can never be removed by it.
LEADING_ARTIFACT_PATTERN = re.compile(r"^[\s.,;:]+")
TRAILING_CODE_PATTERN = re.compile(rf"\s*\({NICE_DOCUMENT_CODE}\)\s*$", re.IGNORECASE)


@dataclass
class ParsedDocument:
    """Chunks plus the counters the ingestion report needs."""

    chunks: list[DocumentChunk]
    page_count: int
    pages_without_text: int
    disclaimer_pages: int
    chunks_considered: int
    chunks_dropped: int


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip().lower()


def _is_running_header(line: str, normalised_title: str) -> bool:
    """True when a line is the repeated document title used as a header/footer."""
    if not normalised_title:
        return False

    candidate = TRAILING_CODE_PATTERN.sub("", _normalise(line)).strip()

    if not candidate:
        return False

    if candidate == normalised_title:
        return True

    # Long titles wrap, leaving a leading fragment on its own layout line.
    return (
        len(candidate) >= MINIMUM_WRAPPED_HEADER_LENGTH
        and normalised_title.startswith(candidate)
    )


def clean_page_text(raw_text: str, document_title: str = "") -> str:
    """Strip layout boilerplate from one page, preserving clinical wording."""
    normalised_title = _normalise(document_title)

    kept_lines: list[str] = []

    for raw_line in (raw_text or "").splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if any(pattern.search(line) for pattern in BOILERPLATE_LINE_PATTERNS):
            continue

        if _is_running_header(line, normalised_title):
            continue

        kept_lines.append(line)

    removed = 0

    while kept_lines and removed < MAXIMUM_TRAILING_NOISE_LINES:
        if any(pattern.match(kept_lines[-1]) for pattern in TRAILING_NOISE_PATTERNS):
            kept_lines.pop()
            removed += 1
            continue
        break

    return re.sub(r"\s+", " ", " ".join(kept_lines)).strip()


def clean_text(text: str) -> str:
    """Backwards-compatible wrapper kept for existing callers.

    Prefer `clean_page_text`, which can also strip the repeated document title.
    """
    return clean_page_text(text)


def strip_leading_artifacts(text: str) -> str:
    """Remove leading whitespace and dangling separator punctuation.

    The recursive splitter cuts on ". ", which leaves some chunks starting with
    an orphaned ". ". This is presentation cleanup at the chunk boundary only:
    no character inside the retained wording is altered, and nothing is
    removed from the end of the chunk.
    """
    return LEADING_ARTIFACT_PATTERN.sub("", text).strip()


def is_disclaimer_page(text: str) -> bool:
    """True for the standard NICE 'Your responsibility' page."""
    normalised = _normalise(text)
    matches = sum(1 for marker in DISCLAIMER_MARKERS if marker in normalised)

    return matches >= MINIMUM_DISCLAIMER_MARKERS


def _looks_like_table_of_contents(text: str) -> bool:
    return len(DOT_LEADER_PATTERN.findall(text)) >= 2


def _boilerplate_ratio(text: str) -> float:
    if not text:
        return 0.0

    matched_characters = 0

    for pattern in RESIDUAL_BOILERPLATE_PATTERNS:
        matched_characters += sum(len(match.group(0)) for match in pattern.finditer(text))

    return matched_characters / len(text)


def is_useful_chunk(text: str) -> bool:
    """Conservative retention test applied after cleaning."""
    stripped = text.strip()

    if len(stripped) < MINIMUM_CHUNK_LENGTH:
        return False

    if len(stripped.split()) < MINIMUM_CHUNK_WORDS:
        return False

    if _looks_like_table_of_contents(stripped):
        return False

    if _boilerplate_ratio(stripped) > MAXIMUM_BOILERPLATE_RATIO:
        return False

    return True


def _fallback_entry(pdf_path: Path) -> ManifestEntry:
    """Provenance stand-in for callers that parse a PDF outside the manifest."""
    slug = re.sub(r"[^a-z0-9]+", "_", pdf_path.stem.lower()).strip("_")

    return ManifestEntry(
        document_id=slug or "unknown_document",
        title=pdf_path.stem,
        local_filename=pdf_path.name,
    )


def build_chunk_id(document_id: str, page_number: int, chunk_index: int) -> str:
    return f"{document_id}_p{page_number}_c{chunk_index}"


def parse_pdf_with_stats(
    pdf_path: str | Path,
    document: ManifestEntry | None = None,
) -> ParsedDocument:
    """Parse one PDF into provenance-carrying chunks plus ingestion counters."""
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if document is None:
        document = _fallback_entry(pdf_path)

    reader = PdfReader(str(pdf_path))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
    )

    chunks: list[DocumentChunk] = []
    pages_without_text = 0
    disclaimer_pages = 0
    chunks_considered = 0

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception:
            raw_text = ""

        page_text = clean_page_text(raw_text, document.title)

        if not page_text:
            pages_without_text += 1
            continue

        if is_disclaimer_page(page_text):
            disclaimer_pages += 1
            continue

        for chunk_index, raw_chunk in enumerate(splitter.split_text(page_text)):
            chunks_considered += 1

            # Cleaned before the retention test so length is measured on the
            # text that is actually stored.
            chunk_text = strip_leading_artifacts(raw_chunk)

            if not is_useful_chunk(chunk_text):
                continue

            chunks.append(
                DocumentChunk(
                    chunk_id=build_chunk_id(document.document_id, page_number, chunk_index),
                    document_id=document.document_id,
                    document_title=document.title,
                    organization=document.organization,
                    domain=document.domain,
                    topic=document.topic,
                    source_filename=document.local_filename or pdf_path.name,
                    source_url=document.source_url,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    text=chunk_text,
                )
            )

    return ParsedDocument(
        chunks=chunks,
        page_count=len(reader.pages),
        pages_without_text=pages_without_text,
        disclaimer_pages=disclaimer_pages,
        chunks_considered=chunks_considered,
        chunks_dropped=chunks_considered - len(chunks),
    )


def parse_pdf(
    pdf_path: str | Path,
    document: ManifestEntry | None = None,
) -> list[DocumentChunk]:
    """Parse one PDF into chunks. Signature stays compatible with earlier callers."""
    return parse_pdf_with_stats(pdf_path, document).chunks
