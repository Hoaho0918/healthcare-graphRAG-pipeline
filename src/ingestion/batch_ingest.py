"""Manifest-driven batch ingestion for the local clinical guideline corpus.

Reads data/manifests/clinical_guidelines.csv, parses every document marked
`downloaded` that exists in data/raw/, and writes:

    data/processed/document_chunks.json   combined chunks (git-ignored)
    data/processed/ingestion_report.csv   metadata and counts only (committed)

No LLM is involved. One failing PDF never aborts the batch.
"""

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.ingestion.manifest_loader import (
    MANIFEST_PATH,
    RAW_DIR,
    ManifestEntry,
    ManifestError,
    load_manifest,
    select_ingestable,
)
from src.ingestion.pdf_parser import parse_pdf_with_stats
from src.models.document_chunk import DocumentChunk

CHUNKS_PATH = Path("data/processed/document_chunks.json")
REPORT_PATH = Path("data/processed/ingestion_report.csv")

HASH_BLOCK_SIZE = 1024 * 1024
ERROR_MESSAGE_LIMIT = 300

STATUS_SUCCESS = "success"
STATUS_EMPTY = "empty"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
STATUS_DUPLICATE = "duplicate"

BLOCKING_STATUSES = (STATUS_FAILED, STATUS_EMPTY)

REPORT_COLUMNS = (
    "document_id",
    "local_filename",
    "parse_status",
    "page_count",
    "chunk_count",
    "pages_without_text",
    "disclaimer_pages",
    "chunks_dropped",
    "file_size_bytes",
    "sha256",
    "duplicate_of",
    "source_url_present",
    "error_message",
    "processed_at",
)


@dataclass
class DocumentReport:
    document_id: str
    local_filename: str
    parse_status: str
    page_count: int = 0
    chunk_count: int = 0
    pages_without_text: int = 0
    disclaimer_pages: int = 0
    chunks_dropped: int = 0
    file_size_bytes: int = 0
    sha256: str = ""
    duplicate_of: str = ""
    source_url_present: str = "false"
    error_message: str = ""
    processed_at: str = ""


@dataclass
class BatchResult:
    chunks: list[DocumentChunk] = field(default_factory=list)
    reports: list[DocumentReport] = field(default_factory=list)
    duplicate_chunk_ids: list[str] = field(default_factory=list)
    duplicate_hash_count: int = 0


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(HASH_BLOCK_SIZE), b""):
            digest.update(block)

    return digest.hexdigest()


def ingest_documents(
    entries: list[ManifestEntry],
    raw_dir: Path,
    timestamp: str,
) -> BatchResult:
    """Parse each entry independently. Errors are recorded, never raised."""
    result = BatchResult()
    hashes_seen: dict[str, str] = {}

    for entry in entries:
        pdf_path = entry.pdf_path(raw_dir)

        report = DocumentReport(
            document_id=entry.document_id,
            local_filename=entry.local_filename,
            parse_status=STATUS_SUCCESS,
            source_url_present="true" if entry.source_url.strip() else "false",
            processed_at=timestamp,
        )

        try:
            report.file_size_bytes = pdf_path.stat().st_size
            report.sha256 = sha256_of_file(pdf_path)
        except Exception as error:
            report.parse_status = STATUS_FAILED
            report.error_message = _format_error(error)
            result.reports.append(report)
            continue

        first_seen = hashes_seen.get(report.sha256)

        if first_seen is not None:
            report.parse_status = STATUS_DUPLICATE
            report.duplicate_of = first_seen
            report.error_message = (
                f"identical file content to {first_seen}; not parsed to avoid duplicate chunks"
            )
            result.duplicate_hash_count += 1
            result.reports.append(report)
            continue

        hashes_seen[report.sha256] = entry.document_id

        try:
            parsed = parse_pdf_with_stats(pdf_path, entry)
        except Exception as error:
            report.parse_status = STATUS_FAILED
            report.error_message = _format_error(error)
            result.reports.append(report)
            continue

        report.page_count = parsed.page_count
        report.chunk_count = len(parsed.chunks)
        report.pages_without_text = parsed.pages_without_text
        report.disclaimer_pages = parsed.disclaimer_pages
        report.chunks_dropped = parsed.chunks_dropped

        if report.page_count == 0:
            report.parse_status = STATUS_EMPTY
            report.error_message = "PDF contains no pages"
        elif report.chunk_count == 0:
            report.parse_status = STATUS_EMPTY
            report.error_message = "no usable text extracted from any page"

        result.chunks.extend(parsed.chunks)
        result.reports.append(report)

    result.duplicate_chunk_ids = _find_duplicate_chunk_ids(result.chunks)

    return result


def _format_error(error: Exception) -> str:
    message = f"{type(error).__name__}: {error}".replace("\n", " ")

    if len(message) > ERROR_MESSAGE_LIMIT:
        message = message[: ERROR_MESSAGE_LIMIT - 3] + "..."

    return message


def _find_duplicate_chunk_ids(chunks: list[DocumentChunk]) -> list[str]:
    counts = Counter(chunk.chunk_id for chunk in chunks)

    return sorted(chunk_id for chunk_id, count in counts.items() if count > 1)


def write_chunks(chunks: list[DocumentChunk], output_path: Path) -> None:
    """Replace the combined output atomically so reruns never duplicate chunks."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            [chunk.model_dump() for chunk in chunks],
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    os.replace(temporary_path, output_path)


def write_report(reports: list[DocumentReport], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=REPORT_COLUMNS)
        writer.writeheader()

        for report in reports:
            writer.writerow({column: asdict(report)[column] for column in REPORT_COLUMNS})


def print_summary(result: BatchResult, chunks_written: bool, output_path: Path) -> None:
    by_status: dict[str, int] = defaultdict(int)

    for report in result.reports:
        by_status[report.parse_status] += 1

    print("\nPer-document results")
    print("-" * 104)
    print(f"{'document_id':42} {'status':10} {'pages':>6} {'chunks':>7} {'dropped':>8}  note")
    print("-" * 104)

    for report in sorted(result.reports, key=lambda item: item.document_id):
        note = report.error_message[:30]
        print(
            f"{report.document_id:42} {report.parse_status:10} "
            f"{report.page_count:6} {report.chunk_count:7} {report.chunks_dropped:8}  {note}"
        )

    print("-" * 104)

    total_pages = sum(report.page_count for report in result.reports)

    print("\nBatch summary")
    print(f"  documents processed      : {len(result.reports)}")

    for status in sorted(by_status):
        print(f"    {status:22} : {by_status[status]}")

    print(f"  pages read               : {total_pages}")
    print(f"  chunks produced          : {len(result.chunks)}")
    print(f"  duplicate file hashes    : {result.duplicate_hash_count}")
    print(f"  duplicate chunk ids      : {len(result.duplicate_chunk_ids)}")

    missing_urls = [
        report.document_id
        for report in result.reports
        if report.source_url_present == "false"
    ]

    if missing_urls:
        print(
            f"\n  WARNING: {len(missing_urls)} document(s) have an empty source_url in the manifest."
        )
        print("  Chunks carry an empty string. Enrich the manifest in a later milestone.")

    if chunks_written:
        print(f"\nWrote combined chunks to {output_path}")
    else:
        print(f"\nDid NOT write {output_path}. Existing file left untouched.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch-ingest manifest-listed clinical guideline PDFs into chunks.",
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write the combined chunk file even if some documents failed.",
    )

    args = parser.parse_args(argv)
    timestamp = utc_timestamp()

    try:
        entries = load_manifest(args.manifest)
    except ManifestError as error:
        print(f"ERROR: {error}")
        return 2

    ingestable, skipped = select_ingestable(entries, args.raw_dir)

    print(f"Manifest rows            : {len(entries)}")
    print(f"Selected for ingestion   : {len(ingestable)}")
    print(f"Skipped before parsing   : {len(skipped)}")

    for skip in skipped:
        print(f"  SKIP {skip.document_id}: {skip.reason}")

    result = ingest_documents(ingestable, args.raw_dir, timestamp)

    result.reports.extend(
        DocumentReport(
            document_id=skip.document_id,
            local_filename=skip.local_filename,
            parse_status=STATUS_SKIPPED,
            error_message=skip.reason,
            processed_at=timestamp,
        )
        for skip in skipped
    )

    write_report(result.reports, args.report)
    print(f"\nWrote ingestion report to {args.report}")

    blocking = [
        report for report in result.reports if report.parse_status in BLOCKING_STATUSES
    ]

    chunks_written = False

    if result.duplicate_chunk_ids:
        print(
            f"\nERROR: {len(result.duplicate_chunk_ids)} duplicate chunk id(s) detected, "
            "for example: " + ", ".join(result.duplicate_chunk_ids[:3])
        )
        print("Refusing to write the combined chunk file.")
    elif not result.chunks:
        print("\nERROR: no chunks produced. Refusing to write the combined chunk file.")
    elif blocking and not args.allow_partial:
        print(
            f"\n{len(blocking)} document(s) failed or produced no text. "
            "Re-run with --allow-partial to write a partial corpus anyway."
        )
    else:
        write_chunks(result.chunks, args.output)
        chunks_written = True

    print_summary(result, chunks_written, args.output)

    problems = bool(result.duplicate_chunk_ids) or bool(blocking) or not result.chunks

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
