"""Load and validate the clinical guideline CSV manifest.

The manifest is the single source of truth for which documents enter the
pipeline and what provenance each chunk carries. Nothing here touches PDFs.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

MANIFEST_PATH = Path("data/manifests/clinical_guidelines.csv")
RAW_DIR = Path("data/raw")

INGESTABLE_STATUS = "downloaded"

REQUIRED_COLUMNS = (
    "document_id",
    "title",
    "organization",
    "domain",
    "topic",
    "publication_year",
    "version",
    "local_filename",
    "source_url",
    "download_url",
    "approved_for_download",
    "format",
    "status",
    "license_notes",
    "retrieved_at",
)


class ManifestError(RuntimeError):
    """Raised when the manifest is structurally unusable."""


class ManifestEntry(BaseModel):
    document_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    organization: str = ""
    domain: str = ""
    topic: str = ""
    publication_year: str = ""
    version: str = ""
    local_filename: str = Field(..., min_length=1)
    source_url: str = ""
    download_url: str = ""
    approved_for_download: str = ""
    format: str = ""
    status: str = ""
    license_notes: str = ""
    retrieved_at: str = ""

    def pdf_path(self, raw_dir: Path = RAW_DIR) -> Path:
        return Path(raw_dir) / self.local_filename

    def is_marked_downloaded(self) -> bool:
        return self.status.strip().lower() == INGESTABLE_STATUS


@dataclass(frozen=True)
class SkippedDocument:
    document_id: str
    local_filename: str
    reason: str


def load_manifest(manifest_path: str | Path = MANIFEST_PATH) -> list[ManifestEntry]:
    """Read the manifest into validated entries, sorted by document_id.

    Sorting makes every downstream artefact deterministic regardless of the
    row order in the CSV.
    """
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise ManifestError(f"Manifest not found: {manifest_path}")

    with manifest_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []

        missing_columns = [
            column for column in REQUIRED_COLUMNS if column not in fieldnames
        ]

        if missing_columns:
            raise ManifestError(
                f"Manifest is missing required columns: {', '.join(missing_columns)}"
            )

        entries: list[ManifestEntry] = []

        for row_number, row in enumerate(reader, start=2):
            if not any((value or "").strip() for value in row.values()):
                continue

            cleaned = {
                column: (row.get(column) or "").strip() for column in REQUIRED_COLUMNS
            }

            try:
                entries.append(ManifestEntry(**cleaned))
            except Exception as error:
                raise ManifestError(
                    f"Invalid manifest row {row_number}: {error}"
                ) from error

    _reject_duplicate_document_ids(entries)

    return sorted(entries, key=lambda entry: entry.document_id)


def _reject_duplicate_document_ids(entries: list[ManifestEntry]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []

    for entry in entries:
        if entry.document_id in seen:
            duplicates.append(entry.document_id)
        seen.add(entry.document_id)

    if duplicates:
        raise ManifestError(
            "Duplicate document_id values in manifest: " + ", ".join(sorted(set(duplicates)))
        )


def select_ingestable(
    entries: list[ManifestEntry],
    raw_dir: str | Path = RAW_DIR,
) -> tuple[list[ManifestEntry], list[SkippedDocument]]:
    """Split entries into those ready to parse and those skipped, with reasons."""
    raw_dir = Path(raw_dir)

    ingestable: list[ManifestEntry] = []
    skipped: list[SkippedDocument] = []

    for entry in entries:
        if not entry.is_marked_downloaded():
            skipped.append(
                SkippedDocument(
                    document_id=entry.document_id,
                    local_filename=entry.local_filename,
                    reason=f"status is '{entry.status or 'empty'}', expected '{INGESTABLE_STATUS}'",
                )
            )
            continue

        if not entry.pdf_path(raw_dir).exists():
            skipped.append(
                SkippedDocument(
                    document_id=entry.document_id,
                    local_filename=entry.local_filename,
                    reason=f"file not found in {raw_dir}",
                )
            )
            continue

        ingestable.append(entry)

    return ingestable, skipped
