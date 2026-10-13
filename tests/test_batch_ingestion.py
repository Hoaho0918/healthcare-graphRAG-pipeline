"""Integrity checks for the batch ingestion output.

Unit tests for cleaning/filtering run anywhere. The corpus tests are skipped
if data/processed/document_chunks.json has not been built yet, since that
file is git-ignored local output.
"""

import json
import re
from pathlib import Path

import pytest

from src.ingestion.manifest_loader import load_manifest
from src.ingestion.pdf_parser import (
    build_chunk_id,
    clean_page_text,
    is_disclaimer_page,
    is_useful_chunk,
    strip_leading_artifacts,
)

CHUNKS_PATH = Path("data/processed/document_chunks.json")

CHUNK_ID_PATTERN = re.compile(r"^(?P<document_id>.+)_p(?P<page>\d+)_c(?P<index>\d+)$")

REQUIRED_NON_EMPTY_FIELDS = (
    "chunk_id",
    "document_id",
    "document_title",
    "organization",
    "domain",
    "topic",
    "source_filename",
    "text",
)

# source_url is intentionally allowed to be empty until the manifest is enriched.
REQUIRED_PRESENT_FIELDS = REQUIRED_NON_EMPTY_FIELDS + (
    "source_url",
    "page_number",
    "chunk_index",
)


@pytest.fixture(scope="module")
def chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        pytest.skip(
            f"{CHUNKS_PATH} not found. Run: python -m src.ingestion.batch_ingest"
        )

    with CHUNKS_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def test_corpus_is_not_empty(chunks: list[dict]) -> None:
    assert chunks, "document_chunks.json contains no chunks"


def test_chunk_ids_are_globally_unique(chunks: list[dict]) -> None:
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    duplicates = {chunk_id for chunk_id in chunk_ids if chunk_ids.count(chunk_id) > 1}

    assert not duplicates, f"duplicate chunk ids: {sorted(duplicates)[:5]}"


def test_chunk_ids_follow_the_naming_convention(chunks: list[dict]) -> None:
    for chunk in chunks:
        match = CHUNK_ID_PATTERN.match(chunk["chunk_id"])

        assert match, f"malformed chunk id: {chunk['chunk_id']}"
        assert match.group("document_id") == chunk["document_id"]
        assert int(match.group("page")) == chunk["page_number"]
        assert int(match.group("index")) == chunk["chunk_index"]


def test_chunk_ids_are_rebuildable_from_metadata(chunks: list[dict]) -> None:
    for chunk in chunks:
        expected = build_chunk_id(
            chunk["document_id"], chunk["page_number"], chunk["chunk_index"]
        )

        assert chunk["chunk_id"] == expected


def test_required_metadata_is_present(chunks: list[dict]) -> None:
    for chunk in chunks:
        missing = [field for field in REQUIRED_PRESENT_FIELDS if field not in chunk]

        assert not missing, f"{chunk.get('chunk_id')} is missing {missing}"

        empty = [
            field
            for field in REQUIRED_NON_EMPTY_FIELDS
            if not str(chunk[field]).strip()
        ]

        assert not empty, f"{chunk['chunk_id']} has empty {empty}"


def test_page_and_index_bounds(chunks: list[dict]) -> None:
    for chunk in chunks:
        assert chunk["page_number"] >= 1
        assert chunk["chunk_index"] >= 0


def test_every_document_id_exists_in_the_manifest(chunks: list[dict]) -> None:
    manifest_ids = {entry.document_id for entry in load_manifest()}
    corpus_ids = {chunk["document_id"] for chunk in chunks}

    assert corpus_ids <= manifest_ids, f"unknown document ids: {corpus_ids - manifest_ids}"


def test_filenames_match_the_manifest(chunks: list[dict]) -> None:
    expected = {entry.document_id: entry.local_filename for entry in load_manifest()}

    for chunk in chunks:
        assert chunk["source_filename"] == expected[chunk["document_id"]]


def test_no_chunk_starts_with_a_splitter_artifact(chunks: list[dict]) -> None:
    offenders = [
        chunk["chunk_id"]
        for chunk in chunks
        if chunk["text"] != strip_leading_artifacts(chunk["text"])
    ]

    assert not offenders, f"chunks with leading artifacts: {offenders[:5]}"


def test_strip_leading_artifacts_removes_orphaned_separator() -> None:
    assert strip_leading_artifacts(". Offer metformin.") == "Offer metformin."
    assert strip_leading_artifacts("   Offer metformin.") == "Offer metformin."
    assert strip_leading_artifacts(", and review annually.") == "and review annually."


def test_strip_leading_artifacts_preserves_wording_and_internal_punctuation() -> None:
    text = (
        "1.6.5 Measure HbA1c at 3-monthly to 6-monthly intervals (e.g. 48 mmol/mol, "
        "6.5%); review, then tailor to the person's needs. [2015]"
    )

    assert strip_leading_artifacts(text) == text


def test_strip_leading_artifacts_does_not_touch_the_end_of_a_chunk() -> None:
    assert strip_leading_artifacts(". Review the care plan...") == "Review the care plan..."


def test_strip_leading_artifacts_never_removes_letters_or_digits() -> None:
    for text in ("48 mmol/mol target", "HbA1c monitoring", "0.5 mg daily"):
        assert strip_leading_artifacts(text) == text


def test_cleaning_removes_nice_layout_boilerplate() -> None:
    page = (
        "1.2.1 Offer structured education to adults with type 2 diabetes. [2009] \n"
        "Type 2 diabetes in adults: management (NG28)\n"
        "© NICE 2026. All rights reserved. Subject to Notice of rights "
        "(https://www.nice.org.uk/terms-and-\n"
        "conditions#notice-of-rights).\n"
        "Page 8 of\n"
        "131"
    )

    cleaned = clean_page_text(page, "Type 2 diabetes in adults: management")

    assert cleaned == (
        "1.2.1 Offer structured education to adults with type 2 diabetes. [2009]"
    )


def test_cleaning_preserves_clinical_wording() -> None:
    sentence = (
        "Offer metformin as the initial drug treatment for adults with type 2 diabetes."
    )

    assert clean_page_text(sentence + "\nPage 4 of\n99", "Anything") == sentence


def test_recommendation_survives_a_surviving_footer_phrase() -> None:
    text = (
        "1.6.5 Measure HbA1c levels at 3-monthly to 6-monthly intervals in adults with "
        "type 2 diabetes, tailored to individual needs. All rights reserved."
    )

    assert is_useful_chunk(text)


def test_short_clinical_recommendation_is_retained() -> None:
    text = (
        "1.1.1 Adopt an individualised approach to diabetes care for adults with "
        "type 2 diabetes. [2015]"
    )

    assert is_useful_chunk(text)


def test_table_of_contents_chunk_is_rejected() -> None:
    text = (
        "Contents Using this guideline ............................................ 6 "
        "Individualised care ................................................... 7"
    )

    assert not is_useful_chunk(text)


def test_very_short_chunk_is_rejected() -> None:
    assert not is_useful_chunk("Education")


def test_disclaimer_page_is_detected() -> None:
    text = (
        "Your responsibility The recommendations in this guideline represent the view "
        "of NICE, arrived at after careful consideration of the evidence available. "
        "It is not mandatory to apply the recommendations. All problems should be "
        "reported using the Yellow Card Scheme."
    )

    assert is_disclaimer_page(text)


def test_clinical_page_is_not_mistaken_for_the_disclaimer() -> None:
    text = (
        "1.3.1 Provide individualised and ongoing nutritional advice from a healthcare "
        "professional with specific expertise and competencies in nutrition. [2015]"
    )

    assert not is_disclaimer_page(text)
