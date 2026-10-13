"""Validation for the reviewed clinical triplets.

These tests need no database. They prove provenance integrity: every triplet
cites a real chunk, and its evidence_text is a verbatim substring of that
chunk. They do not and cannot judge clinical correctness.
"""

import json
from pathlib import Path

import pytest

from src.graph.triplet_schema import (
    ALLOWED_NODE_LABELS,
    ALLOWED_RELATION_TYPES,
    ReviewedTriplet,
    TripletValidationError,
    entity_key,
    load_triplets,
)

CHUNKS_PATH = Path("data/processed/document_chunks.json")

VALID_TRIPLET = {
    "triplet_id": "test001",
    "source": {"label": "Drug", "name": "Metformin"},
    "relation": "TREATS",
    "target": {"label": "Condition", "name": "Type 2 diabetes"},
    "chunk_id": "nice_type2_adults_001_p71_c0",
    "document_id": "nice_type2_adults_001",
    "source_filename": "type-2-diabetes-in-adults-management.pdf",
    "page_number": 71,
    "source_url": "",
    "evidence_text": "first-line therapy with metformin",
    "confidence": "high",
    "review_status": "approved",
}


@pytest.fixture(scope="module")
def triplets() -> list[ReviewedTriplet]:
    return load_triplets()


@pytest.fixture(scope="module")
def chunks_by_id() -> dict[str, dict]:
    if not CHUNKS_PATH.exists():
        pytest.skip(f"{CHUNKS_PATH} not found. Run: python -m src.ingestion.batch_ingest")

    with CHUNKS_PATH.open(encoding="utf-8") as file:
        return {chunk["chunk_id"]: chunk for chunk in json.load(file)}


def test_triplet_count_is_in_range(triplets: list[ReviewedTriplet]) -> None:
    assert 20 <= len(triplets) <= 30


def test_labels_and_relations_are_allow_listed(triplets: list[ReviewedTriplet]) -> None:
    for triplet in triplets:
        assert triplet.source.label in ALLOWED_NODE_LABELS
        assert triplet.target.label in ALLOWED_NODE_LABELS
        assert triplet.relation in ALLOWED_RELATION_TYPES


def test_every_triplet_is_approved(triplets: list[ReviewedTriplet]) -> None:
    for triplet in triplets:
        assert triplet.review_status == "approved"


def test_entity_keys_are_stable_and_unique_per_name(triplets: list[ReviewedTriplet]) -> None:
    for triplet in triplets:
        assert triplet.source.key == entity_key(triplet.source.label, triplet.source.name)
        assert triplet.target.key == entity_key(triplet.target.label, triplet.target.name)


def test_every_triplet_cites_a_real_chunk(
    triplets: list[ReviewedTriplet], chunks_by_id: dict[str, dict]
) -> None:
    missing = [t.triplet_id for t in triplets if t.chunk_id not in chunks_by_id]

    assert not missing, f"triplets citing unknown chunks: {missing}"


def test_evidence_text_is_verbatim_from_the_cited_chunk(
    triplets: list[ReviewedTriplet], chunks_by_id: dict[str, dict]
) -> None:
    for triplet in triplets:
        chunk_text = chunks_by_id[triplet.chunk_id]["text"]

        assert triplet.evidence_text in chunk_text, (
            f"{triplet.triplet_id}: evidence_text is not a verbatim substring of "
            f"{triplet.chunk_id}"
        )


def test_provenance_matches_the_cited_chunk(
    triplets: list[ReviewedTriplet], chunks_by_id: dict[str, dict]
) -> None:
    for triplet in triplets:
        chunk = chunks_by_id[triplet.chunk_id]

        assert triplet.document_id == chunk["document_id"]
        assert triplet.source_filename == chunk["source_filename"]
        assert triplet.page_number == chunk["page_number"]
        assert triplet.source_url == chunk["source_url"]


def test_unsupported_node_label_is_rejected() -> None:
    payload = {**VALID_TRIPLET, "source": {"label": "Procedure", "name": "Surgery"}}

    with pytest.raises(ValueError, match="unsupported node label"):
        ReviewedTriplet(**payload)


def test_unsupported_relation_type_is_rejected() -> None:
    payload = {**VALID_TRIPLET, "relation": "CAUSES"}

    with pytest.raises(ValueError, match="unsupported relation"):
        ReviewedTriplet(**payload)


def test_unsupported_direction_is_rejected() -> None:
    payload = {
        **VALID_TRIPLET,
        "source": {"label": "Condition", "name": "Type 2 diabetes"},
        "relation": "TREATS",
        "target": {"label": "Drug", "name": "Metformin"},
    }

    with pytest.raises(ValueError, match="unsupported direction"):
        ReviewedTriplet(**payload)


def test_unapproved_triplet_is_rejected() -> None:
    payload = {**VALID_TRIPLET, "review_status": "pending_review"}

    with pytest.raises(ValueError, match="review_status must be 'approved'"):
        ReviewedTriplet(**payload)


def test_missing_triplet_file_fails_safely(tmp_path: Path) -> None:
    with pytest.raises(TripletValidationError, match="not found"):
        load_triplets(tmp_path / "does_not_exist.json")
