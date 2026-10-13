"""Schema and validation for manually reviewed clinical triplets.

Nothing here calls an LLM or extracts facts. It only validates a hand-reviewed
file against a closed allow-list so unsupported labels, relation types or
directions fail before anything reaches the graph.
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

TRIPLETS_PATH = Path("data/reviewed/clinical_triplets.json")

ALLOWED_NODE_LABELS = frozenset(
    {
        "Condition",
        "Drug",
        "Technology",
        "Population",
        "Complication",
        "Recommendation",
    }
)

ALLOWED_RELATION_TYPES = frozenset(
    {
        "TREATS",
        "RECOMMENDS",
        "HAS_COMPLICATION",
        "APPLIES_TO",
        "USES_TECHNOLOGY",
    }
)

# (source_label, relation, target_label) combinations that are permitted.
ALLOWED_DIRECTIONS = frozenset(
    {
        ("Drug", "TREATS", "Condition"),
        ("Technology", "TREATS", "Condition"),
        ("Recommendation", "RECOMMENDS", "Drug"),
        ("Recommendation", "RECOMMENDS", "Technology"),
        ("Condition", "HAS_COMPLICATION", "Complication"),
        ("Recommendation", "APPLIES_TO", "Population"),
        ("Recommendation", "APPLIES_TO", "Condition"),
        ("Drug", "APPLIES_TO", "Population"),
        ("Technology", "APPLIES_TO", "Population"),
        ("Condition", "USES_TECHNOLOGY", "Technology"),
        ("Population", "USES_TECHNOLOGY", "Technology"),
    }
)

ALLOWED_CONFIDENCE = frozenset({"high", "medium", "low"})
REQUIRED_REVIEW_STATUS = "approved"


class TripletValidationError(ValueError):
    """Raised when the reviewed triplet file violates the allow-list."""


def entity_key(label: str, name: str) -> str:
    """Stable key for the Entity.key uniqueness constraint."""
    return f"{label}:{normalise_name(name)}"


def normalise_name(name: str) -> str:
    return " ".join(name.split()).strip().lower()


class TripletNode(BaseModel):
    label: str
    name: str = Field(..., min_length=1)

    @field_validator("label")
    @classmethod
    def label_must_be_allowed(cls, value: str) -> str:
        if value not in ALLOWED_NODE_LABELS:
            raise ValueError(
                f"unsupported node label '{value}'. Allowed: {sorted(ALLOWED_NODE_LABELS)}"
            )

        return value

    @property
    def key(self) -> str:
        return entity_key(self.label, self.name)

    @property
    def name_normalized(self) -> str:
        return normalise_name(self.name)


class ReviewedTriplet(BaseModel):
    triplet_id: str = Field(..., min_length=1)
    source: TripletNode
    relation: str
    target: TripletNode

    chunk_id: str = Field(..., min_length=1)
    document_id: str = Field(..., min_length=1)
    source_filename: str = Field(..., min_length=1)
    page_number: int = Field(..., ge=1)
    source_url: str = ""
    evidence_text: str = Field(..., min_length=1)
    confidence: str
    review_status: str

    @field_validator("relation")
    @classmethod
    def relation_must_be_allowed(cls, value: str) -> str:
        if value not in ALLOWED_RELATION_TYPES:
            raise ValueError(
                f"unsupported relation '{value}'. Allowed: {sorted(ALLOWED_RELATION_TYPES)}"
            )

        return value

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_allowed(cls, value: str) -> str:
        if value not in ALLOWED_CONFIDENCE:
            raise ValueError(
                f"unsupported confidence '{value}'. Allowed: {sorted(ALLOWED_CONFIDENCE)}"
            )

        return value

    @field_validator("review_status")
    @classmethod
    def must_be_approved(cls, value: str) -> str:
        if value != REQUIRED_REVIEW_STATUS:
            raise ValueError(
                f"review_status must be '{REQUIRED_REVIEW_STATUS}', got '{value}'. "
                "Unreviewed triplets are never loaded."
            )

        return value

    @model_validator(mode="after")
    def direction_must_be_allowed(self) -> "ReviewedTriplet":
        direction = (self.source.label, self.relation, self.target.label)

        if direction not in ALLOWED_DIRECTIONS:
            raise ValueError(
                f"unsupported direction {direction[0]} -[{direction[1]}]-> {direction[2]}"
            )

        return self


def load_triplets(path: str | Path = TRIPLETS_PATH) -> list[ReviewedTriplet]:
    """Load and fully validate the reviewed triplet file."""
    path = Path(path)

    if not path.exists():
        raise TripletValidationError(f"Reviewed triplet file not found: {path}")

    with path.open(encoding="utf-8") as file:
        payload = json.load(file)

    raw_triplets = payload.get("triplets")

    if not isinstance(raw_triplets, list) or not raw_triplets:
        raise TripletValidationError(f"{path} contains no 'triplets' list")

    triplets: list[ReviewedTriplet] = []
    seen_ids: set[str] = set()

    for position, raw in enumerate(raw_triplets, start=1):
        try:
            triplet = ReviewedTriplet(**raw)
        except Exception as error:
            raise TripletValidationError(
                f"triplet #{position} ({raw.get('triplet_id', 'no id')}) is invalid: {error}"
            ) from error

        if triplet.triplet_id in seen_ids:
            raise TripletValidationError(f"duplicate triplet_id: {triplet.triplet_id}")

        seen_ids.add(triplet.triplet_id)
        triplets.append(triplet)

    return triplets
