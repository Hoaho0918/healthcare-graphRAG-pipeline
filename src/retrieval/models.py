"""Typed result schema shared by vector and graph retrieval.

One schema for both sources. Scores are NOT blended: cosine similarity and
human review confidence are different scales, so each result keeps its native
score and declares its `score_type`.

Educational retrieval prototype. Not clinical decision support.
"""

from typing import Literal

from pydantic import BaseModel, Field

RetrievalSource = Literal["vector", "graph"]
ScoreType = Literal["cosine_similarity", "review_confidence"]
GraphStatus = Literal["ok", "unavailable", "disabled"]

# Deterministic ordering weight for reviewed graph evidence.
CONFIDENCE_SCORES: dict[str, float] = {"high": 1.0, "medium": 0.75, "low": 0.5}
DEFAULT_CONFIDENCE_SCORE = 0.25


def confidence_score(confidence: str | None) -> float:
    return CONFIDENCE_SCORES.get((confidence or "").lower(), DEFAULT_CONFIDENCE_SCORE)


class RetrievalResult(BaseModel):
    rank: int = Field(..., ge=1)
    retrieval_source: RetrievalSource
    score: float
    score_type: ScoreType

    document_id: str
    document_title: str = ""
    topic: str = ""
    domain: str = ""
    source_filename: str
    page_number: int = Field(..., ge=1)
    chunk_id: str
    source_url: str = ""
    evidence_text: str

    # Populated only for graph-derived results.
    relationship_type: str | None = None
    review_status: str | None = None
    confidence: str | None = None
    source_entity: str | None = None
    target_entity: str | None = None

    # Dedup bookkeeping: set when the same chunk was also found by vector search.
    vector_score: float | None = None
    also_found_by: list[RetrievalSource] = Field(default_factory=list)


class HybridRetrievalResponse(BaseModel):
    query: str
    results: list[RetrievalResult] = Field(default_factory=list)

    graph_status: GraphStatus = "ok"
    graph_warning: str | None = None

    vector_hits: int = 0
    graph_hits: int = 0
    duplicates_merged: int = 0
    matched_conditions: list[str] = Field(default_factory=list)

    @property
    def graph_available(self) -> bool:
        return self.graph_status == "ok"
