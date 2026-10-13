"""Typed schema for evidence-grounded answers.

No model generates prose here. An answer is a numbered set of verbatim quotes
from retrieved source text, plus provenance. The layer draws no clinical
conclusion of its own.

Educational retrieval prototype. Not clinical decision support.
"""

from pydantic import BaseModel, Field

from src.retrieval.models import GraphStatus, RetrievalSource, ScoreType

DISCLAIMER = (
    "Educational retrieval prototype. Not clinical decision support. "
    "Quotes are reproduced from the indexed guideline PDFs; verify against the "
    "cited source before any clinical use."
)

ABSTENTION_TEXT = (
    "I couldn't find sufficient evidence in the indexed sources to answer that."
)


class Citation(BaseModel):
    marker: str
    rank: int = Field(..., ge=1)
    retrieval_source: RetrievalSource
    score: float
    score_type: ScoreType

    document_id: str
    document_title: str = ""
    source_filename: str
    page_number: int = Field(..., ge=1)
    chunk_id: str
    source_url: str = ""

    evidence_text: str

    # Present only when the evidence came from the reviewed graph.
    relationship_type: str | None = None
    review_status: str | None = None
    confidence: str | None = None

    def provenance_line(self) -> str:
        """Human-readable provenance for this citation."""
        parts = [
            f"{self.source_filename}, page {self.page_number}",
            f"chunk {self.chunk_id}",
            f"retrieved by {self.retrieval_source}",
        ]

        if self.retrieval_source == "graph":
            parts.append(f"relation {self.relationship_type}")
            parts.append(f"review {self.review_status}")

        if self.source_url:
            parts.append(self.source_url)

        return " | ".join(parts)


class GroundedAnswer(BaseModel):
    question: str
    answer_text: str
    citations: list[Citation] = Field(default_factory=list)
    insufficient_evidence: bool = False

    graph_status: GraphStatus = "ok"
    graph_warning: str | None = None
    warnings: list[str] = Field(default_factory=list)

    disclaimer: str = DISCLAIMER

    @property
    def citation_count(self) -> int:
        return len(self.citations)

    @property
    def vector_citations(self) -> int:
        return sum(1 for c in self.citations if c.retrieval_source == "vector")

    @property
    def graph_citations(self) -> int:
        return sum(1 for c in self.citations if c.retrieval_source == "graph")
