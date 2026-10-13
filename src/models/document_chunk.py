from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """One retrievable passage of clinical guideline text with full provenance.

    Field order is deliberate: it is the order used in `document_chunks.json`,
    so the serialised output stays readable and diff-friendly.
    """

    chunk_id: str = Field(
        ...,
        min_length=1,
        description="Globally unique, stable identifier: {document_id}_p{page_number}_c{chunk_index}",
    )
    document_id: str = Field(
        ...,
        min_length=1,
        description="Manifest identifier of the source document",
    )
    document_title: str = Field(
        ...,
        min_length=1,
        description="Full published title of the source guideline",
    )
    organization: str = Field(
        default="",
        description="Publishing body, for example NICE",
    )
    domain: str = Field(
        default="",
        description="Broad clinical domain from the manifest",
    )
    topic: str = Field(
        default="",
        description="Narrow topic label from the manifest",
    )
    source_filename: str = Field(
        ...,
        min_length=1,
        description="Original PDF filename in data/raw/",
    )
    source_url: str = Field(
        default="",
        description="Public landing page for the guideline. May be empty until the manifest is enriched.",
    )
    page_number: int = Field(
        ...,
        ge=1,
        description="One-indexed page number from the original PDF",
    )
    chunk_index: int = Field(
        ...,
        ge=0,
        description=(
            "Position of the chunk within its page, taken from the original splitter "
            "output. Gaps are expected and intentional: filtered chunks keep their "
            "original index so IDs stay stable across reruns."
        ),
    )
    text: str = Field(
        ...,
        min_length=1,
        description="Cleaned text content for retrieval and graph extraction",
    )
