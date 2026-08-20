from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str = Field(
        ...,
        description="Unique identifier for one text chunk",
    )
    source_file: str = Field(
        ...,
        description="Original PDF filename",
    )
    page_number: int = Field(
        ...,
        ge=1,
        description="One-indexed page number from the original PDF",
    )
    chunk_index: int = Field(
        ...,
        ge=0,
        description="Position of the chunk within its page",
    )
    text: str = Field(
        ...,
        min_length=1,
        description="Cleaned text content for retrieval and graph extraction",
    )
