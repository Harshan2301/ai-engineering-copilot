from pydantic import BaseModel, Field
from typing import Optional


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    doc_id: Optional[str] = Field(None, description="Filter to a specific document")
    top_k: int = Field(5, ge=1, le=20)


class SourceChunk(BaseModel):
    text: str
    doc_id: str
    filename: str
    page: int
    score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]
    model: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)
    doc_id: Optional[str] = None
    top_k: int = Field(8, ge=1, le=30)


class SearchResponse(BaseModel):
    query: str
    results: list[SourceChunk]
    total: int


class SummarizeRequest(BaseModel):
    doc_id: str = Field(..., description="Document ID to summarize")
    style: str = Field(
        "technical",
        description="Summary style: 'technical', 'executive', 'bullet'",
    )


class SummarizeResponse(BaseModel):
    doc_id: str
    filename: str
    summary: str
    style: str
    word_count: int
