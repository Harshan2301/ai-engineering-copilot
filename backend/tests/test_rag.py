"""
Tests for RAG endpoints: /rag/ask, /rag/search, /rag/summarize.
Run with: pytest tests/test_rag.py -v
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# /rag/ask
# ---------------------------------------------------------------------------

@patch("app.services.rag_service.answer_question")
def test_ask_success(mock_ask):
    from app.schemas.rag import AskResponse, SourceChunk
    mock_ask.return_value = AskResponse(
        question="What is a PID controller?",
        answer="A PID controller is a control loop mechanism...",
        sources=[
            SourceChunk(
                text="PID stands for Proportional-Integral-Derivative...",
                doc_id="doc-1",
                filename="controls.pdf",
                page=3,
                score=0.91,
            )
        ],
        model="gemini-1.5-pro",
    )
    resp = client.post(
        "/api/v1/rag/ask",
        json={"question": "What is a PID controller?", "top_k": 5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert len(data["sources"]) == 1
    assert data["sources"][0]["score"] == 0.91


def test_ask_too_short_question():
    resp = client.post(
        "/api/v1/rag/ask",
        json={"question": "hi", "top_k": 5},
    )
    assert resp.status_code == 422  # Pydantic min_length validation


def test_ask_missing_question():
    resp = client.post("/api/v1/rag/ask", json={"top_k": 5})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /rag/search
# ---------------------------------------------------------------------------

@patch("app.services.rag_service.semantic_search")
def test_search_success(mock_search):
    from app.schemas.rag import SearchResponse, SourceChunk
    mock_search.return_value = SearchResponse(
        query="thermal runaway",
        results=[
            SourceChunk(
                text="Thermal runaway is a condition...",
                doc_id="doc-2",
                filename="battery.pdf",
                page=7,
                score=0.85,
            )
        ],
        total=1,
    )
    resp = client.post(
        "/api/v1/rag/search",
        json={"query": "thermal runaway", "top_k": 8},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["results"][0]["filename"] == "battery.pdf"


def test_search_empty_query():
    resp = client.post("/api/v1/rag/search", json={"query": "x", "top_k": 5})
    assert resp.status_code == 422  # min_length=2 enforced by Pydantic


# ---------------------------------------------------------------------------
# /rag/summarize
# ---------------------------------------------------------------------------

@patch("app.services.rag_service.summarize_document")
def test_summarize_success(mock_summarize):
    from app.schemas.rag import SummarizeResponse
    mock_summarize.return_value = SummarizeResponse(
        doc_id="doc-1",
        filename="controls.pdf",
        summary="This document covers PID control theory...",
        style="technical",
        word_count=120,
    )
    resp = client.post(
        "/api/v1/rag/summarize",
        json={"doc_id": "doc-1", "style": "technical"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["word_count"] == 120
    assert data["style"] == "technical"


@patch("app.services.rag_service.summarize_document")
def test_summarize_doc_not_found(mock_summarize):
    mock_summarize.side_effect = ValueError("No content found for document ID 'bad-id'.")
    resp = client.post(
        "/api/v1/rag/summarize",
        json={"doc_id": "bad-id", "style": "executive"},
    )
    assert resp.status_code == 404


def test_summarize_missing_doc_id():
    resp = client.post("/api/v1/rag/summarize", json={"style": "technical"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Source chunk score validation
# ---------------------------------------------------------------------------

def test_source_chunk_score_range():
    """Score must be between 0.0 and 1.0 (cosine similarity)."""
    from app.schemas.rag import SourceChunk
    chunk = SourceChunk(
        text="sample", doc_id="d", filename="f.pdf", page=1, score=0.75
    )
    assert 0.0 <= chunk.score <= 1.0
