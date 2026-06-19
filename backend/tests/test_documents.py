"""
Tests for the document ingestion endpoints.
Run with: pytest tests/test_documents.py -v
"""
import io
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_pdf_bytes() -> bytes:
    """Create a minimal valid PDF in-memory (no external files needed)."""
    import pypdf
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "AI Engineering Copilot"


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "chromadb" in data
    assert "gemini" in data


# ---------------------------------------------------------------------------
# Document upload validation
# ---------------------------------------------------------------------------

def test_upload_rejects_non_pdf():
    resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("test.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 415


def test_upload_rejects_empty_file():
    resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert resp.status_code == 400


@patch("app.services.document_service.ingest_pdf")
def test_upload_pdf_success(mock_ingest, minimal_pdf_bytes):
    from app.schemas.documents import DocumentUploadResponse
    mock_ingest.return_value = DocumentUploadResponse(
        doc_id="test-doc-id-123",
        filename="test.pdf",
        pages=1,
        chunks=3,
        message="Successfully ingested 'test.pdf' with 3 chunks.",
    )
    resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("test.pdf", minimal_pdf_bytes, "application/pdf")},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["doc_id"] == "test-doc-id-123"
    assert data["chunks"] == 3


# ---------------------------------------------------------------------------
# Document list
# ---------------------------------------------------------------------------

@patch("app.services.document_service.list_documents")
def test_list_documents(mock_list):
    from app.schemas.documents import DocumentListResponse, DocumentMeta
    mock_list.return_value = DocumentListResponse(
        documents=[
            DocumentMeta(
                doc_id="abc",
                filename="spec.pdf",
                pages=10,
                chunks=42,
                uploaded_at="2025-01-01T00:00:00Z",
            )
        ],
        total=1,
    )
    resp = client.get("/api/v1/documents/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["documents"][0]["filename"] == "spec.pdf"


# ---------------------------------------------------------------------------
# Document delete
# ---------------------------------------------------------------------------

@patch("app.services.document_service.delete_document")
def test_delete_document(mock_delete):
    from app.schemas.documents import DeleteDocumentResponse
    mock_delete.return_value = DeleteDocumentResponse(
        doc_id="abc",
        message="Document 'abc' deleted successfully.",
    )
    resp = client.delete("/api/v1/documents/abc")
    assert resp.status_code == 200
    assert resp.json()["doc_id"] == "abc"


# ---------------------------------------------------------------------------
# Chunking utility
# ---------------------------------------------------------------------------

def test_chunk_text_basic():
    from app.services.document_service import _chunk_text
    text = "A" * 2000
    chunks = _chunk_text(text, chunk_size=800, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 800 for c in chunks)


def test_chunk_text_empty():
    from app.services.document_service import _chunk_text
    assert _chunk_text("", 800, 100) == []
    assert _chunk_text("   ", 800, 100) == []
