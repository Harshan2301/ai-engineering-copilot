"""
httpx-based API client for the Streamlit frontend.
All communication with the FastAPI backend goes through this module.
"""
import logging
from typing import Optional, Any

import httpx

logger = logging.getLogger("frontend.api_client")

# Default backend URL (overridden by environment variable via st.secrets or env)
import os
BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000/api/v1")
TIMEOUT = 120.0  # seconds — long for summarize requests


def _get_client() -> httpx.Client:
    return httpx.Client(base_url=BASE_URL, timeout=TIMEOUT)


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

def upload_document(file_bytes: bytes, filename: str) -> dict:
    """Upload a PDF to the backend for ingestion."""
    with _get_client() as client:
        response = client.post(
            "/documents/upload",
            files={"file": (filename, file_bytes, "application/pdf")},
        )
        response.raise_for_status()
        return response.json()


def list_documents() -> list[dict]:
    """Fetch all uploaded documents."""
    with _get_client() as client:
        response = client.get("/documents/")
        response.raise_for_status()
        return response.json().get("documents", [])


def delete_document(doc_id: str) -> dict:
    """Delete a document by ID."""
    with _get_client() as client:
        response = client.delete(f"/documents/{doc_id}")
        response.raise_for_status()
        return response.json()


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------

def ask_question(
    question: str,
    doc_id: Optional[str] = None,
    top_k: int = 5,
) -> dict:
    """Send a RAG question and return the answer + sources."""
    payload: dict[str, Any] = {"question": question, "top_k": top_k}
    if doc_id:
        payload["doc_id"] = doc_id

    with _get_client() as client:
        response = client.post("/rag/ask", json=payload)
        response.raise_for_status()
        return response.json()


def semantic_search(
    query: str,
    doc_id: Optional[str] = None,
    top_k: int = 8,
) -> dict:
    """Run semantic search and return ranked chunks."""
    payload: dict[str, Any] = {"query": query, "top_k": top_k}
    if doc_id:
        payload["doc_id"] = doc_id

    with _get_client() as client:
        response = client.post("/rag/search", json=payload)
        response.raise_for_status()
        return response.json()


def summarize_document(doc_id: str, style: str = "technical") -> dict:
    """Request an AI summary of a document."""
    with _get_client() as client:
        response = client.post(
            "/rag/summarize",
            json={"doc_id": doc_id, "style": style},
        )
        response.raise_for_status()
        return response.json()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def health_check() -> dict:
    """Check backend health status."""
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(
                os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/api/v1") + "/health"
            )
            response.raise_for_status()
            return response.json()
    except Exception as exc:
        return {"status": "unreachable", "error": str(exc)}
