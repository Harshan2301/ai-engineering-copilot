"""
Document service — PDF parsing, text chunking, embedding, and ChromaDB upsert.
"""
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

import pypdf

from app.config import get_settings
from app.core.chroma import get_chroma_manager
from app.core.gemini import get_gemini_client
from app.schemas.documents import (
    DocumentMeta,
    DocumentUploadResponse,
    DocumentListResponse,
    DeleteDocumentResponse,
)

logger = logging.getLogger("app.services.document")
settings = get_settings()


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Sliding-window character-level chunker."""
    if not text.strip():
        return []
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# PDF parsing
# ---------------------------------------------------------------------------

def _parse_pdf(file_bytes: bytes) -> tuple[list[str], int]:
    """
    Extract per-page text from PDF bytes.
    Returns (list_of_page_texts, num_pages).
    """
    import io
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return pages, len(pages)


# ---------------------------------------------------------------------------
# Public service functions
# ---------------------------------------------------------------------------

def ingest_pdf(
    file_bytes: bytes,
    filename: str,
) -> DocumentUploadResponse:
    """Parse PDF → chunk → embed → store in ChromaDB."""
    doc_id = str(uuid.uuid4())
    gemini = get_gemini_client()
    chroma = get_chroma_manager()

    logger.info("Ingesting PDF '%s' (doc_id=%s)", filename, doc_id)

    # 1. Parse
    page_texts, num_pages = _parse_pdf(file_bytes)
    full_text = "\n\n".join(page_texts)
    if not full_text.strip():
        raise ValueError("PDF appears to be empty or contains only images.")

    # 2. Chunk with page tracking
    all_chunks, all_embeddings, all_metadatas = [], [], []
    for page_num, page_text in enumerate(page_texts, start=1):
        page_chunks = _chunk_text(
            page_text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP
        )
        for chunk in page_chunks:
            embedding = gemini.embed(chunk)
            all_chunks.append(chunk)
            all_embeddings.append(embedding)
            all_metadatas.append(
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    "page": page_num,
                    "uploaded_at": datetime.now(timezone.utc).isoformat(),
                }
            )

    if not all_chunks:
        raise ValueError("No text could be extracted from the PDF.")

    # 3. Upsert to ChromaDB
    chroma.upsert_chunks(doc_id, all_chunks, all_embeddings, all_metadatas)

    logger.info(
        "Ingested '%s': %d pages, %d chunks", filename, num_pages, len(all_chunks)
    )
    return DocumentUploadResponse(
        doc_id=doc_id,
        filename=filename,
        pages=num_pages,
        chunks=len(all_chunks),
        message=f"Successfully ingested '{filename}' with {len(all_chunks)} chunks.",
    )


def list_documents() -> DocumentListResponse:
    """Return all unique documents stored in ChromaDB."""
    chroma = get_chroma_manager()
    raw_docs = chroma.list_documents()

    docs = []
    seen = {}
    for meta in raw_docs:
        did = meta.get("doc_id", "")
        if did not in seen:
            seen[did] = True
            # Count chunks for this doc
            chunk_count = len(chroma.get_chunks_by_doc(did))
            docs.append(
                DocumentMeta(
                    doc_id=did,
                    filename=meta.get("filename", "unknown"),
                    pages=int(meta.get("page", 0)),
                    chunks=chunk_count,
                    uploaded_at=meta.get("uploaded_at", ""),
                )
            )

    return DocumentListResponse(documents=docs, total=len(docs))


def delete_document(doc_id: str) -> DeleteDocumentResponse:
    """Delete all chunks for a document from ChromaDB."""
    chroma = get_chroma_manager()
    chroma.delete_document(doc_id)
    logger.info("Deleted document doc_id=%s", doc_id)
    return DeleteDocumentResponse(
        doc_id=doc_id,
        message=f"Document '{doc_id}' deleted successfully.",
    )
