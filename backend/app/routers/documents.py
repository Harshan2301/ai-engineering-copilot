"""
Documents router — upload, list, and delete PDF documents.
"""
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, status

from app.config import get_settings
from app.schemas.documents import (
    DocumentUploadResponse,
    DocumentListResponse,
    DeleteDocumentResponse,
)
from app.services.document_service import ingest_pdf, list_documents, delete_document

logger = logging.getLogger("app.routers.documents")
settings = get_settings()

router = APIRouter(prefix="/documents", tags=["Documents"])

ALLOWED_CONTENT_TYPES = {"application/pdf"}
MAX_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a PDF document for ingestion",
)
async def upload_document(file: UploadFile = File(...)):
    # Validate content type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Only PDF files are accepted. Got: {file.content_type}",
        )

    file_bytes = await file.read()

    # Validate file size
    if len(file_bytes) > MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {settings.MAX_FILE_SIZE_MB} MB limit.",
        )

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        result = ingest_pdf(file_bytes, file.filename or "document.pdf")
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error during ingestion: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error during document ingestion.",
        )

    return result


@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List all uploaded documents",
)
async def get_documents():
    try:
        return list_documents()
    except Exception as exc:
        logger.exception("Error listing documents: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to retrieve document list.")


@router.delete(
    "/{doc_id}",
    response_model=DeleteDocumentResponse,
    summary="Delete a document and all its chunks",
)
async def remove_document(doc_id: str):
    try:
        return delete_document(doc_id)
    except Exception as exc:
        logger.exception("Error deleting document %s: %s", doc_id, exc)
        raise HTTPException(status_code=500, detail="Failed to delete document.")
