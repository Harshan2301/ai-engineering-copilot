from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class DocumentUploadResponse(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunks: int
    message: str


class DocumentMeta(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunks: int
    uploaded_at: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentMeta]
    total: int


class DeleteDocumentResponse(BaseModel):
    doc_id: str
    message: str
