"""
RAG router — question answering, semantic search, document summarization.
"""
import logging
from fastapi import APIRouter, HTTPException, status

from app.schemas.rag import (
    AskRequest, AskResponse,
    SearchRequest, SearchResponse,
    SummarizeRequest, SummarizeResponse,
)
from app.services.rag_service import answer_question, semantic_search, summarize_document

logger = logging.getLogger("app.routers.rag")

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question using RAG over uploaded documents",
)
async def ask(req: AskRequest):
    try:
        return answer_question(req)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        logger.exception("Error in /rag/ask: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG pipeline error: {str(exc)}",
        )


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic similarity search over document chunks",
)
async def search(req: SearchRequest):
    try:
        return semantic_search(req)
    except Exception as exc:
        logger.exception("Error in /rag/search: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search error: {str(exc)}",
        )


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    summary="Generate an AI summary of an uploaded document",
)
async def summarize(req: SummarizeRequest):
    try:
        return summarize_document(req)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        logger.exception("Error in /rag/summarize: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarization error: {str(exc)}",
        )
