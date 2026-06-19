"""
RAG service — retrieval-augmented generation using ChromaDB + Gemini.
Handles: question answering, semantic search, document summarization.
"""
import logging
from typing import Optional

from app.config import get_settings
from app.core.chroma import get_chroma_manager
from app.core.gemini import get_gemini_client
from app.schemas.rag import (
    AskRequest, AskResponse,
    SearchRequest, SearchResponse,
    SummarizeRequest, SummarizeResponse,
    SourceChunk,
)

logger = logging.getLogger("app.services.rag")
settings = get_settings()

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_QA_SYSTEM = """You are an expert AI Engineering Copilot. Answer the user's question
EXCLUSIVELY based on the provided context chunks from their documents.
If the answer is not found in the context, say so clearly.
Always cite which source document/page your answer comes from.
Be precise, concise, and technically accurate."""

_SUMMARY_STYLES = {
    "technical": (
        "Create a detailed technical summary covering: architecture, key algorithms, "
        "data flows, APIs, dependencies, and important implementation details."
    ),
    "executive": (
        "Create a concise executive summary (3-5 bullet points) covering: "
        "purpose, key findings, risks, and recommendations."
    ),
    "bullet": (
        "Summarize the document as a structured bullet-point list grouped by topic. "
        "Be thorough but concise."
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chroma_results_to_chunks(results: dict) -> list[SourceChunk]:
    """Convert raw ChromaDB query results to SourceChunk list."""
    chunks = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for text, meta, dist in zip(docs, metas, distances):
        # ChromaDB cosine distance → similarity score (1 - dist)
        score = round(1.0 - float(dist), 4)
        chunks.append(
            SourceChunk(
                text=text,
                doc_id=meta.get("doc_id", ""),
                filename=meta.get("filename", "unknown"),
                page=int(meta.get("page", 0)),
                score=score,
            )
        )
    return chunks


def _build_context(chunks: list[SourceChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[Source {i} — {c.filename}, page {c.page}]\n{c.text}"
        )
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public service functions
# ---------------------------------------------------------------------------

def answer_question(req: AskRequest) -> AskResponse:
    """RAG pipeline: embed question → retrieve → generate answer."""
    gemini = get_gemini_client()
    chroma = get_chroma_manager()

    logger.info("RAG ask: '%s'", req.question[:80])

    # 1. Embed the query
    q_embedding = gemini.embed_query(req.question)

    # 2. Retrieve top-k chunks
    where = {"doc_id": req.doc_id} if req.doc_id else None
    results = chroma.query(q_embedding, n_results=req.top_k, where=where)
    source_chunks = _chroma_results_to_chunks(results)

    if not source_chunks:
        return AskResponse(
            question=req.question,
            answer="No relevant content found in the uploaded documents. Please upload relevant PDFs first.",
            sources=[],
            model=settings.GEMINI_MODEL,
        )

    # 3. Build prompt with context
    context = _build_context(source_chunks)
    prompt = (
        f"Context from documents:\n\n{context}\n\n"
        f"Question: {req.question}\n\n"
        f"Answer (cite sources by filename and page number):"
    )

    # 4. Generate answer
    answer = gemini.generate(prompt, system_instruction=_QA_SYSTEM)

    logger.info("Generated answer (%d chars) from %d sources", len(answer), len(source_chunks))
    return AskResponse(
        question=req.question,
        answer=answer,
        sources=source_chunks,
        model=settings.GEMINI_MODEL,
    )


def semantic_search(req: SearchRequest) -> SearchResponse:
    """Pure semantic search — returns ranked chunks without generation."""
    gemini = get_gemini_client()
    chroma = get_chroma_manager()

    logger.info("Semantic search: '%s'", req.query[:80])

    q_embedding = gemini.embed_query(req.query)
    where = {"doc_id": req.doc_id} if req.doc_id else None
    results = chroma.query(q_embedding, n_results=req.top_k, where=where)
    chunks = _chroma_results_to_chunks(results)

    return SearchResponse(query=req.query, results=chunks, total=len(chunks))


def summarize_document(req: SummarizeRequest) -> SummarizeResponse:
    """Retrieve all chunks for a document and generate a Gemini summary."""
    gemini = get_gemini_client()
    chroma = get_chroma_manager()

    logger.info("Summarizing doc_id=%s style=%s", req.doc_id, req.style)

    # Get all document chunks
    chunks = chroma.get_chunks_by_doc(req.doc_id)
    if not chunks:
        raise ValueError(f"No content found for document ID '{req.doc_id}'.")

    # Get filename from metadata
    raw = chroma.collection.get(
        where={"doc_id": req.doc_id},
        include=["metadatas"],
        limit=1,
    )
    filename = "unknown"
    if raw.get("metadatas"):
        filename = raw["metadatas"][0].get("filename", "unknown")

    # Truncate to avoid token limits (~100k chars ≈ 25k tokens)
    combined = "\n\n".join(chunks)
    if len(combined) > 100_000:
        combined = combined[:100_000] + "\n\n[... content truncated for summary ...]"

    style_instruction = _SUMMARY_STYLES.get(req.style, _SUMMARY_STYLES["technical"])
    prompt = (
        f"Document: {filename}\n\n"
        f"Content:\n{combined}\n\n"
        f"Instructions: {style_instruction}"
    )

    summary = gemini.generate(
        prompt,
        system_instruction="You are a technical documentation expert. Summarize the document precisely.",
    )

    word_count = len(summary.split())
    logger.info("Summary generated: %d words", word_count)

    return SummarizeResponse(
        doc_id=req.doc_id,
        filename=filename,
        summary=summary,
        style=req.style,
        word_count=word_count,
    )
