"""
FastAPI application entry point.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.core.logging_config import logger
from app.core.chroma import get_chroma_manager
from app.core.gemini import get_gemini_client
from app.routers import documents, rag

settings = get_settings()


# ---------------------------------------------------------------------------
# Lifespan — startup & shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise singletons on startup."""
    logger.info("=== AI Engineering Copilot starting up ===")
    settings.ensure_dirs()

    try:
        get_chroma_manager()
        logger.info("ChromaDB ready ✓")
    except Exception as exc:
        logger.error("ChromaDB init failed: %s", exc)

    try:
        get_gemini_client()
        logger.info("Gemini client ready ✓")
    except Exception as exc:
        logger.warning("Gemini client init failed (check GEMINI_API_KEY): %s", exc)

    yield

    logger.info("=== AI Engineering Copilot shutting down ===")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Engineering Copilot",
    description=(
        "Production-ready RAG API powered by Gemini + ChromaDB.\n\n"
        "Upload PDFs, ask questions, perform semantic search, and summarize documents."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(documents.router, prefix="/api/v1")
app.include_router(rag.router, prefix="/api/v1")


# ---------------------------------------------------------------------------
# Health & root
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "AI Engineering Copilot",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    chroma_ok = False
    try:
        chroma = get_chroma_manager()
        count = chroma.count()
        chroma_ok = True
    except Exception:
        count = -1

    gemini_ok = False
    try:
        get_gemini_client()
        gemini_ok = True
    except Exception:
        pass

    return {
        "status": "healthy" if (chroma_ok and gemini_ok) else "degraded",
        "chromadb": {"ok": chroma_ok, "total_chunks": count},
        "gemini": {"ok": gemini_ok, "model": settings.GEMINI_MODEL},
    }


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred."},
    )
