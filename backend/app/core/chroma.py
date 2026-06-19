"""
ChromaDB persistent client and collection helpers.
"""
import logging
from functools import lru_cache
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings

logger = logging.getLogger("app.core.chroma")
settings = get_settings()


class ChromaManager:
    """Wraps a persistent ChromaDB client with helper methods."""

    def __init__(self):
        settings.ensure_dirs()
        self._client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._get_or_create_collection(
            settings.CHROMA_COLLECTION_NAME
        )
        logger.info(
            "ChromaDB ready — collection '%s' at '%s'",
            settings.CHROMA_COLLECTION_NAME,
            settings.CHROMA_PERSIST_DIR,
        )

    # ------------------------------------------------------------------
    # Collection helpers
    # ------------------------------------------------------------------
    def _get_or_create_collection(self, name: str):
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self):
        return self._collection

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def upsert_chunks(
        self,
        doc_id: str,
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> int:
        """Upsert text chunks with pre-computed embeddings."""
        ids = [f"{doc_id}__chunk_{i}" for i in range(len(chunks))]
        self._collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.debug("Upserted %d chunks for doc_id=%s", len(chunks), doc_id)
        return len(chunks)

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: Optional[dict] = None,
    ) -> dict:
        """Cosine-similarity search, returns Chroma results dict."""
        kwargs = dict(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where
        return self._collection.query(**kwargs)

    def delete_document(self, doc_id: str) -> None:
        """Remove all chunks belonging to a document."""
        self._collection.delete(where={"doc_id": doc_id})
        logger.info("Deleted all chunks for doc_id=%s", doc_id)

    def list_documents(self) -> list[dict]:
        """Return unique document metadata entries."""
        results = self._collection.get(include=["metadatas"])
        seen, docs = set(), []
        for meta in results["metadatas"]:
            did = meta.get("doc_id")
            if did and did not in seen:
                seen.add(did)
                docs.append(meta)
        return docs

    def get_chunks_by_doc(self, doc_id: str) -> list[str]:
        """Retrieve all text chunks for a given document."""
        results = self._collection.get(
            where={"doc_id": doc_id},
            include=["documents"],
        )
        return results.get("documents", [])

    def count(self) -> int:
        return self._collection.count()


@lru_cache(maxsize=1)
def get_chroma_manager() -> ChromaManager:
    return ChromaManager()
