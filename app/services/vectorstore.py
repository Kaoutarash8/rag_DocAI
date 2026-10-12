"""Accès ChromaDB (similarité cosinus) : indexation, suppression, recherche."""

import logging
import threading
from dataclasses import dataclass

from app.core.config import get_settings

log = logging.getLogger(__name__)
_collection = None
_lock = threading.Lock()


@dataclass
class Hit:
    text: str
    source: str
    document_id: int
    chunk_index: int
    score: float | None  # similarité cosinus (None = non applicable)


def _col():
    global _collection
    with _lock:
        if _collection is None:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            s = get_settings()
            client = chromadb.PersistentClient(
                path=str(s.vectorstore_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            _collection = client.get_or_create_collection(
                name=s.vectorstore_collection, metadata={"hnsw:space": "cosine"}
            )
    return _collection


def index_document(document_id: int, filename: str, chunks: list[str],
                   embeddings: list[list[float]]) -> None:
    col = _col()
    col.delete(where={"document_id": document_id})  # ré-indexation propre
    if not chunks:
        return
    col.upsert(
        ids=[f"{document_id}::{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=chunks,
        metadatas=[{"document_id": document_id, "source": filename, "chunk_index": i}
                   for i in range(len(chunks))],
    )


def delete_document(document_id: int) -> None:
    _col().delete(where={"document_id": document_id})


def search(query_embedding: list[float], top_k: int,
           document_ids: list[int] | None = None) -> list[Hit]:
    kwargs = {"query_embeddings": [query_embedding], "n_results": top_k}
    if document_ids:
        kwargs["where"] = {"document_id": {"$in": document_ids}}
    res = _col().query(**kwargs)
    hits = []
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append(Hit(text, meta["source"], meta["document_id"],
                        meta["chunk_index"], round(1.0 - dist, 4)))
    return hits


def get_document_chunks(document_id: int, limit: int) -> list[Hit]:
    """Premiers passages d'un document, dans l'ordre (pour les résumés)."""
    res = _col().get(where={"document_id": document_id}, include=["documents", "metadatas"])
    items = sorted(zip(res["documents"], res["metadatas"]), key=lambda x: x[1]["chunk_index"])[:limit]
    return [Hit(t, m["source"], m["document_id"], m["chunk_index"], None) for t, m in items]


def ping() -> int:
    return _col().count()