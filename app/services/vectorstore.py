"""Accès ChromaDB (similarité cosinus) : une collection par profil d'indexation."""

import logging
import threading
import time
from dataclasses import dataclass

from app.core.config import get_settings
from app.services.content import Chunk

log = logging.getLogger(__name__)
_client = None
_lock = threading.Lock()
_versions: dict[str, int] = {}   # incrémenté à chaque écriture : invalide l'index BM25 en cache


@dataclass
class Hit:
    text: str
    source: str
    document_id: int
    chunk_index: int
    score: float | None  # score final (voir score_kind) ; None = non applicable
    page: int | None = None
    section: str = ""
    kind: str = "text"
    page_end: int | None = None
    score_kind: str = "cosine"   # cosine | rrf | rerank
    cosine: float | None = None
    bm25: float | None = None
    rrf: float | None = None
    rerank: float | None = None

    def __post_init__(self) -> None:
        if self.cosine is None and self.score_kind == "cosine":
            self.cosine = self.score

    @property
    def key(self) -> tuple[int, int]:
        return (self.document_id, self.chunk_index)


def _get_client():
    global _client
    with _lock:
        if _client is None:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            s = get_settings()
            cfg = ChromaSettings(anonymized_telemetry=False)
            if s.chroma_host:  # serveur partagé : indispensable dès qu'il y a plusieurs process (API + workers)
                _client = chromadb.HttpClient(host=s.chroma_host, port=s.chroma_port, settings=cfg)
            else:
                _client = chromadb.PersistentClient(path=str(s.vectorstore_dir), settings=cfg)
    return _client


def _col(name: str):
    return _get_client().get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def version(collection: str) -> int:
    """Compteur d'écritures de la collection (invalide l'index BM25 en cache).

    Avec Redis il est partagé : un worker qui indexe invalide le cache de TOUTES les instances API.
    """
    from app.core.redis import get_redis

    client = get_redis()
    if client is not None:
        try:
            return int(client.get(f"vsver:{collection}") or 0)
        except Exception:
            return time.time_ns()  # Redis KO : valeur changeante = cache toujours reconstruit (correct, plus lent)
    return _versions.get(collection, 0)


def _bump(collection: str) -> None:
    _versions[collection] = _versions.get(collection, 0) + 1
    from app.core.redis import get_redis

    client = get_redis()
    if client is not None:
        try:
            client.incr(f"vsver:{collection}")
        except Exception:
            log.warning("Version d'index non publiée dans Redis", exc_info=True)


def _metadata(document_id: int, filename: str, c: Chunk) -> dict:
    """Chroma n'accepte ni None ni objets : les champs absents sont omis."""
    meta = {"document_id": document_id, "source": filename, "chunk_index": c.index, "kind": c.kind}
    if c.page is not None:
        meta["page"] = c.page
    if c.page_end is not None:
        meta["page_end"] = c.page_end
    if c.section:
        meta["section"] = c.section[:500]
    return meta


def make_hit(text: str, meta: dict, score: float | None, kind: str = "cosine") -> Hit:
    return Hit(text, meta["source"], meta["document_id"], meta["chunk_index"], score,
               meta.get("page"), meta.get("section", ""), meta.get("kind", "text"),
               meta.get("page_end"), kind)


def index_chunks(collection: str, document_id: int, filename: str, chunks: list[Chunk],
                 embeddings: list[list[float]]) -> None:
    """Ajoute un lot de chunks (upsert idempotent). Appelé une fois par lot de pages."""
    if not chunks:
        return
    _col(collection).upsert(
        ids=[f"{document_id}::{c.index}" for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[_metadata(document_id, filename, c) for c in chunks],
    )
    _bump(collection)


def _existing_names() -> set[str]:
    return {getattr(c, "name", c) for c in _get_client().list_collections()}


def delete_document(document_id: int) -> None:
    """Supprime le document de TOUTES les collections de profils (même non indexées ensuite)."""
    from app.services import profiles

    existing = _existing_names()
    for name in profiles.all_collections():
        if name in existing:
            _col(name).delete(where={"document_id": document_id})
            _bump(name)


def search(collection: str, query_embedding: list[float], top_k: int,
           document_ids: list[int] | None = None) -> list[Hit]:
    kwargs = {"query_embeddings": [query_embedding], "n_results": top_k}
    if document_ids:
        kwargs["where"] = {"document_id": {"$in": document_ids}}
    res = _col(collection).query(**kwargs)
    return [make_hit(t, m, round(1.0 - d, 4))
            for t, m, d in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]


def get_all(collection: str) -> tuple[list[str], list[dict]]:
    """(textes, métadonnées) de toute la collection, dans un ordre stable : sert à bâtir BM25."""
    res = _col(collection).get(include=["documents", "metadatas"])
    items = sorted(zip(res["documents"], res["metadatas"]),
                   key=lambda x: (x[1]["document_id"], x[1]["chunk_index"]))
    return [t for t, _ in items], [m for _, m in items]


def count(collection: str) -> int:
    return _col(collection).count()


def ping() -> int:
    """Nombre total de passages (toutes collections de profils existantes)."""
    from app.services import profiles

    existing = _existing_names()
    return sum(_col(n).count() for n in profiles.all_collections() if n in existing)
