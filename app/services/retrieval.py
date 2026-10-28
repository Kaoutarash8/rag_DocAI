"""Récupération : dense (Chroma) -> [BM25 + fusion RRF] -> [reranker cross-encoder] -> top_k.

Seuils : `min_score` (cosinus) filtre la liste dense AVANT fusion ; BM25 est filtré par couverture
des termes (`bm25_min_coverage`) ; le reranker peut avoir son propre seuil (`rerank_min_score`).
Quand l'utilisateur choisit des documents, aucun seuil n'est appliqué (on garde leurs meilleurs
passages, comme avant).
"""

import logging
import threading
import time
from collections.abc import Collection
from dataclasses import dataclass, field

from app.core.errors import ProfileError
from app.services import embeddings, reranker, vectorstore
from app.services.bm25 import BM25Index, tokenize
from app.services.fusion import rrf
from app.services.profiles import Profile
from app.services.vectorstore import Hit

log = logging.getLogger(__name__)


@dataclass
class _Lexical:
    version: int
    index: BM25Index
    texts: list[str]
    metas: list[dict]
    by_doc: dict[int, list[int]] = field(default_factory=dict)


_cache: dict[str, _Lexical] = {}
_lock = threading.Lock()


def _lexical(collection: str) -> _Lexical:
    """Index BM25 de la collection, reconstruit seulement si elle a changé (process unique)."""
    v = vectorstore.version(collection)
    with _lock:
        cached = _cache.get(collection)
        if cached and cached.version == v:
            return cached
        texts, metas = vectorstore.get_all(collection)
        by_doc: dict[int, list[int]] = {}
        for i, m in enumerate(metas):
            by_doc.setdefault(m["document_id"], []).append(i)
        lex = _Lexical(v, BM25Index([tokenize(t) for t in texts]), texts, metas, by_doc)
        _cache[collection] = lex
        return lex


def lexical_search(p: Profile, question: str, n: int,
                   document_ids: list[int] | None) -> list[Hit]:
    lex = _lexical(p.collection)
    allowed = None
    if document_ids:
        allowed = {i for d in document_ids for i in lex.by_doc.get(d, [])}
    ranked = lex.index.search(tokenize(question), n, allowed, p.bm25_min_coverage)
    hits = []
    for i, score in ranked:
        h = vectorstore.make_hit(lex.texts[i], lex.metas[i], None)
        h.bm25 = round(score, 4)
        hits.append(h)
    return hits


def fuse(dense: list[Hit], lexical: list[Hit], k: int) -> list[Hit]:
    by_key: dict = {h.key: h for h in dense}
    for h in lexical:
        known = by_key.get(h.key)
        if known:
            known.bm25 = h.bm25
        else:
            by_key[h.key] = h
    out = []
    for key, score in rrf([[h.key for h in dense], [h.key for h in lexical]], k):
        h = by_key[key]
        h.rrf, h.score, h.score_kind = score, round(score, 5), "rrf"
        out.append(h)
    return out


def retrieve(question: str, p: Profile, document_ids: list[int] | None = None,
             top_k: int | None = None, timings: dict | None = None, *,
             allowed_ids: Collection[int] | None) -> list[Hit]:
    """Retourne au plus `top_k` passages (défaut : celui du profil). `timings` reçoit les ms/étape.

    `allowed_ids` est OBLIGATOIRE (mot-clé) : les documents que l'appelant a le droit de lire,
    calculés en SQL (propriétaire + partages). Liste vide = aucun résultat, sans interroger l'index.
    `None` = aucun contrôle d'accès, réservé aux scripts hors API (benchmark).
    `document_ids` = sélection de l'utilisateur, toujours intersectée avec `allowed_ids`.
    """
    top_k = top_k or p.top_k
    t = timings if timings is not None else {}
    gated = not document_ids  # seuils de score seulement si l'utilisateur n'a pas choisi de documents

    scope: set[int] | None = None
    ids = list(document_ids) if document_ids else None
    if allowed_ids is not None:
        scope = set(allowed_ids)
        if document_ids:
            scope &= set(document_ids)
        if not scope:
            return []
        ids = sorted(scope)

    n = max(p.candidates, top_k) if (p.hybrid or p.rerank) else top_k

    t0 = time.perf_counter()
    qvec = embeddings.embed_query(p.embedding_model, question)
    t["embed_ms"] = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    dense = vectorstore.search(p.collection, qvec, n, ids)
    t["dense_ms"] = (time.perf_counter() - t0) * 1000
    if scope is not None:  # défense en profondeur : même si l'index renvoyait trop, rien ne sort du périmètre
        dense = [h for h in dense if h.document_id in scope]
    if not dense:
        _warn_if_not_indexed(p)
    cands = [h for h in dense if h.cosine >= p.min_score] if gated else dense

    if p.hybrid:
        t0 = time.perf_counter()
        lexical = lexical_search(p, question, n, ids)
        if scope is not None:
            lexical = [h for h in lexical if h.document_id in scope]
        cands = fuse(cands, lexical, p.rrf_k)
        t["bm25_fusion_ms"] = (time.perf_counter() - t0) * 1000

    if p.rerank and cands:
        t0 = time.perf_counter()
        cands = reranker.rerank(p.reranker_model, question, cands, p.rerank_top_n)
        t["rerank_ms"] = (time.perf_counter() - t0) * 1000
        if gated and p.rerank_min_score is not None:
            cands = [h for h in cands if h.rerank >= p.rerank_min_score]
    return cands[:top_k]


def _warn_if_not_indexed(p: Profile) -> None:
    """Profil vide alors que d'autres collections contiennent des documents : erreur explicite."""
    from app.services import profiles

    try:
        if vectorstore.count(p.collection) > 0:
            return
        others = [c for c in profiles.all_collections() if c != p.collection]
        existing = vectorstore._existing_names()
        if any(c in existing and vectorstore.count(c) > 0 for c in others):
            raise ProfileError(
                f"Le profil « {p.name} » n'est pas indexé : ajoutez-le à INDEX_PROFILES puis "
                "relancez `python -m scripts.reindex`.")
    except ProfileError:
        raise
    except Exception:  # le contrôle ne doit jamais casser une recherche
        log.debug("Contrôle d'indexation ignoré", exc_info=True)
