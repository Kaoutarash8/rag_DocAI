"""Reranking par cross-encoder (ex. BAAI/bge-reranker-v2-m3) : lit (question, passage) ensemble."""

import threading

_models: dict[str, object] = {}
_lock = threading.Lock()


def get_model(name: str):
    with _lock:
        if name not in _models:
            from sentence_transformers import CrossEncoder

            _models[name] = CrossEncoder(name, max_length=512)
    return _models[name]


def rerank(name: str, query: str, hits: list, top_n: int) -> list:
    """Reclasse les `top_n` premiers candidats ; score = probabilité de pertinence (0..1)."""
    cands = hits[:top_n]
    if not cands:
        return []
    pairs = [(query, f"{h.section}\n{h.text}" if h.section else h.text) for h in cands]
    scores = get_model(name).predict(pairs, batch_size=16, show_progress_bar=False)
    for h, s in zip(cands, scores):
        h.rerank = float(s)
        h.score, h.score_kind = h.rerank, "rerank"
    return sorted(cands, key=lambda h: -h.rerank)
