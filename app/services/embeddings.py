"""Modèles d'embeddings interchangeables (MiniLM, bge-m3, multilingual-e5...), chargés à la demande.

Les modèles e5 exigent les préfixes "query: " / "passage: " ; bge-m3 n'en a pas besoin.
"""

import logging
import re
import threading
from dataclasses import dataclass

from app.core.config import get_settings

log = logging.getLogger(__name__)
_models: dict[str, object] = {}
_lock = threading.Lock()


@dataclass(frozen=True)
class ModelSpec:
    query_prefix: str = ""
    passage_prefix: str = ""
    max_seq_length: int | None = None   # None = valeur par défaut du modèle
    batch_size: int = 32


def spec_for(name: str) -> ModelSpec:
    n = name.lower()
    if re.search(r"(^|[-/])e5([-_]|$)", n):
        return ModelSpec("query: ", "passage: ", 512, 16)
    if "bge-m3" in n:  # 8192 par défaut : on borne pour maîtriser mémoire et latence sur CPU
        return ModelSpec(max_seq_length=1024, batch_size=8)
    return ModelSpec()


def get_model(name: str | None = None):
    """Chargé une seule fois par nom (paresseusement), partagé entre les requêtes."""
    name = name or get_settings().embedding_model
    with _lock:
        if name not in _models:
            from sentence_transformers import SentenceTransformer

            log.info("Chargement du modèle d'embeddings %s", name)
            model = SentenceTransformer(name)
            if spec_for(name).max_seq_length:
                model.max_seq_length = spec_for(name).max_seq_length
            _models[name] = model
    return _models[name]


def _encode(name: str, texts: list[str], prefix: str) -> list[list[float]]:
    spec = spec_for(name)
    vectors = get_model(name).encode(
        [prefix + t for t in texts], batch_size=spec.batch_size,
        normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_passages(name: str, texts: list[str]) -> list[list[float]]:
    return _encode(name, texts, spec_for(name).passage_prefix)


def embed_query(name: str, text: str) -> list[float]:
    return _encode(name, [text], spec_for(name).query_prefix)[0]
