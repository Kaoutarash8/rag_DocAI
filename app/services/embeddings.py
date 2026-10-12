import logging
import threading

from app.core.config import get_settings

log = logging.getLogger(__name__)
_model = None
_lock = threading.Lock()


def get_model():
    """Chargé une seule fois (paresseusement), partagé entre les requêtes."""
    global _model
    with _lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer

            name = get_settings().embedding_model
            log.info("Chargement du modèle d'embeddings %s", name)
            _model = SentenceTransformer(name)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    vectors = get_model().encode(
        texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
    )
    return vectors.tolist()
