"""Versionnage de l'index : nom de collection versionné et registre (modèle + paramètres)."""

import json
import logging

from app.core.config import get_settings
from app.database import session_scope
from app.models.index_collection import IndexCollection

log = logging.getLogger(__name__)


def versioned(name: str) -> str:
    """`documents` -> `documents_v3` quand INDEX_VERSION=v3."""
    return f"{name}_{get_settings().index_version}"


def register_collections(specs) -> None:
    """Enregistre (une fois) quel modèle/quels paramètres ont alimenté chaque collection."""
    version = get_settings().index_version
    try:
        with session_scope() as db:
            for sp in specs:
                if db.get(IndexCollection, sp.collection) is None:
                    db.add(IndexCollection(
                        name=sp.collection, index_version=version, embedding_model=sp.embedding_model,
                        chunk_strategy=sp.strategy, chunk_size=sp.size, chunk_overlap=sp.overlap,
                        params=json.dumps({"strategy": sp.strategy, "size": sp.size, "overlap": sp.overlap,
                                           "embedding_model": sp.embedding_model})))
    except Exception:  # le registre ne doit jamais faire échouer une ingestion
        log.warning("Registre des collections indisponible", exc_info=True)
