"""Profils de recherche : une combinaison (chunking, embedding, récupération) nommée.

Chaque profil a sa propre collection vectorielle et ses propres `top_k` / `min_score` (les
cosinus de MiniLM, e5 et bge-m3 n'ont pas la même échelle : un seuil unique n'a pas de sens).
Plusieurs profils peuvent partager une collection (ex. dense / hybride / hybride+rerank sur le
même index) à condition d'avoir le même chunking et le même modèle d'embedding.

Surcharge par l'environnement : RAG_PROFILES_JSON='{"mon-profil": {"base": "e5-hybrid", "top_k": 8}}'
(`base` = profil existant dont on hérite ; sans `base`, `collection` et `embedding_model` sont requis).
"""

import json
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.core.config import get_settings
from app.core.errors import ProfileError
from app.services.versioning import versioned


class Profile(BaseModel):
    name: str
    collection: str
    embedding_model: str
    chunk_strategy: Literal["fixed", "page", "section", "sentence"] = "fixed"
    chunk_size: int = Field(100, ge=20)       # mots
    chunk_overlap: int = Field(20, ge=0)
    top_k: int = Field(5, ge=1, le=20)
    min_score: float = Field(0.15, ge=-1.0, le=1.0)   # cosinus minimal (liste dense)
    hybrid: bool = False                      # dense + BM25 fusionnés par RRF
    rrf_k: int = Field(60, ge=1)
    bm25_min_coverage: float = Field(0.5, ge=0.0, le=1.0)
    candidates: int = Field(20, ge=1, le=100)  # candidats par retriever (hybride / rerank)
    rerank: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_n: int = Field(20, ge=1, le=100)
    rerank_min_score: float | None = None     # probabilité de pertinence minimale (0..1)

    @model_validator(mode="after")
    def _check(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"{self.name} : chunk_overlap doit être < chunk_size")
        return self


@dataclass(frozen=True)
class IndexSpec:
    """Ce qu'il faut indexer : un jeu de chunks + embeddings par collection."""
    collection: str
    embedding_model: str
    strategy: str
    size: int
    overlap: int


def index_spec(p: Profile) -> IndexSpec:
    return IndexSpec(p.collection, p.embedding_model, p.chunk_strategy, p.chunk_size, p.chunk_overlap)


# Valeurs de min_score = points de départ à calibrer avec scripts/benchmark.py
# (les cosinus e5 sont comprimés vers le haut, ceux de bge-m3 et MiniLM plus étalés).
E5 = "intfloat/multilingual-e5-base"
BGE = "BAAI/bge-m3"


def _builtin() -> dict[str, Profile]:
    s = get_settings()
    base = {"top_k": 5}
    e5 = {**base, "embedding_model": E5, "chunk_size": 250, "chunk_overlap": 50, "min_score": 0.75}
    profiles = [
        Profile(name="baseline", collection=versioned(s.vectorstore_collection), embedding_model=s.embedding_model,
                chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap,
                top_k=s.rag_top_k, min_score=s.rag_min_score),
        # Comparaison des modèles (chunks fixes de 250 mots ; MiniLM tronque à ~128 tokens : témoin)
        Profile(name="minilm-250", collection=versioned("docs_minilm_fixed250"), chunk_size=250, chunk_overlap=50,
                embedding_model=s.embedding_model, top_k=5, min_score=s.rag_min_score),
        Profile(name="e5-fixed", collection=versioned("docs_e5_fixed"), chunk_strategy="fixed", **e5),
        Profile(name="bge-fixed", collection=versioned("docs_bge_fixed"), embedding_model=BGE, chunk_size=250,
                chunk_overlap=50, top_k=5, min_score=0.35),
        # Comparaison des stratégies de chunking (modèle e5 fixé)
        Profile(name="e5-sentence", collection=versioned("docs_e5_sentence"), chunk_strategy="sentence", **e5),
        Profile(name="e5-section", collection=versioned("docs_e5_section"), chunk_strategy="section", **e5),
        Profile(name="e5-page", collection=versioned("docs_e5_page"), chunk_strategy="page", **{**e5, "chunk_size": 300}),
        # Récupération : même index que e5-fixed, seule la recherche change
        Profile(name="e5-hybrid", collection=versioned("docs_e5_fixed"), chunk_strategy="fixed", hybrid=True, **e5),
        Profile(name="e5-hybrid-rerank", collection=versioned("docs_e5_fixed"), chunk_strategy="fixed",
                hybrid=True, rerank=True, rerank_min_score=None, **e5),
    ]
    return {p.name: p for p in profiles}


def all_profiles() -> dict[str, Profile]:
    profs = _builtin()
    raw = get_settings().rag_profiles_json.strip()
    if raw:
        try:
            custom = json.loads(raw)
            assert isinstance(custom, dict)
        except (ValueError, AssertionError) as e:
            raise ValueError(f"RAG_PROFILES_JSON invalide : {e}") from e
        for name, over in custom.items():
            over = dict(over)
            parent = over.pop("base", name if name in profs else None)
            data = profs[parent].model_dump() if parent else {}
            data.update(over, name=name)
            profs[name] = Profile(**data)
    _check_shared_collections(profs)
    return profs


def _check_shared_collections(profs: dict[str, Profile]) -> None:
    seen: dict[str, tuple[str, IndexSpec]] = {}
    for p in profs.values():
        spec = index_spec(p)
        if p.collection in seen and seen[p.collection][1] != spec:
            raise ValueError(f"Les profils {seen[p.collection][0]} et {p.name} partagent la collection "
                             f"{p.collection} avec un chunking/embedding différent")
        seen.setdefault(p.collection, (p.name, spec))


def get_profile(name: str | None = None) -> Profile:
    profs = all_profiles()
    name = name or get_settings().default_profile
    if name not in profs:
        raise ProfileError(f"Profil inconnu : {name} (disponibles : {', '.join(sorted(profs))})")
    return profs[name]


def index_specs() -> list[IndexSpec]:
    """Collections alimentées à l'ingestion (la première = référence pour `chunk_count`)."""
    profs = all_profiles()
    raw = get_settings().index_profiles.strip()
    names = list(profs) if raw == "all" else [n.strip() for n in raw.split(",") if n.strip()]
    default = get_settings().default_profile
    names = [default] + [n for n in names if n != default]  # le défaut est indexé en premier
    unknown = [n for n in names if n not in profs]
    if unknown:
        raise ValueError(f"INDEX_PROFILES : profil(s) inconnu(s) {unknown}")
    specs: dict[str, IndexSpec] = {}
    for n in names:
        specs.setdefault(profs[n].collection, index_spec(profs[n]))
    return list(specs.values())


def all_collections() -> list[str]:
    return sorted({p.collection for p in all_profiles().values()})
