"""Configuration centralisée, validée au démarrage (pydantic-settings)."""

import secrets
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Général ---
    env: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"
    database_url: str = "postgresql://postgres:postgres@localhost:5432/mon_projet"
    auto_create_tables: bool = False  # dev/tests uniquement ; en prod : Alembic
    cors_origins: str = ""  # liste séparée par des virgules ; vide = pas de CORS

    # --- Sécurité : comptes, JWT, limitation de débit ---
    jwt_secret: str = ""  # >= 32 caractères en production (ex. `openssl rand -hex 32`)
    jwt_algorithm: Literal["HS256", "HS384", "HS512"] = "HS256"
    access_token_minutes: int = Field(60, ge=1, le=60 * 24 * 7)
    allow_registration: bool = True  # False = seul un admin crée des comptes (POST /admin/users)
    password_min_length: int = Field(10, ge=8, le=128)
    password_scrypt_n: int = Field(2**15, ge=2**4)  # coût du hachage ; valeur basse réservée aux tests
    rate_limit_per_minute: int = 60  # par utilisateur ; 0 = désactivé
    auth_rate_limit_per_minute: int = 10  # login/register : par IP et par e-mail ; 0 = désactivé
    redis_url: str = ""  # vide = limiteur en mémoire, pas de verrous ni de versions partagées

    # --- Exploitation ---
    queue_backend: Literal["thread", "celery"] = "thread"  # celery : worker(s) séparés + Redis
    chroma_host: str = ""  # vide = ChromaDB embarqué (1 seul process) ; sinon serveur Chroma
    chroma_port: int = 8000
    index_version: str = Field("v2", pattern=r"^[A-Za-z0-9]{1,12}$")  # suffixe des collections
    log_format: Literal["text", "json"] = "text"
    metrics_token: str = ""  # si défini, GET /metrics exige « Authorization: Bearer <token> »
    worker_metrics_port: int = 9101

    # --- Fichiers ---
    data_dir: Path = Path("data")
    max_upload_mb: int = Field(200, ge=1, le=2000)
    max_pdf_pages: int = Field(3000, ge=1)
    max_sheet_rows: int = Field(200_000, ge=1)  # lignes lues par feuille Excel

    # --- OCR ---
    ocr_lang: str = "fra"  # fra | eng | fra+eng
    pdf_render_dpi: int = Field(200, ge=72, le=400)
    min_chars_per_page: int = 20  # sous ce seuil, la page est passée en OCR
    extract_tables: bool = True   # tableaux des PDF natifs -> Markdown

    # --- Figures (VLM) : désactivé par défaut ---
    vlm_provider: Literal["none", "ollama", "openai"] = "none"
    vlm_model: str = "qwen2.5vl:7b"  # ollama ; ex. gpt-4o-mini pour openai
    vlm_timeout_s: float = 90.0
    vlm_dpi: int = Field(150, ge=72, le=300)
    vlm_max_figures: int = Field(40, ge=1)  # plafond par document (coût / durée)

    # --- Indexation ---
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    # Ce modèle tronque à ~128 tokens : des chunks courts sont indispensables.
    chunk_size: int = Field(100, ge=20)  # mots
    chunk_overlap: int = Field(20, ge=0)
    vectorstore_collection: str = "documents"  # nom logique ; la collection réelle = <nom>_<index_version>
    page_batch_size: int = Field(20, ge=1)  # pages traitées (chunk+embed+index) par lot
    worker_threads: int = Field(1, ge=1, le=8)
    max_attempts: int = Field(3, ge=1)
    warmup_models: bool = False

    # --- Profils de recherche (stratégie de chunking + embedding + récupération) ---
    default_profile: str = "baseline"
    # Réécriture de la question en requête de recherche autonome (suivis « et pour les femmes ? », « à quelle page… »)
    query_rewrite: Literal["auto", "always", "off"] = "auto"
    index_profiles: str = "baseline"   # profils alimentés à l'ingestion : "a,b,c" ou "all"
    rag_profiles_json: str = ""        # surcharge/ajout de profils : {"nom": {"champ": valeur}}

    # --- Résumé hiérarchique ---
    summary_block_words: int = Field(1500, ge=200)   # taille d'un bloc résumé en une passe
    summary_max_llm_calls: int = Field(40, ge=4)     # plafond d'appels LLM par résumé

    # --- RAG / LLM ---
    rag_top_k: int = Field(5, ge=1, le=20)
    rag_min_score: float = Field(0.15, ge=-1.0, le=1.0)  # similarité cosinus
    llm_provider: Literal["ollama", "groq", "openai", "mock"] = "ollama"  # mock : tests de charge / CI
    llm_mock_delay_ms: int = Field(0, ge=0)
    llm_timeout_s: float = 60.0
    llm_max_tokens: int = 800
    ollama_model: str = "llama3"
    ollama_base_url: str = "http://localhost:11434"
    groq_model: str = "llama-3.1-8b-instant"
    groq_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str = ""

    @model_validator(mode="after")
    def _check(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP doit être inférieur à CHUNK_SIZE")
        if not self.jwt_secret:
            if self.env == "production":
                raise ValueError("JWT_SECRET est obligatoire quand ENV=production")
            # dev/tests : secret éphémère (les jetons ne survivent pas à un redémarrage)
            self.jwt_secret = secrets.token_urlsafe(48)
        if self.env == "production" and len(self.jwt_secret) < 32:
            raise ValueError("JWT_SECRET doit contenir au moins 32 caractères")
        if self.queue_backend == "celery":
            if not self.redis_url:
                raise ValueError("QUEUE_BACKEND=celery nécessite REDIS_URL")
            if not self.chroma_host:
                raise ValueError("Plusieurs processus (API + workers) exigent un serveur Chroma : "
                                 "définissez CHROMA_HOST (ChromaDB embarqué = un seul process)")
        return self

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def text_dir(self) -> Path:
        return self.data_dir / "text"

    @property
    def vectorstore_dir(self) -> Path:
        return self.data_dir / "vectorstore"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    for d in (s.upload_dir, s.text_dir, s.vectorstore_dir):
        d.mkdir(parents=True, exist_ok=True)
    return s
