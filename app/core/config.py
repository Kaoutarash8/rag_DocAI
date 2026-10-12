"""Configuration centralisée, validée au démarrage (pydantic-settings)."""

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

    # --- Sécurité ---
    api_key: str = ""  # vide = authentification désactivée (dev)
    rate_limit_per_minute: int = 60  # 0 = désactivé

    # --- Fichiers ---
    data_dir: Path = Path("data")
    max_upload_mb: int = Field(25, ge=1, le=500)
    max_pdf_pages: int = Field(100, ge=1)

    # --- OCR ---
    ocr_lang: str = "fra"  # fra | eng | fra+eng
    pdf_render_dpi: int = Field(200, ge=72, le=400)
    min_chars_per_page: int = 20  # sous ce seuil, la page est passée en OCR

    # --- Indexation ---
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    # Ce modèle tronque à ~128 tokens : des chunks courts sont indispensables.
    chunk_size: int = Field(100, ge=20)  # mots
    chunk_overlap: int = Field(20, ge=0)
    vectorstore_collection: str = "documents_v2"
    worker_threads: int = Field(1, ge=1, le=8)
    max_attempts: int = Field(3, ge=1)
    warmup_models: bool = False

    # --- RAG / LLM ---
    rag_top_k: int = Field(5, ge=1, le=20)
    rag_min_score: float = Field(0.15, ge=-1.0, le=1.0)  # similarité cosinus
    llm_provider: Literal["ollama", "groq", "openai"] = "ollama"
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
        if self.env == "production" and not self.api_key:
            raise ValueError("API_KEY est obligatoire quand ENV=production")
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
