import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import get_settings
from app.models.document import DocumentStatus
from app.models.user import UserRole

_EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,189}\.[^@\s]{2,}$")


def normalize_email(v: str) -> str:
    v = v.strip().lower()
    if len(v) > 254 or not _EMAIL.match(v):
        raise ValueError("Adresse e-mail invalide")
    return v


class LoginIn(BaseModel):
    email: str
    password: str = Field(min_length=1, max_length=128)
    _email = field_validator("email")(normalize_email)


class RegisterIn(LoginIn):
    @field_validator("password")
    @classmethod
    def _policy(cls, v: str) -> str:
        n = get_settings().password_min_length
        if len(v) < n:
            raise ValueError(f"Le mot de passe doit contenir au moins {n} caractères")
        return v


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    email: str
    role: UserRole
    is_active: bool = True
    created_at: datetime | None = None


class AdminUserCreate(RegisterIn):
    role: UserRole = UserRole.USER


class AdminUserPatch(BaseModel):
    role: UserRole | None = None
    is_active: bool | None = None


class ShareIn(BaseModel):
    email: str
    _email = field_validator("email")(normalize_email)


class ShareOut(BaseModel):
    user_id: int
    email: str
    permission: str = "read"


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    owner_id: int | None = None
    access: Literal["owner", "shared", "admin"] = "owner"   # droit de l'appelant sur ce document
    filename: str
    file_type: str
    size_bytes: int
    status: DocumentStatus
    chunk_count: int
    pages_total: int = 0
    pages_done: int = 0
    error_message: str | None = None
    created_at: datetime
    processed_at: datetime | None = None


class UploadOut(DocumentOut):
    duplicate: bool = False
    message: str = ""


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(None, ge=1, le=20)   # None = valeur du profil
    profile: str | None = Field(None, max_length=64)  # None = DEFAULT_PROFILE
    document_ids: list[int] | None = Field(default=None, max_length=100)
    scope: Literal["accessible", "all"] = "accessible"   # "all" : administrateurs seulement
    history: list[HistoryTurn] = Field(default_factory=list, max_length=12)   # messages précédents de la conversation


class SourceOut(BaseModel):
    ref: int = 0            # numéro de l'extrait : c'est le [n] cité dans la réponse (PAS un numéro de page)
    cited: bool = False     # l'extrait est cité dans la réponse
    source: str
    document_id: int
    chunk_index: int
    page: int | None = None
    page_end: int | None = None
    section: str = ""
    kind: str = "text"
    score: float | None = None
    score_kind: str = "cosine"   # cosine | rrf | rerank
    snippet: str


class AskResponse(BaseModel):
    question: str
    answer: str
    profile: str = ""
    search_query: str | None = None   # requête réellement utilisée pour chercher (après réécriture)
    sources: list[SourceOut]

class SummaryPart(BaseModel):
    title: str
    pages: str = ""
    summary: str


class SummaryOut(BaseModel):
    document_id: int
    summary: str
    parts: list[SummaryPart]
    truncated: bool = False   # document trop long pour le plafond d'appels LLM : début seulement
    llm_calls: int = 0
    generated_at: str = ""
