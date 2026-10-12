from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.document import DocumentStatus


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    filename: str
    file_type: str
    size_bytes: int
    status: DocumentStatus
    chunk_count: int
    error_message: str | None = None
    created_at: datetime
    processed_at: datetime | None = None


class UploadOut(DocumentOut):
    duplicate: bool = False
    message: str = ""


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)
    document_ids: list[int] | None = Field(default=None, max_length=100)


class SourceOut(BaseModel):
    source: str
    document_id: int
    chunk_index: int
    score: float | None = None
    snippet: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceOut]