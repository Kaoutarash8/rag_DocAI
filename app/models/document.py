import enum
from datetime import datetime, timezone

from sqlalchemy import BigInteger, Column, DateTime, Enum, Integer, String, Text

from app.database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class DocumentStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)          # nom d'origine (nettoyé)
    stored_name = Column(String(80), nullable=False, unique=True)  # uuid sur disque
    file_type = Column(String(10), nullable=False)          # pdf | image
    size_bytes = Column(BigInteger, nullable=False)
    sha256 = Column(String(64), nullable=False, unique=True, index=True)  # déduplication
    status = Column(
        Enum(DocumentStatus, native_enum=False, length=20,
             values_callable=lambda e: [m.value for m in e]),
        default=DocumentStatus.PENDING, nullable=False, index=True,
    )
    chunk_count = Column(Integer, nullable=False, default=0)
    attempts = Column(Integer, nullable=False, default=0)
    extracted_text_path = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_now, nullable=False)
    processed_at = Column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"<Document id={self.id} {self.filename!r} {self.status}>"
