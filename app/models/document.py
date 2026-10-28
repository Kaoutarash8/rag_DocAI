import enum
from datetime import datetime, timezone

from sqlalchemy import (BigInteger, Column, DateTime, Enum, ForeignKey, Integer, String, Text,
                        UniqueConstraint)

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
    # Le contenu n'est unique que PAR propriétaire : une unicité globale révélerait à B qu'A possède
    # le même fichier (et lierait leurs documents).
    __table_args__ = (UniqueConstraint("owner_id", "sha256", name="uq_documents_owner_sha256"),)

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="RESTRICT"), nullable=True, index=True)
    filename = Column(String(255), nullable=False)          # nom d'origine (nettoyé)
    stored_name = Column(String(80), nullable=False, unique=True)  # uuid sur disque
    file_type = Column(String(10), nullable=False)          # pdf | image | docx | pptx | xlsx | txt | md
    size_bytes = Column(BigInteger, nullable=False)
    sha256 = Column(String(64), nullable=False, index=True)  # déduplication (par propriétaire)
    status = Column(
        Enum(DocumentStatus, native_enum=False, length=20,
             values_callable=lambda e: [m.value for m in e]),
        default=DocumentStatus.PENDING, nullable=False, index=True,
    )
    chunk_count = Column(Integer, nullable=False, default=0)
    pages_total = Column(Integer, nullable=False, default=0)   # unités à traiter (pages, diapos...)
    pages_done = Column(Integer, nullable=False, default=0)    # progression du traitement
    attempts = Column(Integer, nullable=False, default=0)
    index_version = Column(String(20), nullable=True)           # version de l'index qui contient ce document
    extracted_text_path = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_now, nullable=False)
    processed_at = Column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"<Document id={self.id} {self.filename!r} {self.status}>"
