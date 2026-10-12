"""Enregistrement d'un fichier déjà validé (partagé par l'API et les scripts)."""

import logging
import uuid
from pathlib import Path

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.document import Document, DocumentStatus

log = logging.getLogger(__name__)


def create_document(db: Session, tmp_path: Path, filename: str, file_type: str,
                    sha256: str, size: int) -> tuple[Document, bool]:
    """Retourne (document, créé). Un fichier au contenu identique n'est pas ré-importé."""
    existing = db.query(Document).filter(Document.sha256 == sha256).first()
    if existing:
        tmp_path.unlink(missing_ok=True)
        return existing, False

    s = get_settings()
    stored = f"{uuid.uuid4().hex}{Path(filename).suffix.lower()}"
    final = s.upload_dir / stored
    tmp_path.replace(final)
    doc = Document(filename=filename, stored_name=stored, file_type=file_type,
                   size_bytes=size, sha256=sha256, status=DocumentStatus.PENDING)
    db.add(doc)
    try:
        db.commit()
    except IntegrityError:  # import concurrent du même fichier
        db.rollback()
        final.unlink(missing_ok=True)
        return db.query(Document).filter(Document.sha256 == sha256).one(), False
    db.refresh(doc)
    return doc, True
