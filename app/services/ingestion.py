"""Enregistrement d'un fichier déjà validé (partagé par l'API et les scripts)."""

import logging
import uuid
from pathlib import Path

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.document import Document, DocumentStatus

log = logging.getLogger(__name__)


def _existing(db: Session, owner_id: int | None, sha256: str):
    return db.query(Document).filter(Document.owner_id == owner_id, Document.sha256 == sha256)


def create_document(db: Session, tmp_path: Path, filename: str, file_type: str,
                    sha256: str, size: int, owner_id: int | None) -> tuple[Document, bool]:
    """Retourne (document, créé). Un contenu identique n'est pas ré-importé POUR LE MÊME PROPRIÉTAIRE ;
    un autre utilisateur qui envoie le même fichier obtient son propre document (isolation)."""
    existing = _existing(db, owner_id, sha256).first()
    if existing:
        tmp_path.unlink(missing_ok=True)
        return existing, False

    s = get_settings()
    stored = f"{uuid.uuid4().hex}{Path(filename).suffix.lower()}"
    final = s.upload_dir / stored
    tmp_path.replace(final)
    doc = Document(owner_id=owner_id, filename=filename, stored_name=stored, file_type=file_type,
                   size_bytes=size, sha256=sha256, status=DocumentStatus.PENDING)
    db.add(doc)
    try:
        db.commit()
    except IntegrityError:  # import concurrent du même fichier par le même propriétaire
        db.rollback()
        final.unlink(missing_ok=True)
        return _existing(db, owner_id, sha256).one(), False
    db.refresh(doc)
    return doc, True
