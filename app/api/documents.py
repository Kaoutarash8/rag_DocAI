import hashlib
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from app.api.schemas import DocumentOut, UploadOut
from app.core.config import get_settings
from app.core.security import rate_limit, require_api_key
from app.database import get_db
from app.models.document import Document, DocumentStatus
from app.services import processing, vectorstore
from app.services.files import ALLOWED_EXT, sanitize_filename, sniff_file_type
from app.services.ingestion import create_document

log = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"],
                   dependencies=[Depends(require_api_key)])


def _get_or_404(db: Session, document_id: int) -> Document:
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(404, "Document introuvable")
    return doc


@router.post("/upload", response_model=UploadOut, status_code=202,
             dependencies=[Depends(rate_limit)])
def upload_document(response: Response, file: UploadFile = File(...),
                    db: Session = Depends(get_db)):
    s = get_settings()
    filename = sanitize_filename(file.filename)
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(415, "Type non supporté (PDF, PNG, JPG, WEBP, BMP, TIFF)")

    tmp = s.upload_dir / f".tmp-{uuid.uuid4().hex}"
    digest, size, head = hashlib.sha256(), 0, b""
    try:
        with tmp.open("wb") as out:
            while chunk := file.file.read(1024 * 1024):
                head = head or chunk[:16]
                size += len(chunk)
                if size > s.max_upload_bytes:
                    raise HTTPException(413, f"Fichier trop volumineux (max {s.max_upload_mb} Mo)")
                digest.update(chunk)
                out.write(chunk)
        if size == 0:
            raise HTTPException(400, "Fichier vide")
        file_type = sniff_file_type(head, ext)
        if not file_type:
            raise HTTPException(415, "Le contenu ne correspond pas à l'extension du fichier")
        doc, created = create_document(db, tmp, filename, file_type, digest.hexdigest(), size)
    finally:
        tmp.unlink(missing_ok=True)

    if created:
        processing.submit(doc.id)
        message = "Document reçu, traitement en cours."
    else:
        response.status_code = 200
        message = "Ce document existe déjà (contenu identique)."
    return UploadOut.model_validate(doc, from_attributes=True).model_copy(
        update={"duplicate": not created, "message": message})


@router.get("", response_model=list[DocumentOut])
def list_documents(response: Response, db: Session = Depends(get_db),
                   limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0),
                   status: DocumentStatus | None = None):
    q = db.query(Document)
    if status:
        q = q.filter(Document.status == status)
    response.headers["X-Total-Count"] = str(q.count())
    return q.order_by(Document.created_at.desc(), Document.id.desc()).offset(offset).limit(limit).all()


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: int, db: Session = Depends(get_db)):
    return _get_or_404(db, document_id)


@router.get("/{document_id}/text", response_class=PlainTextResponse)
def get_document_text(document_id: int, db: Session = Depends(get_db)):
    doc = _get_or_404(db, document_id)
    if not doc.extracted_text_path or not Path(doc.extracted_text_path).exists():
        raise HTTPException(404, "Texte non disponible (document non indexé)")
    return Path(doc.extracted_text_path).read_text(encoding="utf-8")


@router.post("/{document_id}/reprocess", response_model=DocumentOut, status_code=202,
             dependencies=[Depends(rate_limit)])
def reprocess_document(document_id: int, db: Session = Depends(get_db)):
    doc = _get_or_404(db, document_id)
    if doc.status == DocumentStatus.PROCESSING:
        raise HTTPException(409, "Traitement déjà en cours")
    doc.status, doc.attempts, doc.error_message = DocumentStatus.PENDING, 0, None
    db.commit()
    db.refresh(doc)
    processing.submit(doc.id)
    return doc


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: int, db: Session = Depends(get_db)):
    doc = _get_or_404(db, document_id)
    if doc.status == DocumentStatus.PROCESSING:
        raise HTTPException(409, "Traitement en cours, réessayez plus tard")
    s = get_settings()
    vectorstore.delete_document(doc.id)
    (s.upload_dir / doc.stored_name).unlink(missing_ok=True)
    if doc.extracted_text_path:
        Path(doc.extracted_text_path).unlink(missing_ok=True)
    db.delete(doc)
    db.commit()
    return Response(status_code=204)
