import hashlib
import logging
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from app.api.deps import current_user, rate_limit
from app.api.schemas import DocumentOut, ShareIn, ShareOut, SummaryOut, UploadOut
from app.core.config import get_settings
from app.database import get_db
from app.models.document import Document, DocumentStatus
from app.models.user import DocumentShare, User
from app.services import access, processing, summary, vectorstore
from app.services.files import (ALLOWED_EXT, SUPPORTED_LABEL, sanitize_filename,
                                sniff_file_type)
from app.services.ingestion import create_document

log = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"], dependencies=[Depends(current_user)])


def _out(doc: Document, user: User, shared: set[int]) -> DocumentOut:
    out = DocumentOut.model_validate(doc, from_attributes=True)
    return out.model_copy(update={"access": access.access_label(user, doc, shared)})


@router.post("/upload", response_model=UploadOut, status_code=202, dependencies=[Depends(rate_limit)])
def upload_document(response: Response, file: UploadFile = File(...),
                    db: Session = Depends(get_db), user: User = Depends(current_user)):
    s = get_settings()
    filename = sanitize_filename(file.filename)
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(415, f"Type non supporté ({SUPPORTED_LABEL})")

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
        file_type = sniff_file_type(head, ext, tmp)
        if not file_type:
            raise HTTPException(415, "Le contenu ne correspond pas à l'extension du fichier")
        doc, created = create_document(db, tmp, filename, file_type, digest.hexdigest(), size, user.id)
    finally:
        tmp.unlink(missing_ok=True)

    if created:
        processing.submit(doc.id)
        message = "Document reçu, traitement en cours."
    else:
        response.status_code = 200
        message = "Ce document existe déjà (contenu identique)."
    return UploadOut.model_validate(doc, from_attributes=True).model_copy(
        update={"duplicate": not created, "message": message, "access": "owner"})


@router.get("", response_model=list[DocumentOut])
def list_documents(response: Response, db: Session = Depends(get_db), user: User = Depends(current_user),
                   limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0),
                   status: DocumentStatus | None = None,
                   scope: Literal["accessible", "mine", "all"] = "accessible"):
    """Vos documents et ceux partagés avec vous. `scope=mine` : les vôtres ; `scope=all` : admin."""
    q = access.accessible_query(db, user, scope)
    if status:
        q = q.filter(Document.status == status)
    response.headers["X-Total-Count"] = str(q.count())
    docs = q.order_by(Document.created_at.desc(), Document.id.desc()).offset(offset).limit(limit).all()
    shared = access.shared_ids(db, user)
    return [_out(d, user, shared) for d in docs]


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    doc = access.get_document(db, user, document_id)
    return _out(doc, user, access.shared_ids(db, user))


@router.get("/{document_id}/text", response_class=PlainTextResponse)
def get_document_text(document_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    doc = access.get_document(db, user, document_id)
    if not doc.extracted_text_path or not Path(doc.extracted_text_path).exists():
        raise HTTPException(404, "Texte non disponible (document non indexé)")
    return Path(doc.extracted_text_path).read_text(encoding="utf-8")


@router.post("/{document_id}/summary", response_model=SummaryOut, dependencies=[Depends(rate_limit)])
def summarize_document(document_id: int, refresh: bool = False, db: Session = Depends(get_db),
                       user: User = Depends(current_user)):
    """Résumé hiérarchique par sections (map-reduce). Mis en cache ; `refresh=true` le régénère.

    Peut prendre du temps sur un gros document (un appel LLM par bloc de ~1500 mots).
    Les lecteurs d'un partage peuvent le demander ; seul le propriétaire (ou un admin) peut le régénérer."""
    doc = access.get_document(db, user, document_id, manage=refresh)
    if doc.status != DocumentStatus.INDEXED:
        raise HTTPException(409, "Le document n'est pas encore indexé")
    try:
        result = summary.summarize_document(doc.id, doc.filename, refresh)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return {"document_id": doc.id, **result}


@router.post("/{document_id}/reprocess", response_model=DocumentOut, status_code=202,
             dependencies=[Depends(rate_limit)])
def reprocess_document(document_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    doc = access.get_document(db, user, document_id, manage=True)
    if doc.status == DocumentStatus.PROCESSING:
        raise HTTPException(409, "Traitement déjà en cours")
    doc.status, doc.attempts, doc.error_message = DocumentStatus.PENDING, 0, None
    db.commit()
    db.refresh(doc)
    processing.submit(doc.id)
    return _out(doc, user, access.shared_ids(db, user))


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    doc = access.get_document(db, user, document_id, manage=True)
    if doc.status == DocumentStatus.PROCESSING:
        raise HTTPException(409, "Traitement en cours, réessayez plus tard")
    s = get_settings()
    vectorstore.delete_document(doc.id)
    summary.cleanup(doc.id)
    (s.upload_dir / doc.stored_name).unlink(missing_ok=True)
    if doc.extracted_text_path:
        Path(doc.extracted_text_path).unlink(missing_ok=True)
    db.query(DocumentShare).filter(DocumentShare.document_id == doc.id).delete()
    db.delete(doc)
    db.commit()
    return Response(status_code=204)


# ---------- partage en lecture ----------

@router.get("/{document_id}/shares", response_model=list[ShareOut])
def list_shares(document_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    doc = access.get_document(db, user, document_id, manage=True)
    rows = (db.query(DocumentShare, User.email).join(User, User.id == DocumentShare.user_id)
            .filter(DocumentShare.document_id == doc.id).order_by(DocumentShare.id).all())
    return [ShareOut(user_id=sh.user_id, email=email, permission=sh.permission) for sh, email in rows]


@router.post("/{document_id}/shares", response_model=ShareOut, status_code=201)
def share_document(document_id: int, body: ShareIn, db: Session = Depends(get_db),
                   user: User = Depends(current_user)):
    """Partage en lecture avec un utilisateur existant. Effet immédiat, y compris sur la recherche."""
    doc = access.get_document(db, user, document_id, manage=True)
    target = db.query(User).filter(User.email == body.email, User.is_active.is_(True)).first()
    if not target:
        raise HTTPException(404, "Utilisateur introuvable")
    if target.id == doc.owner_id:
        raise HTTPException(400, "Le propriétaire a déjà accès à ce document")
    share = db.query(DocumentShare).filter(
        DocumentShare.document_id == doc.id, DocumentShare.user_id == target.id).first()
    if not share:
        share = DocumentShare(document_id=doc.id, user_id=target.id, permission="read")
        db.add(share)
        db.commit()
    return ShareOut(user_id=target.id, email=target.email, permission=share.permission)


@router.delete("/{document_id}/shares/{user_id}", status_code=204)
def revoke_share(document_id: int, user_id: int, db: Session = Depends(get_db),
                 user: User = Depends(current_user)):
    doc = access.get_document(db, user, document_id, manage=True)
    n = db.query(DocumentShare).filter(
        DocumentShare.document_id == doc.id, DocumentShare.user_id == user_id).delete()
    db.commit()
    if not n:
        raise HTTPException(404, "Partage introuvable")
    return Response(status_code=204)
