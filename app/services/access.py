"""Droits d'accès aux documents : propriétaire, partage en lecture, rôle admin.

Règles :
- propriétaire : lit, modifie (reprocess), supprime, partage ;
- destinataire d'un partage : lit (liste, texte, résumé, questions) ;
- admin : lit et gère tout via /documents ; sa recherche /ask reste limitée à ses documents
  (+ partagés) sauf `scope="all"` explicite.
Un document inaccessible répond 404 (jamais 403) : on ne révèle pas son existence.
"""

from fastapi import HTTPException
from sqlalchemy import or_, select
from sqlalchemy.orm import Query, Session

from app.models.document import Document
from app.models.user import DocumentShare, User, UserRole


def is_admin(user: User) -> bool:
    return user.role == UserRole.ADMIN


def _shared_with(user: User):
    return select(DocumentShare.document_id).where(DocumentShare.user_id == user.id)


def accessible_query(db: Session, user: User, scope: str = "accessible") -> Query:
    """Documents lisibles par `user`. scope : accessible (propres + partagés) | mine | all (admin)."""
    q = db.query(Document)
    if scope == "all":
        if not is_admin(user):
            raise HTTPException(403, "Réservé aux administrateurs")
        return q
    if scope == "mine":
        return q.filter(Document.owner_id == user.id)
    return q.filter(or_(Document.owner_id == user.id, Document.id.in_(_shared_with(user))))


def accessible_ids(db: Session, user: User, scope: str = "accessible") -> list[int]:
    """Identifiants dont la recherche (vectorielle ET BM25) a le droit de renvoyer des passages."""
    return [i for (i,) in accessible_query(db, user, scope).with_entities(Document.id)]


def shared_ids(db: Session, user: User) -> set[int]:
    return set(db.execute(_shared_with(user)).scalars())


def get_document(db: Session, user: User, document_id: int, *, manage: bool = False) -> Document:
    doc = db.get(Document, document_id)
    if doc is None:
        raise HTTPException(404, "Document introuvable")
    owner = doc.owner_id == user.id
    admin = is_admin(user)
    if not (owner or admin):
        shared = db.query(DocumentShare.id).filter(
            DocumentShare.document_id == doc.id, DocumentShare.user_id == user.id).first()
        if not shared:
            raise HTTPException(404, "Document introuvable")
        if manage:
            raise HTTPException(403, "Seul le propriétaire peut effectuer cette action")
    return doc


def access_label(user: User, doc: Document, shared: set[int]) -> str:
    if doc.owner_id == user.id:
        return "owner"
    if doc.id in shared:
        return "shared"
    return "admin" if is_admin(user) else "shared"
