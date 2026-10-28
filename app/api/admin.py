from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import rate_limit, require_admin
from app.api.schemas import AdminUserCreate, AdminUserPatch, UserOut
from app.core.auth import hash_password
from app.core.config import get_settings
from app.database import get_db
from app.models.index_collection import IndexCollection
from app.models.user import User, UserRole

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin), Depends(rate_limit)])


@router.get("/users", response_model=list[UserOut])
def list_users(db: Session = Depends(get_db), limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)):
    return db.query(User).order_by(User.id).offset(offset).limit(limit).all()


@router.post("/users", response_model=UserOut, status_code=201)
def create_user(body: AdminUserCreate, db: Session = Depends(get_db)):
    """Création d'un compte (y compris admin) ; utile quand ALLOW_REGISTRATION=false."""
    user = User(email=body.email, password_hash=hash_password(body.password), role=body.role)
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "Cette adresse e-mail est déjà utilisée")
    db.refresh(user)
    return user


@router.patch("/users/{user_id}", response_model=UserOut)
def update_user(user_id: int, body: AdminUserPatch, db: Session = Depends(get_db),
                admin: User = Depends(require_admin)):
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(404, "Utilisateur introuvable")
    if user.id == admin.id and (body.is_active is False or (body.role and body.role != UserRole.ADMIN)):
        raise HTTPException(400, "Vous ne pouvez pas vous retirer vos propres droits ni désactiver votre compte")
    if body.role is not None:
        user.role = body.role
    if body.is_active is not None:
        user.is_active = body.is_active
    db.commit()
    db.refresh(user)
    return user


@router.get("/system")
def system_info(db: Session = Depends(get_db)):
    """Versions en service : application, index, modèles, LLM, et registre des collections vectorielles."""
    from app.services import profiles

    s = get_settings()
    specs = profiles.index_specs()
    registry = {c.name: {"index_version": c.index_version, "embedding_model": c.embedding_model,
                         "chunking": f"{c.chunk_strategy} {c.chunk_size}/{c.chunk_overlap}",
                         "created_at": c.created_at.isoformat()} for c in db.query(IndexCollection)}
    return {
        "app_version": "2.3.0", "index_version": s.index_version, "default_profile": s.default_profile,
        "queue_backend": s.queue_backend, "vectorstore": "chroma-server" if s.chroma_host else "chroma-embedded",
        "llm": {"provider": s.llm_provider,
                "model": {"ollama": s.ollama_model, "groq": s.groq_model, "openai": s.openai_model}.get(s.llm_provider)},
        "active_collections": [{"name": sp.collection, "embedding_model": sp.embedding_model,
                                "chunking": f"{sp.strategy} {sp.size}/{sp.overlap}"} for sp in specs],
        "registry": registry,
    }
