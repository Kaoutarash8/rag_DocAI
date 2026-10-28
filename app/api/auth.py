from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import auth_rate_limit, current_user
from app.api.schemas import LoginIn, RegisterIn, TokenOut, UserOut
from app.core import ratelimit
from app.core.auth import create_access_token, hash_password, verify_dummy, verify_password
from app.core.config import get_settings
from app.database import get_db
from app.models.user import User, UserRole

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserOut, status_code=201, dependencies=[Depends(auth_rate_limit)])
def register(body: RegisterIn, db: Session = Depends(get_db)):
    """Crée un compte standard (jamais admin : voir `scripts.create_admin` et POST /admin/users)."""
    if not get_settings().allow_registration:
        raise HTTPException(403, "Les inscriptions sont désactivées")
    if db.query(User.id).filter(User.email == body.email).first():
        raise HTTPException(409, "Cette adresse e-mail est déjà utilisée")
    user = User(email=body.email, password_hash=hash_password(body.password), role=UserRole.USER)
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "Cette adresse e-mail est déjà utilisée")
    db.refresh(user)
    return user


@router.post("/login", response_model=TokenOut, dependencies=[Depends(auth_rate_limit)])
def login(body: LoginIn, db: Session = Depends(get_db)):
    # Limite aussi par e-mail : un botnet qui change d'IP ne peut pas forcer un compte précis.
    ratelimit.enforce(f"auth-email:{body.email}", get_settings().auth_rate_limit_per_minute)
    user = db.query(User).filter(User.email == body.email).first()
    ok = verify_password(body.password, user.password_hash) if user else (verify_dummy(body.password) or False)
    if not (user and ok and user.is_active):
        raise HTTPException(401, "Identifiants invalides", headers={"WWW-Authenticate": "Bearer"})
    user.last_login_at = datetime.now(timezone.utc).replace(tzinfo=None)
    db.commit()
    token, ttl = create_access_token(user.id)
    return TokenOut(access_token=token, expires_in=ttl)


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)):
    return user
