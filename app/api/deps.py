"""Dépendances FastAPI : utilisateur courant (JWT), rôle admin, limitation de débit."""

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core import ratelimit
from app.core.auth import decode_access_token
from app.core.config import get_settings
from app.database import get_db
from app.models.user import User, UserRole

_bearer = HTTPBearer(auto_error=False)
_UNAUTHORIZED = {"WWW-Authenticate": "Bearer"}


def current_user(request: Request,
                 creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
                 db: Session = Depends(get_db)) -> User:
    if creds is None:
        raise HTTPException(401, "Authentification requise", headers=_UNAUTHORIZED)
    uid = decode_access_token(creds.credentials)
    user = db.get(User, uid) if uid is not None else None
    if user is None or not user.is_active:  # rôle et statut sont relus en base : effet immédiat
        raise HTTPException(401, "Jeton invalide ou expiré", headers=_UNAUTHORIZED)
    request.state.user_id = user.id  # lu par le middleware (journaux)
    return user


def require_admin(user: User = Depends(current_user)) -> User:
    if user.role != UserRole.ADMIN:
        raise HTTPException(403, "Réservé aux administrateurs")
    return user


def rate_limit(user: User = Depends(current_user)) -> None:
    """Limite par utilisateur (et non par IP : plusieurs utilisateurs peuvent partager une adresse)."""
    ratelimit.enforce(f"user:{user.id}", get_settings().rate_limit_per_minute)


def auth_rate_limit(request: Request) -> None:
    """Login/inscription : limite par IP (X-Forwarded-For fourni par le reverse proxy de confiance)."""
    ip = request.client.host if request.client else "unknown"
    ratelimit.enforce(f"auth-ip:{ip}", get_settings().auth_rate_limit_per_minute)
