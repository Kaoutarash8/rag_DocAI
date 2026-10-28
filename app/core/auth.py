"""Mots de passe (scrypt, bibliothèque standard) et jetons d'accès JWT."""

import base64
import hashlib
import hmac
import os
import time
import unicodedata
import uuid

import jwt

from app.core.config import get_settings

_R, _P = 8, 3  # scrypt : r=8, p=3 ; N vient de PASSWORD_SCRYPT_N (2**15 par défaut)
_dummy_hash: str | None = None


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _scrypt(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    pw = unicodedata.normalize("NFKC", password).encode("utf-8")
    return hashlib.scrypt(pw, salt=salt, n=n, r=r, p=p, dklen=32, maxmem=256 * 1024 * 1024)


def hash_password(password: str) -> str:
    """`scrypt$N$r$p$sel$empreinte` : les paramètres sont stockés, donc relevables sans migration."""
    n = get_settings().password_scrypt_n
    salt = os.urandom(16)
    return f"scrypt${n}${_R}${_P}${_b64(salt)}${_b64(_scrypt(password, salt, n, _R, _P))}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, expected = stored.split("$")
        if scheme != "scrypt":
            return False
        got = _scrypt(password, base64.b64decode(salt), int(n), int(r), int(p))
        return hmac.compare_digest(got, base64.b64decode(expected))
    except Exception:
        return False


def verify_dummy(password: str) -> None:
    """Même coût qu'une vérification réelle quand l'e-mail est inconnu (anti-énumération par le temps)."""
    global _dummy_hash
    if _dummy_hash is None:
        _dummy_hash = hash_password("dummy-password-for-timing")
    verify_password(password, _dummy_hash)


def create_access_token(user_id: int) -> tuple[str, int]:
    """(jeton, durée de validité en secondes). Le rôle n'est PAS dans le jeton : il est relu en base."""
    s = get_settings()
    now = int(time.time())
    ttl = s.access_token_minutes * 60
    payload = {"sub": str(user_id), "iat": now, "exp": now + ttl, "jti": uuid.uuid4().hex, "typ": "access"}
    return jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_algorithm), ttl


def decode_access_token(token: str) -> int | None:
    """Identifiant utilisateur, ou None si le jeton est invalide, expiré ou falsifié."""
    s = get_settings()
    try:
        payload = jwt.decode(token, s.jwt_secret, algorithms=[s.jwt_algorithm], leeway=10,
                             options={"require": ["exp", "iat", "sub"]})
        if payload.get("typ") != "access":
            return None
        return int(payload["sub"])
    except (jwt.PyJWTError, ValueError, TypeError):
        return None
