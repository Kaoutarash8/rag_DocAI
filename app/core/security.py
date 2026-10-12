"""Authentification par clé d'API et limitation de débit (en mémoire)."""

import hmac
import threading
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request

from app.core.config import get_settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = get_settings().api_key
    if not expected:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Clé d'API invalide ou manquante")


class _RateLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window: float = 60.0) -> bool:
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            return True


_limiter = _RateLimiter()


def rate_limit(request: Request) -> None:
    limit = get_settings().rate_limit_per_minute
    if limit <= 0:
        return
    key = request.client.host if request.client else "anonymous"
    if not _limiter.allow(key, limit):
        raise HTTPException(
            status_code=429, detail="Trop de requêtes, réessayez dans une minute",
            headers={"Retry-After": "60"},
        )
