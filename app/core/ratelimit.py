"""Limitation de débit : Redis (partagé entre instances) avec repli en mémoire (par process).

Fenêtre fixe d'une minute : simple et atomique avec Redis. Un client peut, au pire, enchaîner
2x la limite autour d'un changement de fenêtre ; suffisant contre les abus, pas un quota de facturation.
"""

import logging
import math
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException

from app.core.redis import get_redis

log = logging.getLogger(__name__)


class MemoryLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window: float = 60.0) -> tuple[bool, int]:
        """(autorisé, secondes avant nouvel essai)."""
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= limit:
                return False, max(1, math.ceil(window - (now - q[0])))
            q.append(now)
            return True, 0


class RedisLimiter:
    def __init__(self, client) -> None:
        self.r = client

    def allow(self, key: str, limit: int, window: float = 60.0) -> tuple[bool, int]:
        k = f"rl:{key}"
        pipe = self.r.pipeline()
        pipe.set(k, 0, ex=int(window), nx=True)  # crée la fenêtre (avec expiration) si absente
        pipe.incr(k)
        pipe.ttl(k)
        _, count, ttl = pipe.execute()
        if ttl is None or ttl < 0:  # clé sans expiration (ne devrait pas arriver) : on la répare
            self.r.expire(k, int(window))
            ttl = int(window)
        return count <= limit, max(1, int(ttl))


_memory = MemoryLimiter()


def enforce(key: str, limit: int, window: float = 60.0) -> None:
    """Lève 429 (avec Retry-After) si `key` dépasse `limit` requêtes par fenêtre ; limit<=0 = désactivé."""
    if limit <= 0:
        return
    client = get_redis()
    if client is not None:
        try:
            allowed, retry = RedisLimiter(client).allow(key, limit, window)
        except Exception:  # Redis indisponible : on dégrade (limite par process) plutôt que de bloquer l'API
            log.warning("Redis indisponible pour la limitation de débit : repli en mémoire", exc_info=True)
            allowed, retry = _memory.allow(key, limit, window)
    else:
        allowed, retry = _memory.allow(key, limit, window)
    if not allowed:
        raise HTTPException(429, "Trop de requêtes, réessayez plus tard",
                            headers={"Retry-After": str(retry)})
