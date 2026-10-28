"""Client Redis partagé (limiteur de débit, verrous, versions d'index, file Celery)."""

import threading

from app.core.config import get_settings

_client = None
_override = None
_lock = threading.Lock()


def get_redis():
    """Client Redis, ou None si REDIS_URL est vide (mode mono-process)."""
    global _client
    if _override is not None:
        return _override
    url = get_settings().redis_url
    if not url:
        return None
    with _lock:
        if _client is None:
            import redis

            _client = redis.Redis.from_url(
                url, socket_timeout=2, socket_connect_timeout=2,
                health_check_interval=30, decode_responses=True)
    return _client


def set_client(client) -> None:
    """Injecte un client (tests) ; None pour revenir au comportement normal."""
    global _override
    _override = client
