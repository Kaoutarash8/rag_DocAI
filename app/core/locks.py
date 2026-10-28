"""Bail Redis par document : un seul worker traite un document à la fois.

Le bail expire seul (TTL) si le worker meurt, et il est prolongé à chaque page traitée.
Sans Redis (mode mono-process), aucun verrou : `acquire` renvoie None.
"""

import uuid

from app.core.redis import get_redis

TTL_SECONDS = 600

_EXTEND = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('expire', KEYS[1], ARGV[2]) else return 0 end"
_RELEASE = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end"


class Busy(Exception):
    """Le document est déjà en cours de traitement par un autre worker."""


class Lease:
    def __init__(self, client, key: str, token: str) -> None:
        self.r, self.key, self.token = client, key, token

    def extend(self) -> None:
        self.r.eval(_EXTEND, 1, self.key, self.token, TTL_SECONDS)

    def release(self) -> None:
        try:
            self.r.eval(_RELEASE, 1, self.key, self.token)
        except Exception:  # le bail expirera de lui-même
            pass


def acquire(document_id: int) -> Lease | None:
    client = get_redis()
    if client is None:
        return None
    key, token = f"lock:process:{document_id}", uuid.uuid4().hex
    if not client.set(key, token, nx=True, ex=TTL_SECONDS):
        raise Busy(document_id)
    return Lease(client, key, token)
