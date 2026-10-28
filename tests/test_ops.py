"""Limiteur Redis par utilisateur, verrous de traitement, versions d'index partagées, logs JSON, métriques."""

import json
import logging

import pytest

from app.core import locks, ratelimit
from app.core.config import get_settings
from app.core.redis import set_client
from app.services import vectorstore
from tests.conftest import auth, new_user


class FakeRedis:
    """Sous-ensemble de redis-py suffisant pour le limiteur, les verrous et les versions (horloge pilotable)."""

    def __init__(self):
        self.now, self.d, self.exp = 0.0, {}, {}

    def _alive(self, k):
        if k in self.exp and self.exp[k] <= self.now:
            self.d.pop(k, None), self.exp.pop(k, None)
        return k in self.d

    def set(self, k, v, ex=None, nx=False):
        if nx and self._alive(k):
            return None
        self.d[k] = str(v)
        if ex:
            self.exp[k] = self.now + ex
        return True

    def get(self, k):
        return self.d[k] if self._alive(k) else None

    def incr(self, k):
        self._alive(k)
        self.d[k] = str(int(self.d.get(k, 0)) + 1)
        return int(self.d[k])

    def ttl(self, k):
        if not self._alive(k):
            return -2
        return int(self.exp[k] - self.now) if k in self.exp else -1

    def expire(self, k, s):
        self.exp[k] = self.now + s

    def eval(self, script, n, key, *args):
        if not self._alive(key) or self.d[key] != args[0]:
            return 0
        if script == locks._RELEASE:
            del self.d[key]
        else:
            self.exp[key] = self.now + int(args[1])
        return 1

    def pipeline(self):
        outer, ops = self, []

        class Pipe:
            def __getattr__(self, name):
                return lambda *a, **k: ops.append((name, a, k)) or self

            def execute(self):
                res = [getattr(outer, n)(*a, **k) for n, a, k in ops]
                ops.clear()
                return res
        return Pipe()


@pytest.fixture()
def redis():
    r = FakeRedis()
    set_client(r)
    yield r
    set_client(None)
    ratelimit._memory = ratelimit.MemoryLimiter()


def test_redis_limiter_is_shared_between_instances_and_resets(redis):
    one, two = ratelimit.RedisLimiter(redis), ratelimit.RedisLimiter(redis)  # deux instances d'API
    assert [one.allow("user:1", 3)[0], two.allow("user:1", 3)[0], one.allow("user:1", 3)[0]] == [True] * 3
    allowed, retry = two.allow("user:1", 3)
    assert allowed is False and 1 <= retry <= 60
    assert one.allow("user:2", 3)[0] is True  # limite PAR utilisateur
    redis.now += 61
    assert two.allow("user:1", 3)[0] is True  # nouvelle fenêtre


def test_enforce_raises_429_with_retry_after(redis):
    from fastapi import HTTPException
    ratelimit.enforce("user:9", 1)
    with pytest.raises(HTTPException) as e:
        ratelimit.enforce("user:9", 1)
    assert e.value.status_code == 429 and int(e.value.headers["Retry-After"]) >= 1
    ratelimit.enforce("user:9", 0)  # 0 = désactivé


def test_enforce_falls_back_to_memory_when_redis_is_down(redis):
    from fastapi import HTTPException

    class Down(FakeRedis):
        def pipeline(self):
            raise ConnectionError("redis down")
    set_client(Down())
    ratelimit.enforce("user:5", 1)           # pas d'erreur 500 : repli en mémoire
    with pytest.raises(HTTPException):
        ratelimit.enforce("user:5", 1)


def test_api_rate_limit_is_per_user(client, redis, monkeypatch):
    monkeypatch.setattr(get_settings(), "rate_limit_per_minute", 2)
    a, b = new_user(client), new_user(client)
    codes = [client.post("/ask", json={"question": "une question"}, headers=auth(a)).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    assert client.post("/ask", json={"question": "une question"}, headers=auth(b)).status_code == 200


def test_login_is_throttled_per_ip_and_per_email(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "auth_rate_limit_per_minute", 3)
    ratelimit._memory = ratelimit.MemoryLimiter()
    codes = [client.post("/auth/login", json={"email": "victime@test.io", "password": "mauvais-mot-de-passe"},
                         headers={"Authorization": ""}).status_code for _ in range(5)]
    assert codes[:3] == [401] * 3 and codes[3:] == [429, 429]
    ratelimit._memory = ratelimit.MemoryLimiter()


def test_processing_lease_prevents_double_processing(redis):
    lease = locks.acquire(7)
    with pytest.raises(locks.Busy):
        locks.acquire(7)
    lease.extend()
    lease.release()
    assert locks.acquire(7) is not None


def test_lease_expires_if_worker_dies_and_cannot_be_stolen_by_stale_holder(redis):
    old = locks.acquire(8)
    redis.now += locks.TTL_SECONDS + 1          # le worker est mort : le bail a expiré
    new = locks.acquire(8)                      # un autre worker reprend
    old.release()                               # l'ancien se réveille : ne doit PAS libérer le bail du nouveau
    with pytest.raises(locks.Busy):
        locks.acquire(8)
    new.release()


def test_no_redis_means_no_lock():
    set_client(None)
    assert locks.acquire(1) is None


def test_vector_version_is_shared_through_redis(redis):
    assert vectorstore.version("c1") == 0
    vectorstore._bump("c1")                      # écriture faite par un worker
    assert vectorstore.version("c1") == 1        # visible par l'API : le cache BM25 sera reconstruit


def test_json_log_format():
    from app.core.logs import JsonFormatter
    rec = logging.LogRecord("app", logging.INFO, __file__, 1, "GET /x -> 200", (), None)
    rec.request_id, rec.user_id, rec.duration_ms = "abc", 3, 12
    out = json.loads(JsonFormatter().format(rec))
    assert out["msg"] == "GET /x -> 200" and out["request_id"] == "abc" and out["user_id"] == 3
    assert out["level"] == "INFO" and "ts" in out


def test_metrics_endpoint_uses_route_templates(client, monkeypatch):
    client.get("/documents/123456")
    body = client.get("/metrics").text
    assert 'route="/documents/{document_id}"' in body and "123456" not in body
    assert "docai_http_request_duration_seconds_bucket" in body
    assert "docai_documents" in body
    monkeypatch.setattr(get_settings(), "metrics_token", "jeton-metrics")
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"Authorization": "Bearer jeton-metrics"}).status_code == 200


def test_config_guards(monkeypatch):
    from pydantic import ValidationError
    from app.core.config import Settings
    with pytest.raises(ValidationError):
        Settings(env="production", jwt_secret="")
    with pytest.raises(ValidationError):
        Settings(env="production", jwt_secret="trop-court")
    with pytest.raises(ValidationError):  # plusieurs process sans serveur Chroma = corruption de l'index
        Settings(env="test", queue_backend="celery", redis_url="redis://x", chroma_host="")
    Settings(env="production", jwt_secret="x" * 40)
