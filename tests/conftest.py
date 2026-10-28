import os
import tempfile
import uuid

_tmp = tempfile.mkdtemp()
os.environ.update({
    "ENV": "test", "DATABASE_URL": f"sqlite:///{_tmp}/test.db", "DATA_DIR": f"{_tmp}/data",
    "AUTO_CREATE_TABLES": "true", "RATE_LIMIT_PER_MINUTE": "0", "AUTH_RATE_LIMIT_PER_MINUTE": "0",
    "JWT_SECRET": "test-secret-test-secret-test-secret-0123", "PASSWORD_SCRYPT_N": "16",
    "REDIS_URL": "", "QUEUE_BACKEND": "thread",
})

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.services import processing  # noqa: E402

PDF = b"%PDF-1.4\n% fake pdf for tests\n"
PASSWORD = "mot-de-passe-solide-123"


def new_user(client, email=None, password=PASSWORD) -> dict:
    """Crée un compte et renvoie ses en-têtes d'authentification (+ id/email)."""
    email = email or f"u-{uuid.uuid4().hex[:8]}@test.io"
    r = client.post("/auth/register", json={"email": email, "password": password})
    assert r.status_code == 201, r.text
    uid = r.json()["id"]
    tok = client.post("/auth/login", json={"email": email, "password": password}).json()["access_token"]
    return {"Authorization": f"Bearer {tok}", "id": uid, "email": email}


def auth(h: dict) -> dict:
    return {"Authorization": h["Authorization"]}


def make_admin(email: str) -> None:
    from app.database import session_scope
    from app.models.user import User, UserRole
    with session_scope() as db:
        db.query(User).filter(User.email == email).update({"role": UserRole.ADMIN})


@pytest.fixture()
def client(monkeypatch):
    """Client déjà authentifié en tant qu'utilisateur par défaut (les tests d'isolation passent leurs propres en-têtes)."""
    submitted = []
    monkeypatch.setattr(processing, "submit", submitted.append)
    with TestClient(app) as c:
        c.submitted = submitted
        c.headers.update(auth(new_user(c)))
        yield c


@pytest.fixture()
def anon(monkeypatch):
    """Client sans jeton."""
    monkeypatch.setattr(processing, "submit", lambda i: None)
    with TestClient(app) as c:
        yield c
