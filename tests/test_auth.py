import time

import jwt

from app.core.auth import create_access_token, hash_password, verify_password
from app.core.config import get_settings
from tests.conftest import PASSWORD, auth, make_admin, new_user


def test_register_login_me(anon):
    h = new_user(anon, "Alice@Test.io")  # l'e-mail est normalisé en minuscules
    me = anon.get("/auth/me", headers=auth(h)).json()
    assert me["email"] == "alice@test.io" and me["role"] == "user"
    assert "password" not in str(me)


def test_register_rejects_duplicates_and_weak_passwords(anon):
    new_user(anon, "dup@test.io")
    assert anon.post("/auth/register", json={"email": "DUP@test.io", "password": PASSWORD}).status_code == 409
    assert anon.post("/auth/register", json={"email": "faible@test.io", "password": "court"}).status_code == 422
    assert anon.post("/auth/register", json={"email": "pas-un-mail", "password": PASSWORD}).status_code == 422


def test_login_failures_are_indistinguishable(anon):
    new_user(anon, "bob@test.io")
    bad_pw = anon.post("/auth/login", json={"email": "bob@test.io", "password": "mauvais-mot-de-passe"})
    unknown = anon.post("/auth/login", json={"email": "inconnu@test.io", "password": "mauvais-mot-de-passe"})
    assert bad_pw.status_code == unknown.status_code == 401
    assert bad_pw.json() == unknown.json()


def test_register_cannot_choose_role(anon):
    r = anon.post("/auth/register", json={"email": "sneaky@test.io", "password": PASSWORD, "role": "admin"})
    assert r.status_code == 201 and r.json()["role"] == "user"


def test_registration_can_be_disabled(anon, monkeypatch):
    monkeypatch.setattr(get_settings(), "allow_registration", False)
    assert anon.post("/auth/register", json={"email": "x@test.io", "password": PASSWORD}).status_code == 403


def test_expired_tampered_and_foreign_tokens_are_rejected(anon, monkeypatch):
    h = new_user(anon)
    s = get_settings()
    assert anon.get("/auth/me", headers=auth(h)).status_code == 200
    expired = jwt.encode({"sub": str(h["id"]), "iat": int(time.time()) - 7200, "exp": int(time.time()) - 3600,
                          "typ": "access"}, s.jwt_secret, algorithm="HS256")
    assert anon.get("/auth/me", headers={"Authorization": f"Bearer {expired}"}).status_code == 401
    forged = jwt.encode({"sub": str(h["id"]), "iat": int(time.time()), "exp": int(time.time()) + 600,
                         "typ": "access"}, "autre-secret-autre-secret-autre-secret", algorithm="HS256")
    assert anon.get("/auth/me", headers={"Authorization": f"Bearer {forged}"}).status_code == 401
    none_alg = jwt.encode({"sub": str(h["id"]), "exp": int(time.time()) + 600}, key="", algorithm="none")
    assert anon.get("/auth/me", headers={"Authorization": f"Bearer {none_alg}"}).status_code == 401
    refresh = jwt.encode({"sub": str(h["id"]), "iat": int(time.time()), "exp": int(time.time()) + 600,
                          "typ": "refresh"}, s.jwt_secret, algorithm="HS256")
    assert anon.get("/auth/me", headers={"Authorization": f"Bearer {refresh}"}).status_code == 401


def test_deactivated_user_loses_access_immediately(anon):
    admin = new_user(anon)
    make_admin(admin["email"])
    user = new_user(anon)
    assert anon.get("/auth/me", headers=auth(user)).status_code == 200
    assert anon.patch(f"/admin/users/{user['id']}", json={"is_active": False}, headers=auth(admin)).status_code == 200
    assert anon.get("/auth/me", headers=auth(user)).status_code == 401
    login = anon.post("/auth/login", json={"email": user["email"], "password": PASSWORD})
    assert login.status_code == 401


def test_role_is_read_from_database_not_from_token(anon):
    user = new_user(anon)
    assert anon.get("/admin/users", headers=auth(user)).status_code == 403
    make_admin(user["email"])  # promu : le même jeton fonctionne maintenant
    assert anon.get("/admin/users", headers=auth(user)).status_code == 200


def test_admin_cannot_demote_or_deactivate_self(anon):
    admin = new_user(anon)
    make_admin(admin["email"])
    r = anon.patch(f"/admin/users/{admin['id']}", json={"role": "user"}, headers=auth(admin))
    assert r.status_code == 400
    r = anon.patch(f"/admin/users/{admin['id']}", json={"is_active": False}, headers=auth(admin))
    assert r.status_code == 400


def test_password_hash_roundtrip_and_salt():
    h1, h2 = hash_password("un-mot-de-passe"), hash_password("un-mot-de-passe")
    assert h1 != h2 and h1.startswith("scrypt$")
    assert verify_password("un-mot-de-passe", h1) and not verify_password("autre", h1)
    assert not verify_password("x", "format$invalide")


def test_token_contains_no_role():
    tok, ttl = create_access_token(42)
    payload = jwt.decode(tok, get_settings().jwt_secret, algorithms=["HS256"])
    assert payload["sub"] == "42" and "role" not in payload and ttl > 0
