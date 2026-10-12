from app.core.config import get_settings
from tests.conftest import PDF


def upload(client, name="a.pdf", data=PDF):
    return client.post("/documents/upload", files={"file": (name, data, "application/pdf")})


def test_health(client):
    assert client.get("/health").json() == {"status": "healthy"}


def test_upload_list_get_delete(client, monkeypatch):
    from app.services import vectorstore
    monkeypatch.setattr(vectorstore, "delete_document", lambda i: None)
    r = upload(client)
    assert r.status_code == 202 and r.json()["status"] == "pending"
    doc_id = r.json()["id"]
    assert client.submitted == [doc_id]

    lst = client.get("/documents")
    assert lst.status_code == 200 and isinstance(lst.json(), list)
    assert lst.headers["X-Total-Count"] == "1"
    assert client.get(f"/documents/{doc_id}").json()["filename"] == "a.pdf"
    assert client.get("/documents?status=failed").json() == []

    assert client.delete(f"/documents/{doc_id}").status_code == 204
    assert client.get(f"/documents/{doc_id}").status_code == 404


def test_duplicate_content_not_reimported(client):
    first = upload(client, "x.pdf", PDF + b"unique1").json()
    second = upload(client, "renamed.pdf", PDF + b"unique1")
    assert second.status_code == 200 and second.json()["duplicate"] is True
    assert second.json()["id"] == first["id"]


def test_rejects_bad_extension_and_content(client):
    assert upload(client, "evil.exe", b"MZ....").status_code == 415
    assert upload(client, "fake.pdf", b"not a pdf at all").status_code == 415
    assert upload(client, "empty.pdf", b"").status_code == 400


def test_rejects_too_large(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "max_upload_mb", 1)
    assert upload(client, "big.pdf", PDF + b"0" * (1024 * 1024 + 1)).status_code == 413


def test_path_traversal_filename_is_sanitized(client):
    r = upload(client, "../../etc/passwd.pdf", PDF + b"trav")
    assert r.json()["filename"] == "passwd.pdf"


def test_api_key_required_when_configured(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "api_key", "secret")
    assert client.get("/documents").status_code == 401
    assert client.get("/documents", headers={"X-API-Key": "bad"}).status_code == 401
    assert client.get("/documents", headers={"X-API-Key": "secret"}).status_code == 200
    assert client.get("/health").status_code == 200  # reste public


def test_ask_validation(client):
    assert client.post("/ask", json={"question": "ab"}).status_code == 422
    assert client.post("/ask", json={"question": "valide ?", "top_k": 99}).status_code == 422


def test_ask_no_hits_and_with_hits(client, monkeypatch):
    from app.services import embeddings, llm, vectorstore
    from app.services.vectorstore import Hit
    monkeypatch.setattr(embeddings, "embed", lambda t: [[0.1, 0.2]])
    monkeypatch.setattr(vectorstore, "search", lambda *a, **k: [])
    r = client.post("/ask", json={"question": "Quel est le prix ?"})
    assert r.status_code == 200 and r.json()["sources"] == []

    hit = Hit("Le prix est de 100 euros.", "contrat.pdf", 1, 0, 0.8)
    low = Hit("hors sujet", "autre.pdf", 2, 0, 0.01)
    monkeypatch.setattr(vectorstore, "search", lambda *a, **k: [hit, low])
    monkeypatch.setattr(llm, "generate_answer", lambda q, c: f"ok {len(c)}")
    body = client.post("/ask", json={"question": "Quel est le prix ?"}).json()
    assert body["answer"] == "ok 1"  # le chunk sous le seuil est écarté
    assert body["sources"][0]["source"] == "contrat.pdf"


def test_llm_failure_returns_502(client, monkeypatch):
    from app.core.errors import LLMError
    from app.services import embeddings, llm, vectorstore
    from app.services.vectorstore import Hit
    monkeypatch.setattr(embeddings, "embed", lambda t: [[0.1]])
    monkeypatch.setattr(vectorstore, "search", lambda *a, **k: [Hit("t", "s", 1, 0, 0.9)])

    def boom(q, c):
        raise LLMError("down")
    monkeypatch.setattr(llm, "generate_answer", boom)
    assert client.post("/ask", json={"question": "une question"}).status_code == 502
