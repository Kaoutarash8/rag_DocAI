"""Isolation entre utilisateurs : A ne voit jamais les documents de B (SQL, fichiers, recherche dense, BM25)."""

from app.database import session_scope
from app.models.document import Document, DocumentStatus
from app.services import embeddings, llm, rag, retrieval, vectorstore
from app.services.vectorstore import Hit
from tests.conftest import PDF, auth, make_admin, new_user


def upload(client, h, name="a.pdf", content=b"x"):
    return client.post("/documents/upload", files={"file": (name, PDF + content, "application/pdf")}, headers=auth(h))


def mark_indexed(doc_id, tmp_path, text="contenu extrait"):
    p = tmp_path / f"{doc_id}.txt"
    p.write_text(text, encoding="utf-8")
    with session_scope() as db:
        d = db.get(Document, doc_id)
        d.status, d.extracted_text_path = DocumentStatus.INDEXED, str(p)


class FakeIndex:
    """Index vectoriel MALVEILLANT : ignore le filtre et renvoie toujours les passages de A et de B."""

    def __init__(self, monkeypatch, doc_a, doc_b):
        self.calls, self.chunks = [], []
        monkeypatch.setattr(embeddings, "embed_query", lambda m, q: [0.0])
        monkeypatch.setattr(vectorstore, "count", lambda c: 1)

        def search(col, vec, n, ids=None):
            self.calls.append(ids)
            return [Hit("secret de A", "a.pdf", doc_a, 0, 0.9), Hit("note de B", "b.pdf", doc_b, 0, 0.9)]
        monkeypatch.setattr(vectorstore, "search", search)
        monkeypatch.setattr(llm, "generate_answer", lambda q, chunks: self.chunks.append(chunks) or "ok")


def ask(client, h, **body):
    return client.post("/ask", json={"question": "Que dit le document ?", **body}, headers=auth(h))


def test_listing_and_direct_access_are_isolated(client, tmp_path):
    a, b = new_user(client), new_user(client)
    doc = upload(client, a, content=b"iso1").json()["id"]
    mark_indexed(doc, tmp_path)

    assert client.get("/documents", headers=auth(b)).json() == []
    assert client.get("/documents", headers=auth(b)).headers["X-Total-Count"] == "0"
    for method, url in [("get", f"/documents/{doc}"), ("get", f"/documents/{doc}/text"),
                        ("post", f"/documents/{doc}/summary"), ("post", f"/documents/{doc}/reprocess"),
                        ("delete", f"/documents/{doc}"), ("get", f"/documents/{doc}/shares")]:
        r = getattr(client, method)(url, headers=auth(b))
        assert r.status_code == 404, (method, url, r.status_code)  # 404, pas 403 : l'existence n'est pas révélée
    assert client.get(f"/documents/{doc}/text", headers=auth(a)).status_code == 200
    assert client.get(f"/documents/{doc}", headers=auth(a)).json()["access"] == "owner"


def test_same_content_gives_each_user_their_own_document(client):
    a, b = new_user(client), new_user(client)
    da = upload(client, a, content=b"same").json()
    db_ = upload(client, b, content=b"same")
    assert db_.status_code == 202 and db_.json()["duplicate"] is False  # pas de « existe déjà » : fuite d'existence
    assert db_.json()["id"] != da["id"]
    assert upload(client, a, content=b"same").json()["duplicate"] is True  # dédup toujours active pour A


def test_ask_never_returns_other_users_passages_even_if_index_ignores_filter(client, monkeypatch):
    a, b = new_user(client), new_user(client)
    da = upload(client, a, content=b"q1").json()["id"]
    db_ = upload(client, b, content=b"q2").json()["id"]
    idx = FakeIndex(monkeypatch, da, db_)

    body = ask(client, b).json()
    assert idx.calls == [[db_]]                       # la requête vectorielle est bornée aux documents de B
    assert [s["source"] for s in body["sources"]] == ["b.pdf"]   # et le post-filtre écarte les passages de A
    assert all("secret de A" not in c["text"] for c in idx.chunks[0])


def test_user_without_documents_never_queries_the_index(client, monkeypatch):
    a, c = new_user(client), new_user(client)
    da = upload(client, a, content=b"q3").json()["id"]
    idx = FakeIndex(monkeypatch, da, da)
    body = ask(client, c).json()
    assert idx.calls == [] and body["sources"] == []   # liste vide != « pas de filtre »


def test_ask_on_foreign_document_ids_is_404(client, monkeypatch):
    a, b = new_user(client), new_user(client)
    da = upload(client, a, content=b"q4").json()["id"]
    db_ = upload(client, b, content=b"q5").json()["id"]
    idx = FakeIndex(monkeypatch, da, db_)
    assert ask(client, b, document_ids=[da]).status_code == 404
    assert ask(client, b, document_ids=[db_, da]).status_code == 404
    assert idx.calls == []
    assert ask(client, b, document_ids=[db_]).status_code == 200


def test_bm25_branch_is_scoped_too(client, monkeypatch):
    a, b = new_user(client), new_user(client)
    da = upload(client, a, content=b"q6").json()["id"]
    db_ = upload(client, b, content=b"q7").json()["id"]
    monkeypatch.setattr(embeddings, "embed_query", lambda m, q: [0.0])
    monkeypatch.setattr(vectorstore, "search", lambda *a_, **k: [])
    monkeypatch.setattr(vectorstore, "count", lambda c: 1)
    monkeypatch.setattr(vectorstore, "version", lambda c: object().__hash__())  # force la reconstruction BM25
    monkeypatch.setattr(vectorstore, "get_all", lambda col: (
        ["Le code erreur XJ-4412 est confidentiel", "Autre sujet sans rapport"],
        [{"document_id": da, "chunk_index": 0, "source": "a.pdf"},
         {"document_id": db_, "chunk_index": 0, "source": "b.pdf"}]))
    monkeypatch.setattr(llm, "generate_answer", lambda q, c: "ok")
    q = {"question": "erreur XJ-4412", "profile": "e5-hybrid"}
    assert client.post("/ask", json=q, headers=auth(b)).json()["sources"] == []
    assert [s["document_id"] for s in client.post("/ask", json=q, headers=auth(a)).json()["sources"]] == [da]


def test_retrieve_requires_explicit_scope(monkeypatch):
    import inspect
    sig = inspect.signature(retrieval.retrieve)
    assert sig.parameters["allowed_ids"].default is inspect.Parameter.empty  # pas de défaut : jamais « tout » par oubli
    assert sig.parameters["allowed_ids"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "allowed_ids" in inspect.signature(rag.answer_question).parameters


def test_sharing_grants_read_only_and_revocation_is_immediate(client, monkeypatch, tmp_path):
    a, b, c = new_user(client), new_user(client), new_user(client)
    da = upload(client, a, content=b"s1").json()["id"]
    db_ = upload(client, b, content=b"s2").json()["id"]
    mark_indexed(da, tmp_path)
    idx = FakeIndex(monkeypatch, da, db_)
    monkeypatch.setattr(vectorstore, "delete_document", lambda i: None)

    r = client.post(f"/documents/{da}/shares", json={"email": b["email"]}, headers=auth(a))
    assert r.status_code == 201 and r.json()["permission"] == "read"
    assert client.post(f"/documents/{da}/shares", json={"email": b["email"]}, headers=auth(a)).status_code == 201  # idempotent

    listing = {d["id"]: d["access"] for d in client.get("/documents", headers=auth(b)).json()}
    assert listing == {da: "shared", db_: "owner"}
    assert client.get(f"/documents/{da}/text", headers=auth(b)).status_code == 200
    ask(client, b)
    assert idx.calls[-1] == sorted([da, db_])  # le partage élargit le périmètre de B, rien de plus

    # lecture seule : ni suppression, ni retraitement, ni régénération du résumé, ni re-partage
    assert client.delete(f"/documents/{da}", headers=auth(b)).status_code == 403
    assert client.post(f"/documents/{da}/reprocess", headers=auth(b)).status_code == 403
    assert client.post(f"/documents/{da}/summary?refresh=true", headers=auth(b)).status_code == 403
    assert client.post(f"/documents/{da}/shares", json={"email": c["email"]}, headers=auth(b)).status_code == 403
    assert client.get(f"/documents/{da}/shares", headers=auth(b)).status_code == 403

    # un tiers n'y gagne rien
    assert client.get(f"/documents/{da}", headers=auth(c)).status_code == 404

    # révocation : effet immédiat, y compris sur la recherche
    assert client.delete(f"/documents/{da}/shares/{b['id']}", headers=auth(a)).status_code == 204
    assert client.get(f"/documents/{da}", headers=auth(b)).status_code == 404
    ask(client, b)
    assert idx.calls[-1] == [db_]
    assert client.delete(f"/documents/{da}/shares/{b['id']}", headers=auth(a)).status_code == 404


def test_share_validations(client):
    a, b = new_user(client), new_user(client)
    da = upload(client, a, content=b"s3").json()["id"]
    assert client.post(f"/documents/{da}/shares", json={"email": "inconnu@test.io"}, headers=auth(a)).status_code == 404
    assert client.post(f"/documents/{da}/shares", json={"email": a["email"]}, headers=auth(a)).status_code == 400
    assert client.post(f"/documents/{da}/shares", json={"email": "pas-un-mail"}, headers=auth(a)).status_code == 422


def test_deleting_a_document_removes_its_shares(client, monkeypatch):
    a, b = new_user(client), new_user(client)
    da = upload(client, a, content=b"s4").json()["id"]
    monkeypatch.setattr(vectorstore, "delete_document", lambda i: None)
    client.post(f"/documents/{da}/shares", json={"email": b["email"]}, headers=auth(a))
    assert client.delete(f"/documents/{da}", headers=auth(a)).status_code == 204
    assert client.get("/documents", headers=auth(b)).json() == []
    from app.models.user import DocumentShare
    with session_scope() as db:
        assert db.query(DocumentShare).filter(DocumentShare.document_id == da).count() == 0


def test_admin_privileges(client, monkeypatch):
    a, admin = new_user(client), new_user(client)
    make_admin(admin["email"])
    da = upload(client, a, content=b"adm1").json()["id"]
    db_ = upload(client, admin, content=b"adm2").json()["id"]
    monkeypatch.setattr(vectorstore, "delete_document", lambda i: None)

    assert client.get("/documents?scope=all", headers=auth(a)).status_code == 403
    assert da in {d["id"] for d in client.get("/documents?scope=all", headers=auth(admin)).json()}
    assert da not in {d["id"] for d in client.get("/documents", headers=auth(admin)).json()}  # défaut : les siens
    assert client.get(f"/documents/{da}", headers=auth(admin)).json()["access"] == "admin"

    idx = FakeIndex(monkeypatch, da, db_)
    assert ask(client, a, scope="all").status_code == 403          # scope=all : admin seulement
    ask(client, admin)
    assert idx.calls[-1] == [db_]                                  # la recherche de l'admin reste bornée à ses documents
    ask(client, admin, scope="all")
    assert da in idx.calls[-1] and db_ in idx.calls[-1]

    assert client.delete(f"/documents/{da}", headers=auth(admin)).status_code == 204


def test_orphan_documents_are_visible_to_admins_only(client):
    a, admin = new_user(client), new_user(client)
    make_admin(admin["email"])
    da = upload(client, a, content=b"orph").json()["id"]
    with session_scope() as db:
        db.get(Document, da).owner_id = None
    assert client.get(f"/documents/{da}", headers=auth(a)).status_code == 404
    assert client.get(f"/documents/{da}", headers=auth(admin)).status_code == 200
