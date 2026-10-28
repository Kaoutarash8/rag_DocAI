from app.core.config import get_settings
from app.database import session_scope
from app.models.document import Document, DocumentStatus
from app.services import embeddings, extraction, processing, vectorstore
from app.services.content import PageContent, Segment
from tests.conftest import PDF


def _make_doc(client, content):
    return client.post("/documents/upload",
                       files={"file": ("p.pdf", PDF + content, "application/pdf")}).json()["id"]


def test_process_success(client, monkeypatch):
    doc_id = _make_doc(client, b"ok1")
    stored = []
    pages = [PageContent(n, [Segment("mot " * 250, "text", f"Section {n}")]) for n in (1, 2, 3)]
    monkeypatch.setattr(extraction, "iter_pages", lambda p, t: (3, iter(pages)))
    monkeypatch.setattr(embeddings, "embed_passages", lambda m, ch: [[0.0]] * len(ch))
    monkeypatch.setattr(vectorstore, "delete_document", lambda i: None)
    monkeypatch.setattr(vectorstore, "index_chunks",
                        lambda col, i, f, ch, v: stored.extend(ch))
    monkeypatch.setattr(get_settings(), "page_batch_size", 2)  # 2 lots : 2 pages puis 1
    processing.process_document(doc_id)
    with session_scope() as db:
        d = db.get(Document, doc_id)
        assert d.status == DocumentStatus.INDEXED and d.chunk_count == len(stored) > 3
        assert d.pages_total == 3 and d.pages_done == 3
    # chaque chunk garde sa page et sa section, et les index sont globaux et uniques
    assert {c.page for c in stored} == {1, 2, 3}
    assert all(c.section == f"Section {c.page}" for c in stored)
    assert [c.index for c in stored] == list(range(len(stored)))


def test_process_failure_is_recorded(client, monkeypatch):
    doc_id = _make_doc(client, b"fail1")

    def boom(p, t):
        raise ValueError("PDF illisible")
    monkeypatch.setattr(vectorstore, "delete_document", lambda i: None)
    monkeypatch.setattr(extraction, "iter_pages", boom)
    processing.process_document(doc_id)
    with session_scope() as db:
        d = db.get(Document, doc_id)
        assert d.status == DocumentStatus.FAILED and "illisible" in d.error_message


def test_gives_up_after_max_attempts(client, monkeypatch):
    doc_id = _make_doc(client, b"retry1")
    monkeypatch.setattr(vectorstore, "delete_document", lambda i: None)
    monkeypatch.setattr(extraction, "iter_pages",
                        lambda p, t: (_ for _ in ()).throw(RuntimeError("x")))
    for _ in range(5):
        processing.process_document(doc_id)
    with session_scope() as db:
        d = db.get(Document, doc_id)
        assert d.status == DocumentStatus.FAILED and d.attempts == 3


def test_deleted_during_processing_cleans_up(client, monkeypatch):
    doc_id = _make_doc(client, b"gone1")
    deleted = []
    pages = [PageContent(1, [Segment("mot " * 50)]), PageContent(2, [Segment("mot " * 50)])]
    monkeypatch.setattr(extraction, "iter_pages", lambda p, t: (2, iter(pages)))
    monkeypatch.setattr(embeddings, "embed_passages", lambda m, ch: [[0.0]] * len(ch))
    monkeypatch.setattr(vectorstore, "delete_document", deleted.append)
    monkeypatch.setattr(vectorstore, "index_chunks", lambda *a: None)
    monkeypatch.setattr(get_settings(), "page_batch_size", 1)

    def vanish(document_id, **kw):  # le document disparaît au premier lot
        with session_scope() as db:
            db.delete(db.get(Document, document_id))
        return False
    monkeypatch.setattr(processing, "_set_progress",
                        lambda i, **kw: kw.get("total") is not None or vanish(i))
    processing.process_document(doc_id)
    assert deleted  # les vecteurs sont nettoyés, pas de document fantôme
