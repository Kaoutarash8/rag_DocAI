from app.database import session_scope
from app.models.document import Document, DocumentStatus
from app.services import chunking, embeddings, ocr, processing, vectorstore
from app.services.ocr import Extraction
from tests.conftest import PDF


def _make_doc(client, content):
    return client.post("/documents/upload",
                       files={"file": ("p.pdf", PDF + content, "application/pdf")}).json()["id"]


def test_process_success(client, monkeypatch):
    doc_id = _make_doc(client, b"ok1")
    indexed = {}
    monkeypatch.setattr(ocr, "extract", lambda p, t: Extraction("mot " * 250, 1, 0))
    monkeypatch.setattr(embeddings, "embed", lambda ch: [[0.0]] * len(ch))
    monkeypatch.setattr(vectorstore, "index_document",
                        lambda i, f, ch, v: indexed.update(id=i, n=len(ch)))
    # le fichier doit exister sur disque : upload l'a créé
    processing.process_document(doc_id)
    with session_scope() as db:
        d = db.get(Document, doc_id)
        assert d.status == DocumentStatus.INDEXED and d.chunk_count == indexed["n"] > 1


def test_process_failure_is_recorded(client, monkeypatch):
    doc_id = _make_doc(client, b"fail1")

    def boom(p, t):
        raise ValueError("PDF illisible")
    monkeypatch.setattr(ocr, "extract", boom)
    processing.process_document(doc_id)
    with session_scope() as db:
        d = db.get(Document, doc_id)
        assert d.status == DocumentStatus.FAILED and "illisible" in d.error_message


def test_gives_up_after_max_attempts(client, monkeypatch):
    doc_id = _make_doc(client, b"retry1")
    monkeypatch.setattr(ocr, "extract", lambda p, t: (_ for _ in ()).throw(RuntimeError("x")))
    for _ in range(5):
        processing.process_document(doc_id)
    with session_scope() as db:
        d = db.get(Document, doc_id)
        assert d.status == DocumentStatus.FAILED and d.attempts == 3
