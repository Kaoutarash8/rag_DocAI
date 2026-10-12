"""File de traitement bornée : OCR -> chunks -> embeddings -> vectorstore.

Les documents bloqués en PENDING/PROCESSING (crash, redémarrage) sont repris
au démarrage. Pour passer à l'échelle, remplacer l'executor par Celery/RQ/ARQ.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from app.core.config import get_settings
from app.database import session_scope
from app.models.document import Document, DocumentStatus
from app.services import chunking, embeddings, ocr, vectorstore

log = logging.getLogger(__name__)
_executor: ThreadPoolExecutor | None = None


def start() -> None:
    global _executor
    _executor = ThreadPoolExecutor(
        max_workers=get_settings().worker_threads, thread_name_prefix="ingest"
    )
    recover_pending()


def stop() -> None:
    if _executor:
        _executor.shutdown(wait=False, cancel_futures=True)


def submit(document_id: int) -> None:
    if _executor is None:
        start()
    _executor.submit(process_document, document_id)


def recover_pending() -> None:
    with session_scope() as db:
        ids = [d.id for d in db.query(Document).filter(
            Document.status.in_([DocumentStatus.PENDING, DocumentStatus.PROCESSING]))]
    for i in ids:
        log.info("Reprise du document %s", i)
        _executor.submit(process_document, i)


def _fail(document_id: int, message: str) -> None:
    with session_scope() as db:
        doc = db.get(Document, document_id)
        if doc:
            doc.status = DocumentStatus.FAILED
            doc.error_message = message[:1000]


def process_document(document_id: int) -> None:
    s = get_settings()
    with session_scope() as db:
        doc = db.get(Document, document_id)
        if doc is None:
            return
        if doc.attempts >= s.max_attempts:
            doc.status = DocumentStatus.FAILED
            doc.error_message = f"Abandon après {doc.attempts} tentatives"
            return
        doc.status = DocumentStatus.PROCESSING
        doc.attempts += 1
        doc.error_message = None
        path = s.upload_dir / doc.stored_name
        file_type, filename = doc.file_type, doc.filename

    try:  # travail lourd HORS transaction
        if not path.exists():
            raise FileNotFoundError("Fichier source introuvable sur le disque")
        result = ocr.extract(path, file_type)
        if not result.text.strip():
            raise ValueError("Aucun texte détecté dans le document")
        chunks = chunking.chunk_text(result.text, s.chunk_size, s.chunk_overlap)
        vectors = embeddings.embed(chunks)
        vectorstore.index_document(document_id, filename, chunks, vectors)
        txt_path = s.text_dir / f"{document_id}.txt"
        txt_path.write_text(result.text, encoding="utf-8")
    except Exception as e:
        log.exception("Échec du traitement du document %s", document_id)
        _fail(document_id, f"{type(e).__name__}: {e}")
        return

    with session_scope() as db:
        doc = db.get(Document, document_id)
        if doc:  # peut avoir été supprimé pendant le traitement
            doc.status = DocumentStatus.INDEXED
            doc.chunk_count = len(chunks)
            doc.extracted_text_path = str(txt_path)
            doc.processed_at = datetime.now(timezone.utc).replace(tzinfo=None)
        else:
            vectorstore.delete_document(document_id)
            txt_path.unlink(missing_ok=True)
    log.info("Document %s indexé (%d chunks, %d pages, %d en OCR)",
             document_id, len(chunks), result.pages, result.ocr_pages)
