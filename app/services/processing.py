"""Traitement des documents : OCR -> chunks -> embeddings -> vectorstore.

Deux backends (QUEUE_BACKEND) :
- `thread` (défaut) : file bornée dans le process de l'API ; les documents restés en PENDING/PROCESSING
  (crash, redémarrage) sont repris au démarrage ;
- `celery` : l'API publie une tâche dans Redis, un ou plusieurs workers (`app.worker`) la traitent.
  Un bail Redis par document évite qu'un même document soit traité par deux workers.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from app.core import locks, metrics
from app.core.config import get_settings
from app.database import session_scope
from app.models.document import Document, DocumentStatus
from app.services import embeddings, extraction, profiles, summary, vectorstore, versioning
from app.services.chunking import Chunker

log = logging.getLogger(__name__)
_executor: ThreadPoolExecutor | None = None


def _celery() -> bool:
    return get_settings().queue_backend == "celery"


def start() -> None:
    """Thread : démarre l'executor et reprend les documents bloqués. Celery : rien (c'est le worker qui s'en charge)."""
    global _executor
    if _celery():
        return
    _executor = ThreadPoolExecutor(
        max_workers=get_settings().worker_threads, thread_name_prefix="ingest"
    )
    recover_pending()


def stop() -> None:
    if _executor:
        _executor.shutdown(wait=False, cancel_futures=True)


def submit(document_id: int) -> None:
    if _celery():
        from app.worker import process_document_task

        process_document_task.delay(document_id)
        return
    if _executor is None:
        start()
    _executor.submit(process_document, document_id)


def wait_idle() -> None:
    """Scripts : attend la fin des traitements (thread). Avec Celery, les workers continuent seuls."""
    if _executor:
        _executor.shutdown(wait=True)


def recover_pending() -> None:
    with session_scope() as db:
        ids = [d.id for d in db.query(Document).filter(
            Document.status.in_([DocumentStatus.PENDING, DocumentStatus.PROCESSING]))]
    for i in ids:
        log.info("Reprise du document %s", i)
        submit(i)


def _fail(document_id: int, message: str) -> None:
    with session_scope() as db:
        doc = db.get(Document, document_id)
        if doc:
            doc.status = DocumentStatus.FAILED
            doc.error_message = message[:1000]


def process_document(document_id: int) -> None:
    """Point d'entrée d'un traitement. Lève `locks.Busy` si un autre worker traite déjà ce document."""
    lease = locks.acquire(document_id)  # None sans Redis
    try:
        _process(document_id, lease)
    finally:
        if lease:
            lease.release()


def _process(document_id: int, lease) -> None:
    s = get_settings()
    t0 = time.perf_counter()
    with session_scope() as db:
        doc = db.get(Document, document_id)
        if doc is None:
            return
        if doc.status == DocumentStatus.INDEXED:  # tâche en double (reprise après crash) : déjà fait
            return
        if doc.attempts >= s.max_attempts:
            doc.status = DocumentStatus.FAILED
            doc.error_message = f"Abandon après {doc.attempts} tentatives"
            metrics.INGEST_DOCS.labels("failed").inc()
            return
        doc.status = DocumentStatus.PROCESSING
        doc.attempts += 1
        doc.error_message = None
        path = s.upload_dir / doc.stored_name
        file_type, filename = doc.file_type, doc.filename

    txt_path = s.text_dir / f"{document_id}.txt"
    try:  # travail lourd HORS transaction
        if not path.exists():
            raise FileNotFoundError("Fichier source introuvable sur le disque")
        stats = _index_in_batches(document_id, filename, path, file_type, txt_path, lease)
    except _Cancelled:
        log.info("Document %s supprimé pendant le traitement : nettoyage", document_id)
        vectorstore.delete_document(document_id)
        txt_path.unlink(missing_ok=True)
        summary.cleanup(document_id)
        return
    except Exception as e:
        log.exception("Échec du traitement du document %s", document_id)
        vectorstore.delete_document(document_id)  # pas de chunks partiels orphelins
        txt_path.unlink(missing_ok=True)
        summary.cleanup(document_id)
        _fail(document_id, f"{type(e).__name__}: {e}")
        metrics.INGEST_DOCS.labels("failed").inc()
        metrics.ERRORS.labels("ingestion").inc()
        return

    with session_scope() as db:
        doc = db.get(Document, document_id)
        if doc:  # peut avoir été supprimé pendant le traitement
            doc.status = DocumentStatus.INDEXED
            doc.chunk_count = stats["chunks"]
            doc.pages_done = doc.pages_total
            doc.index_version = s.index_version
            doc.extracted_text_path = str(txt_path)
            doc.processed_at = datetime.now(timezone.utc).replace(tzinfo=None)
        else:
            vectorstore.delete_document(document_id)
            txt_path.unlink(missing_ok=True)
            summary.cleanup(document_id)
    metrics.INGEST_DOCS.labels("indexed").inc()
    metrics.INGEST_DURATION.observe(time.perf_counter() - t0)
    log.info("Document %s indexé (%d chunks, %d unités, %d en OCR, %d tableaux, %d figures)",
             document_id, stats["chunks"], stats["units"], stats["ocr"],
             stats["tables"], stats["figures"])


class _Cancelled(Exception):
    pass


def _set_progress(document_id: int, *, total: int | None = None, done: int | None = None) -> bool:
    """Met à jour la progression ; False si le document a été supprimé entre-temps."""
    with session_scope() as db:
        doc = db.get(Document, document_id)
        if doc is None:
            return False
        if total is not None:
            doc.pages_total = total
        if done is not None:
            doc.pages_done = done
    return True


def _index_in_batches(document_id: int, filename: str, path, file_type: str, txt_path, lease=None) -> dict:
    """Extraction -> chunks -> embeddings -> index, par lots de pages (mémoire bornée).

    Une passe d'extraction alimente TOUTES les collections de INDEX_PROFILES (un Chunker par
    collection). Le texte et la structure (`segments.jsonl`, pour les résumés) sont écrits au fil
    de l'eau ; un échec à mi-parcours est nettoyé par l'appelant.
    """
    s = get_settings()
    specs = profiles.index_specs()
    versioning.register_collections(specs)
    total, pages = extraction.iter_pages(path, file_type)
    if not _set_progress(document_id, total=total, done=0):
        raise _Cancelled
    vectorstore.delete_document(document_id)  # ré-indexation propre (toutes collections)
    summary.cleanup(document_id)

    chunkers = {sp.collection: Chunker(sp.strategy, sp.size, sp.overlap) for sp in specs}
    stats = {"chunks": 0, "units": 0, "ocr": 0, "tables": 0, "figures": 0}
    batch: list = []

    def index(chunks_by_spec) -> None:
        for n, (sp, chunks) in enumerate(chunks_by_spec):
            if not chunks:
                continue
            vectors = embeddings.embed_passages(sp.embedding_model, [c.text for c in chunks])
            vectorstore.index_chunks(sp.collection, document_id, filename, chunks, vectors)
            if n == 0:  # la 1re collection (profil par défaut) sert de référence au compteur
                stats["chunks"] += len(chunks)

    def flush() -> None:
        if not batch:
            return
        index([(sp, chunkers[sp.collection].feed(batch)) for sp in specs])
        batch.clear()
        if not _set_progress(document_id, done=stats["units"]):
            raise _Cancelled

    with txt_path.open("w", encoding="utf-8") as out, \
            summary.segments_path(document_id).open("w", encoding="utf-8") as segs:
        for page in pages:
            if lease:
                lease.extend()  # le worker est vivant : on prolonge le bail Redis
            text = page.text
            if text.strip():
                label = f"\n\n--- page {page.page} ---\n" if page.page else "\n\n"
                out.write(label + text)
            for sg in page.segments:
                if sg.text.strip():
                    segs.write(json.dumps({"page": page.page, "section": sg.section, "kind": sg.kind,
                                           "text": sg.text}, ensure_ascii=False) + "\n")
            stats["units"] += 1
            stats["ocr"] += page.ocr
            stats["tables"] += sum(sg.kind == "table" for sg in page.segments)
            stats["figures"] += sum(sg.kind == "figure" for sg in page.segments)
            batch.append(page)
            if len(batch) >= s.page_batch_size:
                flush()
        flush()
        index([(sp, chunkers[sp.collection].finish()) for sp in specs])  # sections encore ouvertes

    if stats["chunks"] == 0:
        raise ValueError("Aucun texte détecté dans le document")
    return stats
