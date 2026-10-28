"""Worker Celery : traite les documents mis en file par l'API (QUEUE_BACKEND=celery).

    celery -A app.worker:celery_app worker --loglevel=INFO --concurrency=1

Monter en charge : `docker compose up -d --scale worker=3`. Chaque worker charge ses propres modèles
(OCR, embeddings) : prévoir la RAM en conséquence. Prérequis : serveur Chroma (CHROMA_HOST) et Redis.
"""

import logging

from celery import Celery
from celery.signals import worker_ready

from app.core.config import get_settings
from app.core.locks import Busy
from app.core.logs import setup_logging

setup_logging()
log = logging.getLogger(__name__)
_s = get_settings()

celery_app = Celery("docai", broker=_s.redis_url or "memory://", backend=None)
celery_app.conf.update(
    task_acks_late=True,                 # la tâche n'est acquittée qu'une fois terminée : reprise si le worker meurt
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,        # tâches très longues : un worker ne réserve pas de file d'avance
    task_ignore_result=True,
    broker_connection_retry_on_startup=True,
    broker_transport_options={"visibility_timeout": 6 * 3600},  # > durée max d'un document, sinon re-livraison
    timezone="UTC",
)


@celery_app.task(name="docai.process_document", bind=True, max_retries=30)
def process_document_task(self, document_id: int) -> None:
    from app.services import processing

    try:
        processing.process_document(document_id)
    except Busy:  # un autre worker a le bail (ou un worker mort dont le bail n'a pas encore expiré)
        raise self.retry(countdown=60)


@worker_ready.connect
def _on_ready(**_):
    from app.core import metrics
    from app.services import processing

    try:
        metrics.start_worker_server(_s.worker_metrics_port)
    except OSError:  # port déjà pris (plusieurs workers dans le même conteneur)
        log.warning("Port de métriques %s indisponible", _s.worker_metrics_port)
    processing.recover_pending()  # republie les documents PENDING/PROCESSING (doublons neutralisés par le bail)
