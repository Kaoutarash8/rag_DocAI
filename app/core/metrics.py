"""Métriques Prometheus : latence HTTP, erreurs, tokens LLM, étapes de recherche, ingestion, file d'attente.

Exposées par l'API sur GET /metrics et, pour les workers Celery, sur WORKER_METRICS_PORT.
Les libellés `route` sont les gabarits (`/documents/{document_id}`), jamais les URL réelles : cardinalité bornée.
"""

import logging

from prometheus_client import (CONTENT_TYPE_LATEST, REGISTRY, Counter, Histogram,
                               generate_latest, start_http_server)
from prometheus_client.core import GaugeMetricFamily

log = logging.getLogger(__name__)
_LAT = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)

HTTP_REQUESTS = Counter("docai_http_requests_total", "Requêtes HTTP", ["method", "route", "status"])
HTTP_LATENCY = Histogram("docai_http_request_duration_seconds", "Latence HTTP", ["method", "route"], buckets=_LAT)
ERRORS = Counter("docai_errors_total", "Erreurs applicatives", ["kind"])  # llm | internal | ingestion
LLM_TOKENS = Counter("docai_llm_tokens_total", "Tokens LLM", ["provider", "model", "kind"])  # prompt | completion
LLM_LATENCY = Histogram("docai_llm_request_duration_seconds", "Latence des appels LLM", ["provider"], buckets=_LAT)
RETRIEVAL_LATENCY = Histogram("docai_retrieval_stage_duration_seconds", "Latence par étape de recherche",
                              ["profile", "stage"], buckets=_LAT)
INGEST_DOCS = Counter("docai_ingestion_documents_total", "Documents traités", ["status"])  # indexed | failed
INGEST_DURATION = Histogram("docai_ingestion_duration_seconds", "Durée de traitement d'un document",
                            buckets=(1, 5, 15, 30, 60, 120, 300, 600, 1800, 3600))


def observe_http(method: str, route: str, status: int, seconds: float) -> None:
    HTTP_REQUESTS.labels(method, route, str(status)).inc()
    HTTP_LATENCY.labels(method, route).observe(seconds)


def observe_retrieval(profile: str, timings: dict) -> None:
    for key, ms in timings.items():
        if key.endswith("_ms"):
            RETRIEVAL_LATENCY.labels(profile, key[:-3]).observe(ms / 1000)


def record_llm(provider: str, model: str, prompt_tokens: int | None, completion_tokens: int | None,
               seconds: float) -> None:
    LLM_LATENCY.labels(provider).observe(seconds)
    if prompt_tokens:
        LLM_TOKENS.labels(provider, model, "prompt").inc(prompt_tokens)
    if completion_tokens:
        LLM_TOKENS.labels(provider, model, "completion").inc(completion_tokens)


class _QueueCollector:
    """Calculé à chaque scrape : documents par statut (SQL) et longueur de la file Celery (Redis)."""

    def collect(self):
        by_status = GaugeMetricFamily("docai_documents", "Documents par statut", labels=["status"])
        try:
            from sqlalchemy import func

            from app.database import SessionLocal
            from app.models.document import Document

            with SessionLocal() as db:
                for status, n in db.query(Document.status, func.count(Document.id)).group_by(Document.status):
                    by_status.add_metric([getattr(status, "value", str(status))], n)
        except Exception:
            log.debug("Métrique docai_documents indisponible", exc_info=True)
        yield by_status

        queue = GaugeMetricFamily("docai_queue_length", "Tâches d'ingestion en attente dans Redis")
        try:
            from app.core.redis import get_redis

            client = get_redis()
            if client is not None:
                queue.add_metric([], client.llen("celery"))
        except Exception:
            log.debug("Métrique docai_queue_length indisponible", exc_info=True)
        yield queue


_collector_registered = False


def register_queue_collector() -> None:
    global _collector_registered
    if not _collector_registered:
        REGISTRY.register(_QueueCollector())
        _collector_registered = True


def render() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


def start_worker_server(port: int) -> None:
    start_http_server(port)
