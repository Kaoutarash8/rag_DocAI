"""Importer tous les modèles ici : les clés étrangères (documents.owner_id -> users.id)
doivent être résolvables dans TOUT processus (API, worker Celery, scripts)."""
from app.models import document, index_collection, user  # noqa: F401