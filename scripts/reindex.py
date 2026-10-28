"""Ré-indexe tous les documents (après changement de INDEX_PROFILES, de stratégie ou de modèle).

    docker compose exec api python -m scripts.reindex
"""
from app.database import SessionLocal
from app.models.document import Document, DocumentStatus
from app.services import processing, profiles


def main() -> None:
    specs = profiles.index_specs()
    print("Collections alimentées :", ", ".join(f"{s.collection} ({s.strategy}, {s.size} mots, {s.embedding_model})"
                                               for s in specs))
    processing.start()
    with SessionLocal() as db:
        docs = db.query(Document).filter(Document.status.in_([DocumentStatus.INDEXED, DocumentStatus.FAILED])).all()
        for d in docs:
            d.status, d.attempts, d.error_message = DocumentStatus.PENDING, 0, None
        db.commit()
        ids = [d.id for d in docs]
    for i in ids:
        processing.submit(i)
    print(f"{len(ids)} document(s) en cours de ré-indexation...")
    processing.wait_idle()  # thread : attend ; celery : les workers traitent en arrière-plan


if __name__ == "__main__":
    main()
