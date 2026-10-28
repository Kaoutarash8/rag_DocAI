from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String, Text

from app.database import Base


class IndexCollection(Base):
    """Registre des collections vectorielles : quel modèle et quels paramètres les ont produites."""
    __tablename__ = "index_collections"

    name = Column(String(120), primary_key=True)            # ex. documents_v3
    index_version = Column(String(20), nullable=False)
    embedding_model = Column(String(200), nullable=False)
    chunk_strategy = Column(String(20), nullable=False)
    chunk_size = Column(Integer, nullable=False)
    chunk_overlap = Column(Integer, nullable=False)
    params = Column(Text, nullable=True)                    # JSON : spécification complète
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), nullable=False)
