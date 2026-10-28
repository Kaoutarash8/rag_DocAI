"""versionnage de l'index : registre des collections et version par document

Revision ID: 0004
"""
import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"


def upgrade():
    op.add_column("documents", sa.Column("index_version", sa.String(20), nullable=True))
    op.create_table(
        "index_collections",
        sa.Column("name", sa.String(120), primary_key=True),
        sa.Column("index_version", sa.String(20), nullable=False),
        sa.Column("embedding_model", sa.String(200), nullable=False),
        sa.Column("chunk_strategy", sa.String(20), nullable=False),
        sa.Column("chunk_size", sa.Integer, nullable=False),
        sa.Column("chunk_overlap", sa.Integer, nullable=False),
        sa.Column("params", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False),
    )


def downgrade():
    op.drop_table("index_collections")
    op.drop_column("documents", "index_version")
