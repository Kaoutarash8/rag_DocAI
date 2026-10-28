"""progression du traitement (pages_total, pages_done)

Revision ID: 0002
"""
import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"


def upgrade():
    op.add_column("documents", sa.Column("pages_total", sa.Integer, nullable=False, server_default="0"))
    op.add_column("documents", sa.Column("pages_done", sa.Integer, nullable=False, server_default="0"))


def downgrade():
    op.drop_column("documents", "pages_done")
    op.drop_column("documents", "pages_total")
