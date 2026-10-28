"""comptes utilisateurs, propriétaire des documents, partages

Revision ID: 0003

Les documents existants gardent `owner_id` NULL (visibles des seuls admins) : les attribuer avec
    python -m scripts.create_admin --email ... --adopt-orphans
"""
import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"


def upgrade():
    op.create_table(
        "users",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("email", sa.String(254), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("role", sa.String(20), nullable=False, server_default="user"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("last_login_at", sa.DateTime, nullable=True),
    )
    op.create_index("ix_users_id", "users", ["id"])
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.add_column("documents", sa.Column("owner_id", sa.Integer, sa.ForeignKey("users.id", ondelete="RESTRICT"),
                                         nullable=True))
    op.create_index("ix_documents_owner_id", "documents", ["owner_id"])
    # Unicité du contenu par propriétaire (et non plus globale). Nom par défaut PostgreSQL de `unique=True`.
    op.execute("ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_sha256_key")
    op.create_unique_constraint("uq_documents_owner_sha256", "documents", ["owner_id", "sha256"])

    op.create_table(
        "document_shares",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("document_id", sa.Integer, sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("permission", sa.String(10), nullable=False, server_default="read"),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.UniqueConstraint("document_id", "user_id", name="uq_document_shares_doc_user"),
    )
    op.create_index("ix_document_shares_document_id", "document_shares", ["document_id"])
    op.create_index("ix_document_shares_user_id", "document_shares", ["user_id"])


def downgrade():
    op.drop_table("document_shares")
    op.drop_constraint("uq_documents_owner_sha256", "documents", type_="unique")
    op.create_unique_constraint("documents_sha256_key", "documents", ["sha256"])
    op.drop_index("ix_documents_owner_id", table_name="documents")
    op.drop_column("documents", "owner_id")
    op.drop_table("users")
