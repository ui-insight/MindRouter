"""Add reranking modality to models and requests tables.

Revision ID: 026
Revises: 025
Create Date: 2026-02-22
"""

from alembic import op


revision = "026"
down_revision = "025"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE models MODIFY COLUMN modality "
        "ENUM('chat','completion','embedding','multimodal','reranking') "
        "NOT NULL DEFAULT 'chat'"
    )
    op.execute(
        "ALTER TABLE requests MODIFY COLUMN modality "
        "ENUM('chat','completion','embedding','multimodal','reranking') "
        "NULL"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE models MODIFY COLUMN modality "
        "ENUM('chat','completion','embedding','multimodal') "
        "NOT NULL DEFAULT 'chat'"
    )
    op.execute(
        "ALTER TABLE requests MODIFY COLUMN modality "
        "ENUM('chat','completion','embedding','multimodal') "
        "NULL"
    )
