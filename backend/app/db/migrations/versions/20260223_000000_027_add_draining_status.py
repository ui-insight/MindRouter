"""Add draining status to backends table.

Revision ID: 027
Revises: 026
Create Date: 2026-02-23
"""

from alembic import op


revision = "027"
down_revision = "026"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE backends MODIFY COLUMN status "
        "ENUM('healthy','unhealthy','disabled','draining','unknown') "
        "NOT NULL DEFAULT 'unknown'"
    )


def downgrade() -> None:
    # Move any draining backends to disabled before shrinking the enum
    op.execute("UPDATE backends SET status = 'disabled' WHERE status = 'draining'")
    op.execute(
        "ALTER TABLE backends MODIFY COLUMN status "
        "ENUM('healthy','unhealthy','disabled','unknown') "
        "NOT NULL DEFAULT 'unknown'"
    )
