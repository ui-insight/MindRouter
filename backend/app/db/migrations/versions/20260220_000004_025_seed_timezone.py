"""Seed app.timezone default config value.

Revision ID: 025
Revises: 024
Create Date: 2026-02-20
"""

from alembic import op


revision = "025"
down_revision = "024"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "INSERT INTO app_config (`key`, value, description) VALUES "
        "('app.timezone', '\"America/Los_Angeles\"', "
        "'IANA timezone for displaying dates in the web UI')"
    )


def downgrade() -> None:
    op.execute("DELETE FROM app_config WHERE `key` = 'app.timezone'")
