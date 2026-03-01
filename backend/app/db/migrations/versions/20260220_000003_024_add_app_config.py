"""Add app_config key-value table with chat config seed data.

Revision ID: 024
Revises: 023
Create Date: 2026-02-20
"""

from alembic import op
import sqlalchemy as sa


revision = "024"
down_revision = "023"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "app_config",
        sa.Column("key", sa.String(100), primary_key=True),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("description", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )

    # Seed default chat config
    op.execute(
        "INSERT INTO app_config (`key`, value, description) VALUES "
        "('chat.core_models', "
        "'[\"openai/gpt-oss-20b\",\"openai/gpt-oss-120b\",\"qwen/qwen3.5-400b\"]', "
        "'List of model names shown by default in the chat dropdown'), "
        "('chat.default_model', "
        "'\"openai/gpt-oss-120b\"', "
        "'Default model pre-selected in the chat dropdown')"
    )


def downgrade() -> None:
    op.drop_table("app_config")
