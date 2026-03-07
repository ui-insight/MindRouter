############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 029_seed_retention_config.py: Seed default retention
#     configuration values in the app_config table.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Seed default data retention config values.

Revision ID: 029
Revises: 028
Create Date: 2026-03-07
"""

import json

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic
revision = "029"
down_revision = "028"
branch_labels = None
depends_on = None

_DEFAULTS = [
    ("retention.requests.tier1_days", 90, "Days before request data is archived (0 = disabled)"),
    ("retention.requests.tier2_days", 730, "Days archived requests are kept before purge"),
    ("retention.chat.tier1_days", 90, "Days before chat data is archived (0 = disabled)"),
    ("retention.chat.tier2_days", 730, "Days archived chat data is kept before purge"),
    ("retention.telemetry.tier1_days", 30, "Days before telemetry is purged (0 = disabled)"),
    ("retention.telemetry.tier2_days", 0, "Days archived telemetry is kept (0 = no archival)"),
    ("retention.cleanup_interval", 3600, "Seconds between retention runs"),
    ("retention.batch_size", 500, "Rows per archive+delete batch"),
]


def upgrade() -> None:
    app_config = sa.table(
        "app_config",
        sa.column("key", sa.String),
        sa.column("value", sa.Text),
        sa.column("description", sa.String),
    )

    for key, value, description in _DEFAULTS:
        # Only insert if not already present (idempotent)
        conn = op.get_bind()
        existing = conn.execute(
            sa.text("SELECT 1 FROM app_config WHERE `key` = :k"),
            {"k": key},
        ).fetchone()
        if not existing:
            op.execute(
                app_config.insert().values(
                    key=key,
                    value=json.dumps(value),
                    description=description,
                )
            )


def downgrade() -> None:
    for key, _, _ in _DEFAULTS:
        op.execute(
            sa.text("DELETE FROM app_config WHERE `key` = :k"),
            {"k": key},
        )
