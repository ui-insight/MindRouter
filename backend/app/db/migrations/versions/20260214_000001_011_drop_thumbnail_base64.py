############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# 011_drop_thumbnail_base64.py: Drop thumbnail_base64 column
#     after migrate_thumbnails.py has moved data to filesystem
#
# IMPORTANT: Run scripts/migrate_thumbnails.py BEFORE this migration.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Drop thumbnail_base64 column from chat_attachments

Revision ID: 011
Revises: 010
Create Date: 2026-02-14 00:00:01.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.mysql import MEDIUMTEXT

# revision identifiers, used by Alembic
revision = "011"
down_revision = "010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("chat_attachments", "thumbnail_base64")


def downgrade() -> None:
    op.add_column(
        "chat_attachments",
        sa.Column("thumbnail_base64", MEDIUMTEXT, nullable=True),
    )
