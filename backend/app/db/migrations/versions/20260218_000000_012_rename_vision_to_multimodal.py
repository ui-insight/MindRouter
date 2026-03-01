############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# 012_rename_vision_to_multimodal.py: Rename supports_vision
#     to supports_multimodal, add multimodal_override, update
#     modality enum from 'vision' to 'multimodal'
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Rename vision to multimodal across backends, models, and modality enum

Revision ID: 012
Revises: 011
Create Date: 2026-02-18 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic
revision = "012"
down_revision = "011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Rename columns on backends and models tables
    op.alter_column(
        "backends",
        "supports_vision",
        new_column_name="supports_multimodal",
        existing_type=sa.Boolean(),
        existing_nullable=False,
        existing_server_default=sa.text("0"),
    )
    op.alter_column(
        "models",
        "supports_vision",
        new_column_name="supports_multimodal",
        existing_type=sa.Boolean(),
        existing_nullable=False,
        existing_server_default=sa.text("0"),
    )

    # 2. Add multimodal_override column to models (nullable, default NULL)
    op.add_column(
        "models",
        sa.Column("multimodal_override", sa.Boolean(), nullable=True),
    )

    # 3. Expand modality enum to include 'multimodal' on both tables
    #    MariaDB requires ALTER COLUMN to change enum values
    op.execute(
        "ALTER TABLE models MODIFY COLUMN modality "
        "ENUM('chat','completion','embedding','vision','multimodal') NOT NULL DEFAULT 'chat'"
    )
    op.execute(
        "ALTER TABLE requests MODIFY COLUMN modality "
        "ENUM('chat','completion','embedding','vision','multimodal') NOT NULL"
    )

    # 4. Migrate existing 'vision' rows to 'multimodal'
    op.execute("UPDATE models SET modality = 'multimodal' WHERE modality = 'vision'")
    op.execute("UPDATE requests SET modality = 'multimodal' WHERE modality = 'vision'")

    # 5. Shrink enum to remove 'vision'
    op.execute(
        "ALTER TABLE models MODIFY COLUMN modality "
        "ENUM('chat','completion','embedding','multimodal') NOT NULL DEFAULT 'chat'"
    )
    op.execute(
        "ALTER TABLE requests MODIFY COLUMN modality "
        "ENUM('chat','completion','embedding','multimodal') NOT NULL"
    )


def downgrade() -> None:
    # 1. Expand enum to include 'vision' again
    op.execute(
        "ALTER TABLE models MODIFY COLUMN modality "
        "ENUM('chat','completion','embedding','vision','multimodal') NOT NULL DEFAULT 'chat'"
    )
    op.execute(
        "ALTER TABLE requests MODIFY COLUMN modality "
        "ENUM('chat','completion','embedding','vision','multimodal') NOT NULL"
    )

    # 2. Migrate 'multimodal' back to 'vision'
    op.execute("UPDATE models SET modality = 'vision' WHERE modality = 'multimodal'")
    op.execute("UPDATE requests SET modality = 'vision' WHERE modality = 'multimodal'")

    # 3. Shrink enum to remove 'multimodal'
    op.execute(
        "ALTER TABLE models MODIFY COLUMN modality "
        "ENUM('chat','completion','embedding','vision') NOT NULL DEFAULT 'chat'"
    )
    op.execute(
        "ALTER TABLE requests MODIFY COLUMN modality "
        "ENUM('chat','completion','embedding','vision') NOT NULL"
    )

    # 4. Drop multimodal_override column
    op.drop_column("models", "multimodal_override")

    # 5. Rename columns back
    op.alter_column(
        "backends",
        "supports_multimodal",
        new_column_name="supports_vision",
        existing_type=sa.Boolean(),
        existing_nullable=False,
        existing_server_default=sa.text("0"),
    )
    op.alter_column(
        "models",
        "supports_multimodal",
        new_column_name="supports_vision",
        existing_type=sa.Boolean(),
        existing_nullable=False,
        existing_server_default=sa.text("0"),
    )
