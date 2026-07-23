############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 065_add_video_modality_and_engine.py: Add the video_generation
#     modality, the video backend engine type, and the per-user
#     video_generation_enabled access flag (default off).
#
# Mirrors 057 (image_generation modality + diffusion engine) and
# 058 (image_generation_enabled) — see docs/video-generation-plan.md.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add video_generation modality, video engine, and user access flag.

Revision ID: 065
Revises: 064

MariaDB / OPS NOTES (DDL here is NON-TRANSACTIONAL — a failure mid-migration
leaves partial state that must be cleaned up by hand; each statement below is
independent, apply/inspect in order):

  1. `requests` is the largest table in prod. We only APPEND one value to the
     end of an 8-value ENUM, so storage stays 1 byte and MariaDB can do this
     in place without a rewrite. We request ALGORITHM=INSTANT, LOCK=NONE
     explicitly rather than letting the server choose. FALLBACK: if the server
     rejects INSTANT on this version, re-run the single failing statement
     without the ALGORITHM/LOCK clause (as migration 057 did successfully at a
     smaller table size) — it will pick INPLACE. Dry-run against a restored
     copy first.
  2. `models` and `backends` are small; plain ALTER is fine.
  3. The `users` column add is a metadata-only default; trivial.
"""

import sqlalchemy as sa
from alembic import op

revision = "065"
down_revision = "064"
branch_labels = None
depends_on = None

# Full value lists spelled out (do not rely on the ORM enum at migration time).
OLD_MODALITY = (
    "'chat','completion','embedding','multimodal','reranking','tts','stt',"
    "'image_generation'"
)
NEW_MODALITY = (
    "'chat','completion','embedding','multimodal','reranking','tts','stt',"
    "'image_generation','video_generation'"
)
OLD_ENGINE = "'ollama','vllm','diffusion'"
NEW_ENGINE = "'ollama','vllm','diffusion','video'"


def upgrade():
    # 1. requests.modality — append video_generation (large table; instant append)
    op.execute(
        f"ALTER TABLE requests MODIFY COLUMN modality "
        f"ENUM({NEW_MODALITY}) NOT NULL, ALGORITHM=INSTANT, LOCK=NONE"
    )

    # 2. models.modality — append video_generation
    op.execute(
        f"ALTER TABLE models MODIFY COLUMN modality "
        f"ENUM({NEW_MODALITY}) NOT NULL DEFAULT 'chat'"
    )

    # 3. backends.engine — append video
    op.execute(
        f"ALTER TABLE backends MODIFY COLUMN engine "
        f"ENUM({NEW_ENGINE}) NOT NULL"
    )

    # 4. users.video_generation_enabled — per-user access flag, default off
    op.add_column(
        "users",
        sa.Column(
            "video_generation_enabled",
            sa.Boolean(),
            nullable=False,
            server_default="0",
        ),
    )


def downgrade():
    # Reverse order. Narrowing an ENUM (removing a value) is NOT an instant
    # operation, so no ALGORITHM clause here. Any row using the removed value
    # must be reassigned first or the ALTER will fail.
    op.drop_column("users", "video_generation_enabled")

    op.execute(
        f"ALTER TABLE backends MODIFY COLUMN engine "
        f"ENUM({OLD_ENGINE}) NOT NULL"
    )
    op.execute(
        f"ALTER TABLE models MODIFY COLUMN modality "
        f"ENUM({OLD_MODALITY}) NOT NULL DEFAULT 'chat'"
    )
    op.execute(
        f"ALTER TABLE requests MODIFY COLUMN modality "
        f"ENUM({OLD_MODALITY}) NOT NULL"
    )
