############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 089_add_matting_backend_engine.py: Add the 'matting'
#     backend engine type so the matting (background-removal)
#     server behind `background: "transparent"` on the images
#     API can be registered as a backend: health-polled, shown
#     on the backends page and charted through the node's GPU
#     sidecar, while serving NO chat models.
#
# Mirrors 077 (dlp engine) and 087 (decision engine).
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add matting backend engine type.

Revision ID: 089
Revises: 088

MariaDB / OPS NOTES (DDL here is NON-TRANSACTIONAL):

  1. `backends` has tens of rows; this APPENDS one value to the end of an
     8-value ENUM, so the column stays 1 byte and no row rewrite is needed.
     ALGORITHM=INSTANT, LOCK=NONE is requested explicitly. FALLBACK: if the
     server rejects INSTANT, re-run the statement without the ALGORITHM/LOCK
     clause.
  2. The downgrade NARROWS the enum and would corrupt surviving rows, so it
     REFUSES to run while any backend is still engine='matting'.
"""

from alembic import op

revision = "089"
down_revision = "088"
branch_labels = None
depends_on = None

# Spelled out in full: never rely on the ORM enum at migration time.
OLD_ENGINE = "'ollama','vllm','diffusion','video','tts','stt','dlp','decision'"
NEW_ENGINE = "'ollama','vllm','diffusion','video','tts','stt','dlp','decision','matting'"


def upgrade() -> None:
    op.execute(
        f"ALTER TABLE backends MODIFY COLUMN engine "
        f"ENUM({NEW_ENGINE}) NOT NULL, ALGORITHM=INSTANT, LOCK=NONE"
    )


def downgrade() -> None:
    bind = op.get_bind()
    remaining = bind.exec_driver_sql(
        "SELECT COUNT(*) FROM backends WHERE engine = 'matting'"
    ).scalar()
    if remaining:
        raise RuntimeError(
            f"Cannot downgrade: {remaining} backend row(s) still use "
            f"engine='matting'. Delete or re-engine them first "
            f"(SELECT id, name FROM backends WHERE engine='matting';)."
        )

    op.execute(
        f"ALTER TABLE backends MODIFY COLUMN engine "
        f"ENUM({OLD_ENGINE}) NOT NULL"
    )
