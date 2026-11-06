############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 090_add_reasoning_history_override.py: Add the per-model
#     reasoning_history_override column: whether the earlier
#     assistant reasoning clients send back is forwarded to the
#     model (NULL = the fleet setting reasoning.forward_history).
#
# Mirrors the *_override columns added in 018.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add models.reasoning_history_override.

Revision ID: 090
Revises: 089

MariaDB / OPS NOTES (DDL here is NON-TRANSACTIONAL):

  `models` has a few hundred rows; a trailing NULLable column with no
  default is an instant metadata change on MariaDB 10.3+, so
  ALGORITHM=INSTANT, LOCK=NONE is requested explicitly. FALLBACK: if the
  server rejects INSTANT, re-run the statement without the ALGORITHM/LOCK
  clause.
"""

from alembic import op

revision = "090"
down_revision = "089"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE models ADD COLUMN reasoning_history_override BOOLEAN NULL, "
        "ALGORITHM=INSTANT, LOCK=NONE"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE models DROP COLUMN reasoning_history_override")
