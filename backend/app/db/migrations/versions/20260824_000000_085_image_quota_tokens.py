############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 085_image_quota_tokens.py: seed the flat per-image quota
#     charge (Admin -> Images).
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Flat quota cost per generated image.

Revision ID: 085
Revises: 084

Diffusion backends return no `usage` block, so image requests fell through to
the prompt-text estimator and were billed for the length of the prompt string:
measured in production, a 24-character prompt cost 6 tokens and a 485-character
one cost 121, with completion_tokens 0 on every single request. Meanwhile each
image occupies one of nine max_concurrent=1 diffusion workers for ~7s of
exclusive GPU time — the scarcest capacity on the cluster.

The default of 10,000 is one average chat exchange here (measured: 10,663
total tokens across 122,913 completed chat requests), so the rule is simply
"an image costs about as much as a conversation turn". At current volume
(~4,600 images/30d) that adds ~46M tokens against ~3,441M of chat, so it
changes per-user fairness without distorting cluster accounting.

0 disables the charge, mirroring search.quota_tokens_per_request.
"""

import json

from alembic import op

revision = "085"
down_revision = "084"
branch_labels = None
depends_on = None

_KEY = "img.quota_tokens_per_image"
_DEFAULT = 10000


def upgrade() -> None:
    op.get_bind().exec_driver_sql(
        "INSERT IGNORE INTO app_config (`key`, value, description) VALUES (%s, %s, %s)",
        (_KEY, json.dumps(_DEFAULT), "Flat quota tokens charged per generated image (added in 085)"),
    )


def downgrade() -> None:
    op.get_bind().exec_driver_sql(
        "DELETE FROM app_config WHERE `key` = %s", (_KEY,)
    )
