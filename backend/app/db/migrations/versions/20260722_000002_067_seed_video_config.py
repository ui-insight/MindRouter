############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 067_seed_video_config.py: Seed default vid.* config and video
#     retention values in the app_config table (idempotent).
#
# Follows the 029 seed template (backtick-quoted `key`, SELECT-then-INSERT).
# Per-model tunables live here (editable from /admin/video-config with no
# redeploy), NOT in settings.py. See docs/video-generation-plan.md.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Seed default vid.* config + video retention.

Revision ID: 067
Revises: 066
Create Date: 2026-07-22
"""

import json

import sqlalchemy as sa
from alembic import op

revision = "067"
down_revision = "066"
branch_labels = None
depends_on = None

# (key, value, description). vid.enabled is seeded FALSE — deploying the
# migration must NOT turn the feature on; an admin flips it deliberately.
_DEFAULTS = [
    ("vid.enabled", False, "Master switch for video generation (off by default; enable in admin)"),
    ("vid.default_model", "lightricks/ltx-2.3-distilled", "Default video model (served-model-name)"),
    ("vid.default_size", "1280x704", "Default resolution WIDTHxHEIGHT"),
    ("vid.allowed_sizes", "1280x704,704x1280,1024x576,768x448", "Legal preset resolutions (torch.compile warm set)"),
    ("vid.default_seconds", 5, "Default clip duration (seconds)"),
    ("vid.allowed_durations", "4,5,8,10,12,15,20", "Legal clip durations (seconds)"),
    ("vid.default_fps", 24, "Default frames per second"),
    ("vid.default_quality", "standard", "Default quality tier (draft|standard|final)"),
    ("vid.max_shots", 1, "Max shots per job (v1 = single clip; raised in Phase 2)"),
    ("vid.max_total_seconds", 90, "Max total video length per job (seconds)"),
    ("vid.max_continue_chain", 3, "Max last-frame chain length before drift (set from Phase 4 measurement)"),
    ("vid.max_concurrent_jobs_per_user", 1, "In-flight jobs allowed per user (fairness on a 1-render GPU)"),
    ("vid.max_retries_per_shot", 2, "Per-shot render retries (connect/5xx-at-submit only)"),
    ("vid.token_cost_per_second", 2000, "Token-equivalent charged per second of output (PLACEHOLDER — set from Phase 0)"),
    ("vid.user_storage_cap_gb", 50, "Per-user video storage cap (GB)"),
    ("vid.policy", "", "Natural-language content policy for the LLM judge (empty = no judge in v1)"),
    ("vid.judge_model", "", "Primary content-policy judge model (empty = disabled)"),
    ("vid.keyframe_model", "", "FLUX model for keyframe/reference generation (Phase 3; empty = disabled)"),
    ("vid.watermark_text", "AI-generated — University of Idaho MindRouter", "Machine-generated disclosure label"),
    ("retention.user_videos_days", 90, "Days before a user's generated videos are purged (0 = keep forever)"),
    ("retention.video_shot_intermediates_days", 7, "Days before per-shot intermediate renders are purged"),
]


def upgrade() -> None:
    app_config = sa.table(
        "app_config",
        sa.column("key", sa.String),
        sa.column("value", sa.Text),
        sa.column("description", sa.String),
    )
    conn = op.get_bind()
    for key, value, description in _DEFAULTS:
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
        op.execute(sa.text("DELETE FROM app_config WHERE `key` = :k"), {"k": key})
