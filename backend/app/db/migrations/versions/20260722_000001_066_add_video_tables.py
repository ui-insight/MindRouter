############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 066_add_video_tables.py: Create the four video-generation tables
#     (video_projects, video_assets, video_jobs, video_shots).
#
# Full schema created up front (see docs/video-generation-plan.md) so later
# phases (i2v, keyframes, storyboards, source clips, assembly) are additive
# code, not a schema redo. v1 only exercises single-shot generated projects.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Create video_projects / video_assets / video_jobs / video_shots.

Revision ID: 066
Revises: 065

NOTE: MariaDB DDL is non-transactional — if this migration fails midway,
manual cleanup may be required. All four tables are new (no lock risk on
existing data). FKs are declared inline in create_table, so error 1553 is not
triggered now — but ANY future migration dropping an FK-backed index on these
tables MUST drop the FK constraint first (MariaDB 1553).

Create Date: 2026-07-22
"""

import sqlalchemy as sa
from alembic import op

revision = "066"
down_revision = "065"
branch_labels = None
depends_on = None

# Full ENUM value lists spelled out (do not rely on the ORM at migration time).
QUALITY = sa.Enum("draft", "standard", "final", name="video_quality")
SEED_POLICY = sa.Enum("fixed", "per_shot", name="video_seed_policy")
ASSET_KIND = sa.Enum(
    "reference", "keyframe", "upload", "source_clip", "shot_output", "final",
    name="video_asset_kind",
)
JOB_STATUS = sa.Enum(
    "queued", "planning", "rendering", "assembling", "completed", "failed", "cancelled",
    name="video_job_status",
)
SHOT_STATUS = sa.Enum(
    "pending", "queued", "rendering", "rendered", "failed", "skipped",
    name="video_shot_status",
)
SHOT_TYPE = sa.Enum("generated", "source", name="video_shot_type")
TRANSITION = sa.Enum("cut", "crossfade", "continue", name="video_transition")


def upgrade() -> None:
    # 1. video_projects
    op.create_table(
        "video_projects",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("title", sa.String(200), nullable=True),
        sa.Column("model", sa.String(200), nullable=False),
        sa.Column("size", sa.String(20), nullable=False),
        sa.Column("fps", sa.Integer(), nullable=False, server_default="24"),
        sa.Column("quality", QUALITY, nullable=False, server_default="standard"),
        sa.Column("style_prompt", sa.Text(), nullable=True),
        sa.Column("seed_policy", SEED_POLICY, nullable=False, server_default="fixed"),
        sa.Column("base_seed", sa.BigInteger(), nullable=True),
        sa.Column("storyboard", sa.JSON(), nullable=True),
        sa.Column("generate_audio", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_video_projects_user_id", "video_projects", ["user_id"])
    op.create_index("ix_video_projects_created_at", "video_projects", ["created_at"])

    # 2. video_assets
    op.create_table(
        "video_assets",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("project_id", sa.BigInteger(), sa.ForeignKey("video_projects.id"), nullable=True),
        sa.Column("kind", ASSET_KIND, nullable=False),
        sa.Column("entity_label", sa.String(100), nullable=True),
        sa.Column("storage_path", sa.String(500), nullable=False),
        sa.Column("content_type", sa.String(60), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("poster_path", sa.String(500), nullable=True),
        sa.Column("policy_verdict", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_video_assets_user_id", "video_assets", ["user_id"])
    op.create_index("ix_video_assets_project_id", "video_assets", ["project_id"])
    op.create_index("ix_video_assets_sha256", "video_assets", ["sha256"])

    # 3. video_jobs
    op.create_table(
        "video_jobs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("job_uuid", sa.String(64), nullable=False),
        sa.Column("project_id", sa.BigInteger(), sa.ForeignKey("video_projects.id"), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("api_key_id", sa.Integer(), sa.ForeignKey("api_keys.id"), nullable=True),
        sa.Column("request_id", sa.BigInteger(), sa.ForeignKey("requests.id"), nullable=True),
        sa.Column("status", JOB_STATUS, nullable=False, server_default="queued"),
        sa.Column("progress", sa.Float(), nullable=False, server_default="0"),
        sa.Column("shots_total", sa.Integer(), nullable=True),
        sa.Column("shots_done", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("priority", sa.Float(), nullable=False, server_default="0"),
        sa.Column("claimed_by", sa.String(64), nullable=True),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("gpu_seconds", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("token_equivalent", sa.BigInteger(), nullable=True),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("output_asset_id", sa.BigInteger(), sa.ForeignKey("video_assets.id"), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("callback_url", sa.String(500), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_uuid", name="uq_video_jobs_job_uuid"),
    )
    op.create_index("ix_video_jobs_status_priority", "video_jobs", ["status", "priority", "id"])
    op.create_index("ix_video_jobs_user_id", "video_jobs", ["user_id"])
    op.create_index("ix_video_jobs_created_at", "video_jobs", ["created_at"])
    op.create_index("ix_video_jobs_heartbeat", "video_jobs", ["status", "heartbeat_at"])

    # 4. video_shots
    op.create_table(
        "video_shots",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.BigInteger(), sa.ForeignKey("video_jobs.id"), nullable=False),
        sa.Column("shot_index", sa.Integer(), nullable=False),
        sa.Column("shot_type", SHOT_TYPE, nullable=False, server_default="generated"),
        sa.Column("prompt", sa.Text(), nullable=True),
        sa.Column("seconds", sa.Float(), nullable=False),
        sa.Column("num_frames", sa.Integer(), nullable=True),
        sa.Column("transition", TRANSITION, nullable=False, server_default="cut"),
        sa.Column("transition_seconds", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("camera", sa.String(60), nullable=True),
        sa.Column("seed", sa.BigInteger(), nullable=True),
        sa.Column("first_frame_asset_id", sa.BigInteger(), sa.ForeignKey("video_assets.id"), nullable=True),
        sa.Column("last_frame_asset_id", sa.BigInteger(), sa.ForeignKey("video_assets.id"), nullable=True),
        sa.Column("source_asset_id", sa.BigInteger(), sa.ForeignKey("video_assets.id"), nullable=True),
        sa.Column("trim_start_seconds", sa.Float(), nullable=True),
        sa.Column("trim_end_seconds", sa.Float(), nullable=True),
        sa.Column("reference_asset_ids", sa.JSON(), nullable=True),
        sa.Column("status", SHOT_STATUS, nullable=False, server_default="pending"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("backend_id", sa.Integer(), sa.ForeignKey("backends.id"), nullable=True),
        sa.Column("backend_job_id", sa.String(120), nullable=True),
        sa.Column("output_asset_id", sa.BigInteger(), sa.ForeignKey("video_assets.id"), nullable=True),
        sa.Column("render_ms", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id", "shot_index", name="uq_video_shots_job_index"),
    )
    op.create_index("ix_video_shots_status", "video_shots", ["status"])


def downgrade() -> None:
    # Reverse creation order to respect FK dependencies.
    op.drop_index("ix_video_shots_status", table_name="video_shots")
    op.drop_table("video_shots")

    op.drop_index("ix_video_jobs_heartbeat", table_name="video_jobs")
    op.drop_index("ix_video_jobs_created_at", table_name="video_jobs")
    op.drop_index("ix_video_jobs_user_id", table_name="video_jobs")
    op.drop_index("ix_video_jobs_status_priority", table_name="video_jobs")
    op.drop_table("video_jobs")

    op.drop_index("ix_video_assets_sha256", table_name="video_assets")
    op.drop_index("ix_video_assets_project_id", table_name="video_assets")
    op.drop_index("ix_video_assets_user_id", table_name="video_assets")
    op.drop_table("video_assets")

    op.drop_index("ix_video_projects_created_at", table_name="video_projects")
    op.drop_index("ix_video_projects_user_id", table_name="video_projects")
    op.drop_table("video_projects")
