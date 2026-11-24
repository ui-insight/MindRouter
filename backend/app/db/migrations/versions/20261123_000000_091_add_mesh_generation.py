############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 091_add_mesh_generation.py: Image-to-3D mesh generation.
#     Adds the 'mesh_generation' modality, the 'mesh' backend
#     engine, the per-user users.mesh_generation_enabled flag,
#     the mesh_jobs table, and the mesh.* / retention defaults.
#
# Mirrors 065-067 (video). See docs/mesh-api.md.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
#
############################################################

"""Add mesh generation (modality, engine, user flag, mesh_jobs, config).

Revision ID: 091
Revises: 090

MariaDB / OPS NOTES (DDL here is NON-TRANSACTIONAL — a failure mid-migration
leaves partial state that must be cleaned up by hand; apply/inspect in order):

  1. `requests` is the largest table in prod. We only APPEND one value to the
     end of a 9-value ENUM, so storage stays 1 byte and MariaDB can do it in
     place. ALGORITHM=INSTANT, LOCK=NONE is requested explicitly. FALLBACK: if
     the server rejects INSTANT, re-run that single statement without the
     ALGORITHM/LOCK clause (it will pick INPLACE).
  2. `models` and `backends` are small; `backends` also requests INSTANT.
  3. `users.mesh_generation_enabled` is a metadata-only default (off).
  4. `mesh_jobs` is new (no lock risk). Its FKs are declared inline, so any
     future migration dropping an FK-backed index on it MUST drop the FK
     constraint first (MariaDB 1553).
  5. mesh.enabled is seeded FALSE: deploying this migration does not turn the
     feature on; an admin enables it, and each user, deliberately.
  6. The downgrade NARROWS the enums and REFUSES to run while any row still
     uses the new values.
"""

import json

import sqlalchemy as sa
from alembic import op

revision = "091"
down_revision = "090"
branch_labels = None
depends_on = None

# Full value lists spelled out (never rely on the ORM enum at migration time).
OLD_MODALITY = (
    "'chat','completion','embedding','multimodal','reranking','tts','stt',"
    "'image_generation','video_generation'"
)
NEW_MODALITY = OLD_MODALITY + ",'mesh_generation'"
OLD_ENGINE = "'ollama','vllm','diffusion','video','tts','stt','dlp','decision','matting'"
NEW_ENGINE = OLD_ENGINE + ",'mesh'"

JOB_STATUS = sa.Enum(
    "queued", "rendering", "completed", "failed", "cancelled",
    name="mesh_job_status",
)

# (key, value, description). Seeded only when absent, so re-running or an
# admin's earlier value is never overwritten.
_DEFAULTS = [
    ("mesh.enabled", False, "Master switch for image-to-mesh generation (off by default; enable in admin)"),
    ("mesh.default_model", "default-mesh", "Default mesh model or alias (served-model-name)"),
    ("mesh.token_cost_textured", 10000, "Token-equivalent charged per textured mesh job"),
    ("mesh.token_cost_untextured", 5000, "Token-equivalent charged per shape-only (untextured) mesh job"),
    ("mesh.user_storage_cap_gb", 20, "Per-user mesh storage cap (GB): inputs + GLBs + previews"),
    ("mesh.max_concurrent_jobs_per_user", 2, "In-flight mesh jobs allowed per user (fairness on a one-job-at-a-time GPU)"),
    ("mesh.max_image_upload_mb", 10, "Largest input image accepted (MB)"),
    ("mesh.default_texture_size", 2048, "Default PBR texture size (px)"),
    ("mesh.allowed_texture_sizes", "1024,2048,4096", "Allowed PBR texture sizes (px)"),
    # 40k faces: smooth on organic shapes; fewer faces barely save time or
    # size (textures dominate) but facet visibly. 300k: 1M-face jobs held a GPU
    # for minutes (Nov 2026 measurements).
    ("mesh.default_target_faces", 40000, "Default face count the mesh is decimated toward"),
    ("mesh.min_target_faces", 5000, "Smallest target face count a request may ask for"),
    ("mesh.max_target_faces", 300000, "Largest target face count a request may ask for"),
    # Per-model caps and settings (Admin -> Meshes). Hunyuan3D starts capped at
    # 100k faces, the largest size measured (about 90 s per job on an H200).
    ("mesh.model_settings", {"tencent/hunyuan3d-2.1": {"max_faces": 100000}},
     "Per-model caps and settings: {model: {max_faces, max_texture_size, options}}"),
    ("mesh.license_acceptance", {}, "Model licenses accepted by an admin: {family: {accepted_by, accepted_at}}"),
    ("mesh.background_removal", True, "Cut the subject out with the matting server (img.matting_url) before submitting"),
    ("retention.user_meshes_days", 90, "Days before a user's finished mesh jobs and files are purged (0 = keep forever)"),
]


# The starting default mesh model: Hunyuan3D 2.1 (most realistic textures in
# the Nov 2026 bake-off). mesh.default_model points at this alias; it is seeded
# only when no default-mesh alias exists, and admins can repoint it.
DEFAULT_MESH_ALIAS = ("default-mesh", "tencent/hunyuan3d-2.1", "Default image-to-mesh model (Hunyuan3D 2.1)")


def upgrade() -> None:
    # 1. requests.modality — append mesh_generation (large table; instant append)
    op.execute(
        f"ALTER TABLE requests MODIFY COLUMN modality "
        f"ENUM({NEW_MODALITY}) NOT NULL, ALGORITHM=INSTANT, LOCK=NONE"
    )
    # 2. models.modality — append mesh_generation
    op.execute(
        f"ALTER TABLE models MODIFY COLUMN modality "
        f"ENUM({NEW_MODALITY}) NOT NULL DEFAULT 'chat'"
    )
    # 3. backends.engine — append mesh
    op.execute(
        f"ALTER TABLE backends MODIFY COLUMN engine "
        f"ENUM({NEW_ENGINE}) NOT NULL, ALGORITHM=INSTANT, LOCK=NONE"
    )
    # 4. users.mesh_generation_enabled — per-user access flag, default off
    op.add_column(
        "users",
        sa.Column("mesh_generation_enabled", sa.Boolean(), nullable=False, server_default="0"),
    )
    # 5. mesh_jobs
    op.create_table(
        "mesh_jobs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("job_uuid", sa.String(64), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("api_key_id", sa.Integer(), sa.ForeignKey("api_keys.id"), nullable=True),
        sa.Column("request_id", sa.BigInteger(), sa.ForeignKey("requests.id"), nullable=True),
        sa.Column("model", sa.String(255), nullable=False),
        sa.Column("status", JOB_STATUS, nullable=False, server_default="queued"),
        sa.Column("progress", sa.Float(), nullable=False, server_default="0"),
        sa.Column("stage", sa.String(32), nullable=True),
        sa.Column("priority", sa.Float(), nullable=False, server_default="0"),
        sa.Column("seed", sa.BigInteger(), nullable=True),
        sa.Column("texture", sa.Boolean(), nullable=False, server_default="1"),
        sa.Column("texture_size", sa.Integer(), nullable=False, server_default="2048"),
        sa.Column("target_faces", sa.Integer(), nullable=False, server_default="40000"),
        sa.Column("remove_background", sa.String(8), nullable=False, server_default="auto"),
        sa.Column("model_options", sa.JSON(), nullable=True),
        sa.Column("input_content_type", sa.String(64), nullable=True),
        sa.Column("input_size_bytes", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("output_size_bytes", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("output_sha256", sa.String(64), nullable=True),
        sa.Column("preview_size_bytes", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("faces", sa.Integer(), nullable=True),
        sa.Column("vertices", sa.Integer(), nullable=True),
        sa.Column("claimed_by", sa.String(64), nullable=True),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("backend_id", sa.Integer(), sa.ForeignKey("backends.id"), nullable=True),
        sa.Column("backend_job_id", sa.String(120), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("gpu_seconds", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("token_equivalent", sa.BigInteger(), nullable=True),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_uuid", name="uq_mesh_jobs_job_uuid"),
    )
    op.create_index("ix_mesh_jobs_status_priority", "mesh_jobs", ["status", "priority", "id"])
    op.create_index("ix_mesh_jobs_user_id", "mesh_jobs", ["user_id"])
    op.create_index("ix_mesh_jobs_created_at", "mesh_jobs", ["created_at"])
    op.create_index("ix_mesh_jobs_heartbeat", "mesh_jobs", ["status", "heartbeat_at"])

    # 6. Config defaults (insert only when absent).
    app_config = sa.table(
        "app_config",
        sa.column("key", sa.String),
        sa.column("value", sa.Text),
        sa.column("description", sa.String),
    )
    conn = op.get_bind()
    for key, value, description in _DEFAULTS:
        existing = conn.execute(
            sa.text("SELECT 1 FROM app_config WHERE `key` = :k"), {"k": key}
        ).fetchone()
        if not existing:
            op.execute(
                app_config.insert().values(
                    key=key, value=json.dumps(value), description=description
                )
            )

    # 7. The default-mesh alias (insert only when absent).
    alias, target, description = DEFAULT_MESH_ALIAS
    if not conn.execute(
        sa.text("SELECT 1 FROM model_aliases WHERE alias_name = :a"), {"a": alias}
    ).fetchone():
        conn.execute(
            sa.text(
                "INSERT INTO model_aliases (alias_name, target_model, description) "
                "VALUES (:a, :t, :d)"
            ),
            {"a": alias, "t": target, "d": description},
        )


def downgrade() -> None:
    bind = op.get_bind()
    for sql, what in (
        ("SELECT COUNT(*) FROM backends WHERE engine = 'mesh'", "backend row(s) with engine='mesh'"),
        ("SELECT COUNT(*) FROM models WHERE modality = 'mesh_generation'", "model row(s) with modality='mesh_generation'"),
        ("SELECT COUNT(*) FROM requests WHERE modality = 'mesh_generation'", "request row(s) with modality='mesh_generation'"),
    ):
        remaining = bind.exec_driver_sql(sql).scalar()
        if remaining:
            raise RuntimeError(
                f"Cannot downgrade: {remaining} {what} remain. Delete or "
                f"reassign them first; narrowing the ENUM would corrupt them."
            )

    for key, _, _ in _DEFAULTS:
        op.execute(sa.text("DELETE FROM app_config WHERE `key` = :k").bindparams(k=key))
    # Remove the alias only if it is still the one we seeded.
    op.execute(
        sa.text("DELETE FROM model_aliases WHERE alias_name = :a AND target_model = :t").bindparams(
            a=DEFAULT_MESH_ALIAS[0], t=DEFAULT_MESH_ALIAS[1]
        )
    )

    # Dropping the table drops its indexes with it. Dropping the FK-backed
    # indexes one by one first would fail with MariaDB error 1553.
    op.drop_table("mesh_jobs")
    op.drop_column("users", "mesh_generation_enabled")
    op.execute(f"ALTER TABLE backends MODIFY COLUMN engine ENUM({OLD_ENGINE}) NOT NULL")
    op.execute(f"ALTER TABLE models MODIFY COLUMN modality ENUM({OLD_MODALITY}) NOT NULL DEFAULT 'chat'")
    op.execute(f"ALTER TABLE requests MODIFY COLUMN modality ENUM({OLD_MODALITY}) NOT NULL")
