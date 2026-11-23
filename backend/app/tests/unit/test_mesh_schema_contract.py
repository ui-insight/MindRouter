"""Contract tests for the image-to-mesh foundation (migration 091 + models).

Source inspection only: tests must NEVER import backend.app.db.models /
services.inference / core.telemetry.* at module level (see TESTING.md), so the
model layer, the migration, the compose passthrough and the retention wiring
are checked as text, the same way test_video_schema_contract.py does it.
"""

import ast
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
MIGRATION_091 = "backend/app/db/migrations/versions/20261123_000000_091_add_mesh_generation.py"


def _read(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return f.read()


def _enum_values(src, class_name):
    """The string values of a `class X(str, PyEnum)` body, in order."""
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return [
                stmt.value.value
                for stmt in node.body
                if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            ]
    raise AssertionError(f"{class_name} not found")


def _migration_list(src, name):
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id == name:
            return node.value
    raise AssertionError(f"{name} not found")


def _sql_values(expr_src):
    return re.findall(r"'([a-z_]+)'", expr_src)


def test_models_declare_mesh_members():
    src = _read("backend/app/db/models.py")
    assert _enum_values(src, "BackendEngine")[-1] == "mesh"
    assert _enum_values(src, "Modality")[-1] == "mesh_generation"
    assert _enum_values(src, "MeshJobStatus") == ["queued", "rendering", "completed", "failed", "cancelled"]
    assert re.search(r"mesh_generation_enabled: Mapped\[bool\] = mapped_column\(Boolean, default=False, nullable=False, server_default=\"0\"\)", src)
    assert '__tablename__ = "mesh_jobs"' in src


def test_migration_enum_lists_match_the_models():
    """The migration spells the ENUMs out; they must equal the ORM's, in order
    (MariaDB ENUM storage is positional)."""
    models = _read("backend/app/db/models.py")
    mig = _read(MIGRATION_091)
    ns = {}
    exec(compile(ast.Module(body=[
        n for n in ast.parse(mig).body
        if isinstance(n, ast.Assign) and n.targets[0].id in ("OLD_MODALITY", "NEW_MODALITY", "OLD_ENGINE", "NEW_ENGINE")
    ], type_ignores=[]), "m", "exec"), ns)
    assert _sql_values(ns["NEW_MODALITY"]) == _enum_values(models, "Modality")
    assert _sql_values(ns["NEW_ENGINE"]) == _enum_values(models, "BackendEngine")
    # Only one value appended to each.
    assert _sql_values(ns["NEW_MODALITY"])[:-1] == _sql_values(ns["OLD_MODALITY"])
    assert _sql_values(ns["NEW_ENGINE"])[:-1] == _sql_values(ns["OLD_ENGINE"])


def test_migration_chain_and_job_status_enum():
    mig = _read(MIGRATION_091)
    assert 'revision = "091"' in mig and 'down_revision = "090"' in mig
    status_call = re.search(r"JOB_STATUS = sa\.Enum\((.*?)name=", mig, re.S).group(1)
    assert re.findall(r'"([a-z_]+)"', status_call) == _enum_values(_read("backend/app/db/models.py"), "MeshJobStatus")
    # The large-table ALTER is an instant append.
    assert "ALTER TABLE requests MODIFY COLUMN modality" in mig
    assert "ALGORITHM=INSTANT, LOCK=NONE" in mig


def test_migration_columns_match_the_model():
    """Every MeshJob column exists in the create_table call, and vice versa."""
    models = _read("backend/app/db/models.py")
    body = models[models.index("class MeshJob(Base, TimestampMixin)"):]
    body = body[: body.index("__table_args__")]
    model_cols = set(re.findall(r"^    ([a-z_0-9]+): Mapped\[", body, re.M)) | {"created_at", "updated_at"}
    mig = _read(MIGRATION_091)
    table = mig[mig.index('"mesh_jobs",'): mig.index("op.create_index(\"ix_mesh_jobs_status_priority\"")]
    mig_cols = set(re.findall(r'sa\.Column\("([a-z_0-9]+)"', table))
    assert model_cols == mig_cols


def test_mesh_is_seeded_off_and_retention_has_a_window():
    mig = _read(MIGRATION_091)
    assert '("mesh.enabled", False,' in mig
    assert '("retention.user_meshes_days", 90,' in mig
    assert '("mesh.token_cost_textured", 10000,' in mig
    assert '("mesh.user_storage_cap_gb", 20,' in mig
    retention = _read("backend/app/services/retention.py")
    assert '"retention.user_meshes_days": 90' in retention
    assert '("meshes", _meshes)' in retention


def test_request_archival_detaches_mesh_jobs():
    """mesh_jobs.request_id is a RESTRICT FK: archival must SET NULL it first,
    or deleting an archived request raises IntegrityError."""
    retention = _read("backend/app/services/retention.py")
    detach = retention[retention.index("async def _detach_request_references"):]
    detach = detach[: detach.index("\nasync def ", 10)]
    assert "update(MeshJob)" in detach and "MeshJob.request_id.in_(request_ids)" in detach


def test_user_backend_and_key_deletion_know_about_mesh_jobs():
    crud = _read("backend/app/db/crud.py")
    delete_user = crud[crud.index("async def delete_user("): crud.index("# API Key CRUD")]
    assert "delete(MeshJob).where(MeshJob.user_id == user_id)" in delete_user
    # Requests are deleted after mesh jobs (mesh_jobs.request_id references them).
    assert delete_user.index("delete(MeshJob)") < delete_user.index("delete(Request).where(Request.user_id")
    delete_backend = crud[crud.index("async def delete_backend("):]
    delete_backend = delete_backend[: delete_backend.index("\nasync def ", 10)]
    assert "update(MeshJob).where(MeshJob.backend_id == backend_id)" in delete_backend
    assert '("mesh_jobs", MeshJob, MeshJob.api_key_id)' in crud


def _crud_function(name):
    crud = _read("backend/app/db/crud.py")
    body = crud[crud.index(f"async def {name}("):]
    return body[: body.index("\nasync def ", 10)]


def test_job_state_changes_are_exactly_once():
    """Terminal transitions are conditional UPDATEs, and only the caller whose
    UPDATE matched refunds, so racing callers can't double-refund or
    resurrect a finished job."""
    end = _crud_function("_end_mesh_job")
    assert "MeshJob.status.in_(MESH_ACTIVE_STATUSES)" in end
    assert "ended = (result.rowcount or 0) == 1" in end and "if ended and row.token_equivalent" in end
    for name in ("fail_mesh_job", "mark_mesh_job_cancelled"):
        assert "_end_mesh_job(" in _crud_function(name), name
    complete = _crud_function("complete_mesh_job")
    assert "MeshJob.status == MeshJobStatus.RENDERING" in complete and "rowcount" in complete
    requeue = _crud_function("requeue_mesh_job")
    assert "job.status != MeshJobStatus.RENDERING" in requeue and "if job.cancel_requested:" in requeue
    cancel = _crud_function("request_cancel_mesh_job")
    assert "MeshJob.status == MeshJobStatus.QUEUED" in cancel and "rowcount" in cancel


def test_cancel_flagged_jobs_are_still_claimed_and_missing_rows_count_as_cancelled():
    claim = _crud_function("claim_next_mesh_job")
    assert "cancel_requested" not in claim.split('"""')[2]   # code, not the docstring
    assert "return True if row is None" in _crud_function("is_mesh_job_cancel_requested")


def test_every_mesh_setting_is_passed_through_compose():
    """pydantic-settings never sees .env inside the container: each mesh_*
    setting must have its own line in docker-compose.yml."""
    settings = _read("backend/app/settings.py")
    compose = _read("docker-compose.yml")
    fields = re.findall(r"^    (mesh_[a-z_]+):", settings, re.M)
    assert len(fields) >= 10
    for field in fields:
        assert f"- {field.upper()}=${{{field.upper()}:-" in compose, field
    assert "/data/mesh" in compose


def test_registry_maps_mesh_backends_to_mesh_modality():
    reg = _read("backend/app/core/telemetry/registry.py")
    assert re.search(
        r"elif backend\.engine == BackendEngine\.MESH:\s+modality = Modality\.MESH_GENERATION", reg
    )


def test_mesh_models_stay_out_of_the_llm_catalogs():
    """/v1/models et al. publish an allow-list; it must not grow mesh."""
    src = _read("backend/app/api/models_api.py")
    block = src[src.index("CATALOG_MODALITIES = frozenset("): src.index("def is_catalog_model")]
    assert "MESH" not in block
