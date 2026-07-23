"""Contract tests for the video-generation foundation layer (v1).

Source inspection (no db/telemetry imports) for the model-layer and migration
pieces, plus direct import of the import-chain-clean modules (canonical_schemas,
settings) to exercise the schemas and the docker-compose passthrough rule.

Guards the foundation described in docs/video-generation-plan.md so later phases
build on a stable base. See MEMORY: tests must NEVER import
backend.app.db.models / services.inference / core.telemetry.* at module level.
"""

import importlib.util
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

MIGRATION_065 = (
    "backend/app/db/migrations/versions/"
    "20260722_000000_065_add_video_modality_and_engine.py"
)
MIGRATION_066 = (
    "backend/app/db/migrations/versions/20260722_000001_066_add_video_tables.py"
)
MIGRATION_067 = (
    "backend/app/db/migrations/versions/20260722_000002_067_seed_video_config.py"
)


def _read(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return f.read()


def _load_canonical():
    """Direct-load canonical_schemas.py (pydantic-only, import-chain-clean) so
    this test is immune to sys.modules pollution from sibling spec-load tests."""
    spec = importlib.util.spec_from_file_location(
        "canonical_schemas_vid",
        os.path.join(ROOT, "backend/app/core/canonical_schemas.py"),
        submodule_search_locations=[],
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- Canonical schemas ----------------------------------------------------

def test_video_request_defaults_match_v1_preset_shape():
    cs = _load_canonical()
    r = cs.CanonicalVideoRequest(model="lightricks/ltx-2.3-distilled", prompt="a fox")
    assert r.size == "1280x704"
    assert r.seconds == "5"          # string, per OpenAI video convention
    assert r.fps == 24
    assert r.quality == "standard"
    # v1 is text-to-video only: no image/keyframe/storyboard fields yet.
    assert not hasattr(r, "input_reference")
    assert not hasattr(r, "storyboard")


def test_video_job_object_is_openai_shaped():
    cs = _load_canonical()
    j = cs.CanonicalVideoJob(id="vid-deadbeef", created_at=1, model="m")
    assert j.object == "video"
    assert j.status == cs.VideoJobStatus.QUEUED
    # Stock SDK polling loops expect in_progress, not "processing".
    assert cs.VideoJobStatus.IN_PROGRESS.value == "in_progress"
    assert {s.value for s in cs.VideoJobStatus} == {
        "queued", "in_progress", "completed", "failed", "cancelled"
    }


# --- Model-layer enums / column (source inspection, no db import) ----------

def test_models_declare_video_enum_members():
    src = _read("backend/app/db/models.py")
    assert 'VIDEO = "video"' in src                       # BackendEngine
    assert 'VIDEO_GENERATION = "video_generation"' in src  # Modality
    # per-user access flag, default off, mirroring image_generation_enabled
    assert "video_generation_enabled" in src
    assert 'server_default="0"' in src


def test_scheduler_job_modality_has_video():
    src = _read("backend/app/core/scheduler/queue.py")
    assert 'VIDEO_GENERATION = "video_generation"' in src


def test_registry_discovers_video_modality():
    # A video-engine backend's model must be discovered as VIDEO_GENERATION,
    # else the runner's get_backends_with_model(..., VIDEO_GENERATION) can't route.
    src = _read("backend/app/core/telemetry/registry.py")
    assert "BackendEngine.VIDEO" in src
    assert "Modality.VIDEO_GENERATION" in src


# --- Migration 065 (source inspection) ------------------------------------

def test_migration_065_chain_and_enum_widening():
    src = _read(MIGRATION_065)
    assert 'revision = "065"' in src
    assert 'down_revision = "064"' in src
    # widens all three enums with the appended values
    assert "'image_generation','video_generation'" in src
    assert "'diffusion','video'" in src
    # requests is the big table: instant append requested explicitly
    assert "ALGORITHM=INSTANT, LOCK=NONE" in src
    # adds the user flag
    assert 'add_column(' in src and '"video_generation_enabled"' in src


def test_migration_066_creates_four_tables_with_indexes():
    src = _read(MIGRATION_066)
    assert 'revision = "066"' in src
    assert 'down_revision = "065"' in src
    for table in ("video_projects", "video_assets", "video_jobs", "video_shots"):
        assert f'create_table(\n        "{table}"' in src or f'"{table}"' in src
    # the claim-query index and crash re-adoption index
    assert "ix_video_jobs_status_priority" in src
    assert "ix_video_jobs_heartbeat" in src
    # source_clip provisions (decision #4) are present from v1
    assert "source_clip" in src
    assert "source_asset_id" in src
    # unique shot index
    assert "uq_video_shots_job_index" in src


def test_migration_067_seeds_config_off_by_default():
    src = _read(MIGRATION_067)
    assert 'revision = "067"' in src
    assert 'down_revision = "066"' in src
    # deploying the migration must NOT enable the feature
    assert '("vid.enabled", False' in src
    # storage cap = 50 (locked decision #6) and retention seeded
    assert '("vid.user_storage_cap_gb", 50' in src
    assert '"retention.user_videos_days"' in src
    # idempotent SELECT-then-INSERT with backtick-quoted key (029 template)
    assert "SELECT 1 FROM app_config WHERE `key`" in src


def test_migration_chain_is_linear_065_066_067():
    revs = {}
    for path in (MIGRATION_065, MIGRATION_066, MIGRATION_067):
        src = _read(path)
        rev = re.search(r'revision = "(\d+)"', src).group(1)
        down = re.search(r'down_revision = "(\d+)"', src).group(1)
        revs[rev] = down
    assert revs == {"065": "064", "066": "065", "067": "066"}


# --- docker-compose passthrough rule (the #2 project-rule violation) -------

def test_every_video_setting_has_docker_compose_passthrough():
    # Source-inspect settings.py for video_* field names (no import — immune to
    # sys.modules pollution) and assert each has a docker-compose.yml passthrough.
    import re

    settings_src = _read("backend/app/settings.py")
    compose = _read("docker-compose.yml")
    video_fields = re.findall(r"^\s{4}(video_[a-z0-9_]+)\s*:", settings_src, re.MULTILINE)
    assert video_fields, "expected video_* settings to exist"
    missing = [f for f in video_fields if f.upper() not in compose]
    assert not missing, f"settings missing docker-compose passthrough: {missing}"
