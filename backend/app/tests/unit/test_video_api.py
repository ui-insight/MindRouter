"""Unit tests for the public video API (/v1/videos) — v1 text-to-video.

Route-level tests: handlers are awaited as plain coroutines with hand-built
args (no TestClient). The module is direct-loaded via spec_from_file_location
with the heavy DB/registry chains stubbed, so this never imports
backend.app.db.models / services.inference / core.telemetry.* at module level
(see MEMORY). Mirrors the isolation pattern in test_voice_api.py.

Covers:
- _job_to_dict shape + status mapping (fine-grained -> OpenAI external)
- create_video gates: disabled (503), user flag off (403), missing prompt (400),
  disallowed size (400), disallowed duration (400), bad quality (400),
  over per-user concurrency (429), model not found (404)
- create_video happy path: returns 'queued', persists, does NOT block
- get_video: 404 for missing / other-user id (no existence leak)
- cancel_video: flags cancel, cancels a queued job
- list_video_models: capability shape
"""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

_api_dir = Path(__file__).resolve().parents[2] / "api"
_core_dir = Path(__file__).resolve().parents[2] / "core"

# Load the REAL, dependency-free pathsafe (only imports os/re) so video_api's
# `from backend.app.core.pathsafe import ...` resolves to genuine containment
# behavior even though backend.app.core is stubbed below (self-contained; not
# reliant on another test importing it first).
_ps_spec = importlib.util.spec_from_file_location(
    "backend.app.core.pathsafe", _core_dir / "pathsafe.py"
)
_pathsafe = importlib.util.module_from_spec(_ps_spec)
_ps_spec.loader.exec_module(_pathsafe)


class _JS:
    QUEUED = "queued"
    PLANNING = "planning"
    RENDERING = "rendering"
    ASSEMBLING = "assembling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# models is enum-heavy; provide a light stub with the members video_api uses.
_models_stub = MagicMock()
_models_stub.BackendEngine = SimpleNamespace(VIDEO="video")
_models_stub.Modality = SimpleNamespace(VIDEO_GENERATION="video_generation")
_models_stub.VideoJobStatus = _JS
_models_stub.VideoJob = MagicMock
_models_stub.ApiKey = MagicMock
_models_stub.User = MagicMock


# The API modules import the pure `model_availability` helper (404-vs-503
# decision). It has no heavy dependencies, so load the REAL module rather than
# stubbing it — a MagicMock here would make every model look unavailable.
def _load_model_availability():
    import importlib.util as _ilu

    _s = _ilu.spec_from_file_location(
        "backend.app.api.model_availability",
        _api_dir / "model_availability.py",
        submodule_search_locations=[],
    )
    _m = _ilu.module_from_spec(_s)
    _s.loader.exec_module(_m)
    return _m

_STUBS = {
    "backend.app.api.model_availability": _load_model_availability(),
    "backend": MagicMock(),
    "backend.app": MagicMock(),
    "backend.app.api": MagicMock(),
    "backend.app.api.auth": MagicMock(),
    "backend.app.core": MagicMock(),
    "backend.app.core.pathsafe": _pathsafe,
    "backend.app.core.telemetry": MagicMock(),
    "backend.app.core.telemetry.registry": MagicMock(),
    "backend.app.db": MagicMock(),
    "backend.app.db.crud": MagicMock(),
    "backend.app.db.session": MagicMock(),
    "backend.app.db.models": _models_stub,
    "backend.app.settings": MagicMock(
        get_settings=MagicMock(return_value=SimpleNamespace(video_storage_path="/data/video")),
    ),
    "backend.app.logging_config": MagicMock(
        get_logger=MagicMock(return_value=MagicMock()),
        bind_request_context=MagicMock(),
    ),
}

# Force our own clean stubs (so a sibling test's leftover pollution can't break
# the load), then restore sys.modules to its exact prior state afterward (so we
# don't pollute siblings). The loaded module keeps its own references.
_MISSING = object()
_prev = {name: sys.modules.get(name, _MISSING) for name in _STUBS}
for _name, _stub in _STUBS.items():
    sys.modules[_name] = _stub

_spec = importlib.util.spec_from_file_location(
    "video_api", _api_dir / "video_api.py", submodule_search_locations=[]
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

for _name, _old in _prev.items():
    if _old is _MISSING:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _old

create_video = _mod.create_video
create_video_asset = _mod.create_video_asset
get_video = _mod.get_video
cancel_video = _mod.cancel_video
list_videos = _mod.list_videos
list_video_models = _mod.list_video_models
get_video_content = _mod.get_video_content
_job_to_dict = _mod._job_to_dict


# --- helpers --------------------------------------------------------------

def _make_config(overrides=None):
    """Return an async side_effect for crud.get_config_json keyed on the value's
    key, falling back to the caller-supplied default."""
    overrides = overrides or {}

    async def _get(db, key, default=None):
        return overrides.get(key, default)

    return _get


def _auth():
    user = SimpleNamespace(id=7, video_generation_enabled=True)
    api_key = SimpleNamespace(id=3)
    return (user, api_key)


def _request(body):
    req = MagicMock()
    req.json = AsyncMock(return_value=body)
    req.headers = {}
    req.client = SimpleNamespace(host="127.0.0.1")
    return req


def _patch_crud(monkeypatch, config=None, active_jobs=0, reserve_ok=True, cost=10000, storage_bytes=0):
    monkeypatch.setattr(_mod.crud, "get_config_json", _make_config(config))
    monkeypatch.setattr(
        _mod.crud, "count_active_video_jobs_for_user", AsyncMock(return_value=active_jobs)
    )
    monkeypatch.setattr(
        _mod.crud, "get_user_video_storage_bytes", AsyncMock(return_value=storage_bytes)
    )
    monkeypatch.setattr(_mod.crud, "compute_video_token_cost", AsyncMock(return_value=cost))
    monkeypatch.setattr(_mod.crud, "reserve_video_tokens", AsyncMock(return_value=reserve_ok))
    monkeypatch.setattr(_mod.crud, "incr_quota_redis", AsyncMock())
    monkeypatch.setattr(
        _mod.crud, "create_request", AsyncMock(return_value=SimpleNamespace(id=100))
    )
    monkeypatch.setattr(
        _mod.crud, "create_video_project",
        AsyncMock(return_value=SimpleNamespace(id=11, model="lightricks/ltx-2.3-distilled",
                                               size="1280x704", fps=24, quality="standard")),
    )
    monkeypatch.setattr(
        _mod.crud, "create_video_job",
        AsyncMock(return_value=_job(status=_JS.QUEUED)),
    )
    monkeypatch.setattr(
        _mod.crud, "create_video_shot",
        AsyncMock(return_value=SimpleNamespace(
            id=5, seed=123, seconds=5.0,
            first_frame_asset_id=None, last_frame_asset_id=None,
        )),
    )


def _patch_registry(monkeypatch, exists=True):
    reg = MagicMock()
    reg.resolve_alias = MagicMock(side_effect=lambda m: (m, None))
    reg.model_exists = AsyncMock(return_value=exists)
    # Unknown-model fixtures: configured mirrors exists (503 is the
    # configured-but-no-healthy-backend case, covered separately).
    reg.model_is_configured = AsyncMock(return_value=exists)
    monkeypatch.setattr(_mod, "get_registry", lambda: reg)


def _job(status=_JS.QUEUED, uuid="vid-abc123", output_asset_id=None):
    now = SimpleNamespace(timestamp=lambda: 1_700_000_000)
    return SimpleNamespace(
        id=42, job_uuid=uuid, status=status, progress=0.0, created_at=now, started_at=None,
        completed_at=None, expires_at=None, output_asset_id=output_asset_id,
        error_code=None, error_message=None, duration_seconds=None, gpu_seconds=0,
        token_equivalent=None, user_id=7, cancel_requested=False, project_id=11,
    )


def _db():
    db = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    return db


# --- _job_to_dict ---------------------------------------------------------

def test_job_to_dict_maps_rendering_to_in_progress():
    d = _job_to_dict(_job(status=_JS.RENDERING))
    assert d["status"] == "in_progress"
    assert d["object"] == "video"
    assert d["content_url"] is None  # no output asset yet


def test_job_to_dict_exposes_content_url_when_output_ready():
    d = _job_to_dict(_job(status=_JS.COMPLETED, output_asset_id=99))
    assert d["status"] == "completed"
    assert d["content_url"] == "/v1/videos/vid-abc123/content"


# --- create_video gates ---------------------------------------------------

@pytest.mark.asyncio
async def test_create_video_disabled_returns_503(monkeypatch):
    _patch_crud(monkeypatch, config={"vid.enabled": False})
    with pytest.raises(HTTPException) as e:
        await create_video(_request({"prompt": "x"}), db=_db(), auth=_auth())
    assert e.value.status_code == 503


@pytest.mark.asyncio
async def test_create_video_user_flag_off_returns_403(monkeypatch):
    _patch_crud(monkeypatch, config={"vid.enabled": True})
    user, api_key = _auth()
    user.video_generation_enabled = False
    with pytest.raises(HTTPException) as e:
        await create_video(_request({"prompt": "x"}), db=_db(), auth=(user, api_key))
    assert e.value.status_code == 403


@pytest.mark.asyncio
async def test_create_video_missing_prompt_returns_400(monkeypatch):
    _patch_crud(monkeypatch, config={"vid.enabled": True})
    with pytest.raises(HTTPException) as e:
        await create_video(_request({}), db=_db(), auth=_auth())
    assert e.value.status_code == 400


@pytest.mark.asyncio
async def test_create_video_disallowed_size_returns_400(monkeypatch):
    _patch_crud(monkeypatch, config={"vid.enabled": True, "vid.allowed_sizes": "1280x704"})
    _patch_registry(monkeypatch)
    with pytest.raises(HTTPException) as e:
        await create_video(_request({"prompt": "x", "size": "9999x9999"}), db=_db(), auth=_auth())
    assert e.value.status_code == 400


@pytest.mark.asyncio
async def test_create_video_out_of_range_duration_returns_400(monkeypatch):
    # Duration is a whole-second range [4, max]; out-of-range and non-integer are 400.
    _patch_crud(monkeypatch, config={"vid.enabled": True, "vid.max_total_seconds": 90})
    _patch_registry(monkeypatch)
    for bad in ("200", "2", "10.5"):
        with pytest.raises(HTTPException) as e:
            await create_video(_request({"prompt": "x", "seconds": bad}), db=_db(), auth=_auth())
        assert e.value.status_code == 400, f"{bad} should be rejected"


@pytest.mark.asyncio
async def test_create_video_bad_quality_returns_400(monkeypatch):
    _patch_crud(monkeypatch, config={"vid.enabled": True})
    _patch_registry(monkeypatch)
    with pytest.raises(HTTPException) as e:
        await create_video(_request({"prompt": "x", "quality": "ultra"}), db=_db(), auth=_auth())
    assert e.value.status_code == 400


@pytest.mark.asyncio
async def test_create_video_model_not_found_returns_404(monkeypatch):
    _patch_crud(monkeypatch, config={"vid.enabled": True})
    _patch_registry(monkeypatch, exists=False)
    with pytest.raises(HTTPException) as e:
        await create_video(_request({"prompt": "x"}), db=_db(), auth=_auth())
    assert e.value.status_code == 404


@pytest.mark.asyncio
async def test_create_video_over_concurrency_returns_429(monkeypatch):
    _patch_crud(monkeypatch, config={"vid.enabled": True, "vid.max_concurrent_jobs_per_user": 1},
                active_jobs=1)
    _patch_registry(monkeypatch)
    with pytest.raises(HTTPException) as e:
        await create_video(_request({"prompt": "x"}), db=_db(), auth=_auth())
    assert e.value.status_code == 429


@pytest.mark.asyncio
async def test_create_video_over_quota_returns_429(monkeypatch):
    _patch_crud(monkeypatch, config={"vid.enabled": True}, reserve_ok=False)
    _patch_registry(monkeypatch)
    with pytest.raises(HTTPException) as e:
        await create_video(_request({"prompt": "x"}), db=_db(), auth=_auth())
    assert e.value.status_code == 429
    # nothing persisted when the reservation is rejected
    _mod.crud.create_video_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_video_over_storage_cap_returns_507(monkeypatch):
    # 50 GB cap, user already sitting at 60 GB → blocked before any reservation.
    _patch_crud(
        monkeypatch,
        config={"vid.enabled": True, "vid.user_storage_cap_gb": 50},
        storage_bytes=60 * 1024**3,
    )
    _patch_registry(monkeypatch)
    with pytest.raises(HTTPException) as e:
        await create_video(_request({"prompt": "x"}), db=_db(), auth=_auth())
    assert e.value.status_code == 507
    _mod.crud.reserve_video_tokens.assert_not_awaited()
    _mod.crud.create_video_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_video_under_storage_cap_proceeds(monkeypatch):
    # 10 GB used against a 50 GB cap → the gate does not fire (job is created).
    _patch_crud(
        monkeypatch,
        config={"vid.enabled": True, "vid.user_storage_cap_gb": 50},
        storage_bytes=10 * 1024**3,
    )
    _patch_registry(monkeypatch)
    result = await create_video(_request({"prompt": "x"}), db=_db(), auth=_auth())
    assert result["status"] == "queued"
    _mod.crud.create_video_job.assert_awaited()


@pytest.mark.asyncio
async def test_create_video_storage_cap_zero_disables_check(monkeypatch):
    # cap_gb=0 disables the gate even if usage is huge.
    _patch_crud(
        monkeypatch,
        config={"vid.enabled": True, "vid.user_storage_cap_gb": 0},
        storage_bytes=999 * 1024**3,
    )
    _patch_registry(monkeypatch)
    result = await create_video(_request({"prompt": "x"}), db=_db(), auth=_auth())
    assert result["status"] == "queued"


# --- POST /v1/videos/assets (public keyframe upload) — gate coverage ------
class _FakeUpload:
    def __init__(self, content_type="image/png", data=b"\x89PNG" + b"x" * 20):
        self.content_type = content_type
        self._data = data

    async def read(self):
        return self._data


def _auth_no_video():
    return (SimpleNamespace(id=7, video_generation_enabled=False), SimpleNamespace(id=3))


@pytest.mark.asyncio
async def test_video_asset_disabled_returns_503(monkeypatch):
    _patch_crud(monkeypatch, config={"vid.enabled": False})
    with pytest.raises(HTTPException) as e:
        await create_video_asset(MagicMock(), image=_FakeUpload(), db=_db(), auth=_auth())
    assert e.value.status_code == 503


@pytest.mark.asyncio
async def test_video_asset_user_flag_off_returns_403(monkeypatch):
    _patch_crud(monkeypatch, config={"vid.enabled": True})
    with pytest.raises(HTTPException) as e:
        await create_video_asset(MagicMock(), image=_FakeUpload(), db=_db(), auth=_auth_no_video())
    assert e.value.status_code == 403


@pytest.mark.asyncio
async def test_video_asset_non_image_returns_400(monkeypatch):
    _patch_crud(monkeypatch, config={"vid.enabled": True})
    with pytest.raises(HTTPException) as e:
        await create_video_asset(MagicMock(), image=_FakeUpload(content_type="text/plain"), db=_db(), auth=_auth())
    assert e.value.status_code == 400


@pytest.mark.asyncio
async def test_video_asset_over_storage_cap_returns_507(monkeypatch):
    # 50 GB cap, already at 60 GB → rejected before writing anything.
    _patch_crud(
        monkeypatch,
        config={"vid.enabled": True, "vid.user_storage_cap_gb": 50, "vid.max_image_upload_mb": 10},
        storage_bytes=60 * 1024**3,
    )
    with pytest.raises(HTTPException) as e:
        await create_video_asset(MagicMock(), image=_FakeUpload(), db=_db(), auth=_auth())
    assert e.value.status_code == 507


@pytest.mark.asyncio
async def test_create_video_happy_path_returns_queued(monkeypatch):
    _patch_crud(monkeypatch, config={"vid.enabled": True})
    _patch_registry(monkeypatch)
    db = _db()
    result = await create_video(_request({"prompt": "a fox in snow"}), db=db, auth=_auth())
    assert result["status"] == "queued"
    assert result["object"] == "video"
    assert result["content_url"] is None      # returns the job, never the video
    _mod.crud.reserve_video_tokens.assert_awaited_once()   # quota reserved up front
    _mod.crud.incr_quota_redis.assert_awaited_once()        # synced post-commit
    _mod.crud.create_video_job.assert_awaited_once()
    _mod.crud.create_video_shot.assert_awaited_once()
    db.commit.assert_awaited_once()


# --- get / cancel / models ------------------------------------------------

@pytest.mark.asyncio
async def test_get_video_missing_returns_404(monkeypatch):
    monkeypatch.setattr(_mod.crud, "get_video_job_by_uuid", AsyncMock(return_value=None))
    with pytest.raises(HTTPException) as e:
        await get_video("vid-nope", db=_db(), auth=_auth())
    assert e.value.status_code == 404  # not 403 — no existence leak


@pytest.mark.asyncio
async def test_cancel_video_flags_cancel(monkeypatch):
    job = _job(status=_JS.QUEUED)
    monkeypatch.setattr(_mod.crud, "get_video_job_by_uuid", AsyncMock(return_value=job))
    cancel_mock = AsyncMock()
    monkeypatch.setattr(_mod.crud, "request_cancel_video_job", cancel_mock)
    await cancel_video("vid-abc123", db=_db(), auth=_auth())
    cancel_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_content_missing_job_404(monkeypatch):
    monkeypatch.setattr(_mod.crud, "get_video_job_by_uuid", AsyncMock(return_value=None))
    with pytest.raises(HTTPException) as e:
        await get_video_content("vid-nope", db=_db(), auth=_auth())
    assert e.value.status_code == 404


@pytest.mark.asyncio
async def test_content_not_ready_409(monkeypatch):
    job = _job(status=_JS.RENDERING)  # no output_asset_id
    job.output_asset_id = None
    monkeypatch.setattr(_mod.crud, "get_video_job_by_uuid", AsyncMock(return_value=job))
    with pytest.raises(HTTPException) as e:
        await get_video_content("vid-abc123", db=_db(), auth=_auth())
    assert e.value.status_code == 409


@pytest.mark.asyncio
async def test_content_file_missing_on_disk_404(monkeypatch):
    job = _job(status=_JS.COMPLETED, output_asset_id=5)
    monkeypatch.setattr(_mod.crud, "get_video_job_by_uuid", AsyncMock(return_value=job))
    monkeypatch.setattr(
        _mod.crud, "get_video_asset",
        AsyncMock(return_value=SimpleNamespace(storage_path="/no/such/file.mp4", content_type="video/mp4")),
    )
    with pytest.raises(HTTPException) as e:
        await get_video_content("vid-abc123", db=_db(), auth=_auth())
    assert e.value.status_code == 404


@pytest.mark.asyncio
async def test_content_streams_file_response(monkeypatch, tmp_path):
    import os
    # 2.9.26 serves a path RECONSTRUCTED from clean components —
    # <root>/<user.id>/<video_id>.mp4 (user id 7 from _auth, id 'vid-abc123') —
    # never the stored path. Place the file at the reconstructed location.
    user_dir = tmp_path / "7"
    user_dir.mkdir()
    f = user_dir / "vid-abc123.mp4"
    f.write_bytes(b"\x00\x00\x00\x18ftypmp42fakebytes")
    monkeypatch.setattr(_mod, "get_settings", lambda: SimpleNamespace(video_storage_path=str(tmp_path)))
    job = _job(status=_JS.COMPLETED, output_asset_id=5)
    monkeypatch.setattr(_mod.crud, "get_video_job_by_uuid", AsyncMock(return_value=job))
    monkeypatch.setattr(
        _mod.crud, "get_video_asset",
        # storage_path is deliberately bogus — reconstruction must ignore it.
        AsyncMock(return_value=SimpleNamespace(storage_path="/not/used", content_type="video/mp4")),
    )
    resp = await get_video_content("vid-abc123", db=_db(), auth=_auth())
    # FileResponse streams from disk (Range/206 handled by starlette), never
    # loading the whole file into a single in-memory Response.
    from fastapi.responses import FileResponse
    assert isinstance(resp, FileResponse)
    assert os.path.realpath(resp.path) == os.path.realpath(str(f))
    assert resp.media_type == "video/mp4"


@pytest.mark.asyncio
async def test_content_traversal_video_id_neutralized(monkeypatch, tmp_path):
    # A traversal video_id is neutralized by os.path.basename() in the
    # reconstructed path, so it can never read a file outside <root>/<user>/ —
    # even when the DB lookup is mocked to return a job. (2.9.26 reconstruction.)
    root = tmp_path / "video"
    (root / "7").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "passwd.mp4").write_bytes(b"secret")  # must NOT be served
    monkeypatch.setattr(_mod, "get_settings", lambda: SimpleNamespace(video_storage_path=str(root)))
    job = _job(status=_JS.COMPLETED, output_asset_id=5)
    monkeypatch.setattr(_mod.crud, "get_video_job_by_uuid", AsyncMock(return_value=job))
    monkeypatch.setattr(
        _mod.crud, "get_video_asset",
        AsyncMock(return_value=SimpleNamespace(storage_path="/x", content_type="video/mp4")),
    )
    with pytest.raises(HTTPException) as e:
        await get_video_content("../outside/passwd", db=_db(), auth=_auth())
    assert e.value.status_code == 404


@pytest.mark.asyncio
async def test_list_video_models_shape(monkeypatch):
    _patch_crud(monkeypatch, config={"vid.allowed_sizes": "1280x704", "vid.allowed_durations": "5"})
    reg = MagicMock()
    reg.get_all_backends = AsyncMock(
        return_value=[SimpleNamespace(id=1, engine="video")]
    )
    reg.get_backend_models = AsyncMock(
        return_value=[SimpleNamespace(name="lightricks/ltx-2.3-distilled")]
    )
    monkeypatch.setattr(_mod, "get_registry", lambda: reg)
    out = await list_video_models(db=_db(), auth=_auth())
    assert out["object"] == "list"
    assert out["data"][0]["id"] == "lightricks/ltx-2.3-distilled"
    assert out["data"][0]["supports_text_to_video"] is True
    # Image conditioning (start/end keyframes) ships and is honored end-to-end,
    # so the capability flags must advertise it (stale-False regression, 2.8.42).
    assert out["data"][0]["supports_image_to_video"] is True
    assert out["data"][0]["supports_keyframes"] is True
    assert out["data"][0]["max_shots"] == 1


# --- 2.8.42: seed randomization, request-param echo, negative_prompt honesty


@pytest.mark.asyncio
async def test_omitted_seed_is_randomized_and_persisted(monkeypatch):
    """No client seed -> gateway generates one (never the worker's fixed 42
    fallback), persists it on the shot, and echoes it. Two submissions get
    different seeds, so a second take no longer requires editing the prompt."""
    _patch_crud(monkeypatch, config={"vid.enabled": True})
    _patch_registry(monkeypatch)
    seeds = []
    for _ in range(2):
        await create_video(_request({"prompt": "a fox in snow"}), db=_db(), auth=_auth())
        kwargs = _mod.crud.create_video_shot.await_args.kwargs
        assert kwargs["seed"] is not None
        assert 0 <= kwargs["seed"] < 2**31
        seeds.append(kwargs["seed"])
    assert seeds[0] != seeds[1]  # randbelow(2**31): collision odds ~0


@pytest.mark.asyncio
async def test_explicit_seed_passes_through_unchanged(monkeypatch):
    _patch_crud(monkeypatch, config={"vid.enabled": True})
    _patch_registry(monkeypatch)
    await create_video(_request({"prompt": "a fox in snow", "seed": 7}), db=_db(), auth=_auth())
    assert _mod.crud.create_video_shot.await_args.kwargs["seed"] == 7


@pytest.mark.asyncio
async def test_create_echoes_shot_params(monkeypatch):
    """Create response carries seed/seconds/asset ids so honored params are
    distinguishable from silently-dropped ones (external tester gap #2)."""
    _patch_crud(monkeypatch, config={"vid.enabled": True})
    _patch_registry(monkeypatch)
    result = await create_video(_request({"prompt": "a fox in snow"}), db=_db(), auth=_auth())
    assert result["seed"] == 123          # from the _patch_crud shot mock
    assert result["seconds"] == 5.0
    assert result["start_image_asset_id"] is None
    assert result["end_image_asset_id"] is None


def test_job_to_dict_echoes_shot_fields():
    shot = SimpleNamespace(seed=99, seconds=8.0, first_frame_asset_id=3, last_frame_asset_id=4)
    proj = SimpleNamespace(model="m", size="1280x704", fps=24, quality="standard")
    d = _job_to_dict(_job(), proj, shot)
    assert (d["seed"], d["seconds"]) == (99, 8.0)
    assert (d["start_image_asset_id"], d["end_image_asset_id"]) == (3, 4)
    # list endpoint shape (no shot) must NOT carry the keys
    assert "seed" not in _job_to_dict(_job())


@pytest.mark.asyncio
async def test_get_video_echoes_project_and_shot(monkeypatch):
    """Polls (not just create) include model/size/fps/quality + seed/asset ids."""
    monkeypatch.setattr(_mod.crud, "get_video_job_by_uuid", AsyncMock(return_value=_job()))
    monkeypatch.setattr(
        _mod.crud, "get_video_project",
        AsyncMock(return_value=SimpleNamespace(model="lightricks/ltx-2.3-distilled",
                                               size="1280x704", fps=24, quality="standard")),
    )
    monkeypatch.setattr(
        _mod.crud, "get_video_shots",
        AsyncMock(return_value=[SimpleNamespace(seed=55, seconds=5.0,
                                                first_frame_asset_id=9, last_frame_asset_id=None)]),
    )
    d = await get_video("vid-abc123", db=_db(), auth=_auth())
    assert d["model"] == "lightricks/ltx-2.3-distilled"
    assert d["seed"] == 55
    assert d["start_image_asset_id"] == 9


def test_negative_prompt_not_accepted_and_has_honest_hint():
    """negative_prompt was allowlisted but dropped everywhere (gateway, runner,
    worker) — the distilled model is guidance-free so it CANNOT take effect.
    It must be off VIDEO_ACCEPTED and carry a hint pointing at prose prompting."""
    from backend.app.core.translators import field_validation as fv

    assert "negative_prompt" not in fv.VIDEO_ACCEPTED
    assert "negative_prompt" in fv.VIDEO_DIALECT_HINTS
    hint = fv.VIDEO_DIALECT_HINTS["negative_prompt"]
    assert "guidance-free" in hint and "positive prompt" in hint
    # The other silently-dropped params surfaced by the tester get hints too.
    for f in ("audio_asset_id", "modality_scale", "extend_from_video_id", "enhance_prompt"):
        assert f in fv.VIDEO_DIALECT_HINTS, f
