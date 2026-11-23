"""Unit tests for the public image-to-mesh API (/v1/meshes).

Route handlers are awaited as plain coroutines with hand-built arguments (no
TestClient). mesh_api is spec-loaded with the DB/registry/auth chains stubbed
(it must never import backend.app.db.models / core.telemetry at module level);
pathsafe, mesh_store and model_availability are loaded for real.

Covers: image decoding and sniffing, the job object, every submit gate
(disabled, no access, image, texture/size/faces/seed/background validation,
unknown and non-mesh models, concurrency, storage cap, quota), the happy path
(input stored under a server-chosen name, quota synced after commit), the JSON
and multipart create paths, poll/delete ownership, and content serving.
"""

import asyncio
import base64
import importlib.util
import io
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from PIL import Image

_APP = Path(__file__).resolve().parents[2]
_MISSING = object()


def _load(name, path, register=None):
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[])
    mod = importlib.util.module_from_spec(spec)
    if register:
        sys.modules[register] = mod
    spec.loader.exec_module(mod)
    return mod


class _Status:
    QUEUED = "queued"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_models = MagicMock()
_models.BackendEngine = SimpleNamespace(MESH="mesh", VIDEO="video")
_models.Modality = SimpleNamespace(MESH_GENERATION="mesh_generation")
_models.MeshJobStatus = _Status
_models.MeshJob = MagicMock
_models.ApiKey = MagicMock
_models.User = MagicMock

_services = MagicMock()
_STUB_NAMES = [
    "backend", "backend.app", "backend.app.api", "backend.app.api.auth", "backend.app.api.model_availability",
    "backend.app.core", "backend.app.core.pathsafe", "backend.app.core.telemetry",
    "backend.app.core.telemetry.registry", "backend.app.db", "backend.app.db.session", "backend.app.db.models",
    "backend.app.services", "backend.app.services.mesh_store", "backend.app.settings", "backend.app.logging_config",
]
_prev = {n: sys.modules.get(n, _MISSING) for n in _STUB_NAMES}
for _n in ["backend", "backend.app", "backend.app.api", "backend.app.api.auth", "backend.app.core",
           "backend.app.core.telemetry", "backend.app.core.telemetry.registry", "backend.app.db",
           "backend.app.db.session"]:
    sys.modules[_n] = MagicMock()
sys.modules["backend.app.db.models"] = _models
sys.modules["backend.app.services"] = _services
sys.modules["backend.app.settings"] = MagicMock(get_settings=MagicMock(return_value=SimpleNamespace(mesh_storage_path="/nonexistent")))
sys.modules["backend.app.logging_config"] = MagicMock(get_logger=MagicMock(return_value=MagicMock()), bind_request_context=MagicMock())
_load("ma_for_mesh", _APP / "api" / "model_availability.py", "backend.app.api.model_availability")
_pathsafe = _load("ps_for_mesh_api", _APP / "core" / "pathsafe.py", "backend.app.core.pathsafe")
_store = _load("store_for_mesh_api", _APP / "services" / "mesh_store.py", "backend.app.services.mesh_store")
_services.mesh_store = _store
api = _load("mesh_api_under_test", _APP / "api" / "mesh_api.py")
for _n, _old in _prev.items():
    if _old is _MISSING:
        sys.modules.pop(_n, None)
    else:
        sys.modules[_n] = _old

UUID = "mesh-0123456789abcdef01234567"
MODEL = "microsoft/trellis.2-4b"
_EXTERNAL = {"queued": "queued", "rendering": "in_progress", "completed": "completed", "failed": "failed", "cancelled": "cancelled"}


def _picture(fmt="PNG", size=(64, 64)):
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 120, 200)).save(buf, format=fmt)
    return buf.getvalue()


def _job(status=_Status.QUEUED, **over):
    when = SimpleNamespace(timestamp=lambda: 1_700_000_000)
    job = SimpleNamespace(
        id=1, job_uuid=UUID, user_id=7, model=MODEL, status=status, stage=None, progress=0.0,
        created_at=when, started_at=None, completed_at=None, seed=5, texture=True, texture_size=2048,
        target_faces=100000, remove_background="auto", preview_size_bytes=0, output_size_bytes=0,
        faces=None, vertices=None, error_code=None, error_message=None, gpu_seconds=0, token_equivalent=10000,
    )
    for k, v in over.items():
        setattr(job, k, v)
    return job


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Patch crud/registry/settings for a working default setup; returns knobs."""
    crud = MagicMock()
    config = {"mesh.enabled": True}
    crud.EXTERNAL_MESH_STATUS = _EXTERNAL
    crud.MESH_TERMINAL_STATUSES = (_Status.COMPLETED, _Status.FAILED, _Status.CANCELLED)

    async def get_config_json(db, key, default=None):
        return config.get(key, default)

    crud.get_config_json = get_config_json
    crud.count_active_mesh_jobs_for_user = AsyncMock(return_value=0)
    crud.get_user_mesh_storage_bytes = AsyncMock(return_value=0)
    crud.compute_mesh_token_cost = AsyncMock(return_value=10000)
    crud.reserve_mesh_tokens = AsyncMock(return_value=True)
    crud.incr_quota_redis = AsyncMock()
    crud.create_request = AsyncMock(return_value=SimpleNamespace(id=100))
    crud.create_mesh_job = AsyncMock(side_effect=lambda db, **kw: _job(**{k: v for k, v in kw.items() if k in (
        "seed", "texture", "texture_size", "target_faces", "remove_background")}))
    crud.get_mesh_job_by_uuid = AsyncMock(return_value=None)
    crud.get_mesh_queue_position = AsyncMock(return_value=(1, 1))
    crud.delete_mesh_job = AsyncMock()
    crud.request_cancel_mesh_job = AsyncMock()
    monkeypatch.setattr(api, "crud", crud)

    reg = MagicMock()
    reg.resolve_alias = MagicMock(side_effect=lambda m: ({"default-mesh": MODEL}.get(m, m), None))
    reg.model_exists = AsyncMock(side_effect=lambda m: m in (MODEL, "some/llm"))
    reg.model_is_configured = AsyncMock(return_value=False)
    reg.get_all_backends = AsyncMock(return_value=[SimpleNamespace(id=1, engine="mesh"), SimpleNamespace(id=2, engine="vllm")])
    reg.get_backend_models = AsyncMock(side_effect=lambda bid: [SimpleNamespace(name=MODEL)] if bid == 1 else [SimpleNamespace(name="some/llm")])
    monkeypatch.setattr(api, "get_registry", lambda: reg)
    monkeypatch.setattr(api, "get_settings", lambda: SimpleNamespace(mesh_storage_path=str(tmp_path)))
    return SimpleNamespace(crud=crud, config=config, registry=reg, root=tmp_path)


def _db():
    db = MagicMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    db.refresh = AsyncMock()
    return db


def _user(enabled=True):
    return SimpleNamespace(id=7, mesh_generation_enabled=enabled)


def _submit(image=None, **params):
    return asyncio.run(api.submit_mesh_job(
        _db(), _user(params.pop("_enabled", True)), SimpleNamespace(id=3),
        _picture() if image is None else image, {"image": True, **params},
    ))


def _code(exc_info):
    return exc_info.value.status_code


# -- helpers ----------------------------------------------------------------------

def test_decode_image_field_accepts_base64_and_data_urls():
    raw = _picture()
    b64 = base64.b64encode(raw).decode()
    assert api.decode_image_field(b64) == raw
    assert api.decode_image_field("data:image/png;base64," + b64) == raw
    for bad in (None, "", "   ", "not base64!!", 42):
        with pytest.raises(HTTPException) as info:
            api.decode_image_field(bad)
        assert _code(info) == 400


@pytest.mark.parametrize("fmt,ctype", [("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")])
def test_sniff_image_uses_the_bytes_not_the_claimed_type(fmt, ctype):
    assert api.sniff_image(_picture(fmt)) == ctype


@pytest.mark.parametrize("data", [_picture("GIF"), _picture(size=(8, 8)), b"\x89PNG not really"])
def test_sniff_image_rejects_other_formats_tiny_pictures_and_garbage(data):
    with pytest.raises(HTTPException) as info:
        api.sniff_image(data)
    assert _code(info) == 400


def test_job_object_links_only_finished_outputs(env):
    queued = api.job_to_dict(_job(), queue=(2, 5))
    assert queued["status"] == "queued" and queued["content_url"] is None
    assert (queued["queue_position"], queued["queue_depth"]) == (2, 5)
    done = api.job_to_dict(_job(_Status.COMPLETED, preview_size_bytes=10, output_size_bytes=99, faces=5))
    assert done["content_url"] == f"/v1/meshes/{UUID}/content"
    assert done["preview_url"] == f"/v1/meshes/{UUID}/content?variant=preview"
    assert done["size_bytes"] == 99 and "queue_position" not in done
    running = api.job_to_dict(_job(_Status.RENDERING, stage="texture"))
    assert running["status"] == "in_progress" and running["stage"] == "texture"
    failed = api.job_to_dict(_job(_Status.FAILED, error_code="oom", error_message="no memory"))
    assert failed["error"] == {"code": "oom", "message": "no memory"}


# -- submit gates ------------------------------------------------------------------

def test_disabled_service_is_503(env):
    env.config["mesh.enabled"] = False
    with pytest.raises(HTTPException) as info:
        _submit()
    assert _code(info) == 503


def test_user_without_access_is_403(env):
    with pytest.raises(HTTPException) as info:
        _submit(_enabled=False)
    assert _code(info) == 403


def test_oversized_image_is_400(env):
    env.config["mesh.max_image_upload_mb"] = 1
    with pytest.raises(HTTPException) as info:
        _submit(image=b"x" * (1024 * 1024 + 1))
    assert _code(info) == 400


@pytest.mark.parametrize("params", [
    {"texture_size": 999}, {"texture_size": "big"}, {"target_faces": 10}, {"target_faces": 2_000_000},
    {"target_faces": 1.5}, {"seed": -1}, {"seed": 2**31}, {"remove_background": "maybe"},
    {"texture": "perhaps"},
])
def test_bad_options_are_400(env, params):
    with pytest.raises(HTTPException) as info:
        _submit(**params)
    assert _code(info) == 400


def test_unknown_model_is_404(env):
    with pytest.raises(HTTPException) as info:
        _submit(model="nope/none")
    assert _code(info) == 404


def test_a_chat_model_is_not_a_mesh_model(env):
    with pytest.raises(HTTPException) as info:
        _submit(model="some/llm")
    assert _code(info) == 400


def test_per_user_concurrency_is_429(env):
    env.crud.count_active_mesh_jobs_for_user.return_value = 2
    with pytest.raises(HTTPException) as info:
        _submit()
    assert _code(info) == 429


def test_storage_cap_is_507(env):
    env.crud.get_user_mesh_storage_bytes.return_value = 20 * 1024**3
    with pytest.raises(HTTPException) as info:
        _submit()
    assert _code(info) == 507


def test_insufficient_quota_is_429_and_nothing_is_written(env):
    env.crud.reserve_mesh_tokens.return_value = False
    with pytest.raises(HTTPException) as info:
        _submit()
    assert _code(info) == 429
    env.crud.create_mesh_job.assert_not_called()
    assert not any(env.root.rglob("*.input.*"))


# -- happy path ---------------------------------------------------------------------

def test_submit_stores_the_input_and_queues_the_job(env):
    jpeg = _picture("JPEG")
    out = _submit(image=jpeg, texture="false", texture_size="1024", target_faces="50000",
                  remove_background="on", seed="42")
    assert out["status"] == "queued" and out["id"] == UUID
    kwargs = env.crud.create_mesh_job.call_args.kwargs
    assert kwargs["model"] == MODEL                       # default-mesh alias resolved
    assert kwargs["input_content_type"] == "image/jpeg"   # sniffed, not trusted
    assert kwargs["texture"] is False and kwargs["texture_size"] == 1024
    assert kwargs["target_faces"] == 50000 and kwargs["seed"] == 42 and kwargs["remove_background"] == "on"
    assert kwargs["token_equivalent"] == 10000 and kwargs["job_uuid"].startswith("mesh-")
    stored = _store.input_path(str(env.root), 7, kwargs["job_uuid"], "image/jpeg")
    assert Path(stored).read_bytes() == jpeg
    env.crud.compute_mesh_token_cost.assert_awaited_with(api.crud.compute_mesh_token_cost.call_args.args[0], texture=False)
    env.crud.incr_quota_redis.assert_awaited_once_with(7, 10000)
    assert env.crud.create_request.call_args.kwargs["modality"] == "mesh_generation"


def test_seed_is_random_when_omitted(env):
    _submit()
    seed = env.crud.create_mesh_job.call_args.kwargs["seed"]
    assert isinstance(seed, int) and 0 <= seed < 2**31


def test_a_failed_write_rolls_back_and_raises(env, monkeypatch):
    db = _db()
    db.commit = AsyncMock(side_effect=RuntimeError("db gone"))
    with pytest.raises(RuntimeError):
        asyncio.run(api.submit_mesh_job(db, _user(), SimpleNamespace(id=3), _picture(), {"image": True}))
    db.rollback.assert_awaited_once()
    env.crud.incr_quota_redis.assert_not_called()
    assert not any(env.root.rglob("*.input.*"))


# -- create endpoint (JSON and multipart) -------------------------------------------

def _request(content_type, body=None, form=None):
    req = MagicMock()
    req.headers = {"content-type": content_type}
    req.client = SimpleNamespace(host="127.0.0.1")
    req.json = AsyncMock(return_value=body)
    req.form = AsyncMock(return_value=form)
    return req


def test_create_from_json(env):
    b64 = base64.b64encode(_picture()).decode()
    out = asyncio.run(api.create_mesh(_request("application/json", body={"image": b64, "texture": False}),
                                      _db(), (_user(), SimpleNamespace(id=3))))
    assert out["status"] == "queued"
    assert env.crud.create_mesh_job.call_args.kwargs["texture"] is False


def test_create_from_multipart(env):
    upload = SimpleNamespace(read=AsyncMock(return_value=_picture()))
    out = asyncio.run(api.create_mesh(
        _request("multipart/form-data; boundary=x", form={"image": upload, "target_faces": "20000"}),
        _db(), (_user(), SimpleNamespace(id=3)),
    ))
    assert out["status"] == "queued"
    assert env.crud.create_mesh_job.call_args.kwargs["target_faces"] == 20000


def test_multipart_without_a_file_is_400(env):
    with pytest.raises(HTTPException) as info:
        asyncio.run(api.create_mesh(_request("multipart/form-data; boundary=x", form={"image": "text"}),
                                    _db(), (_user(), SimpleNamespace(id=3))))
    assert _code(info) == 400


# -- poll / delete / content ----------------------------------------------------------

def test_another_users_job_is_404(env):
    with pytest.raises(HTTPException) as info:
        asyncio.run(api.get_mesh(UUID, _db(), (_user(), None)))
    assert _code(info) == 404
    assert env.crud.get_mesh_job_by_uuid.call_args.kwargs["user_id"] == 7


def test_delete_cancels_a_running_job_and_deletes_a_finished_one(env):
    env.crud.get_mesh_job_by_uuid.return_value = _job(_Status.RENDERING)
    asyncio.run(api.delete_mesh(UUID, _db(), (_user(), None)))
    env.crud.request_cancel_mesh_job.assert_awaited_once()
    env.crud.delete_mesh_job.assert_not_called()

    env.crud.get_mesh_job_by_uuid.return_value = _job(_Status.COMPLETED)
    out = asyncio.run(api.delete_mesh(UUID, _db(), (_user(), None)))
    assert out["deleted"] is True
    env.crud.delete_mesh_job.assert_awaited_once()


def test_content_before_completion_is_409(env):
    env.crud.get_mesh_job_by_uuid.return_value = _job(_Status.RENDERING)
    with pytest.raises(HTTPException) as info:
        asyncio.run(api.get_mesh_content(UUID, "glb", _db(), (_user(), None)))
    assert _code(info) == 409


def test_content_serves_the_glb_and_preview(env):
    env.crud.get_mesh_job_by_uuid.return_value = _job(_Status.COMPLETED)
    Path(_store.output_path(str(env.root), 7, UUID, create=True)).write_bytes(b"glTF")
    Path(_store.preview_path(str(env.root), 7, UUID)).write_bytes(b"\x89PNG")
    glb = asyncio.run(api.get_mesh_content(UUID, "glb", _db(), (_user(), None)))
    assert glb.media_type == "model/gltf-binary" and glb.path.endswith(f"/7/{UUID}.glb")
    png = asyncio.run(api.get_mesh_content(UUID, "preview", _db(), (_user(), None)))
    assert png.media_type == "image/png"


def test_content_missing_file_is_404_and_bad_variant_is_400(env):
    env.crud.get_mesh_job_by_uuid.return_value = _job(_Status.COMPLETED)
    with pytest.raises(HTTPException) as info:
        asyncio.run(api.get_mesh_content(UUID, "glb", _db(), (_user(), None)))
    assert _code(info) == 404
    with pytest.raises(HTTPException) as info:
        asyncio.run(api.get_mesh_content(UUID, "obj", _db(), (_user(), None)))
    assert _code(info) == 400


def test_models_endpoint_lists_only_mesh_backends(env):
    out = asyncio.run(api.list_mesh_models(_db(), (_user(), None)))
    assert [m["id"] for m in out["data"]] == [MODEL]
    assert out["data"][0]["texture_sizes"] == [1024, 2048, 4096]
    assert out["default_model"] == "default-mesh"
