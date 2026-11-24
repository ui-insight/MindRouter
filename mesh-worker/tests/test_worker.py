"""Mesh worker HTTP contract tests (mock engine, FastAPI TestClient, no GPU).

Covers every route of the contract the gateway client is written against:
health (incl. mid-job), version, models, capabilities, submit -> poll ->
completed -> fetch (GLB validated structurally, preview a real PNG), the 400
matrix, 404 / 409, X-Worker-Key auth on/off, cancel (queued + in flight),
queue-full 503, per-job timeout with the GPU kept serialized, the stuck
health signal, result TTL cleanup and startup orphan cleanup.
"""

import hashlib
import io
import os
import secrets
import struct
import threading
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app as app_module
from app import create_app
from config import WorkerConfig
from conftest import TERMINAL, b64, body, image_bytes, poll_until
from engines.base import EngineInputError
from glb import GLB_MAGIC, CHUNK_BIN, CHUNK_JSON, inspect_glb, parse_glb


def _cfg(tmp_path, **kw) -> WorkerConfig:
    return WorkerConfig(mode="mock", output_dir=str(tmp_path), **kw)


def _client(tmp_path, **kw) -> TestClient:
    return TestClient(create_app(_cfg(tmp_path, **kw)))


def _files(tmp_path, job_id):
    return [n for n in os.listdir(tmp_path) if n.startswith(job_id)]


# ── health / version / models / capabilities ──────────────────────────────

def test_health_version_models_capabilities(tmp_path):
    with _client(tmp_path) as client:
        h = client.get("/health")
        assert h.status_code == 200
        hb = h.json()
        assert hb["status"] == "ok" and hb["mode"] == "mock"
        assert hb["model"] == "mindrouter/mesh-mock" and hb["busy"] is False

        assert client.get("/version").json() == {"version": app_module.VERSION, "mode": "mock",
                                                 "model": "mindrouter/mesh-mock"}
        # Same shape video-worker returns (VLLMAdapter reads data[].id).
        models = client.get("/v1/models").json()
        assert models["object"] == "list"
        assert [m["id"] for m in models["data"]] == ["mindrouter/mesh-mock"]

        cap = client.get("/v1/capabilities").json()
        assert cap["model"] == "mindrouter/mesh-mock"
        assert cap["inputs"] == ["image"] and cap["formats"] == ["glb"]
        assert cap["supports_texture"] is True
        assert cap["texture_sizes"] == [1024, 2048, 4096]
        assert cap["min_faces"] <= cap["default_faces"] <= cap["max_faces"]
        assert cap["options"] == {}                         # mock: no per-job model settings


def test_served_model_name_override(tmp_path):
    with _client(tmp_path, model_id="microsoft/trellis.2-4b") as client:
        assert client.get("/v1/models").json()["data"][0]["id"] == "microsoft/trellis.2-4b"
        assert client.get("/health").json()["model"] == "microsoft/trellis.2-4b"


# ── happy path ────────────────────────────────────────────────────────────

def _check_glb(data: bytes) -> dict:
    magic, version, length = struct.unpack_from("<III", data, 0)
    assert magic == GLB_MAGIC and version == 2 and length == len(data)
    json_len, json_type = struct.unpack_from("<II", data, 12)
    assert json_type == CHUNK_JSON and json_len % 4 == 0
    bin_len, bin_type = struct.unpack_from("<II", data, 20 + json_len)
    assert bin_type == CHUNK_BIN and bin_len % 4 == 0
    gltf, binary = parse_glb(data)
    assert gltf["asset"]["version"] == "2.0"
    assert len(binary) == gltf["buffers"][0]["byteLength"]
    prim = gltf["meshes"][0]["primitives"][0]
    pos = gltf["accessors"][prim["attributes"]["POSITION"]]
    assert "min" in pos and "max" in pos          # required by the spec
    return gltf


def test_submit_poll_complete_and_fetch(tmp_path):
    with _client(tmp_path) as client:
        r = client.post("/v1/meshes", json=body())
        assert r.status_code == 202
        assert r.json()["status"] == "queued"
        job_id = r.json()["id"]

        final = poll_until(client, job_id)
        assert final["status"] == "completed", final
        assert final["stage"] == "done" and final["progress"] == 100.0
        assert final["error"] is None
        assert isinstance(final["duration_ms"], int) and final["duration_ms"] >= 0
        assert final["faces"] > 0 and final["vertices"] > 0

        glb = client.get(f"/v1/meshes/{job_id}/content")   # default variant = glb
        assert glb.status_code == 200
        assert glb.headers["content-type"] == "model/gltf-binary"
        gltf = _check_glb(glb.content)
        assert gltf["images"][0]["mimeType"] == "image/png"   # texture=true embeds the bake
        assert final["outputs"]["glb"] == {"size_bytes": len(glb.content),
                                           "sha256": hashlib.sha256(glb.content).hexdigest()}
        assert inspect_glb(glb.content) == {"faces": final["faces"], "vertices": final["vertices"]}

        pv = client.get(f"/v1/meshes/{job_id}/content", params={"variant": "preview"})
        assert pv.status_code == 200 and pv.headers["content-type"] == "image/png"
        assert pv.content[:8] == b"\x89PNG\r\n\x1a\n"
        img = Image.open(io.BytesIO(pv.content))
        img.load()
        assert img.size == (512, 512)
        assert final["outputs"]["preview"]["sha256"] == hashlib.sha256(pv.content).hexdigest()

        explicit = client.get(f"/v1/meshes/{job_id}/content", params={"variant": "glb"})
        assert explicit.content == glb.content


def test_range_request_returns_206(tmp_path):
    with _client(tmp_path) as client:
        job_id = client.post("/v1/meshes", json=body()).json()["id"]
        poll_until(client, job_id)
        r = client.get(f"/v1/meshes/{job_id}/content", headers={"Range": "bytes=0-11"})
        assert r.status_code == 206 and r.headers.get("accept-ranges") == "bytes"
        assert r.content[:4] == b"glTF" and len(r.content) == 12


def test_shape_only_has_no_texture(tmp_path):
    with _client(tmp_path) as client:
        job_id = client.post("/v1/meshes", json=body(texture=False)).json()["id"]
        final = poll_until(client, job_id)
        assert final["status"] == "completed"
        gltf = _check_glb(client.get(f"/v1/meshes/{job_id}/content").content)
        assert "images" not in gltf and "textures" not in gltf


def test_output_is_deterministic_per_seed(tmp_path):
    with _client(tmp_path) as client:
        shas = []
        for seed in (7, 7, 8):
            job_id = client.post("/v1/meshes", json=body(seed=seed)).json()["id"]
            shas.append(poll_until(client, job_id)["outputs"]["glb"]["sha256"])
        assert shas[0] == shas[1] and shas[0] != shas[2]


def test_target_faces_steers_the_mesh(tmp_path):
    with _client(tmp_path) as client:
        small = poll_until(client, client.post("/v1/meshes", json=body(target_faces=1000)).json()["id"])
        large = poll_until(client, client.post("/v1/meshes", json=body(target_faces=40000)).json()["id"])
        assert small["faces"] < large["faces"]
        assert abs(small["faces"] - 1000) / 1000 < 0.25


@pytest.mark.parametrize("fmt,mode", [("JPEG", "RGB"), ("WEBP", "RGBA"), ("PNG", "RGB"), ("PNG", "P")])
def test_accepts_png_jpeg_webp(tmp_path, fmt, mode):
    with _client(tmp_path) as client:
        data = image_bytes(fmt=fmt, mode=mode, alpha=(mode == "RGBA"))
        r = client.post("/v1/meshes", json=body(image=b64(data)))
        assert r.status_code == 202, r.text
        assert poll_until(client, r.json()["id"])["status"] == "completed"


def test_data_url_prefix_tolerated(tmp_path):
    with _client(tmp_path) as client:
        r = client.post("/v1/meshes", json=body(image="data:image/png;base64," + b64(image_bytes())))
        assert r.status_code == 202
        poll_until(client, r.json()["id"])


def test_optional_fields_take_defaults(tmp_path):
    with _client(tmp_path) as client:
        r = client.post("/v1/meshes", json={"image": b64(image_bytes())})
        assert r.status_code == 202
        assert poll_until(client, r.json()["id"])["status"] == "completed"


# ── /health stays responsive mid-job; progress is visible ─────────────────

def test_health_responsive_and_progress_visible_during_job(tmp_path):
    with _client(tmp_path, mock_stage_delay=0.25) as client:
        job_id = client.post("/v1/meshes", json=body()).json()["id"]
        status = poll_until(client, job_id, statuses=("in_progress",))
        start = time.monotonic()
        h = client.get("/health")
        assert time.monotonic() - start < 1.0          # nowhere near the job time
        assert h.status_code == 200
        assert h.json()["busy"] is True and h.json()["current_job"] == job_id
        assert status["stage"] in ("preprocess", "shape", "texture", "export")
        assert 0.0 <= status["progress"] < 100.0
        assert status["outputs"] is None and status["faces"] is None

        seen, last = set(), -1.0
        while True:
            s = client.get(f"/v1/meshes/{job_id}").json()
            assert s["progress"] >= last                  # never goes backwards
            last = s["progress"]
            seen.add(s["stage"])
            if s["status"] in TERMINAL:
                break
            time.sleep(0.02)
        assert s["status"] == "completed"
        assert {"shape", "texture", "export"} <= seen
        assert client.get("/health").json()["busy"] is False


# ── 400s ──────────────────────────────────────────────────────────────────

def _truncated_png() -> bytes:
    data = image_bytes()
    return data[: len(data) // 2]


def _transparent_png() -> bytes:
    img = Image.new("RGBA", (64, 64), (255, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _gif() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (1, 2, 3)).save(buf, format="GIF")
    return buf.getvalue()


@pytest.mark.parametrize("overrides", [
    {"image": None},
    {"image": ""},
    {"image": 12345},
    {"image": "not base64 !!!"},
    {"image": b64(b"definitely not an image")},
    {"image": b64(_truncated_png())},
    {"image": b64(_gif())},                               # unsupported format
    {"image": b64(image_bytes(size=(16, 16)))},           # too small
    {"image": b64(_transparent_png())},                   # alpha with no foreground
    {"texture_size": 512},
    {"texture_size": 8192},
    {"texture_size": "2048"},
    {"target_faces": 10},
    {"target_faces": 50_000_000},
    {"target_faces": 1.5},
    {"texture": "yes"},
    {"seed": -1},
    {"seed": 2**40},
    {"seed": True},
    {"seed": "42"},
    {"model": 7},
    {"options": {"steps": 10}},                          # the mock engine takes no options
    {"options": "fast"},
])
def test_bad_input_is_400(tmp_path, overrides):
    with _client(tmp_path) as client:
        r = client.post("/v1/meshes", json=body(**overrides))
        assert r.status_code == 400, (overrides, r.text)
        assert r.json()["detail"]


def test_bad_json_and_non_object_are_400(tmp_path):
    with _client(tmp_path) as client:
        assert client.post("/v1/meshes", content=b"{nope", headers={"content-type": "application/json"}).status_code == 400
        assert client.post("/v1/meshes", json=[1, 2]).status_code == 400


def test_oversized_image_is_400(tmp_path):
    with _client(tmp_path, max_image_bytes=1024) as client:
        r = client.post("/v1/meshes", json=body())
        assert r.status_code in (400, 413)


def test_rejected_request_does_not_create_a_job(tmp_path):
    with _client(tmp_path) as client:
        client.post("/v1/meshes", json=body(texture_size=3))
        assert client.app.state.manager.jobs == {}


# ── 404 / 409 ─────────────────────────────────────────────────────────────

def test_unknown_and_malformed_ids_are_404(tmp_path):
    with _client(tmp_path) as client:
        for job_id in ("mjob-" + "0" * 24, "nope", "..%2F..%2Fetc%2Fpasswd", "mjob-xyz"):
            assert client.get(f"/v1/meshes/{job_id}").status_code == 404
            assert client.get(f"/v1/meshes/{job_id}/content").status_code == 404
            assert client.delete(f"/v1/meshes/{job_id}").status_code == 404


def test_content_before_complete_is_409(tmp_path):
    with _client(tmp_path, mock_stage_delay=0.1) as client:
        job_id = client.post("/v1/meshes", json=body()).json()["id"]
        assert client.get(f"/v1/meshes/{job_id}/content").status_code == 409
        assert client.get(f"/v1/meshes/{job_id}/content", params={"variant": "preview"}).status_code == 409
        poll_until(client, job_id)


def test_bad_variant_is_400(tmp_path):
    with _client(tmp_path) as client:
        job_id = client.post("/v1/meshes", json=body()).json()["id"]
        poll_until(client, job_id)
        assert client.get(f"/v1/meshes/{job_id}/content", params={"variant": "obj"}).status_code == 400


# ── auth (X-Worker-Key) ───────────────────────────────────────────────────

_KEY = secrets.token_hex(16)   # generated per run, never a literal secret


def test_v1_mesh_routes_require_key_when_configured(tmp_path):
    with _client(tmp_path, api_key=_KEY) as client:
        job = "mjob-" + "0" * 24
        calls = [
            ("post", "/v1/meshes", body()), ("get", f"/v1/meshes/{job}", None),
            ("get", f"/v1/meshes/{job}/content", None), ("delete", f"/v1/meshes/{job}", None),
        ]
        for method, path, payload in calls:
            kwargs = {"json": payload} if payload else {}
            assert getattr(client, method)(path, **kwargs).status_code == 401, path
            assert getattr(client, method)(path, headers={"X-Worker-Key": "nope"}, **kwargs).status_code == 401, path
        assert client.app.state.manager.jobs == {}


def test_v1_routes_accept_correct_key(tmp_path):
    h = {"X-Worker-Key": _KEY}
    with _client(tmp_path, api_key=_KEY) as client:
        assert client.get("/v1/models", headers=h).status_code == 200
        assert client.get("/v1/capabilities", headers=h).status_code == 200
        r = client.post("/v1/meshes", json=body(), headers=h)
        assert r.status_code == 202
        job_id = r.json()["id"]
        assert poll_until(client, job_id, headers=h)["status"] == "completed"
        assert client.get(f"/v1/meshes/{job_id}/content", headers=h).status_code == 200
        assert client.delete(f"/v1/meshes/{job_id}", headers=h).status_code == 200


def test_health_version_and_discovery_open_without_key(tmp_path):
    """The gateway registry discovers models with no X-Worker-Key, so
    /v1/models (and the capability list) must answer without it."""
    with _client(tmp_path, api_key=_KEY) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/version").status_code == 200
        models = client.get("/v1/models")
        assert models.status_code == 200 and models.json()["data"][0]["id"]
        assert client.get("/v1/capabilities").status_code == 200


def test_no_key_configured_leaves_routes_open(tmp_path):
    with _client(tmp_path) as client:
        assert client.get("/v1/models").status_code == 200


# ── cancel ────────────────────────────────────────────────────────────────

def test_cancel_queued_job_is_immediate(tmp_path):
    with _client(tmp_path, mock_stage_delay=0.1) as client:
        first = client.post("/v1/meshes", json=body()).json()["id"]
        poll_until(client, first, statuses=("in_progress",))
        second = client.post("/v1/meshes", json=body()).json()["id"]
        r = client.delete(f"/v1/meshes/{second}")
        assert r.status_code == 200
        assert r.json()["status"] == "cancelled" and r.json()["stage"] == "queued"
        assert client.get(f"/v1/meshes/{second}").json()["status"] == "cancelled"
        assert client.app.state.manager.queue_depth() == 0     # slot freed
        assert poll_until(client, first)["status"] == "completed"
        assert client.get(f"/v1/meshes/{second}").json()["status"] == "cancelled"
        assert _files(tmp_path, second) == []


def test_cancel_in_flight_job_stops_at_next_stage(tmp_path):
    with _client(tmp_path, mock_stage_delay=0.2) as client:
        job_id = client.post("/v1/meshes", json=body()).json()["id"]
        poll_until(client, job_id, statuses=("in_progress",))
        r = client.delete(f"/v1/meshes/{job_id}")
        assert r.status_code == 200
        assert r.json()["status"] == "in_progress"            # cooperative
        assert r.json()["cancel_requested"] is True
        final = poll_until(client, job_id)
        assert final["status"] == "cancelled"
        assert final["outputs"] is None and final["stage"] != "done"
        assert client.get(f"/v1/meshes/{job_id}/content").status_code == 409
        assert _files(tmp_path, job_id) == []
        # The worker is free again and still serves.
        nxt = client.post("/v1/meshes", json=body()).json()["id"]
        assert poll_until(client, nxt)["status"] == "completed"


def test_cancel_terminal_job_is_idempotent(tmp_path):
    with _client(tmp_path) as client:
        job_id = client.post("/v1/meshes", json=body()).json()["id"]
        poll_until(client, job_id)
        r = client.delete(f"/v1/meshes/{job_id}")
        assert r.status_code == 200 and r.json()["status"] == "completed"
        assert client.get(f"/v1/meshes/{job_id}/content").status_code == 200


# ── queue full ────────────────────────────────────────────────────────────

def test_queue_full_is_503(tmp_path):
    with _client(tmp_path, max_queue=2, mock_stage_delay=0.2) as client:
        running = client.post("/v1/meshes", json=body()).json()["id"]
        poll_until(client, running, statuses=("in_progress",))
        queued = [client.post("/v1/meshes", json=body()) for _ in range(2)]
        assert [r.status_code for r in queued] == [202, 202]
        r = client.post("/v1/meshes", json=body())
        assert r.status_code == 503
        assert "retry-after" in {k.lower() for k in r.headers}
        assert client.get("/health").json()["queue_depth"] == 2
        # Cancelling a queued job frees a slot at once.
        client.delete(f"/v1/meshes/{queued[0].json()['id']}")
        assert client.post("/v1/meshes", json=body()).status_code == 202
        for job_id in list(client.app.state.manager.jobs):
            client.delete(f"/v1/meshes/{job_id}")
        poll_until(client, running)


# ── timeout / stuck (fake engines injected through build_engine) ──────────

class _BlockingEngine:
    """First job blocks for `block` seconds IGNORING cancel (a stage that never
    reaches a boundary); later jobs delegate to the mock engine."""

    def __init__(self, config, block):
        from engines.mock import MockEngine

        self.mock = MockEngine(config)
        self.block = block
        self.calls = 0
        self.first_done = threading.Event()

    def load(self):
        pass

    def capabilities(self):
        return {"supports_texture": True}

    def generate(self, spec, image, glb_path, preview_path, tracker):
        self.calls += 1
        if self.calls == 1:
            tracker.enter("preprocess", 5.0)
            time.sleep(self.block)
            self.first_done.set()
        return self.mock.generate(spec, image, glb_path, preview_path, tracker)


def test_timeout_fails_job_and_keeps_gpu_serialized(tmp_path, monkeypatch):
    engines = []

    def build(cfg):
        engines.append(_BlockingEngine(cfg, block=0.6))
        return engines[0]

    monkeypatch.setattr(app_module, "build_engine", build)
    with _client(tmp_path, job_timeout_seconds=0.15) as client:
        first = client.post("/v1/meshes", json=body()).json()["id"]
        t0 = time.monotonic()
        failed = poll_until(client, first, statuses=("failed",))
        assert time.monotonic() - t0 < 0.55          # reported before the thread stops
        assert failed["error"]["code"] == "timeout"
        second = client.post("/v1/meshes", json=body()).json()["id"]
        # The GPU thread is still busy with the timed-out job: next job waits.
        assert client.get("/health").json()["busy"] is True
        assert client.get(f"/v1/meshes/{second}").json()["status"] == "queued"
        assert engines[0].first_done.wait(5)
        assert poll_until(client, second)["status"] == "completed"
        assert client.get(f"/v1/meshes/{first}").json()["status"] == "failed"
        assert _files(tmp_path, first) == []


def test_health_503_when_gpu_thread_stuck(tmp_path, monkeypatch):
    holder = []

    def build(cfg):
        holder.append(_BlockingEngine(cfg, block=0.8))
        return holder[0]

    monkeypatch.setattr(app_module, "build_engine", build)
    with _client(tmp_path, job_timeout_seconds=0.1, stuck_grace_seconds=0.1) as client:
        job_id = client.post("/v1/meshes", json=body()).json()["id"]
        poll_until(client, job_id, statuses=("failed",))
        deadline = time.monotonic() + 2
        while client.get("/health").status_code != 503:
            assert time.monotonic() < deadline, "health never reported stuck"
            time.sleep(0.02)
        assert client.get("/health").json()["status"] == "stuck"
        assert holder[0].first_done.wait(5)
        deadline = time.monotonic() + 2
        while client.get("/health").status_code != 200:
            assert time.monotonic() < deadline, "health never recovered"
            time.sleep(0.02)


class _FailingEngine:
    def __init__(self, exc=None, write_bad_glb=False):
        self.exc = exc
        self.write_bad_glb = write_bad_glb

    def load(self):
        pass

    def capabilities(self):
        return {"supports_texture": False}

    def generate(self, spec, image, glb_path, preview_path, tracker):
        tracker.enter("preprocess", 5.0)
        if self.exc:
            raise self.exc
        with open(glb_path, "wb") as fh:
            fh.write(b"glTF" + b"\x00" * 40)
        with open(preview_path, "wb") as fh:
            fh.write(image_bytes())
        return {}


@pytest.mark.parametrize("engine,code", [
    (_FailingEngine(exc=RuntimeError("CUDA out of memory")), "generation_failed"),
    (_FailingEngine(exc=EngineInputError("no foreground object found")), "invalid_input"),
    (_FailingEngine(write_bad_glb=True), "invalid_output"),
])
def test_engine_failures_surface_as_failed(tmp_path, monkeypatch, engine, code):
    monkeypatch.setattr(app_module, "build_engine", lambda cfg: engine)
    with _client(tmp_path) as client:
        # Capabilities reflect the engine (this fake declares no texture support).
        assert client.get("/v1/capabilities").json()["supports_texture"] is False
        job_id = client.post("/v1/meshes", json=body()).json()["id"]
        final = poll_until(client, job_id)
        assert final["status"] == "failed"
        assert final["error"]["code"] == code and final["error"]["message"]
        if code == "generation_failed":     # raw engine text stays in the worker log
            assert "CUDA out of memory" not in final["error"]["message"]
            assert "RuntimeError" in final["error"]["message"]
        assert final["outputs"] is None
        assert client.get(f"/v1/meshes/{job_id}/content").status_code == 409
        assert _files(tmp_path, job_id) == []
        # One bad job never takes the worker down.
        assert client.get("/health").json()["status"] == "ok"


# ── result TTL + restart hygiene ──────────────────────────────────────────

def test_results_expire_after_ttl(tmp_path):
    with _client(tmp_path, result_ttl_seconds=0.3, sweep_interval_seconds=0.05) as client:
        job_id = client.post("/v1/meshes", json=body()).json()["id"]
        poll_until(client, job_id)
        assert len(_files(tmp_path, job_id)) == 2              # .glb + .preview.png
        deadline = time.monotonic() + 3
        while client.get(f"/v1/meshes/{job_id}").status_code != 404:
            assert time.monotonic() < deadline, "result never expired"
            time.sleep(0.05)
        assert client.get(f"/v1/meshes/{job_id}/content").status_code == 404
        assert _files(tmp_path, job_id) == []


def test_sweep_keeps_fresh_and_in_flight_jobs(tmp_path):
    with _client(tmp_path, result_ttl_seconds=100) as client:
        manager = client.app.state.manager
        job_id = client.post("/v1/meshes", json=body()).json()["id"]
        poll_until(client, job_id)
        assert manager.sweep() == 0
        assert manager.sweep(now=time.time() + 101) == 1
        assert manager.get(job_id) is None and _files(tmp_path, job_id) == []


def test_startup_removes_orphaned_outputs(tmp_path):
    orphan = "mjob-" + "a" * 24
    for name in (f"{orphan}.glb", f"{orphan}.preview.png", "keep-me.txt"):
        (tmp_path / name).write_bytes(b"x")
    with _client(tmp_path) as client:
        assert client.get(f"/v1/meshes/{orphan}").status_code == 404
        assert sorted(os.listdir(tmp_path)) == ["keep-me.txt"]


def test_worker_logs_info_by_default(tmp_path, monkeypatch):
    # The engines' INFO lines (model loading, the RMBG license guard) must
    # reach the container log, not be dropped at Python's WARNING default.
    import logging
    monkeypatch.delenv("MESH_WORKER_LOG_LEVEL", raising=False)
    with _client(tmp_path):
        assert logging.getLogger("engines.trellis_family").isEnabledFor(logging.INFO)
