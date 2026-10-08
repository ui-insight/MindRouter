"""Worker service tests (mock mode — no GPU).

Separate venv, no gateway import chain, so a real FastAPI TestClient is fine.
Covers the async contract end to end: submit -> poll -> completed -> content,
plus validation, cancel, 404s, capabilities, and the load-bearing invariant
that GET /health stays responsive while a render occupies the executor.
"""

import os
import secrets
import sys
import time

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import WorkerConfig  # noqa: E402
from app import create_app  # noqa: E402


def _client(tmp_path, step_delay=0.0):
    cfg = WorkerConfig(mode="mock", output_dir=str(tmp_path), mock_step_delay=step_delay)
    return TestClient(create_app(cfg))


def _poll_until(client, job_id, terminal=("completed", "failed", "cancelled"), tries=200):
    for _ in range(tries):
        r = client.get(f"/v1/videos/{job_id}")
        assert r.status_code == 200
        body = r.json()
        if body["status"] in terminal:
            return body
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not reach terminal state")


def test_capabilities_and_models(tmp_path):
    with _client(tmp_path) as client:
        cap = client.get("/v1/capabilities").json()
        assert "1280x704" in cap["supported_sizes"]
        assert cap["pipelines"] == ["t2v", "i2v", "keyframes"]
        models = client.get("/v1/models").json()
        assert models["data"][0]["id"] == "lightricks/ltx-2.3-distilled"
        assert client.get("/version").json()["mode"] == "mock"


def test_submit_poll_complete_and_fetch(tmp_path):
    with _client(tmp_path) as client:
        r = client.post("/v1/videos", json={"prompt": "a fox in snow", "size": "1280x704", "seconds": "5"})
        assert r.status_code == 202
        job_id = r.json()["id"]
        assert r.json()["status"] == "queued"

        final = _poll_until(client, job_id)
        assert final["status"] == "completed"
        assert final["progress"] == 100.0
        assert final["total_steps"] == 12

        content = client.get(f"/v1/videos/{job_id}/content")
        assert content.status_code == 200
        assert content.headers["content-type"] == "video/mp4"
        assert len(content.content) > 0


def test_content_before_complete_is_409(tmp_path):
    with _client(tmp_path, step_delay=0.05) as client:
        job_id = client.post(
            "/v1/videos", json={"prompt": "x", "size": "1280x704", "seconds": "5"}
        ).json()["id"]
        # Immediately — job is queued/in_progress, not done.
        r = client.get(f"/v1/videos/{job_id}/content")
        assert r.status_code == 409
        _poll_until(client, job_id)  # let it finish so lifespan teardown is clean


def test_health_responsive_during_render(tmp_path):
    # A slow render occupies the executor; /health must still answer immediately.
    with _client(tmp_path, step_delay=0.1) as client:
        client.post("/v1/videos", json={"prompt": "x", "size": "1280x704", "seconds": "5"})
        start = time.monotonic()
        h = client.get("/health")
        elapsed = time.monotonic() - start
        assert h.status_code == 200
        assert h.json()["status"] == "ok"
        assert elapsed < 1.0  # nowhere near the render time (12 * 0.1s)


def test_range_request_returns_206(tmp_path):
    with _client(tmp_path) as client:
        job_id = client.post(
            "/v1/videos", json={"prompt": "x", "size": "1280x704", "seconds": "5"}
        ).json()["id"]
        _poll_until(client, job_id)
        r = client.get(f"/v1/videos/{job_id}/content", headers={"Range": "bytes=0-15"})
        assert r.status_code == 206
        assert r.headers.get("accept-ranges") == "bytes"
        assert len(r.content) == 16


def test_disallowed_size_rejected(tmp_path):
    with _client(tmp_path) as client:
        r = client.post("/v1/videos", json={"prompt": "x", "size": "9999x9999", "seconds": "5"})
        assert r.status_code == 400


def test_out_of_range_duration_rejected(tmp_path):
    # Duration is a whole-second range [MIN_SECONDS, MAX_SECONDS]; out-of-range
    # and non-integer values are 400. In-range values (e.g. 60) are accepted.
    with _client(tmp_path) as client:
        for bad in ("200", "2", "10.5"):
            r = client.post("/v1/videos", json={"prompt": "x", "size": "1280x704", "seconds": bad})
            assert r.status_code == 400, f"{bad} should be rejected"
        ok = client.post("/v1/videos", json={"prompt": "x", "size": "1280x704", "seconds": "60"})
        assert ok.status_code == 202  # in-range now accepted


def test_missing_prompt_rejected(tmp_path):
    with _client(tmp_path) as client:
        r = client.post("/v1/videos", json={"size": "1280x704", "seconds": "5"})
        assert r.status_code == 400


def test_poll_unknown_job_404(tmp_path):
    with _client(tmp_path) as client:
        assert client.get("/v1/videos/wjob-nope").status_code == 404
        assert client.delete("/v1/videos/wjob-nope").status_code == 404


def test_cancel_queued_job(tmp_path):
    # step_delay keeps the first job in-flight so the second stays queued, then
    # gets cancelled before it starts.
    with _client(tmp_path, step_delay=0.05) as client:
        first = client.post("/v1/videos", json={"prompt": "a", "size": "1280x704", "seconds": "5"}).json()["id"]
        second = client.post("/v1/videos", json={"prompt": "b", "size": "1280x704", "seconds": "5"}).json()["id"]
        assert client.delete(f"/v1/videos/{second}").status_code == 200
        body = _poll_until(client, second)
        assert body["status"] == "cancelled"
        _poll_until(client, first)  # drain


# ── Auth (X-Worker-Key) ────────────────────────────────────────────────────
# When WorkerConfig.api_key is set, the /v1/* routes require a matching
# X-Worker-Key header; /health and /version stay open for monitoring. Mirrors
# the sidecar's SIDECAR_SECRET_KEY gate (sidecar/tests/test_gpu_agent_stress.py).

# Generated per run — never a hardcoded secret literal in the tree.
_KEY = secrets.token_hex(16)


def _authed_client(tmp_path):
    cfg = WorkerConfig(mode="mock", output_dir=str(tmp_path), api_key=_KEY)
    return TestClient(create_app(cfg))


def test_v1_routes_require_key_when_configured(tmp_path):
    with _authed_client(tmp_path) as client:
        submit_body = {"prompt": "x", "size": "1280x704", "seconds": "5"}
        # Missing header → 401 on every /v1 route.
        assert client.get("/v1/models").status_code == 401
        assert client.get("/v1/capabilities").status_code == 401
        assert client.post("/v1/videos", json=submit_body).status_code == 401
        assert client.get("/v1/videos/wjob-x").status_code == 401
        assert client.get("/v1/videos/wjob-x/content").status_code == 401
        assert client.delete("/v1/videos/wjob-x").status_code == 401
        # Wrong header → 401.
        assert client.get("/v1/models", headers={"X-Worker-Key": "nope"}).status_code == 401


def test_v1_routes_accept_correct_key(tmp_path):
    with _authed_client(tmp_path) as client:
        h = {"X-Worker-Key": _KEY}
        assert client.get("/v1/models", headers=h).status_code == 200
        assert client.get("/v1/capabilities", headers=h).status_code == 200
        r = client.post(
            "/v1/videos", json={"prompt": "x", "size": "1280x704", "seconds": "5"}, headers=h
        )
        assert r.status_code == 202
        job_id = r.json()["id"]
        # Drain to completion so lifespan teardown is clean (poll needs the key).
        for _ in range(200):
            pr = client.get(f"/v1/videos/{job_id}", headers=h)
            assert pr.status_code == 200
            if pr.json()["status"] in ("completed", "failed", "cancelled"):
                break
            time.sleep(0.01)


def test_health_and_version_open_without_key(tmp_path):
    with _authed_client(tmp_path) as client:
        # Monitoring endpoints must not require the key.
        assert client.get("/health").status_code == 200
        assert client.get("/version").status_code == 200


def test_no_key_configured_leaves_routes_open(tmp_path):
    # Empty api_key = legacy/open behavior (rollout window before the gateway
    # sends the key). No header, still 200.
    with _client(tmp_path) as client:
        assert client.get("/v1/models").status_code == 200


def test_resident_and_layout_settings_from_env(monkeypatch):
    """Weights stay resident and warm by default; the env can switch either off
    and pick the LTX-2.5 split layout."""
    for name in ("VIDEO_WORKER_RESIDENT", "VIDEO_WORKER_WARMUP", "VIDEO_WORKER_LTX_LAYOUT", "VIDEO_WORKER_VIDEO_VAE"):
        monkeypatch.delenv(name, raising=False)
    cfg = WorkerConfig()
    assert (cfg.resident, cfg.warmup, cfg.ltx_layout, cfg.video_vae_file) == (True, True, "monolith", "")

    monkeypatch.setenv("VIDEO_WORKER_RESIDENT", "0")
    monkeypatch.setenv("VIDEO_WORKER_WARMUP", "false")
    monkeypatch.setenv("VIDEO_WORKER_LTX_LAYOUT", "split")
    monkeypatch.setenv("VIDEO_WORKER_VIDEO_VAE", "vae/ltx-2.5-video-vae-conv-bf16.safetensors")
    cfg = WorkerConfig()
    assert (cfg.resident, cfg.warmup, cfg.ltx_layout) == (False, False, "split")

    monkeypatch.setenv("VIDEO_WORKER_RESIDENT", "")   # blank = default
    assert WorkerConfig().resident is True


def test_split_layout_paths_follow_the_hf_repo_layout(tmp_path):
    """LTX-2.5 files resolve under the checkpoint dir in the Hugging Face repo's
    own layout; a VAE override swaps only the video decoder."""
    from engine import LTXEngine

    cfg = WorkerConfig(mode="ltx", checkpoint_dir="/m/ltx-2.5")
    cfg.ltx_layout = "split"
    paths = LTXEngine(cfg)._paths()
    assert paths["transformer"] == "/m/ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
    assert paths["text_encoder"] == "/m/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
    assert paths["video_vae"].endswith("vae/ltx-2.5-video-vae-bf16.safetensors")
    cfg.video_vae_file = "vae/ltx-2.5-video-vae-conv-bf16.safetensors"
    assert LTXEngine(cfg)._paths()["video_vae"] == "/m/ltx-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors"

    cfg.ltx_layout = "monolith"
    assert LTXEngine(cfg)._paths()["gemma"] == "/m/ltx-2.5/gemma-3-12b"


def test_unknown_layout_is_refused():
    from engine import LTXEngine

    cfg = WorkerConfig(mode="ltx")
    cfg.ltx_layout = "fat"
    with pytest.raises(ValueError, match="monolith"):
        LTXEngine(cfg).load()
