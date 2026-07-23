"""Worker service tests (mock mode — no GPU).

Separate venv, no gateway import chain, so a real FastAPI TestClient is fine.
Covers the async contract end to end: submit -> poll -> completed -> content,
plus validation, cancel, 404s, capabilities, and the load-bearing invariant
that GET /health stays responsive while a render occupies the executor.
"""

import os
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
        assert cap["pipelines"] == ["t2v"]
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


def test_disallowed_duration_rejected(tmp_path):
    with _client(tmp_path) as client:
        r = client.post("/v1/videos", json={"prompt": "x", "size": "1280x704", "seconds": "60"})
        assert r.status_code == 400


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
