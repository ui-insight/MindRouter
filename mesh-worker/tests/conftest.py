"""Shared fixtures for the mesh-worker tests (mock mode only — no GPU, no
torch, no upstream model code). Separate venv, no gateway import chain."""

import base64
import io
import os
import sys
import time

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

TERMINAL = ("completed", "failed", "cancelled")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """WorkerConfig reads MESH_WORKER_* at construction: isolate every test
    from the developer's shell."""
    for name in list(os.environ):
        if name.startswith("MESH_WORKER_"):
            monkeypatch.delenv(name, raising=False)


def image_bytes(fmt="PNG", size=(256, 192), mode="RGBA", alpha=True) -> bytes:
    """A red disc; transparent background when alpha=True (the gateway's
    matted RGBA), white background otherwise."""
    bg = (0, 0, 0, 0) if alpha else (255, 255, 255, 255)
    img = Image.new("RGBA", size, bg)
    w, h = size
    ImageDraw.Draw(img).ellipse((w // 5, h // 5, 4 * w // 5, 4 * h // 5), fill=(210, 40, 40, 255))
    if mode != "RGBA":
        img = img.convert(mode)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def body(**overrides):
    req = {
        "model": "mindrouter/mesh-mock",
        "image": b64(image_bytes()),
        "seed": 1234,
        "texture": True,
        "texture_size": 1024,
        "target_faces": 5000,
    }
    req.update(overrides)
    return req


def poll_until(client, job_id, statuses=TERMINAL, timeout=10.0, headers=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get(f"/v1/meshes/{job_id}", headers=headers or {})
        assert r.status_code == 200, r.text
        status = r.json()
        if status["status"] in statuses:
            return status
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} never reached {statuses}")
