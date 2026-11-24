############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# mesh_worker_client.py: HTTP client for a mesh-worker backend
#     (submit / poll / fetch / cancel). The runner depends only
#     on the MeshWorkerClient protocol, so tests inject fakes.
#
# Mirrors video_worker_client.py (shared secret, TLS policy,
# per-call httpx client, bounded streamed downloads).
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
#
############################################################

"""Async mesh-worker HTTP client."""

import hashlib
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Protocol

import httpx

from backend.app.services.video_worker_client import (
    WorkerFetchError,
    WorkerSubmitError,
    _tls_verify,
)
from backend.app.settings import get_settings

__all__ = [
    "ArtifactResult",
    "HttpMeshWorkerClient",
    "MeshWorkerClient",
    "WorkerFetchError",
    "WorkerSubmitError",
]

# The worker's artifact variants (GET /v1/meshes/{id}/content?variant=...).
VARIANT_GLB = "glb"
VARIANT_PREVIEW = "preview"


def _fetch_max_bytes() -> int:
    """Per-artifact download cap from ``mesh_worker_fetch_max_mb`` (<=0 off)."""
    mb = int(getattr(get_settings(), "mesh_worker_fetch_max_mb", 512))
    return mb * 1024 * 1024 if mb > 0 else 0


@dataclass
class ArtifactResult:
    sha256: str
    size_bytes: int


class MeshWorkerClient(Protocol):
    """What the runner needs from a worker (real impl below; fakes in tests)."""

    async def submit(self, base_url: str, payload: Dict[str, Any]) -> str: ...

    async def poll(self, base_url: str, worker_job_id: str) -> Dict[str, Any]: ...

    async def fetch(
        self, base_url: str, worker_job_id: str, variant: str, dest_path: str
    ) -> ArtifactResult: ...

    async def cancel(self, base_url: str, worker_job_id: str) -> None: ...


class HttpMeshWorkerClient:
    """httpx-backed client; a fresh AsyncClient per call. Sends X-Worker-Key
    when ``api_key`` is set; downloads are streamed to disk and capped."""

    def __init__(self, *, control_timeout: float = 60.0, fetch_timeout: float = 600.0, api_key: str = ""):
        self._control_timeout = control_timeout
        self._fetch_timeout = fetch_timeout
        self._api_key = api_key or ""

    def _headers(self) -> Dict[str, str]:
        return {"X-Worker-Key": self._api_key} if self._api_key else {}

    async def submit(self, base_url: str, payload: Dict[str, Any]) -> str:
        url = f"{base_url.rstrip('/')}/v1/meshes"
        try:
            async with httpx.AsyncClient(timeout=self._control_timeout, verify=_tls_verify()) as client:
                resp = await client.post(url, json=payload, headers=self._headers())
        except httpx.HTTPError as exc:
            raise WorkerSubmitError(f"submit connect error: {type(exc).__name__}", retryable=True) from exc
        # 503 = the worker's queue is full: transient, try again later.
        if resp.status_code >= 500:
            raise WorkerSubmitError(f"submit {resp.status_code}", retryable=True)
        if resp.status_code >= 400:
            # The worker's 4xx detail is safe to surface (it validates our own
            # payload, e.g. an image it cannot decode); keep it short.
            raise WorkerSubmitError(
                f"submit rejected: {resp.status_code} {resp.text[:300]}", retryable=False
            )
        worker_job_id = (resp.json() or {}).get("id")
        if not worker_job_id:
            raise WorkerSubmitError("submit response missing job id", retryable=False)
        return str(worker_job_id)

    async def poll(self, base_url: str, worker_job_id: str) -> Dict[str, Any]:
        url = f"{base_url.rstrip('/')}/v1/meshes/{worker_job_id}"
        async with httpx.AsyncClient(timeout=self._control_timeout, verify=_tls_verify()) as client:
            resp = await client.get(url, headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    async def fetch(self, base_url: str, worker_job_id: str, variant: str, dest_path: str) -> ArtifactResult:
        url = f"{base_url.rstrip('/')}/v1/meshes/{worker_job_id}/content"
        hasher = hashlib.sha256()
        size = 0
        max_bytes = _fetch_max_bytes()
        # A name unique to this download, so two downloads of the same file
        # (should that ever happen) can never interleave into one.
        tmp_path = f"{dest_path}.{uuid.uuid4().hex[:12]}.part"
        try:
            async with httpx.AsyncClient(timeout=self._fetch_timeout, verify=_tls_verify()) as client:
                async with client.stream(
                    "GET", url, params={"variant": variant}, headers=self._headers()
                ) as resp:
                    resp.raise_for_status()
                    with open(tmp_path, "wb") as fh:
                        async for chunk in resp.aiter_bytes():
                            size += len(chunk)
                            if max_bytes and size > max_bytes:
                                raise WorkerFetchError(
                                    f"mesh {variant} exceeds the {max_bytes}-byte cap"
                                )
                            fh.write(chunk)
                            hasher.update(chunk)
            # Only a complete download replaces the destination.
            os.replace(tmp_path, dest_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        return ArtifactResult(sha256=hasher.hexdigest(), size_bytes=size)

    async def cancel(self, base_url: str, worker_job_id: str) -> None:
        url = f"{base_url.rstrip('/')}/v1/meshes/{worker_job_id}"
        try:
            async with httpx.AsyncClient(timeout=self._control_timeout, verify=_tls_verify()) as client:
                await client.delete(url, headers=self._headers())
        except httpx.HTTPError:
            # Best effort: the worker frees the GPU at its own deadline anyway.
            pass
