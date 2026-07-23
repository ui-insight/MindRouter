############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# video_worker_client.py: HTTP client for the async video worker.
#
# The worker is submit/poll/fetch/cancel (never one long POST), so the
# gateway runner talks to it in sub-second control-plane calls plus one
# larger artifact fetch. Nothing here touches _proxy_with_retry.
# See docs/video-generation-plan.md.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Async video worker HTTP client (submit / poll / fetch / cancel)."""

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import httpx


class WorkerSubmitError(Exception):
    """Raised when a job cannot be submitted (connect error or 5xx at submit).

    ``retryable`` distinguishes transient submit failures (retry the shot) from
    a render that started and then failed (surface it, never retry)."""

    def __init__(self, message: str, *, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


@dataclass
class FetchResult:
    sha256: str
    size_bytes: int
    duration_ms: Optional[int] = None


class VideoWorkerClient(Protocol):
    """Contract the runner depends on (real impl below; fakes in tests)."""

    async def submit(self, base_url: str, payload: Dict[str, Any]) -> str: ...

    async def poll(self, base_url: str, worker_job_id: str) -> Dict[str, Any]: ...

    async def fetch(self, base_url: str, worker_job_id: str, dest_path: str) -> FetchResult: ...

    async def cancel(self, base_url: str, worker_job_id: str) -> None: ...


class HttpVideoWorkerClient:
    """httpx-backed worker client. A fresh client is used per call (per the
    per-request-client pattern that fixed orphaned generations in v2.4.0)."""

    def __init__(self, *, control_timeout: float = 60.0, fetch_timeout: float = 900.0):
        self._control_timeout = control_timeout
        self._fetch_timeout = fetch_timeout

    async def submit(self, base_url: str, payload: Dict[str, Any]) -> str:
        url = f"{base_url.rstrip('/')}/v1/videos"
        try:
            async with httpx.AsyncClient(timeout=self._control_timeout, verify=False) as client:
                resp = await client.post(url, json=payload)
        except httpx.HTTPError as exc:
            raise WorkerSubmitError(f"submit connect error: {exc}", retryable=True) from exc
        if resp.status_code >= 500:
            raise WorkerSubmitError(f"submit 5xx: {resp.status_code}", retryable=True)
        if resp.status_code >= 400:
            raise WorkerSubmitError(f"submit rejected: {resp.status_code} {resp.text}", retryable=False)
        data = resp.json()
        worker_job_id = data.get("id")
        if not worker_job_id:
            raise WorkerSubmitError("submit response missing job id", retryable=False)
        return worker_job_id

    async def poll(self, base_url: str, worker_job_id: str) -> Dict[str, Any]:
        url = f"{base_url.rstrip('/')}/v1/videos/{worker_job_id}"
        async with httpx.AsyncClient(timeout=self._control_timeout, verify=False) as client:
            resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()

    async def fetch(self, base_url: str, worker_job_id: str, dest_path: str) -> FetchResult:
        url = f"{base_url.rstrip('/')}/v1/videos/{worker_job_id}/content"
        hasher = hashlib.sha256()
        size = 0
        async with httpx.AsyncClient(timeout=self._fetch_timeout, verify=False) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                with open(dest_path, "wb") as fh:
                    async for chunk in resp.aiter_bytes():
                        fh.write(chunk)
                        hasher.update(chunk)
                        size += len(chunk)
        return FetchResult(sha256=hasher.hexdigest(), size_bytes=size)

    async def cancel(self, base_url: str, worker_job_id: str) -> None:
        url = f"{base_url.rstrip('/')}/v1/videos/{worker_job_id}"
        try:
            async with httpx.AsyncClient(timeout=self._control_timeout, verify=False) as client:
                await client.delete(url)
        except httpx.HTTPError:
            # Best effort — the worker frees the GPU on its own deadline anyway.
            pass
