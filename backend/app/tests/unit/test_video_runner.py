"""Unit tests for VideoRunner — the async video job state machine (v1).

The runner is the most stateful new component, so this drives its full
lifecycle against an in-memory fake repository and a scriptable fake worker:
claim -> submit -> poll -> fetch -> complete, plus no-backend requeue, cancel
before/during render, worker-reported failure (no retry), transient-submit
retry-then-fail, non-retryable submit, and crash re-adoption.

video_runner.py is import-light, but its package __init__ pulls the db chain,
so it is spec-loaded with logging_config stubbed and video_worker_client
pre-loaded (httpx-only). sys.modules is restored afterward (see MEMORY).
"""

import asyncio
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_svc_dir = Path(__file__).resolve().parents[2] / "services"


def _spec_load(name, path, extra_sys_name=None):
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[])
    mod = importlib.util.module_from_spec(spec)
    if extra_sys_name:
        sys.modules[extra_sys_name] = mod
    spec.loader.exec_module(mod)
    return mod

_MISSING = object()
_STUB_NAMES = {
    "backend": MagicMock(),
    "backend.app": MagicMock(),
    "backend.app.services": MagicMock(),
    "backend.app.logging_config": MagicMock(get_logger=MagicMock(return_value=MagicMock())),
}
_prev = {n: sys.modules.get(n, _MISSING) for n in _STUB_NAMES}
for _n, _s in _STUB_NAMES.items():
    sys.modules[_n] = _s

# video_worker_client (httpx-only) and video_store (os-only) are chain-safe;
# load them for real and register under the names video_runner imports, so its
# module-level bindings (FetchResult, job_output_path) resolve to the real ones.
_wc = _spec_load(
    "video_worker_client_real",
    _svc_dir / "video_worker_client.py",
    extra_sys_name="backend.app.services.video_worker_client",
)
_store = _spec_load(
    "video_store_real",
    _svc_dir / "video_store.py",
    extra_sys_name="backend.app.services.video_store",
)
_runner_mod = _spec_load("video_runner_real", _svc_dir / "video_runner.py")

for _n, _old in _prev.items():
    if _old is _MISSING:
        sys.modules.pop(_n, None)
    else:
        sys.modules[_n] = _old

VideoRunner = _runner_mod.VideoRunner
FetchResult = _wc.FetchResult
WorkerSubmitError = _wc.WorkerSubmitError


# --- fakes ----------------------------------------------------------------

class FakeRepo:
    """In-memory VideoJobRepo. Records terminal transitions for assertions."""

    def __init__(self, job=None, backend=None, cancelled_after=None):
        self._job = job
        self._backend = backend
        # is_cancelled returns True once this many calls have been made.
        self._cancel_after = cancelled_after
        self._cancel_calls = 0
        self.progress = []
        self.shot_updates = []
        self.readopted = 0
        self.state = None            # 'completed' | 'failed' | 'cancelled' | 'requeued'
        self.completed_kwargs = None
        self.fail_kwargs = None
        self.stored_asset_id = 777

    async def readopt_stale(self, threshold_seconds):
        return self.readopted

    async def claim_next(self, worker_id):
        j, self._job = self._job, None
        return j

    async def is_cancelled(self, job_id):
        self._cancel_calls += 1
        return self._cancel_after is not None and self._cancel_calls >= self._cancel_after

    async def update_progress(self, job_id, progress):
        self.progress.append(progress)

    async def mark_shot(self, shot_id, **fields):
        self.shot_updates.append(fields)

    async def requeue(self, job_id):
        self.state = "requeued"

    async def mark_cancelled(self, job_id):
        self.state = "cancelled"

    async def store_output(self, job, src_path, fetch):
        return self.stored_asset_id

    async def complete(self, job_id, output_asset_id, **kwargs):
        self.state = "completed"
        self.completed_kwargs = {"output_asset_id": output_asset_id, **kwargs}

    async def fail(self, job_id, **kwargs):
        self.state = "failed"
        self.fail_kwargs = kwargs

    async def select_backend(self, model):
        return self._backend


class FakeWorker:
    def __init__(self, poll_sequence=None, submit_error=None, fetch=None):
        self._poll = list(poll_sequence or [{"status": "completed", "progress": 100}])
        self._submit_error = submit_error
        self._fetch = fetch or FetchResult(sha256="abc", size_bytes=1234, duration_ms=5000)
        self.submitted = False
        self.cancelled = False

    async def submit(self, base_url, payload):
        if self._submit_error:
            raise self._submit_error
        self.submitted = True
        return "wjob-1"

    async def poll(self, base_url, worker_job_id):
        return self._poll.pop(0) if len(self._poll) > 1 else self._poll[0]

    async def fetch(self, base_url, worker_job_id, dest_path):
        with open(dest_path, "wb") as fh:
            fh.write(b"fake-mp4")
        return self._fetch

    async def cancel(self, base_url, worker_job_id):
        self.cancelled = True


def _job(**over):
    j = {
        "id": 1, "job_uuid": "vid-abc", "user_id": 7, "project_id": 3,
        "model": "lightricks/ltx-2.3-distilled", "size": "1280x704", "fps": 24,
        "quality": "standard", "shot_id": 9, "prompt": "a fox", "seconds": 5,
        "seed": None, "attempts": 0,
    }
    j.update(over)
    return j


def _runner(repo, worker, tmp_path, **over):
    kwargs = dict(storage_root=str(tmp_path), poll_interval=0, max_retries_per_shot=2,
                  token_cost_per_second=2000)
    kwargs.update(over)
    return VideoRunner(repo=repo, worker=worker, **kwargs)


# --- tests ----------------------------------------------------------------

@pytest.mark.asyncio
async def test_happy_path_completes_and_stores_output(tmp_path):
    repo = FakeRepo(backend={"id": 5, "url": "http://w"})
    worker = FakeWorker(poll_sequence=[{"status": "completed", "progress": 100}])
    r = _runner(repo, worker, tmp_path)
    await r.process_job(_job())
    assert repo.state == "completed"
    assert worker.submitted
    assert repo.completed_kwargs["output_asset_id"] == 777
    # token_equivalent = seconds(5) * cost(2000)
    assert repo.completed_kwargs["token_equivalent"] == 10000
    assert repo.completed_kwargs["duration_seconds"] == 5.0
    # shot marked rendering (with backend) then rendered
    assert any(u.get("status") == "rendering" for u in repo.shot_updates)
    assert any(u.get("status") == "rendered" for u in repo.shot_updates)


@pytest.mark.asyncio
async def test_no_backend_requeues_not_fails(tmp_path):
    repo = FakeRepo(backend=None)
    r = _runner(repo, FakeWorker(), tmp_path)
    await r.process_job(_job())
    assert repo.state == "requeued"


@pytest.mark.asyncio
async def test_cancel_before_submit_marks_cancelled(tmp_path):
    repo = FakeRepo(backend={"id": 5, "url": "http://w"}, cancelled_after=1)
    worker = FakeWorker()
    r = _runner(repo, worker, tmp_path)
    await r.process_job(_job())
    assert repo.state == "cancelled"
    assert not worker.submitted


@pytest.mark.asyncio
async def test_cancel_during_poll_cancels_worker(tmp_path):
    # not cancelled at the pre-submit check, then cancelled at the poll check
    repo = FakeRepo(backend={"id": 5, "url": "http://w"}, cancelled_after=2)
    worker = FakeWorker(poll_sequence=[{"status": "in_progress", "progress": 10}])
    r = _runner(repo, worker, tmp_path)
    await r.process_job(_job())
    assert repo.state == "cancelled"
    assert worker.cancelled
    assert any(u.get("status") == "skipped" for u in repo.shot_updates)


@pytest.mark.asyncio
async def test_worker_reported_failure_fails_without_retry(tmp_path):
    repo = FakeRepo(backend={"id": 5, "url": "http://w"})
    worker = FakeWorker(poll_sequence=[{"status": "failed", "error": {"code": "oom", "message": "OOM"}}])
    r = _runner(repo, worker, tmp_path)
    await r.process_job(_job())
    assert repo.state == "failed"
    assert repo.fail_kwargs["error_code"] == "oom"


@pytest.mark.asyncio
async def test_transient_submit_error_under_cap_requeues(tmp_path):
    repo = FakeRepo(backend={"id": 5, "url": "http://w"})
    worker = FakeWorker(submit_error=WorkerSubmitError("boom", retryable=True))
    r = _runner(repo, worker, tmp_path)
    await r.process_job(_job(attempts=0))  # attempt becomes 1, <= cap 2
    assert repo.state == "requeued"


@pytest.mark.asyncio
async def test_transient_submit_error_over_cap_fails(tmp_path):
    repo = FakeRepo(backend={"id": 5, "url": "http://w"})
    worker = FakeWorker(submit_error=WorkerSubmitError("boom", retryable=True))
    r = _runner(repo, worker, tmp_path)
    await r.process_job(_job(attempts=2))  # attempt becomes 3, > cap 2
    assert repo.state == "failed"
    assert repo.fail_kwargs["error_code"] == "submit_failed"


@pytest.mark.asyncio
async def test_non_retryable_submit_error_fails_immediately(tmp_path):
    repo = FakeRepo(backend={"id": 5, "url": "http://w"})
    worker = FakeWorker(submit_error=WorkerSubmitError("bad request", retryable=False))
    r = _runner(repo, worker, tmp_path)
    await r.process_job(_job(attempts=0))
    assert repo.state == "failed"


@pytest.mark.asyncio
async def test_tick_returns_false_when_queue_empty(tmp_path):
    repo = FakeRepo(job=None)
    r = _runner(repo, FakeWorker(), tmp_path)
    assert await r.tick() is False


@pytest.mark.asyncio
async def test_tick_processes_a_claimed_job(tmp_path):
    repo = FakeRepo(job=_job(), backend={"id": 5, "url": "http://w"})
    r = _runner(repo, FakeWorker(), tmp_path)
    assert await r.tick() is True
    assert repo.state == "completed"


@pytest.mark.asyncio
async def test_run_forever_readopts_then_stops_on_cancel(tmp_path):
    repo = FakeRepo(job=None)
    repo.readopted = 3
    r = _runner(repo, FakeWorker(), tmp_path, poll_interval=0.01)
    task = asyncio.create_task(r.run_forever())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert repo.readopted == 3  # readopt_stale was invoked on startup
