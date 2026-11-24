"""Unit tests for MeshRunner, the image-to-mesh job state machine.

Drives the runner against an in-memory fake repository and a scriptable fake
worker: claim -> cut out -> submit -> poll -> fetch -> complete, plus the
no-backend requeue, cancel, worker failure, submit retries, lost jobs (404 and
repeated poll errors), fetch failures, a missing input, the wall-clock limit,
crash re-adoption, and the background-removal decision.

mesh_runner's package __init__ pulls the DB chain, so the module is
spec-loaded with the backend packages stubbed and its light dependencies
(pathsafe, mesh_store, video/mesh worker clients) loaded for real.
sys.modules is restored afterwards (TESTING.md hygiene rules).
"""

import asyncio
import base64
import importlib.util
import io
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from PIL import Image

_APP = Path(__file__).resolve().parents[2]
_MISSING = object()
_settings = SimpleNamespace(internal_tls_verify=True, mesh_worker_fetch_max_mb=512, video_worker_fetch_max_mb=2048)
_services_pkg = MagicMock()
_STUBS = {
    "backend": MagicMock(),
    "backend.app": MagicMock(),
    "backend.app.core": MagicMock(),
    "backend.app.services": _services_pkg,
    "backend.app.settings": MagicMock(get_settings=lambda: _settings),
    "backend.app.logging_config": MagicMock(get_logger=MagicMock(return_value=MagicMock())),
}
_REAL = [
    "backend.app.core.pathsafe",
    "backend.app.services.video_worker_client",
    "backend.app.services.mesh_worker_client",
    "backend.app.services.mesh_store",
]
_prev = {n: sys.modules.get(n, _MISSING) for n in list(_STUBS) + _REAL}
for _n, _s in _STUBS.items():
    sys.modules[_n] = _s


def _load(name, path, register=None):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if register:
        sys.modules[register] = mod
    spec.loader.exec_module(mod)
    return mod


_load("pathsafe_for_mesh_runner", _APP / "core" / "pathsafe.py", "backend.app.core.pathsafe")
_load("vwc_for_mesh_runner", _APP / "services" / "video_worker_client.py", "backend.app.services.video_worker_client")
_mwc = _load("mwc_for_mesh_runner", _APP / "services" / "mesh_worker_client.py", "backend.app.services.mesh_worker_client")
_store = _load("mesh_store_for_runner", _APP / "services" / "mesh_store.py", "backend.app.services.mesh_store")
_services_pkg.mesh_store = _store  # `from backend.app.services import mesh_store`
runner_mod = _load("mesh_runner_under_test", _APP / "services" / "mesh_runner.py")
for _n, _old in _prev.items():
    if _old is _MISSING:
        sys.modules.pop(_n, None)
    else:
        sys.modules[_n] = _old

MeshRunner = runner_mod.MeshRunner
WorkerSubmitError = _mwc.WorkerSubmitError
ArtifactResult = _mwc.ArtifactResult
MODEL = "microsoft/trellis.2-4b"
UUID = "mesh-0123456789abcdef01234567"


def _png(rgba=False, transparent=False):
    img = Image.new("RGBA" if rgba else "RGB", (32, 32), (200, 10, 10, 0 if transparent else 255) if rgba else (200, 10, 10))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _job(root, *, image=None, **over):
    job = {
        "id": 1, "job_uuid": UUID, "user_id": 5, "model": MODEL, "seed": 7, "texture": True,
        "texture_size": 2048, "target_faces": 50000, "remove_background": "auto",
        "input_content_type": "image/png", "attempts": 0, "token_equivalent": 10000,
        "backend_id": None, "backend_job_id": None, "started_at": datetime.now(timezone.utc),
    }
    job.update(over)
    path = _store.input_path(str(root), job["user_id"], job["job_uuid"], job["input_content_type"], create=True)
    Path(path).write_bytes(image if image is not None else _png())
    return job


class FakeRepo:
    def __init__(self, job=None, backend=None, servable=(MODEL,), cancel_after=None, stale=(), readopt_ok=True):
        self._job = job
        self._backend = backend if backend is not None else {"id": 11, "url": "https://w:8006"}
        self._servable = list(servable)
        self._cancel_after = cancel_after
        self._cancel_calls = 0
        self._stale = list(stale)
        self._readopt_ok = readopt_ok
        self.updates = []
        self.state = None
        self.requeue_kwargs = None
        self.complete_kwargs = None
        self.fail_kwargs = None

    async def list_stale_rendering(self, threshold_seconds):
        s, self._stale = self._stale, []
        return s

    async def readopt(self, job_id, worker_id, threshold_seconds):
        return self._readopt_ok

    async def get_backend(self, backend_id):
        return self._backend or None

    async def servable_models(self):
        return list(self._servable)

    async def model_backends(self):
        # One worker per model by default (ids 11, 12, ... in servable order).
        return {m: [11 + i] for i, m in enumerate(self._servable)}

    blocked = ()           # models the service-on / license gate holds back
    expired = 0

    async def claimable(self, models):
        return [m for m in models if m not in self.blocked]

    async def expire_queued(self):
        return self.expired

    async def claim_next(self, worker_id, models=None, exclude_ids=None):
        self.claim_models = models
        j, self._job = self._job, None
        return j

    async def is_cancelled(self, job_id):
        self._cancel_calls += 1
        return self._cancel_after is not None and self._cancel_calls >= self._cancel_after

    async def update(self, job_id, **fields):
        self.updates.append(fields)

    async def requeue(self, job_id, *, attempts=None, clear_backend=False):
        self.state = "requeued"
        self.requeue_kwargs = {"attempts": attempts, "clear_backend": clear_backend}

    async def mark_cancelled(self, job_id):
        self.state = "cancelled"

    complete_ok = True

    async def complete(self, job_id, **fields):
        self.complete_kwargs = fields
        if self.complete_ok:
            self.state = "completed"
        return self.complete_ok

    async def fail(self, job_id, *, error_code, error_message):
        self.state = "failed"
        self.fail_kwargs = {"error_code": error_code, "error_message": error_message}

    async def select_backend(self, model, exclude=()):
        if not self._backend:
            return None
        i = self._servable.index(model) if model in self._servable else 0
        backend = dict(self._backend, id=self._backend["id"] + i)
        return None if backend["id"] in exclude else backend


class FakeWorker:
    def __init__(self, polls=(), submit_error=None, glb=b"glTF-bytes", preview=b"\x89PNG", fetch_errors=()):
        self._polls = list(polls)
        self.submit_error = submit_error
        self.glb, self.preview = glb, preview
        self.fetch_errors = set(fetch_errors)
        self.submitted = []
        self.cancelled = []
        self.fetched = []

    async def submit(self, base_url, payload):
        self.submitted.append(payload)
        if self.submit_error:
            raise self.submit_error
        return "w-1"

    async def poll(self, base_url, worker_job_id):
        item = self._polls.pop(0) if self._polls else {"status": "in_progress", "progress": 50}
        if isinstance(item, Exception):
            raise item
        return item

    async def fetch(self, base_url, worker_job_id, variant, dest_path):
        self.fetched.append(variant)
        if variant in self.fetch_errors:
            raise _mwc.WorkerFetchError(f"{variant} too big")
        data = self.glb if variant == "glb" else self.preview
        Path(dest_path).write_bytes(data)
        return ArtifactResult(sha256="abc", size_bytes=len(data))

    async def cancel(self, base_url, worker_job_id):
        self.cancelled.append(worker_job_id)


def _runner(repo, worker, root, **kw):
    kw.setdefault("poll_interval", 0)
    return MeshRunner(repo, worker, storage_root=str(root), **kw)


def _404():
    request = httpx.Request("GET", "https://w:8006/v1/meshes/w-1")
    return httpx.HTTPStatusError("gone", request=request, response=httpx.Response(404, request=request))


DONE = {"status": "completed", "progress": 100, "duration_ms": 41_600, "faces": 49_998, "vertices": 25_001}


def test_happy_path_submits_polls_fetches_and_completes(tmp_path):
    repo = FakeRepo(job=_job(tmp_path))
    worker = FakeWorker(polls=[{"status": "in_progress", "progress": 40, "stage": "shape"}, DONE])
    assert asyncio.run(_runner(repo, worker, tmp_path).tick()) is True
    assert repo.state == "completed"
    assert repo.complete_kwargs == {
        "output_size_bytes": len(b"glTF-bytes"), "output_sha256": "abc", "preview_size_bytes": 4,
        "faces": 49_998, "vertices": 25_001, "gpu_seconds": 42,
    }
    assert worker.fetched == ["glb", "preview"]
    assert Path(_store.output_path(str(tmp_path), 5, UUID)).read_bytes() == b"glTF-bytes"
    assert {"backend_id": 11, "backend_job_id": "w-1", "stage": "queued"} in repo.updates
    assert {"progress": 40.0, "stage": "shape"} in repo.updates
    payload = worker.submitted[0]
    assert payload["model"] == MODEL and payload["seed"] == 7
    assert payload["texture"] is True and payload["texture_size"] == 2048 and payload["target_faces"] == 50000
    assert base64.b64decode(payload["image"])[:4] == b"\x89PNG"
    assert "options" not in payload                       # no per-model settings on this job


def test_per_model_settings_ride_along_as_worker_options(tmp_path):
    job = _job(tmp_path)
    job["model_options"] = {"octree_resolution": 384, "steps": 20}
    repo = FakeRepo(job=job)
    worker = FakeWorker(polls=[DONE])
    assert asyncio.run(_runner(repo, worker, tmp_path).tick()) is True
    assert worker.submitted[0]["options"] == {"octree_resolution": 384, "steps": 20}


def test_no_healthy_model_means_nothing_is_claimed(tmp_path):
    repo = FakeRepo(job=_job(tmp_path), servable=())
    assert asyncio.run(_runner(repo, FakeWorker(), tmp_path).tick()) is False
    assert repo.state is None


def test_no_backend_requeues_and_pauses(tmp_path):
    repo = FakeRepo(job=_job(tmp_path), backend={})
    assert asyncio.run(_runner(repo, FakeWorker(), tmp_path).tick()) is False
    assert repo.state == "requeued"


def test_cancel_before_submit(tmp_path):
    repo = FakeRepo(job=_job(tmp_path), cancel_after=1)
    worker = FakeWorker()
    asyncio.run(_runner(repo, worker, tmp_path).tick())
    assert repo.state == "cancelled" and worker.submitted == []


def test_cancel_while_running_tells_the_worker(tmp_path):
    repo = FakeRepo(job=_job(tmp_path), cancel_after=3)
    worker = FakeWorker(polls=[{"status": "in_progress", "progress": 10}] * 5)
    asyncio.run(_runner(repo, worker, tmp_path).tick())
    assert repo.state == "cancelled" and worker.cancelled == ["w-1"]


def test_worker_failure_is_surfaced_not_retried(tmp_path):
    repo = FakeRepo(job=_job(tmp_path))
    worker = FakeWorker(polls=[{"status": "failed", "error": {"code": "oom", "message": "CUDA out of memory"}}])
    asyncio.run(_runner(repo, worker, tmp_path).tick())
    assert repo.fail_kwargs == {"error_code": "oom", "error_message": "CUDA out of memory"}
    assert len(worker.submitted) == 1


def test_retryable_submit_error_requeues_then_fails_at_the_cap(tmp_path):
    repo = FakeRepo(job=_job(tmp_path))
    asyncio.run(_runner(repo, FakeWorker(submit_error=WorkerSubmitError("503", retryable=True)), tmp_path).tick())
    assert repo.state == "requeued" and repo.requeue_kwargs["attempts"] == 1

    repo = FakeRepo(job=_job(tmp_path, attempts=2))
    asyncio.run(_runner(repo, FakeWorker(submit_error=WorkerSubmitError("503", retryable=True)), tmp_path, max_retries=2).tick())
    assert repo.fail_kwargs["error_code"] == "submit_failed"


def test_non_retryable_submit_error_fails_at_once(tmp_path):
    repo = FakeRepo(job=_job(tmp_path))
    asyncio.run(_runner(repo, FakeWorker(submit_error=WorkerSubmitError("400 bad image", retryable=False)), tmp_path).tick())
    assert repo.fail_kwargs["error_code"] == "submit_failed"


def test_worker_404_is_a_lost_job_requeued_from_scratch(tmp_path):
    repo = FakeRepo(job=_job(tmp_path))
    asyncio.run(_runner(repo, FakeWorker(polls=[_404()]), tmp_path).tick())
    assert repo.state == "requeued"
    assert repo.requeue_kwargs == {"attempts": 1, "clear_backend": True}


def test_lost_too_often_fails(tmp_path):
    repo = FakeRepo(job=_job(tmp_path, attempts=2))
    asyncio.run(_runner(repo, FakeWorker(polls=[_404()]), tmp_path, max_retries=2).tick())
    assert repo.fail_kwargs["error_code"] == "worker_lost"


def test_repeated_poll_errors_count_as_lost(tmp_path):
    repo = FakeRepo(job=_job(tmp_path))
    errors = [httpx.ConnectError("down")] * runner_mod.MAX_CONSECUTIVE_POLL_ERRORS
    asyncio.run(_runner(repo, FakeWorker(polls=errors), tmp_path).tick())
    assert repo.state == "requeued"


def test_a_transient_poll_error_is_tolerated(tmp_path):
    repo = FakeRepo(job=_job(tmp_path))
    asyncio.run(_runner(repo, FakeWorker(polls=[httpx.ReadTimeout("slow"), DONE]), tmp_path).tick())
    assert repo.state == "completed"


def test_missing_preview_still_completes(tmp_path):
    repo = FakeRepo(job=_job(tmp_path))
    asyncio.run(_runner(repo, FakeWorker(polls=[DONE], fetch_errors={"preview"}), tmp_path).tick())
    assert repo.state == "completed" and repo.complete_kwargs["preview_size_bytes"] == 0


def test_glb_fetch_failure_fails_the_job(tmp_path):
    repo = FakeRepo(job=_job(tmp_path))
    asyncio.run(_runner(repo, FakeWorker(polls=[DONE], fetch_errors={"glb"}), tmp_path).tick())
    assert repo.fail_kwargs["error_code"] == "fetch_failed"


def test_missing_input_file_fails_cleanly(tmp_path):
    job = _job(tmp_path)
    Path(_store.input_path(str(tmp_path), 5, UUID, "image/png")).unlink()
    repo = FakeRepo(job=job)
    asyncio.run(_runner(repo, FakeWorker(), tmp_path).tick())
    assert repo.fail_kwargs["error_code"] == "input_missing"


def test_wall_clock_limit_cancels_on_the_worker(tmp_path):
    old = datetime.now(timezone.utc) - timedelta(hours=2)
    repo = FakeRepo(job=_job(tmp_path, started_at=old))
    worker = FakeWorker(polls=[{"status": "in_progress"}])
    asyncio.run(_runner(repo, worker, tmp_path, max_wall_seconds=60).tick())
    assert repo.fail_kwargs["error_code"] == "wall_timeout" and worker.cancelled == ["w-1"]


# -- reconciliation -----------------------------------------------------------

async def _reconcile_and_wait(runner):
    count = await runner.reconcile()
    await asyncio.gather(*list(runner._tasks))
    return count


def test_reconcile_finishes_a_job_the_worker_completed(tmp_path):
    stale = _job(tmp_path, backend_id=11, backend_job_id="w-1")
    repo = FakeRepo(stale=[stale])
    assert asyncio.run(_reconcile_and_wait(_runner(repo, FakeWorker(polls=[DONE]), tmp_path))) == 1
    assert repo.state == "completed"


def test_reconcile_resumes_a_running_job_in_its_own_task(tmp_path):
    stale = _job(tmp_path, backend_id=11, backend_job_id="w-1")
    repo = FakeRepo(stale=[stale])
    runner = _runner(repo, FakeWorker(polls=[{"status": "in_progress", "progress": 20}, DONE]), tmp_path)

    async def go():
        await runner.reconcile()
        assert MODEL in runner.busy_models      # held while it is driven
        await asyncio.gather(*list(runner._tasks))

    asyncio.run(go())
    assert repo.state == "completed" and runner.busy_models == set()


def test_reconcile_requeues_a_job_that_died_before_submit(tmp_path):
    repo = FakeRepo(stale=[_job(tmp_path)])
    asyncio.run(_runner(repo, FakeWorker(), tmp_path).reconcile())
    assert repo.state == "requeued"


def test_reconcile_treats_an_unknown_worker_job_as_lost(tmp_path):
    repo = FakeRepo(stale=[_job(tmp_path, backend_id=11, backend_job_id="w-1")])
    asyncio.run(_runner(repo, FakeWorker(polls=[_404()]), tmp_path).reconcile())
    assert repo.state == "requeued" and repo.requeue_kwargs["clear_backend"] is True


def test_reconcile_leaves_a_job_another_runner_took(tmp_path):
    repo = FakeRepo(stale=[_job(tmp_path, backend_id=11, backend_job_id="w-1")], readopt_ok=False)
    asyncio.run(_runner(repo, FakeWorker(polls=[DONE]), tmp_path).reconcile())
    assert repo.state is None


def test_reconcile_fails_a_job_past_the_wall_limit(tmp_path):
    old = datetime.now(timezone.utc) - timedelta(hours=3)
    repo = FakeRepo(stale=[_job(tmp_path, started_at=old, backend_id=11, backend_job_id="w-1")])
    worker = FakeWorker()
    asyncio.run(_runner(repo, worker, tmp_path, max_wall_seconds=60).reconcile())
    assert repo.fail_kwargs["error_code"] == "wall_timeout"
    assert worker.cancelled == ["w-1"]          # the worker's copy is stopped too


def test_reconcile_collects_a_finished_result_even_past_the_wall_limit(tmp_path):
    # The worker finished while the gateway was down for longer than the limit:
    # the result is fetched, not thrown away.
    old = datetime.now(timezone.utc) - timedelta(hours=3)
    repo = FakeRepo(stale=[_job(tmp_path, started_at=old, backend_id=11, backend_job_id="w-1")])
    asyncio.run(_reconcile_and_wait(_runner(repo, FakeWorker(polls=[DONE]), tmp_path, max_wall_seconds=60)))
    assert repo.state == "completed"


def test_reconcile_requeue_counts_as_an_attempt_and_is_capped(tmp_path):
    repo = FakeRepo(stale=[_job(tmp_path, attempts=0)])
    asyncio.run(_runner(repo, FakeWorker(), tmp_path).reconcile())
    assert repo.state == "requeued" and repo.requeue_kwargs["attempts"] == 1
    repo = FakeRepo(stale=[_job(tmp_path, attempts=5)])
    asyncio.run(_runner(repo, FakeWorker(), tmp_path).reconcile())
    assert repo.state == "failed" and repo.fail_kwargs["error_code"] == "worker_lost"


def test_gated_models_are_not_claimed(tmp_path):
    # Service switched off or license withdrawn: queued jobs wait, unclaimed.
    repo = FakeRepo(job=_job(tmp_path))
    repo.blocked = (MODEL,)
    assert asyncio.run(_runner(repo, FakeWorker(), tmp_path).tick()) is False
    assert repo.state is None and not hasattr(repo, "claim_models")


def test_reconcile_expires_long_queued_jobs(tmp_path):
    repo = FakeRepo()
    repo.expired = 2
    asyncio.run(_runner(repo, FakeWorker(), tmp_path).reconcile())   # logs, no error


def test_worker_urls_never_reach_the_user(tmp_path):
    req = httpx.Request("GET", "https://node7.internal:8006/v1/meshes/w-1/content?variant=glb")
    err = httpx.HTTPStatusError("boom", request=req, response=httpx.Response(502, request=req))
    text = runner_mod._fetch_error_text(err)
    assert "node7" not in text and "502" in text
    assert "internal" not in runner_mod._public_text("failed fetching https://node7.internal:8006/x")


# -- review fixes: no self-adoption, exactly-once, backoff, big pictures -------------

def test_reconcile_skips_jobs_this_runner_is_driving(tmp_path):
    stale = _job(tmp_path, backend_id=11, backend_job_id="w-1")
    repo = FakeRepo(stale=[stale])
    worker = FakeWorker(polls=[DONE])
    runner = _runner(repo, worker, tmp_path)
    runner._active[stale["id"]] = MODEL          # e.g. stuck in a slow poll right now
    assert asyncio.run(runner.reconcile()) == 0
    assert repo.state is None and worker.fetched == []


def test_reconcile_resumes_when_the_worker_is_only_unreachable(tmp_path):
    """A connection error at reconcile time is not proof the job is lost:
    resume polling (which tolerates errors) instead of resubmitting a copy."""
    stale = _job(tmp_path, backend_id=11, backend_job_id="w-1")
    repo = FakeRepo(stale=[stale])
    worker = FakeWorker(polls=[httpx.ConnectError("blip"), DONE])
    asyncio.run(_reconcile_and_wait(_runner(repo, worker, tmp_path)))
    assert repo.state == "completed" and worker.submitted == []


def test_lost_after_poll_errors_cancels_on_the_worker_and_backs_off(tmp_path):
    repo = FakeRepo(job=_job(tmp_path))
    worker = FakeWorker(polls=[httpx.ConnectError("down")] * runner_mod.MAX_CONSECUTIVE_POLL_ERRORS)
    runner = _runner(repo, worker, tmp_path, retry_backoff_seconds=15)
    job = asyncio.run(runner.claim())
    assert asyncio.run(runner.process_job(job)) == 15     # hold the model back before a retry
    assert worker.cancelled == ["w-1"] and repo.state == "requeued"


def test_retryable_submit_failure_backs_off_longer_each_time(tmp_path):
    repo = FakeRepo(job=_job(tmp_path, attempts=1))
    runner = _runner(repo, FakeWorker(submit_error=WorkerSubmitError("503", retryable=True)), tmp_path,
                     retry_backoff_seconds=15, max_retries=3)
    job = asyncio.run(runner.claim())
    assert asyncio.run(runner.process_job(job)) == 30


def test_the_poll_loop_beats_the_heartbeat_even_when_polls_fail(tmp_path):
    repo = FakeRepo(job=_job(tmp_path))
    asyncio.run(_runner(repo, FakeWorker(polls=[httpx.ReadTimeout("slow"), httpx.ReadTimeout("slow"), DONE]), tmp_path).tick())
    assert repo.updates.count({}) >= 2 and repo.state == "completed"


def test_outputs_of_a_job_that_ended_meanwhile_are_removed(tmp_path):
    repo = FakeRepo(job=_job(tmp_path))
    repo.complete_ok = False                   # cancelled (or deleted) while downloading
    asyncio.run(_runner(repo, FakeWorker(polls=[DONE]), tmp_path).tick())
    assert not Path(_store.output_path(str(tmp_path), 5, UUID)).exists()
    assert not Path(_store.preview_path(str(tmp_path), 5, UUID)).exists()


def test_large_pictures_are_shrunk_before_cutout_and_submit(tmp_path):
    big = io.BytesIO()
    Image.new("RGB", (4000, 3000), (1, 2, 3)).save(big, format="JPEG")
    seen = []

    async def cutout(image):
        seen.append(Image.open(io.BytesIO(image)).size)
        return None

    payload = _payload(tmp_path, big.getvalue(), "auto", cutout)
    assert seen == [(2048, 1536)]
    assert Image.open(io.BytesIO(base64.b64decode(payload["image"]))).size == (2048, 1536)


def test_prepare_image_keeps_small_pictures_and_alpha():
    small = _png(rgba=True, transparent=True)
    assert runner_mod.prepare_image(small) == (small, True)
    big = io.BytesIO()
    Image.new("RGBA", (3000, 100), (0, 0, 0, 0)).save(big, format="PNG")
    out, transparent = runner_mod.prepare_image(big.getvalue(), max_side=1000)
    assert transparent is True
    assert Image.open(io.BytesIO(out)).mode == "RGBA" and Image.open(io.BytesIO(out)).size == (1000, 33)


# -- concurrency -----------------------------------------------------------------

class TwoModelRepo(FakeRepo):
    """Serves one queued job per model, and records claim filters."""

    def __init__(self, jobs):
        super().__init__(servable=[j["model"] for j in jobs])
        self.queue = list(jobs)
        self.claims = []
        self.completed = []

    async def claim_next(self, worker_id, models=None, exclude_ids=None):
        self.claims.append(list(models))
        for j in self.queue:
            if j["model"] in models:
                self.queue.remove(j)
                return j
        return None

    async def complete(self, job_id, **fields):
        self.completed.append(job_id)


def test_jobs_for_different_models_run_side_by_side(tmp_path):
    a = _job(tmp_path, id=1)
    b = _job(tmp_path, id=2, model="tencent/hunyuan3d-2.1", job_uuid="mesh-ffffffffffffffffffffffff")
    repo = TwoModelRepo([a, b])
    gate = asyncio.Event()

    class SlowWorker(FakeWorker):
        async def poll(self, base_url, worker_job_id):
            await gate.wait()
            return DONE

    worker = SlowWorker()
    runner = _runner(repo, worker, tmp_path)

    async def go():
        first = await runner.claim()
        runner.start(first)
        second = await runner.claim()      # the first model is busy, the other is not
        assert second is not None and second["model"] != first["model"]
        runner.start(second)
        assert await runner.claim() is None  # both models busy now
        for _ in range(300):                  # picture preparation runs in a thread
            if len(worker.submitted) == 2:
                break
            await asyncio.sleep(0.01)
        assert len(worker.submitted) == 2     # both submitted before either finished
        gate.set()
        await asyncio.gather(*list(runner._tasks))

    asyncio.run(go())
    assert sorted(repo.completed) == [1, 2] and runner.busy_models == set()
    assert repo.claims[1] == ["tencent/hunyuan3d-2.1"]


def test_a_requeued_job_releases_its_model_after_a_pause(tmp_path):
    repo = FakeRepo(job=_job(tmp_path), backend={})
    runner = _runner(repo, FakeWorker(), tmp_path, poll_interval=0)

    async def go():
        job = await runner.claim()
        await runner.start(job)

    asyncio.run(go())
    assert repo.state == "requeued" and runner.busy_models == set()


# -- background removal ---------------------------------------------------------

def _payload(tmp_path, image, mode, cutout):
    job = _job(tmp_path, image=image, remove_background=mode)
    runner = _runner(FakeRepo(), FakeWorker(), tmp_path, cutout=cutout)
    return asyncio.run(runner.build_payload(job))


def test_auto_cuts_out_an_opaque_picture(tmp_path):
    calls = []

    async def cutout(image):
        calls.append(image)
        return b"CUT"

    payload = _payload(tmp_path, _png(), "auto", cutout)
    assert len(calls) == 1 and base64.b64decode(payload["image"]) == b"CUT"


def test_auto_keeps_a_picture_that_is_already_cut_out(tmp_path):
    calls = []

    async def cutout(image):
        calls.append(image)
        return b"CUT"

    original = _png(rgba=True, transparent=True)
    payload = _payload(tmp_path, original, "auto", cutout)
    assert calls == [] and base64.b64decode(payload["image"]) == original


def test_off_never_cuts_out_and_on_always_does(tmp_path):
    calls = []

    async def cutout(image):
        calls.append(image)
        return b"CUT"

    _payload(tmp_path, _png(), "off", cutout)
    assert calls == []
    _payload(tmp_path, _png(rgba=True, transparent=True), "on", cutout)
    assert len(calls) == 1


def test_a_failing_cutout_sends_the_original(tmp_path):
    async def cutout(image):
        raise RuntimeError("matting server down")

    original = _png()
    payload = _payload(tmp_path, original, "auto", cutout)
    assert base64.b64decode(payload["image"]) == original


def test_has_transparency():
    assert runner_mod.has_transparency(_png(rgba=True, transparent=True)) is True
    assert runner_mod.has_transparency(_png(rgba=True, transparent=False)) is False
    assert runner_mod.has_transparency(_png()) is False
    assert runner_mod.has_transparency(b"not an image") is False


# -- one job per worker: two workers serving one model run two jobs at once ------

class TwoWorkerRepo(FakeRepo):
    """Two healthy backends serving the same model; records which got which job."""

    def __init__(self, jobs, backend_ids=(11, 12)):
        super().__init__()
        self.queue = list(jobs)
        self.jobs_by_id = {j["id"]: j for j in jobs}
        self.ids = list(backend_ids)
        self.selected = []
        self.completed = []

    async def model_backends(self):
        return {MODEL: list(self.ids)}

    async def claim_next(self, worker_id, models=None, exclude_ids=None):
        self.claims = getattr(self, "claims", [])
        if MODEL not in (models or []):
            return None
        for j in sorted(self.queue, key=lambda j: j["id"]):     # priority, then id
            if j["id"] not in (exclude_ids or ()):
                self.queue.remove(j)
                self.claims.append(j["id"])
                return j
        return None

    async def requeue(self, job_id, *, attempts=None, clear_backend=False):
        self.requeued = getattr(self, "requeued", []) + [job_id]
        self.queue.append(dict(self.jobs_by_id[job_id], attempts=attempts or 0))

    async def select_backend(self, model, exclude=()):
        for i in self.ids:
            if i not in exclude:
                self.selected.append(i)
                return {"id": i, "url": f"https://w{i}:8006"}
        return None

    async def complete(self, job_id, **fields):
        self.completed.append(job_id)
        return True


def test_two_workers_for_one_model_run_two_jobs_at_once(tmp_path):
    jobs = [_job(tmp_path, id=1, job_uuid="mesh-" + "a" * 24), _job(tmp_path, id=2, job_uuid="mesh-" + "b" * 24),
            _job(tmp_path, id=3, job_uuid="mesh-" + "c" * 24)]
    repo = TwoWorkerRepo(jobs)
    runner = _runner(repo, FakeWorker(polls=[DONE] * 4), tmp_path)

    async def go():
        runner.start(await runner.claim())
        runner.start(await runner.claim())          # a second backend is free
        assert await runner.claim() is None          # both busy: the third job waits
        await asyncio.gather(*list(runner._tasks))
        return await runner.claim()                  # both free again

    third = asyncio.run(go())
    assert sorted(repo.selected) == [11, 12]         # each job on its own worker
    assert sorted(repo.completed) == [1, 2] and third is not None and third["id"] == 3
    assert runner._backend_of == {}                  # released when the jobs ended


def test_one_worker_still_means_one_job_at_a_time(tmp_path):
    jobs = [_job(tmp_path, id=1, job_uuid="mesh-" + "a" * 24), _job(tmp_path, id=2, job_uuid="mesh-" + "b" * 24)]
    repo = TwoWorkerRepo(jobs, backend_ids=(11,))
    runner = _runner(repo, FakeWorker(polls=[DONE] * 2), tmp_path)

    async def go():
        runner.start(await runner.claim())
        assert await runner.claim() is None
        await asyncio.gather(*list(runner._tasks))

    asyncio.run(go())
    assert repo.selected == [11]


def test_concurrent_selection_never_puts_two_jobs_on_one_worker(tmp_path):
    """select_backend awaits the registry: both jobs can ask before either has
    recorded its choice. The runner must still give them different workers."""

    class SlowSelectRepo(TwoWorkerRepo):
        async def select_backend(self, model, exclude=()):
            snapshot = set(exclude)
            await asyncio.sleep(0)               # the registry call yields
            await asyncio.sleep(0)
            for i in self.ids:
                if i not in snapshot:
                    self.selected.append(i)
                    return {"id": i, "url": f"https://w{i}:8006"}
            return None

    jobs = [_job(tmp_path, id=1, job_uuid="mesh-" + "a" * 24), _job(tmp_path, id=2, job_uuid="mesh-" + "b" * 24)]
    repo = SlowSelectRepo(jobs)
    urls = []

    class UrlWorker(FakeWorker):
        async def submit(self, base_url, payload):
            urls.append(base_url)
            return await super().submit(base_url, payload)

    runner = _runner(repo, UrlWorker(polls=[DONE] * 2), tmp_path)

    async def go():
        a, b = await runner.claim(), await runner.claim()
        runner.start(a); runner.start(b)
        await asyncio.gather(*list(runner._tasks))

    asyncio.run(go())
    assert sorted(urls) == ["https://w11:8006", "https://w12:8006"]
    assert sorted(repo.completed) == [1, 2]



def test_a_requeued_job_sits_out_its_back_off_and_is_never_run_twice(tmp_path):
    """Two workers; job 1's first submit gets a retryable 503. It goes back to
    the queue but its task holds it for the back-off: claim() must not hand it
    out again meanwhile (that ran it twice, untracked, and doubled up a
    worker). Other jobs use the free slot; job 1 runs once the back-off ends."""
    jobs = [_job(tmp_path, id=i, job_uuid="mesh-" + c * 24) for i, c in ((1, "a"), (2, "b"), (3, "c"))]
    repo = TwoWorkerRepo(jobs)
    running = {11: set(), 12: set()}
    overlap = []

    class Worker(FakeWorker):
        first = True

        async def submit(self, base_url, payload):
            backend = int(base_url.split("w")[1].split(":")[0])
            if self.first and payload is not None and len(self.submitted) == 0:
                self.first = False
                self.submitted.append(payload)
                raise WorkerSubmitError("503 busy", retryable=True)
            self.submitted.append(payload)
            worker_job_id = f"w-{len(self.submitted)}"
            running[backend].add(worker_job_id)
            if len(running[backend]) > 1:
                overlap.append(backend)
            return worker_job_id

        async def poll(self, base_url, worker_job_id):
            await asyncio.sleep(0)
            running[int(base_url.split("w")[1].split(":")[0])].discard(worker_job_id)
            return DONE

    runner = _runner(repo, Worker(), tmp_path, retry_backoff_seconds=0.05)

    async def go():
        runner.start(await runner.claim())                 # job 1 -> 503 -> requeued, backing off
        for _ in range(5):
            await asyncio.sleep(0)
        assert repo.requeued == [1] and 1 in runner._active
        second = await runner.claim()
        assert second is not None and second["id"] == 2    # NOT job 1 again
        runner.start(second)
        assert await runner.claim() is None                # 2 slots: job 1 (backing off) + job 2
        await asyncio.gather(*list(runner._tasks))         # back-off ends, both slots free
        for _ in range(2):
            job = await runner.claim()
            if job:
                runner.start(job)
        await asyncio.gather(*list(runner._tasks))

    asyncio.run(go())
    assert repo.claims.count(1) == 2 and sorted(repo.completed) == [1, 2, 3]
    assert overlap == [] and runner._active == {} and runner._backend_of == {}


def test_tick_releases_its_backend(tmp_path):
    repo = TwoWorkerRepo([_job(tmp_path, id=1), _job(tmp_path, id=2, job_uuid="mesh-" + "b" * 24)], backend_ids=(11,))
    runner = _runner(repo, FakeWorker(polls=[DONE] * 2), tmp_path)
    assert asyncio.run(runner.tick()) is True and runner._backend_of == {}
    assert asyncio.run(runner.tick()) is True              # the one backend is free again
