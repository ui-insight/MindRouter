############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_matting_service.py: Unit tests for the standalone
# matting (background-removal) server (matting_service/),
# which makes the alpha matte behind `background:
# "transparent"` on the images API. Must pass with NO gpu and
# NO torch: the model is replaced through MODEL_FACTORY.
#
# Covers: refusing to start without a key, bearer auth (401),
# open liveness vs keyed health detail, the reply (a PNG
# greyscale matte of the picture's own size, with the model /
# seconds / coverage headers), what is refused before the
# model runs (empty, not an image, a type that lies about
# itself, too many pixels, too many bytes), the bounded queue
# (503), one picture on the model at a time, a caller who
# leaves while waiting, a model failure answered 500 without
# its text, a matte of the wrong shape, the settings read
# from the environment, and the deployment files.
#
############################################################

"""Unit tests for the matting service."""

import asyncio
import io
import logging
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Same pattern as test_clef_service.py: make the repo root importable so
# `matting_service` resolves regardless of pytest rootdir.
_REPO_ROOT = str(Path(__file__).resolve().parents[4])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from matting_service import server  # noqa: E402
from matting_service.server import ServiceConfig  # noqa: E402

KEY = "test-matting-key"
AUTH = {"Authorization": f"Bearer {KEY}"}
SECRET = "SECRET-ERROR-TEXT"
_SERVICE = Path(_REPO_ROOT) / "matting_service"


class FakeEngine:
    """Stands in for the loaded model: the left half of every picture is the
    subject (255), the right half background (0)."""

    device = "cpu"
    half = False

    def __init__(self, config, blocker=None, fail=None, wrong=None):
        self.config = config
        self.seen = []             # (mode, size) of every picture the model was given
        self._blocker, self._fail, self._wrong = blocker, fail, wrong

    def matte(self, image):
        if self._blocker is not None:
            self._blocker.wait(timeout=5)
        if self._fail is not None:
            raise self._fail
        self.seen.append((image.mode, image.size))
        if self._wrong == "size":
            return Image.new("L", (image.width + 1, image.height))
        if self._wrong == "mode":
            return Image.new("RGB", image.size)
        width, height = image.size
        matte = Image.new("L", (width, height), 0)
        matte.paste(255, (0, 0, width // 2, height))
        return matte


def _config(**over):
    return ServiceConfig(api_key=KEY, **over)


def _picture(size=(64, 48), fmt="PNG", mode="RGB", color=(200, 30, 30)):
    out = io.BytesIO()
    Image.new(mode, size, color).save(out, format=fmt)
    return out.getvalue()


@pytest.fixture
def engine_box(monkeypatch):
    """Installs a factory and exposes the engine the service built."""
    box = {"kwargs": {}}

    def factory(config):
        box["engine"] = FakeEngine(config, **box["kwargs"])
        return box["engine"]

    monkeypatch.setattr(server, "MODEL_FACTORY", factory)
    return box


@pytest.fixture
def client(engine_box):
    with TestClient(server.create_app(_config())) as c:
        yield c


# ---------------------------------------------------------------------------
# start-up and auth
# ---------------------------------------------------------------------------

class TestStartAndAuth:
    def test_refuses_to_start_without_a_key(self):
        with pytest.raises(SystemExit):
            server.create_app(ServiceConfig(api_key=None))

    def test_no_auth_must_be_asked_for_explicitly(self, engine_box):
        with TestClient(server.create_app(ServiceConfig(api_key=None, allow_no_auth=True))) as c:
            assert c.post("/v1/matte", content=_picture()).status_code == 200

    @pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": KEY},
                                         {"Authorization": f"Basic {KEY}"}])
    def test_matte_needs_the_bearer_key(self, client, engine_box, headers):
        response = client.post("/v1/matte", content=_picture(), headers=headers)
        assert response.status_code == 401
        assert engine_box["engine"].seen == []          # refused before the model ran

    def test_health_is_open_but_details_need_the_key(self, client):
        assert client.get("/health").json() == {"status": "ok"}
        detail = client.get("/health", headers=AUTH).json()
        assert detail["status"] == "ok" and detail["model"] == "birefnet-dynamic"
        assert detail["source"] == "ZhengPeng7/BiRefNet_dynamic" and detail["device"] == "cpu"
        assert detail["queue_depth"] == 0 and detail["max_queue"] == 16
        assert detail["stats"]["requests"] == 0

    def test_the_model_is_loaded_during_start_up_before_anything_is_served(self, engine_box):
        # uvicorn opens the port only after start-up, so there is no
        # "loading" answer: a server that is starting refuses connections.
        app = server.create_app(_config())
        worker = app.state.worker
        assert worker.ready is False and "engine" not in engine_box
        with TestClient(app) as c:
            assert worker.ready is True and worker.engine is engine_box["engine"]
            assert c.get("/health").json() == {"status": "ok"}
        source = (_SERVICE / "server.py").read_text()
        assert '"loading"' not in source.split("def create_app", 1)[1]


# ---------------------------------------------------------------------------
# the reply
# ---------------------------------------------------------------------------

class TestMatte:
    @pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
    def test_a_picture_comes_back_as_its_matte(self, client, engine_box, fmt):
        response = client.post("/v1/matte", content=_picture(fmt=fmt), headers=AUTH)
        assert response.status_code == 200 and response.headers["content-type"] == "image/png"
        matte = Image.open(io.BytesIO(response.content))
        assert matte.format == "PNG" and matte.mode == "L" and matte.size == (64, 48)
        assert matte.getpixel((0, 0)) == 255 and matte.getpixel((63, 47)) == 0
        assert engine_box["engine"].seen == [("RGB", (64, 48))]

    def test_headers_name_the_model_the_time_and_the_coverage(self, client):
        response = client.post("/v1/matte", content=_picture(), headers=AUTH)
        assert response.headers["x-matting-model"] == "birefnet-dynamic"
        assert float(response.headers["x-matting-seconds"]) >= 0
        assert float(response.headers["x-matting-coverage"]) == pytest.approx(0.5)

    def test_a_picture_with_alpha_is_given_to_the_model_as_rgb(self, client, engine_box):
        response = client.post("/v1/matte", content=_picture(mode="RGBA", color=(1, 2, 3, 0)), headers=AUTH)
        assert response.status_code == 200 and engine_box["engine"].seen == [("RGB", (64, 48))]

    def test_the_content_type_header_is_not_believed(self, client):
        # The gateway sends image/png; a JPEG under that label is still read as what it is.
        response = client.post("/v1/matte", content=_picture(fmt="JPEG"),
                               headers={**AUTH, "Content-Type": "image/png"})
        assert response.status_code == 200

    def test_stats_count_what_was_answered(self, client):
        for _ in range(3):
            client.post("/v1/matte", content=_picture(), headers=AUTH)
        stats = client.get("/health", headers=AUTH).json()["stats"]
        assert stats["requests"] == 3 and stats["answered"] == 3 and stats["failed"] == 0


# ---------------------------------------------------------------------------
# what is refused before the model runs
# ---------------------------------------------------------------------------

class TestRefused:
    @pytest.mark.parametrize("body,needle", [
        (b"", "empty"),
        (b"not an image at all", "PNG, JPEG or WebP"),
        (b"GIF89a" + b"\x00" * 64, "PNG, JPEG or WebP"),
        (b"\x89PNG\r\n\x1a\n" + b"garbage" * 10, "could not be decoded"),
        (b"\xff\xd8\xff" + b"garbage" * 10, "could not be decoded"),
    ])
    def test_a_body_that_is_not_a_picture_is_422(self, client, engine_box, body, needle):
        response = client.post("/v1/matte", content=body, headers=AUTH)
        assert response.status_code == 422 and needle in response.json()["detail"]
        assert engine_box["engine"].seen == []

    def test_too_many_pixels_is_422_before_decoding(self, engine_box):
        with TestClient(server.create_app(_config(max_pixels=64 * 48))) as c:
            assert c.post("/v1/matte", content=_picture(size=(64, 48)), headers=AUTH).status_code == 200
            response = c.post("/v1/matte", content=_picture(size=(65, 48)), headers=AUTH)
            assert response.status_code == 422 and "65 x 48" in response.json()["detail"]

    def test_too_many_bytes_is_413_declared_or_not(self, engine_box):
        picture = _picture()
        with TestClient(server.create_app(_config(max_body_bytes=len(picture)))) as c:
            assert c.post("/v1/matte", content=picture, headers=AUTH).status_code == 200
            assert c.post("/v1/matte", content=picture + b"x", headers=AUTH).status_code == 413

            def chunks():            # no Content-Length: the limit is enforced while reading
                yield picture
                yield b"x"

            assert c.post("/v1/matte", content=chunks(), headers=AUTH).status_code == 413
        assert engine_box["engine"].seen == [("RGB", (64, 48))]

    def test_open_image_checks_size_from_the_header(self):
        assert server.open_image(_picture(size=(10, 10)), max_pixels=100).size == (10, 10)
        with pytest.raises(server.BadRequest):
            server.open_image(_picture(size=(11, 10)), max_pixels=100)


# ---------------------------------------------------------------------------
# when the model is busy or fails
# ---------------------------------------------------------------------------

class TestBusyAndFailure:
    def test_a_full_queue_is_503_with_retry_after(self, engine_box):
        blocker = threading.Event()
        engine_box["kwargs"] = {"blocker": blocker}
        results = []
        with TestClient(server.create_app(_config(max_queue=1))) as c:
            first = threading.Thread(
                target=lambda: results.append(c.post("/v1/matte", content=_picture(), headers=AUTH)))
            first.start()
            worker = c.app.state.worker
            for _ in range(500):                       # until the first picture holds the only place
                if worker.queue_depth() == 1:
                    break
                threading.Event().wait(0.01)
            second = c.post("/v1/matte", content=_picture(), headers=AUTH)
            blocker.set()
            first.join(timeout=5)
        assert second.status_code == 503 and second.headers["retry-after"] == "1"
        assert results[0].status_code == 200
        assert worker.stats.rejected_busy == 1 and worker.queue_depth() == 0

    def test_the_model_sees_one_picture_at_a_time(self, engine_box):
        box = {"now": 0, "most": 0}
        real = FakeEngine.matte

        def counted(self, image):
            box["now"] += 1
            box["most"] = max(box["most"], box["now"])
            time.sleep(0.05)
            try:
                return real(self, image)
            finally:
                box["now"] -= 1

        results = []
        with TestClient(server.create_app(_config())) as c:
            engine_box["engine"].matte = counted.__get__(engine_box["engine"])
            threads = [threading.Thread(
                target=lambda: results.append(c.post("/v1/matte", content=_picture(), headers=AUTH).status_code))
                for _ in range(6)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
            assert c.app.state.worker._executor._max_workers == 1
        assert results == [200] * 6 and box["most"] == 1

    async def test_a_caller_who_leaves_while_waiting_frees_its_work(self):
        class Gone:
            async def is_disconnected(self):
                return True

        class Here:
            async def is_disconnected(self):
                return False

        cancelled = asyncio.Event()

        async def work():
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        with pytest.raises(server.ClientGone):
            await server._unless_disconnected(Gone(), work(), poll_seconds=0.01)
        await asyncio.wait_for(cancelled.wait(), timeout=1)

        async def quick():
            await asyncio.sleep(0.03)
            return "matte"

        assert await server._unless_disconnected(Here(), quick(), poll_seconds=0.01) == "matte"

    def test_a_caller_who_left_is_answered_499(self, engine_box, monkeypatch):
        async def gone(request, work, poll_seconds=0.25):
            work.close()
            raise server.ClientGone()

        monkeypatch.setattr(server, "_unless_disconnected", gone)
        with TestClient(server.create_app(_config())) as c:
            assert c.post("/v1/matte", content=_picture(), headers=AUTH).status_code == 499
            assert c.get("/health", headers=AUTH).json()["stats"]["failed"] == 0

    def test_a_picture_stuck_on_the_model_turns_health_unhealthy(self, engine_box):
        # A hung inference thread behind a live web server would otherwise
        # look healthy for ever. MindRouter's health check reads this word.
        blocker = threading.Event()
        engine_box["kwargs"] = {"blocker": blocker}
        results = []
        with TestClient(server.create_app(_config(stall_seconds=0))) as c:
            assert c.get("/health").json() == {"status": "ok"}             # idle is not stalled
            stuck = threading.Thread(
                target=lambda: results.append(c.post("/v1/matte", content=_picture(), headers=AUTH)))
            stuck.start()
            seen = None
            for _ in range(500):
                seen = c.get("/health")
                if seen.status_code != 200:
                    break
                threading.Event().wait(0.01)
            # Said twice: a status code for anything that reads those, the word for MindRouter's check.
            assert seen.status_code == 503 and seen.json() == {"status": "unhealthy"}
            detail = c.get("/health", headers=AUTH)
            assert detail.status_code == 503 and detail.json()["status"] == "unhealthy"
            assert detail.json()["stall_seconds"] == 0 and detail.json()["queue_depth"] == 1
            # A new picture is refused at once, not queued behind the stuck one.
            started = time.monotonic()
            refused = c.post("/v1/matte", content=_picture(), headers=AUTH)
            assert refused.status_code == 503 and refused.headers["retry-after"] == "30"
            assert time.monotonic() - started < 1.0 and c.app.state.worker.queue_depth() == 1
            assert c.post("/v1/matte", content=_picture()).status_code == 401      # the key is still checked first
            blocker.set()
            stuck.join(timeout=5)
            assert results[0].status_code == 200
            assert c.get("/health").status_code == 200                              # and it recovers by itself
            assert c.post("/v1/matte", content=_picture(), headers=AUTH).status_code == 200

    def test_a_picture_within_the_limit_is_not_a_stall(self, engine_box):
        blocker = threading.Event()
        engine_box["kwargs"] = {"blocker": blocker}
        with TestClient(server.create_app(_config(stall_seconds=60))) as c:
            busy = threading.Thread(target=lambda: c.post("/v1/matte", content=_picture(), headers=AUTH))
            busy.start()
            worker = c.app.state.worker
            for _ in range(500):
                if worker._running_since is not None:
                    break
                threading.Event().wait(0.01)
            assert worker._running_since is not None and c.get("/health").json() == {"status": "ok"}
            blocker.set()
            busy.join(timeout=5)
            assert worker._running_since is None

    def test_the_stall_clock_stops_when_the_model_fails(self, engine_box):
        engine_box["kwargs"] = {"fail": RuntimeError("x")}
        with TestClient(server.create_app(_config(stall_seconds=0))) as c:
            assert c.post("/v1/matte", content=_picture(), headers=AUTH).status_code == 500
            assert c.app.state.worker._running_since is None
            assert c.get("/health").json() == {"status": "ok"}

    def test_the_health_word_is_one_the_gateways_check_reads_as_down(self):
        from backend.app.core.telemetry.adapters.decision import NOT_READY

        assert "unhealthy" in NOT_READY

    def test_a_model_failure_is_500_and_its_text_goes_nowhere(self, engine_box, caplog):
        engine_box["kwargs"] = {"fail": RuntimeError(f"CUDA error near {SECRET}")}
        with caplog.at_level(logging.DEBUG), TestClient(server.create_app(_config())) as c:
            response = c.post("/v1/matte", content=_picture(), headers=AUTH)
            stats = c.get("/health", headers=AUTH).json()["stats"]
        assert response.status_code == 500 and response.json() == {"detail": "matting failed"}
        assert SECRET not in response.text and SECRET not in caplog.text
        assert "matting_failed error_type=RuntimeError" in caplog.text
        assert stats["failed"] == 1 and stats["answered"] == 0

    @pytest.mark.parametrize("wrong", ["size", "mode"])
    def test_a_matte_of_the_wrong_shape_is_never_returned(self, engine_box, wrong):
        engine_box["kwargs"] = {"wrong": wrong}
        with TestClient(server.create_app(_config())) as c:
            assert c.post("/v1/matte", content=_picture(), headers=AUTH).status_code == 500

    def test_the_place_in_the_queue_is_given_back_after_a_failure(self, engine_box):
        engine_box["kwargs"] = {"fail": RuntimeError("x")}
        with TestClient(server.create_app(_config(max_queue=1))) as c:
            for _ in range(3):
                assert c.post("/v1/matte", content=_picture(), headers=AUTH).status_code == 500
            assert c.app.state.worker.queue_depth() == 0


# ---------------------------------------------------------------------------
# settings and deployment files
# ---------------------------------------------------------------------------

class TestSettings:
    def test_defaults(self):
        config = ServiceConfig()
        assert config.model == "ZhengPeng7/BiRefNet_dynamic" and config.revision is None
        assert config.host == "127.0.0.1" and config.side == 1024 and config.half is True

    def test_read_from_the_environment(self, monkeypatch):
        for name, value in {"MATTING_MODEL": "/models/birefnet", "MATTING_REVISION": " abc123 ",
                            "MATTING_SERVED_NAME": "cutout", "MATTING_DEVICE": "cuda:1", "MATTING_HALF": "0",
                            "MATTING_PORT": "18123", "MATTING_API_KEY": " k ", "MATTING_MAX_QUEUE": "3",
                            "MATTING_MAX_PIXELS": "1048576", "MATTING_STALL_SECONDS": "20"}.items():
            monkeypatch.setenv(name, value)
        config = ServiceConfig.from_env()
        assert (config.model, config.revision, config.served_name) == ("/models/birefnet", "abc123", "cutout")
        assert (config.device, config.half, config.port) == ("cuda:1", False, 18123)
        assert (config.api_key, config.max_queue, config.max_pixels) == ("k", 3, 1048576)
        assert config.stall_seconds == 20 and ServiceConfig().stall_seconds == 60

    @pytest.mark.parametrize("name,value", [("MATTING_PORT", "http"), ("MATTING_MAX_QUEUE", "0"),
                                            ("MATTING_SIDE", "32"), ("MATTING_STALL_SECONDS", "1")])
    def test_a_bad_number_stops_the_service_at_start(self, monkeypatch, name, value):
        monkeypatch.setenv(name, value)
        with pytest.raises(SystemExit):
            ServiceConfig.from_env()

    def test_a_blank_key_is_no_key(self, monkeypatch):
        monkeypatch.setenv("MATTING_API_KEY", "   ")
        assert ServiceConfig.from_env().api_key is None

    def test_coverage_is_the_mean_matte_value(self):
        assert server._coverage(Image.new("L", (4, 4), 0)) == 0.0
        assert server._coverage(Image.new("L", (4, 4), 255)) == 1.0
        assert server._coverage(Image.new("L", (4, 4), 51)) == pytest.approx(0.2)


class TestDeploymentFiles:
    def test_unit_template_loads_offline_and_reads_the_key_from_a_file(self):
        unit = (_SERVICE / "deploy" / "matting-service.service").read_text()
        assert "ExecStart=__VENV_DIR__/bin/python -m matting_service" in unit
        assert "EnvironmentFile=__ENV_FILE__" in unit and "MATTING_API_KEY=" not in unit
        # The model's code runs at load: it must be the reviewed copy on disk, never a fresh download.
        assert "Environment=HF_HUB_OFFLINE=1" in unit

    def test_requirements_and_readme_exist(self):
        requirements = (_SERVICE / "requirements.txt").read_text()
        for package in ("torch", "transformers", "timm", "kornia", "einops", "pillow", "fastapi", "uvicorn"):
            assert package in requirements, package
        readme = (_SERVICE / "README.md").read_text()
        assert "MATTING_API_KEY" in readme and "/v1/matte" in readme and "MATTING_REVISION" in readme

    def test_the_real_engine_is_the_default_and_pins_the_revision(self):
        source = (_SERVICE / "server.py").read_text()
        assert "factory = MODEL_FACTORY or BiRefNetEngine" in source
        assert 'kwargs["revision"] = config.revision' in source
        # A repository id with no pinned commit runs whatever code is there today: say so at start.
        engine = source.split("class BiRefNetEngine", 1)[1].split("def matte", 1)[0]
        assert "elif not os.path.isdir(config.model):" in engine and "matting_model_revision_not_pinned" in engine
