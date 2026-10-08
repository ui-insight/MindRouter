############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_video_model_choice.py: More than one video model.
#     Render-time estimates are kept per model (models render at
#     different speeds), the queue uses each job's own model,
#     and the Video tab offers a model picker when there is a
#     choice.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Video model choice: per-model ETAs and the Video tab's model picker."""

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("aiosqlite")

_TEMPLATES = Path(__file__).resolve().parents[2] / "dashboard" / "templates"
OLD, NEW = "lightricks/ltx-2.3-distilled", "lightricks/ltx-2.5-distilled"


@pytest.fixture
async def db():
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from backend.app.db.models import Base

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    names = ("groups", "users", "video_projects", "video_jobs", "video_shots")
    tables = [Base.metadata.tables[t] for t in names]
    async with engine.begin() as conn:
        await conn.run_sync(lambda s: Base.metadata.create_all(s, tables=tables))
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as session:
        yield session
    await engine.dispose()


async def _job(db, n, model, gpu_seconds, seconds=5.0, status=None):
    """One project + one job for `model`; completed unless `status` says otherwise."""
    from backend.app.db.models import VideoJob, VideoJobStatus, VideoProject

    # Explicit ids: SQLite does not auto-number BIGINT primary keys.
    project = VideoProject(id=n + 1, user_id=1, model=model, size="1280x704")
    db.add(project)
    await db.flush()
    job = VideoJob(id=n + 1, job_uuid=f"vid-{n:024d}", project_id=project.id, user_id=1,
                   status=status or VideoJobStatus.COMPLETED, gpu_seconds=gpu_seconds,
                   duration_seconds=seconds, completed_at=datetime.now(timezone.utc))
    db.add(job)
    await db.flush()
    return job


class TestRenderRatioPerModel:
    async def test_each_model_gets_its_own_speed(self, db):
        from backend.app.db import crud

        for n in range(3):
            await _job(db, n, OLD, gpu_seconds=11)        # 2.2 s per output second
        for n in range(3, 6):
            await _job(db, n, NEW, gpu_seconds=30)        # 6.0 s per output second
        await db.commit()

        assert await crud.get_recent_render_ratio(db, model=OLD) == pytest.approx(2.2)
        assert await crud.get_recent_render_ratio(db, model=NEW) == pytest.approx(6.0)

    async def test_only_the_latest_few_jobs_count(self, db):
        # A speed-up (weights kept resident) must show within a few jobs, even
        # with weeks of slower history behind it.
        from backend.app.db import crud

        for n in range(10):
            await _job(db, n, OLD, gpu_seconds=35)        # old, reloading every render
        for n in range(10, 15):
            await _job(db, n, OLD, gpu_seconds=11)        # the newest five
        await db.commit()

        assert await crud.get_recent_render_ratio(db, model=OLD) == pytest.approx(2.2)

    async def test_a_model_without_history_falls_back_to_the_default(self, db):
        from backend.app.db import crud

        await _job(db, 1, OLD, gpu_seconds=11)
        await db.commit()
        assert await crud.get_recent_render_ratio(db, model=NEW) == 6.0
        assert await crud.get_recent_render_ratio(db, model=NEW, default=4.0) == 4.0

    async def test_unfinished_jobs_are_not_history(self, db):
        from backend.app.db import crud
        from backend.app.db.models import VideoJobStatus

        await _job(db, 1, NEW, gpu_seconds=5, status=VideoJobStatus.RENDERING)
        await _job(db, 2, NEW, gpu_seconds=5, status=VideoJobStatus.FAILED)
        await db.commit()
        assert await crud.get_recent_render_ratio(db, model=NEW) == 6.0

    async def test_without_a_model_every_job_counts(self, db):
        from backend.app.db import crud

        await _job(db, 1, OLD, gpu_seconds=10)            # 2.0
        await _job(db, 2, NEW, gpu_seconds=30)            # 6.0
        await db.commit()
        assert await crud.get_recent_render_ratio(db) == pytest.approx(4.0)

    async def test_queue_rows_name_their_model(self, db):
        from backend.app.db import crud
        from backend.app.db.models import VideoJobStatus

        await _job(db, 1, NEW, gpu_seconds=0, status=VideoJobStatus.QUEUED)
        await db.commit()
        rows = await crud.get_active_video_queue(db)
        assert [r["model"] for r in rows] == [NEW]


class TestServableModelsAndClaim:
    @pytest.fixture
    async def fleet(self, db):
        """Three video backends: 2.3 healthy, 2.5 restarting (unhealthy), an old
        disabled one; plus a healthy chat backend that must not count."""
        from backend.app.db.models import Backend, BackendEngine, BackendStatus, Base, Modality, Model

        async with db.bind.begin() as conn:
            await conn.run_sync(lambda s: Base.metadata.create_all(
                s, tables=[Base.metadata.tables["backends"], Base.metadata.tables["models"]]))
        rows = [
            (1, "ltx23", BackendEngine.VIDEO, BackendStatus.HEALTHY, OLD, Modality.VIDEO_GENERATION),
            (2, "ltx25", BackendEngine.VIDEO, BackendStatus.UNHEALTHY, NEW, Modality.VIDEO_GENERATION),
            (3, "old", BackendEngine.VIDEO, BackendStatus.DISABLED, "lightricks/ltx-2.2", Modality.VIDEO_GENERATION),
            (4, "chat", BackendEngine.VLLM, BackendStatus.HEALTHY, "qwen/qwen3.8-27b", Modality.CHAT),
        ]
        for bid, name, engine, status, model, modality in rows:
            db.add(Backend(id=bid, name=name, url=f"https://{name}:8000", engine=engine, status=status))
            db.add(Model(id=bid, backend_id=bid, name=model, modality=modality))
        await db.commit()

    async def test_only_models_with_a_healthy_video_backend(self, db, fleet):
        from backend.app.db import crud
        from backend.app.db.models import Backend, BackendStatus

        assert await crud.get_servable_video_models(db) == [OLD]
        (await db.get(Backend, 2)).status = BackendStatus.HEALTHY      # 2.5 finished warming up
        await db.commit()
        assert await crud.get_servable_video_models(db) == [OLD, NEW]

    async def test_claim_skips_jobs_for_models_that_are_down(self, db):
        from backend.app.db import crud
        from backend.app.db.models import VideoJobStatus

        queued = VideoJobStatus.QUEUED
        await _job(db, 1, NEW, gpu_seconds=0, status=queued)     # oldest, but its model is down
        await _job(db, 2, OLD, gpu_seconds=0, status=queued)
        await db.commit()

        job = await crud.claim_next_video_job(db, "runner-1", models=[OLD])
        assert job.id == 3 and job.status == VideoJobStatus.RENDERING      # the 2.3 job, ids are n + 1
        assert await crud.claim_next_video_job(db, "runner-1", models=[OLD]) is None
        assert await crud.claim_next_video_job(db, "runner-1", models=[]) is None
        assert (await crud.claim_next_video_job(db, "runner-1")).id == 2     # no filter: any model


class TestQueueEstimates:
    async def test_each_queued_job_is_estimated_with_its_own_model(self):
        from backend.app.dashboard import video as video_dash

        rows = [
            {"job_uuid": "vid-a", "user_id": 7, "status": "in_progress", "progress": 50.0, "seconds": 10.0,
             "quality": "standard", "size": "1280x704", "model": OLD, "created_at": 0, "started_at": None},
            {"job_uuid": "vid-b", "user_id": 8, "status": "queued", "progress": 0.0, "seconds": 10.0,
             "quality": "standard", "size": "1280x704", "model": NEW, "created_at": 0, "started_at": None},
        ]
        speeds = {OLD: 2.0, NEW: 5.0}
        ratio = AsyncMock(side_effect=lambda db, model=None: speeds[model])
        with patch.object(video_dash, "_get_video_user", AsyncMock(return_value=(SimpleNamespace(id=7), None))), \
             patch.object(video_dash.crud, "get_active_video_queue", AsyncMock(return_value=rows)), \
             patch.object(video_dash.crud, "get_recent_render_ratio", ratio):
            out = await video_dash.video_queue(SimpleNamespace(), db=None)

        assert [q["est_seconds"] for q in out["queue"]] == [20, 50]
        assert out["queue"][1]["eta_seconds"] == 20          # waits for the 2.3 job ahead of it
        assert ratio.await_count == 2                         # one lookup per model, not per row


class TestVideoTabPicker:
    def _source(self):
        return (_TEMPLATES / "user" / "video.html").read_text()

    def test_template_compiles(self):
        from jinja2 import Environment, FileSystemLoader

        Environment(loader=FileSystemLoader(str(_TEMPLATES)), autoescape=True).get_template("user/video.html")

    def test_picker_only_when_there_is_a_choice_and_it_is_sent(self):
        src = self._source()
        assert "{% if video_models|length > 1 %}" in src
        assert '<label for="vid-model" class="form-label">Model</label>' in src
        assert 'if ($("vid-model")) body.model = $("vid-model").value;' in src

    def test_picker_markup_renders_with_the_default_selected(self):
        from jinja2 import Environment

        src = self._source()
        start = src.index("{% if video_models|length > 1 %}")
        end = src.index("{% endif %}", src.index("{% endfor %}", start)) + len("{% endif %}")
        fragment = Environment(autoescape=True).from_string(src[start:end])

        two = fragment.render(video_models=[OLD, NEW], default_model=NEW)
        assert f'<option value="{NEW}" selected>' in two and f'<option value="{OLD}" >' in two
        assert fragment.render(video_models=[OLD], default_model=OLD).strip() == ""
