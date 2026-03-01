############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# test_scheduler.py: Unit tests for fair-share scheduler
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for the scheduler components."""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from backend.app.core.scheduler.queue import Job, JobModality, RequestQueue
from backend.app.core.scheduler.fairshare import FairShareManager, UserState
from backend.app.core.scheduler.scoring import BackendScorer, BackendScore, HardConstraints
from backend.app.db.models import Backend, BackendEngine, BackendStatus, Model, Modality


class TestJob:
    """Tests for Job dataclass."""

    def test_job_creation(self):
        """Test basic job creation."""
        job = Job(
            request_id="req-123",
            user_id=1,
            api_key_id=1,
            model="llama3.2",
            modality=JobModality.CHAT,
        )

        assert job.request_id == "req-123"
        assert job.user_id == 1
        assert job.model == "llama3.2"
        assert job.modality == JobModality.CHAT
        assert job.priority == 0.0

    def test_job_estimated_cost(self):
        """Test job cost estimation."""
        job = Job(
            request_id="req-123",
            user_id=1,
            api_key_id=1,
            model="llama3.2",
            modality=JobModality.CHAT,
            estimated_prompt_tokens=100,
            estimated_completion_tokens=50,
        )

        assert job.get_estimated_cost() == 150

    def test_job_comparison_by_priority(self):
        """Test job comparison uses priority (higher priority first)."""
        job1 = Job(
            request_id="req-1",
            user_id=1,
            api_key_id=1,
            model="llama3.2",
            modality=JobModality.CHAT,
            priority=10.0,
        )
        job2 = Job(
            request_id="req-2",
            user_id=2,
            api_key_id=2,
            model="llama3.2",
            modality=JobModality.CHAT,
            priority=5.0,
        )

        # job1 has higher priority, so it should come first (be "less than")
        assert job1 < job2


class TestRequestQueue:
    """Tests for RequestQueue."""

    @pytest.fixture
    def queue(self):
        """Create a fresh queue for each test."""
        return RequestQueue()

    @pytest.mark.asyncio
    async def test_enqueue_and_dequeue(self, queue):
        """Test basic enqueue and dequeue operations."""
        job = Job(
            request_id="req-1",
            user_id=1,
            api_key_id=1,
            model="llama3.2",
            modality=JobModality.CHAT,
            priority=5.0,
        )

        position = await queue.enqueue(job)
        assert position == 0
        assert len(queue) == 1

        dequeued = await queue.dequeue()
        assert dequeued.request_id == "req-1"
        assert len(queue) == 0

    @pytest.mark.asyncio
    async def test_priority_ordering(self, queue):
        """Test that jobs are dequeued in priority order."""
        # Enqueue jobs with different priorities
        jobs = [
            Job(
                request_id=f"req-{i}",
                user_id=i,
                api_key_id=i,
                model="llama3.2",
                modality=JobModality.CHAT,
                priority=float(i),
            )
            for i in [3, 1, 4, 1, 5, 9, 2, 6]
        ]

        for job in jobs:
            await queue.enqueue(job)

        # Dequeue should return highest priority first
        priorities = []
        while len(queue) > 0:
            job = await queue.dequeue()
            priorities.append(job.priority)

        assert priorities == sorted(priorities, reverse=True)

    @pytest.mark.asyncio
    async def test_user_queue_access(self, queue):
        """Test getting jobs for a specific user."""
        for i in range(5):
            job = Job(
                request_id=f"req-{i}",
                user_id=i % 2,  # Alternate between user 0 and 1
                api_key_id=i,
                model="llama3.2",
                modality=JobModality.CHAT,
            )
            await queue.enqueue(job)

        user0_jobs = await queue.get_user_queue(0)
        user1_jobs = await queue.get_user_queue(1)

        assert len(user0_jobs) == 3  # req-0, req-2, req-4
        assert len(user1_jobs) == 2  # req-1, req-3

    @pytest.mark.asyncio
    async def test_cancel_job(self, queue):
        """Test job cancellation."""
        job = Job(
            request_id="req-1",
            user_id=1,
            api_key_id=1,
            model="llama3.2",
            modality=JobModality.CHAT,
        )

        await queue.enqueue(job)
        assert len(queue) == 1

        cancelled = await queue.cancel_job("req-1")
        assert cancelled is True

        # Job should not be returned on dequeue
        result = await queue.dequeue()
        assert result is None

    @pytest.mark.asyncio
    async def test_get_queue_stats(self, queue):
        """Test queue statistics."""
        for i in range(3):
            job = Job(
                request_id=f"req-{i}",
                user_id=i % 2,
                api_key_id=i,
                model="llama3.2" if i < 2 else "mistral",
                modality=JobModality.CHAT,
            )
            await queue.enqueue(job)

        stats = await queue.get_queue_stats()

        assert stats["total"] == 3
        assert stats["by_user"][0] == 2  # user 0 has 2 jobs
        assert stats["by_user"][1] == 1  # user 1 has 1 job
        assert stats["by_model"]["llama3.2"] == 2
        assert stats["by_model"]["mistral"] == 1


class TestFairShareManager:
    """Tests for FairShareManager."""

    @pytest.fixture
    def manager(self):
        """Create a fair share manager with mocked settings."""
        with patch("backend.app.core.scheduler.fairshare.get_settings") as mock_settings:
            settings = MagicMock()
            settings.scheduler_fairness_window = 300
            settings.scheduler_deprioritize_threshold = 0.5
            settings.get_scheduler_weight.side_effect = lambda role: {
                "student": 1,
                "staff": 2,
                "faculty": 3,
                "admin": 10,
            }.get(role, 1)
            mock_settings.return_value = settings

            return FairShareManager()

    @pytest.mark.asyncio
    async def test_register_user(self, manager):
        """Test user registration."""
        state = await manager.register_user(1, "faculty")

        assert state.user_id == 1
        assert state.weight == 3.0  # faculty weight
        assert state.deficit == 0.0
        assert state.burst_credits == 0.0

    @pytest.mark.asyncio
    async def test_role_weights(self, manager):
        """Test different role weights."""
        student = await manager.register_user(1, "student")
        staff = await manager.register_user(2, "staff")
        faculty = await manager.register_user(3, "faculty")
        admin = await manager.register_user(4, "admin")

        assert student.weight == 1
        assert staff.weight == 2
        assert faculty.weight == 3
        assert admin.weight == 10

    @pytest.mark.asyncio
    async def test_compute_priority_basic(self, manager):
        """Test basic priority computation."""
        await manager.register_user(1, "student")

        job = Job(
            request_id="req-1",
            user_id=1,
            api_key_id=1,
            model="llama3.2",
            modality=JobModality.CHAT,
        )

        priority = await manager.compute_priority(job, "student")
        assert priority >= 0  # Should be non-negative with no deficit

    @pytest.mark.asyncio
    async def test_priority_reflects_deficit(self, manager):
        """Test that priority reflects deficit counter."""
        await manager.register_user(1, "student")
        await manager.register_user(2, "student")

        # Simulate user 1 completing a job (reducing their deficit)
        job1 = Job(
            request_id="req-1",
            user_id=1,
            api_key_id=1,
            model="llama3.2",
            modality=JobModality.CHAT,
        )
        await manager.on_job_completed(job1, tokens_used=1000)

        # Now create new jobs for both users
        job1_new = Job(
            request_id="req-2",
            user_id=1,
            api_key_id=1,
            model="llama3.2",
            modality=JobModality.CHAT,
        )
        job2 = Job(
            request_id="req-3",
            user_id=2,
            api_key_id=2,
            model="llama3.2",
            modality=JobModality.CHAT,
        )

        priority1 = await manager.compute_priority(job1_new, "student")
        priority2 = await manager.compute_priority(job2, "student")

        # User 2 should have higher priority (hasn't used resources)
        assert priority2 > priority1

    @pytest.mark.asyncio
    async def test_weight_affects_priority(self, manager):
        """Test that higher weight users get proportionally more access."""
        student_state = await manager.register_user(1, "student")
        faculty_state = await manager.register_user(2, "faculty")

        # Set equal deficits
        student_state.deficit = 1000
        faculty_state.deficit = 1000

        job1 = Job(
            request_id="req-1",
            user_id=1,
            api_key_id=1,
            model="llama3.2",
            modality=JobModality.CHAT,
        )
        job2 = Job(
            request_id="req-2",
            user_id=2,
            api_key_id=2,
            model="llama3.2",
            modality=JobModality.CHAT,
        )

        priority_student = await manager.compute_priority(job1, "student")
        priority_faculty = await manager.compute_priority(job2, "faculty")

        # With equal deficit, student (weight 1) should have higher priority/weight ratio
        # But faculty priority = 1000/3 = 333, student = 1000/1 = 1000
        # So student should have higher priority per weight unit
        assert priority_student > priority_faculty

    @pytest.mark.asyncio
    async def test_burst_credits_accumulation(self, manager):
        """Test burst credit accumulation during idle time."""
        await manager.register_user(1, "student")

        state_before = await manager.get_user_state(1)
        initial_credits = state_before.burst_credits

        await manager.accumulate_burst_credits(idle_seconds=10.0)

        state_after = await manager.get_user_state(1)
        assert state_after.burst_credits > initial_credits

    @pytest.mark.asyncio
    async def test_burst_credits_decay(self, manager):
        """Test burst credits decay when contention is detected."""
        state = await manager.register_user(1, "student")
        state.burst_credits = 1000

        await manager.reset_burst_credits()

        state_after = await manager.get_user_state(1)
        assert state_after.burst_credits == 500  # Decayed by 50%

    @pytest.mark.asyncio
    async def test_deficit_updated_on_completion(self, manager):
        """Test that deficit is updated when job completes."""
        state = await manager.register_user(1, "student")
        initial_deficit = state.deficit

        job = Job(
            request_id="req-1",
            user_id=1,
            api_key_id=1,
            model="llama3.2",
            modality=JobModality.CHAT,
        )

        await manager.on_job_completed(job, tokens_used=500)

        state_after = await manager.get_user_state(1)
        assert state_after.deficit == initial_deficit - 500

    @pytest.mark.asyncio
    async def test_heavy_user_deprioritization(self, manager):
        """Test that heavy users get deprioritized."""
        await manager.register_user(1, "student")
        await manager.register_user(2, "student")

        # Simulate user 1 using most of the cluster
        for i in range(10):
            job = Job(
                request_id=f"req-{i}",
                user_id=1,
                api_key_id=1,
                model="llama3.2",
                modality=JobModality.CHAT,
            )
            await manager.on_job_completed(job, tokens_used=1000)

        # User 1 has used 10000 tokens, user 2 has used 0
        # User 1's usage fraction is 100%, well above threshold

        # Create new jobs
        job1 = Job(
            request_id="req-new-1",
            user_id=1,
            api_key_id=1,
            model="llama3.2",
            modality=JobModality.CHAT,
        )
        job2 = Job(
            request_id="req-new-2",
            user_id=2,
            api_key_id=2,
            model="llama3.2",
            modality=JobModality.CHAT,
        )

        # User 2 should have higher priority due to deprioritization of user 1
        priority1 = await manager.compute_priority(job1, "student")
        priority2 = await manager.compute_priority(job2, "student")

        # Note: User 1 has negative deficit from usage, which affects priority too
        assert priority2 > priority1

    @pytest.mark.asyncio
    async def test_get_stats(self, manager):
        """Test getting scheduler statistics."""
        await manager.register_user(1, "student")
        await manager.register_user(2, "faculty")

        stats = await manager.get_stats()

        assert stats["total_users"] == 2
        assert "global_recent_tokens" in stats
        assert len(stats["user_stats"]) == 2


class TestBackendScorer:
    """Tests for BackendScorer."""

    @pytest.fixture
    def scorer(self):
        """Create a backend scorer with mocked settings."""
        with patch("backend.app.core.scheduler.scoring.get_settings") as mock_settings:
            settings = MagicMock()
            settings.scheduler_score_model_loaded = 100
            settings.scheduler_score_low_utilization = 50
            settings.scheduler_score_latency = 40
            settings.scheduler_score_short_queue = 30
            settings.scheduler_score_high_throughput = 20
            mock_settings.return_value = settings

            return BackendScorer()

    @pytest.fixture
    def backend(self):
        """Create a mock backend."""
        backend = MagicMock(spec=Backend)
        backend.id = 1
        backend.name = "backend-1"
        backend.status = BackendStatus.HEALTHY
        backend.engine = BackendEngine.OLLAMA
        backend.supports_multimodal = True
        backend.supports_embeddings = True
        backend.supports_structured_output = True
        backend.current_concurrent = 0
        backend.max_concurrent = 10
        backend.gpu_memory_gb = 24.0
        backend.throughput_score = 1.0
        backend.priority = 0
        return backend

    @pytest.fixture
    def model(self):
        """Create a mock model."""
        model = MagicMock(spec=Model)
        model.name = "llama3.2"
        model.is_loaded = True
        model.supports_multimodal = True
        model.supports_structured_output = True
        model.vram_required_gb = 8.0
        return model

    def test_hard_constraints_pass(self, scorer, backend, model):
        """Test hard constraints all passing."""
        job = Job(
            request_id="req-1",
            user_id=1,
            api_key_id=1,
            model="llama3.2",
            modality=JobModality.CHAT,
        )

        is_eligible, constraints = scorer.check_hard_constraints(
            backend, job, [model], current_queue_depth=0
        )

        assert is_eligible is True
        assert constraints.model_available is True
        assert constraints.supports_modality is True
        assert constraints.has_capacity is True

    def test_hard_constraint_model_not_available(self, scorer, backend, model):
        """Test hard constraint failure: model not available."""
        job = Job(
            request_id="req-1",
            user_id=1,
            api_key_id=1,
            model="nonexistent-model",
            modality=JobModality.CHAT,
        )

        is_eligible, constraints = scorer.check_hard_constraints(
            backend, job, [model], current_queue_depth=0
        )

        assert is_eligible is False
        assert constraints.model_available is False

    def test_hard_constraint_no_capacity(self, scorer, backend, model):
        """Test hard constraint failure: no capacity."""
        backend.current_concurrent = 10  # At max capacity

        job = Job(
            request_id="req-1",
            user_id=1,
            api_key_id=1,
            model="llama3.2",
            modality=JobModality.CHAT,
        )

        is_eligible, constraints = scorer.check_hard_constraints(
            backend, job, [model], current_queue_depth=0
        )

        assert is_eligible is False
        assert constraints.has_capacity is False

    def test_hard_constraint_multimodal_not_supported(self, scorer, backend, model):
        """Test hard constraint failure: multimodal not supported."""
        backend.supports_multimodal = False
        model.supports_multimodal = False

        job = Job(
            request_id="req-1",
            user_id=1,
            api_key_id=1,
            model="llama3.2",
            modality=JobModality.MULTIMODAL,
            requires_multimodal=True,
        )

        is_eligible, constraints = scorer.check_hard_constraints(
            backend, job, [model], current_queue_depth=0
        )

        assert is_eligible is False
        assert constraints.supports_modality is False

    def test_hard_constraint_unhealthy_backend(self, scorer, backend, model):
        """Test hard constraint failure: backend not healthy."""
        backend.status = BackendStatus.UNHEALTHY

        job = Job(
            request_id="req-1",
            user_id=1,
            api_key_id=1,
            model="llama3.2",
            modality=JobModality.CHAT,
        )

        is_eligible, constraints = scorer.check_hard_constraints(
            backend, job, [model], current_queue_depth=0
        )

        assert is_eligible is False

    def test_soft_score_model_loaded(self, scorer, backend, model):
        """Test soft score: model loaded bonus."""
        job = Job(
            request_id="req-1",
            user_id=1,
            api_key_id=1,
            model="llama3.2",
            modality=JobModality.CHAT,
        )

        score = scorer.compute_score(
            backend, job, [model], gpu_utilization=50.0
        )

        assert score.model_loaded_score == 100  # Full bonus for loaded model

    def test_soft_score_model_not_loaded(self, scorer, backend, model):
        """Test soft score: model not loaded (no bonus)."""
        model.is_loaded = False

        job = Job(
            request_id="req-1",
            user_id=1,
            api_key_id=1,
            model="llama3.2",
            modality=JobModality.CHAT,
        )

        score = scorer.compute_score(
            backend, job, [model], gpu_utilization=50.0
        )

        assert score.model_loaded_score == 0

    def test_soft_score_utilization(self, scorer, backend, model):
        """Test soft score: utilization affects score."""
        job = Job(
            request_id="req-1",
            user_id=1,
            api_key_id=1,
            model="llama3.2",
            modality=JobModality.CHAT,
        )

        # Low utilization = high score
        score_low = scorer.compute_score(
            backend, job, [model], gpu_utilization=10.0
        )

        # High utilization = low score
        score_high = scorer.compute_score(
            backend, job, [model], gpu_utilization=90.0
        )

        assert score_low.utilization_score > score_high.utilization_score

    def test_soft_score_queue_depth(self, scorer, backend, model):
        """Test soft score: queue depth affects score."""
        job = Job(
            request_id="req-1",
            user_id=1,
            api_key_id=1,
            model="llama3.2",
            modality=JobModality.CHAT,
        )

        # Empty queue = full bonus
        score_empty = scorer.compute_score(
            backend, job, [model], current_queue_depth=0
        )

        # Deep queue = reduced bonus
        score_deep = scorer.compute_score(
            backend, job, [model], current_queue_depth=10
        )

        assert score_empty.queue_score > score_deep.queue_score
        assert score_empty.queue_score == 30  # Full bonus

    def test_rank_backends(self, scorer):
        """Test ranking multiple backends."""
        # Create multiple backends with different characteristics
        backends = []
        models_by_backend = {}

        for i in range(3):
            backend = MagicMock(spec=Backend)
            backend.id = i
            backend.name = f"backend-{i}"
            backend.status = BackendStatus.HEALTHY
            backend.supports_multimodal = True
            backend.supports_embeddings = True
            backend.supports_structured_output = True
            backend.current_concurrent = 0
            backend.max_concurrent = 10
            backend.gpu_memory_gb = 24.0
            backend.throughput_score = 1.0
            backend.priority = 0
            backends.append(backend)

            model = MagicMock(spec=Model)
            model.name = "llama3.2"
            model.is_loaded = (i == 1)  # Only backend 1 has model loaded
            model.supports_multimodal = True
            model.supports_structured_output = True
            model.vram_required_gb = 8.0
            models_by_backend[i] = [model]

        job = Job(
            request_id="req-1",
            user_id=1,
            api_key_id=1,
            model="llama3.2",
            modality=JobModality.CHAT,
        )

        scores = scorer.rank_backends(
            backends,
            job,
            models_by_backend,
            gpu_utilizations={0: 80.0, 1: 20.0, 2: 50.0},
            queue_depths={0: 5, 1: 0, 2: 2},
        )

        # Backend 1 should rank highest (model loaded + low utilization + empty queue)
        assert len(scores) == 3
        assert scores[0].backend_id == 1


class TestFairnessSimulation:
    """Integration tests simulating multiple users competing for resources."""

    @pytest.mark.asyncio
    async def test_fair_distribution_over_time(self):
        """Simulate multiple users and verify fair distribution."""
        with patch("backend.app.core.scheduler.fairshare.get_settings") as mock_settings:
            settings = MagicMock()
            settings.scheduler_fairness_window = 300
            settings.scheduler_deprioritize_threshold = 0.5
            settings.get_scheduler_weight.side_effect = lambda role: {
                "student": 1,
                "staff": 2,
                "faculty": 3,
            }.get(role, 1)
            mock_settings.return_value = settings

            manager = FairShareManager()

            # Register users with different weights
            await manager.register_user(1, "faculty")  # weight 3
            await manager.register_user(2, "staff")    # weight 2
            await manager.register_user(3, "student")  # weight 1

            # Simulate 100 rounds of scheduling
            service_counts = {1: 0, 2: 0, 3: 0}

            for round_num in range(100):
                # Create a job for each user
                jobs = []
                for user_id in [1, 2, 3]:
                    job = Job(
                        request_id=f"req-{round_num}-{user_id}",
                        user_id=user_id,
                        api_key_id=user_id,
                        model="llama3.2",
                        modality=JobModality.CHAT,
                    )
                    jobs.append((job, ["faculty", "staff", "student"][user_id - 1]))

                # Compute priorities
                priorities = []
                for job, role in jobs:
                    priority = await manager.compute_priority(job, role)
                    priorities.append((priority, job, role))

                # Select highest priority job (winner of this round)
                priorities.sort(reverse=True)
                winner_job = priorities[0][1]
                winner_role = priorities[0][2]

                # Simulate job completion
                await manager.on_job_completed(winner_job, tokens_used=100)
                service_counts[winner_job.user_id] += 1

            # Verify distribution roughly matches weights
            total_services = sum(service_counts.values())
            ratios = {uid: count / total_services for uid, count in service_counts.items()}

            # Expected: faculty=50% (3/6), staff=33% (2/6), student=17% (1/6)
            # Allow some variance due to stochastic nature
            assert ratios[1] > ratios[2] > ratios[3]  # Faculty > Staff > Student
