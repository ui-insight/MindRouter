############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# routing.py: Backend routing combining queue, fair-share, and scoring
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Backend routing - ties together queue, fair-share, and scoring."""

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

from backend.app.core.scheduler.queue import Job, RequestQueue
from backend.app.core.scheduler.fairshare import FairShareManager
from backend.app.core.scheduler.scoring import BackendScorer, BackendScore
from backend.app.db.models import Backend, Model
from backend.app.logging_config import get_logger

logger = get_logger(__name__)


class RoutingDecision:
    """Result of a routing decision."""

    def __init__(
        self,
        backend: Optional[Backend],
        score: Optional[BackendScore],
        reason: str,
        all_scores: Optional[List[BackendScore]] = None,
    ):
        self.backend = backend
        self.score = score
        self.reason = reason
        self.all_scores = all_scores or []

    @property
    def success(self) -> bool:
        """Check if routing was successful."""
        return self.backend is not None


class BackendRouter:
    """
    Routes jobs to backends using fair-share scheduling and backend scoring.

    Responsibilities:
    - Manages the request queue
    - Computes job priorities using fair-share
    - Selects best backend using scoring
    - Tracks active requests per backend
    """

    def __init__(self):
        self.queue = RequestQueue()
        self.fair_share = FairShareManager()
        self.scorer = BackendScorer()

        # Track active requests per backend
        self._backend_queue_depths: Dict[int, int] = {}
        self._lock = asyncio.Lock()

        # Condition notified when a job completes/fails, so waiters can retry routing
        self._capacity_condition = asyncio.Condition()

        # Track jobs waiting for capacity, keyed by model
        self._waiting_jobs: Dict[str, Dict[str, Job]] = defaultdict(dict)

        # Background task handle
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start background maintenance tasks."""
        self._cleanup_task = asyncio.create_task(self._maintenance_loop())
        logger.info("Backend router started")

    async def stop(self) -> None:
        """Stop background tasks."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Backend router stopped")

    async def submit_job(
        self,
        job: Job,
        user_role: str = "",
        user_weight: float = 1.0,
    ) -> int:
        """
        Submit a job to the queue.

        Args:
            job: The job to submit
            user_role: User's role (deprecated, kept for backward compat)
            user_weight: Direct weight from group.scheduler_weight

        Returns:
            Queue position
        """
        # Register user with fair-share manager
        await self.fair_share.register_user(job.user_id, role=user_role, weight=user_weight)

        # Compute initial priority
        priority = await self.fair_share.compute_priority(job, role=user_role, weight=user_weight)
        job.priority = priority

        # Notify fair-share of new job
        await self.fair_share.on_job_queued(job, role=user_role, weight=user_weight)

        # Enqueue
        position = await self.queue.enqueue(job)

        logger.info(
            "job_queued",
            request_id=job.request_id,
            user_id=job.user_id,
            model=job.model,
            priority=priority,
            position=position,
        )

        return position

    async def route_job(
        self,
        job: Job,
        backends: List[Backend],
        backend_models: Dict[int, List[Model]],
        gpu_utilizations: Dict[int, Optional[float]] = None,
        exclude_backend_ids: Optional[Set[int]] = None,
        latency_emas: Optional[Dict[int, float]] = None,
    ) -> RoutingDecision:
        """
        Route a job to the best available backend.

        Args:
            job: The job to route
            backends: Available backends
            backend_models: Models per backend
            gpu_utilizations: GPU utilization per backend
            exclude_backend_ids: Backend IDs to exclude (already tried in retry loop)
            latency_emas: Per-backend latency EMA for scoring

        Returns:
            RoutingDecision with selected backend or failure reason
        """
        # Filter out excluded backends (retry failover)
        if exclude_backend_ids:
            backends = [b for b in backends if b.id not in exclude_backend_ids]

        async with self._lock:
            # Get current queue depths
            queue_depths = dict(self._backend_queue_depths)

        # Score and rank backends
        scores = self.scorer.rank_backends(
            backends=backends,
            job=job,
            backend_models=backend_models,
            gpu_utilizations=gpu_utilizations,
            queue_depths=queue_depths,
            latency_emas=latency_emas,
        )

        if not scores:
            return RoutingDecision(
                backend=None,
                score=None,
                reason="No backends available for model",
            )

        # Check if best backend has negative score (failed constraints)
        best_score = scores[0]
        if best_score.total_score < 0:
            failed = ", ".join(best_score.failed_constraints)
            return RoutingDecision(
                backend=None,
                score=best_score,
                reason=f"All backends failed constraints: {failed}",
                all_scores=scores,
            )

        # Find the backend object
        selected_backend = None
        for backend in backends:
            if backend.id == best_score.backend_id:
                selected_backend = backend
                break

        if not selected_backend:
            return RoutingDecision(
                backend=None,
                score=best_score,
                reason="Backend not found after scoring",
                all_scores=scores,
            )

        # Increment queue depth for selected backend
        async with self._lock:
            self._backend_queue_depths[selected_backend.id] = (
                self._backend_queue_depths.get(selected_backend.id, 0) + 1
            )

        logger.info(
            "job_routed",
            request_id=job.request_id,
            backend_id=selected_backend.id,
            backend_name=selected_backend.name,
            score=best_score.total_score,
        )

        return RoutingDecision(
            backend=selected_backend,
            score=best_score,
            reason="success",
            all_scores=scores,
        )

    async def on_job_started(self, job: Job, backend_id: int) -> None:
        """
        Called when a job starts processing on a backend.

        Args:
            job: The started job
            backend_id: Backend processing the job
        """
        job.assigned_backend_id = backend_id
        await self.fair_share.on_job_started(job)

        logger.debug(
            "job_started",
            request_id=job.request_id,
            backend_id=backend_id,
        )

    async def on_job_completed(
        self,
        job: Job,
        backend_id: int,
        tokens_used: int,
    ) -> None:
        """
        Called when a job completes.

        Args:
            job: The completed job
            backend_id: Backend that processed the job
            tokens_used: Total tokens consumed
        """
        # Update fair-share
        await self.fair_share.on_job_completed(job, tokens_used)

        # Remove job from queue and decrement queue depth
        await self.queue.cancel_job(job.request_id)
        async with self._lock:
            if backend_id in self._backend_queue_depths:
                self._backend_queue_depths[backend_id] = max(
                    0, self._backend_queue_depths[backend_id] - 1
                )
                depth = self._backend_queue_depths[backend_id]
            else:
                depth = 0

        # Auto-complete drain when queue depth hits 0
        if depth == 0:
            await self._try_complete_drain(backend_id)

        # Wake any requests waiting for capacity
        async with self._capacity_condition:
            self._capacity_condition.notify_all()

        logger.info(
            "job_completed",
            request_id=job.request_id,
            backend_id=backend_id,
            tokens_used=tokens_used,
        )

    async def on_job_failed(self, job: Job, backend_id: int) -> None:
        """
        Called when a job fails.

        Args:
            job: The failed job
            backend_id: Backend that attempted the job
        """
        # Remove job from queue and decrement queue depth
        await self.queue.cancel_job(job.request_id)
        async with self._lock:
            if backend_id in self._backend_queue_depths:
                self._backend_queue_depths[backend_id] = max(
                    0, self._backend_queue_depths[backend_id] - 1
                )
                depth = self._backend_queue_depths[backend_id]
            else:
                depth = 0

        # Auto-complete drain when queue depth hits 0
        if depth == 0:
            await self._try_complete_drain(backend_id)

        # Wake any requests waiting for capacity
        async with self._capacity_condition:
            self._capacity_condition.notify_all()

        logger.warning(
            "job_failed",
            request_id=job.request_id,
            backend_id=backend_id,
        )

    async def _try_complete_drain(self, backend_id: int) -> None:
        """If the backend is draining and its queue is empty, transition to DISABLED."""
        try:
            from backend.app.core.telemetry.registry import get_registry
            registry = get_registry()
            await registry.complete_drain(backend_id)
        except Exception as e:
            logger.debug("drain_complete_check_error", backend_id=backend_id, error=str(e))

    async def wait_for_capacity(self, timeout: float = 5.0) -> bool:
        """
        Wait until backend capacity becomes available.

        Returns True if signaled, False on timeout.
        """
        try:
            async with self._capacity_condition:
                await asyncio.wait_for(
                    self._capacity_condition.wait(), timeout=timeout
                )
            return True
        except asyncio.TimeoutError:
            return False

    def register_waiter(self, job: Job) -> None:
        """Register a job as waiting for capacity on its model."""
        self._waiting_jobs[job.model][job.request_id] = job

    def unregister_waiter(self, job: Job) -> None:
        """Remove a job from the waiter set."""
        waiters = self._waiting_jobs.get(job.model)
        if waiters:
            waiters.pop(job.request_id, None)
            if not waiters:
                del self._waiting_jobs[job.model]

    def is_highest_priority_waiter(self, job: Job) -> bool:
        """Return True if job has the highest priority among waiters for its model.

        Higher numeric priority = higher priority.  Ties return True so
        equal-priority jobs proceed on a first-come-first-served basis.
        """
        waiters = self._waiting_jobs.get(job.model)
        if not waiters:
            return True
        max_priority = max(w.priority for w in waiters.values())
        return job.priority >= max_priority

    async def cancel_job(self, request_id: str) -> bool:
        """Cancel a queued job."""
        return await self.queue.cancel_job(request_id)

    async def get_queue_stats(self) -> Dict:
        """Get queue and scheduling statistics."""
        queue_stats = await self.queue.get_queue_stats()
        fair_share_stats = await self.fair_share.get_stats()

        async with self._lock:
            backend_queues = dict(self._backend_queue_depths)

        return {
            "queue": queue_stats,
            "fair_share": fair_share_stats,
            "backend_queues": backend_queues,
        }

    async def recompute_priorities(self) -> None:
        """Recompute priorities for all queued jobs."""
        jobs = await self.queue.get_all_jobs()

        for job in jobs:
            # We'd need user role here - for now use weight from state
            state = await self.fair_share.get_user_state(job.user_id)
            if state:
                # Approximate role from weight
                role = "student"  # Default
                priority = await self.fair_share.compute_priority(job, role)
                await self.queue.update_priority(job.request_id, priority)

    async def _maintenance_loop(self) -> None:
        """Background maintenance tasks."""
        gc_counter = 0
        while True:
            try:
                await asyncio.sleep(30)  # Run every 30 seconds
                gc_counter += 1

                # Clean up old usage history
                await self.fair_share.cleanup_old_history()

                # Rebalance deficits to prevent overflow
                await self.fair_share.rebalance_deficits()

                # Recompute priorities for fairness
                await self.recompute_priorities()

                # Check for idle cluster and accumulate burst credits
                queue_stats = await self.queue.get_queue_stats()
                if queue_stats["total"] == 0:
                    await self.fair_share.accumulate_burst_credits(30)
                else:
                    # Contention detected, decay burst credits
                    await self.fair_share.reset_burst_credits()

                # GC sweep every 60 seconds: evict stale jobs and reconcile
                # backend queue depth counters to prevent phantom queue buildup.
                if gc_counter % 2 == 0:
                    await self._gc_stale_jobs()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("maintenance_error", error=str(e))

    async def _gc_stale_jobs(self) -> None:
        """Garbage-collect stale jobs and reconcile queue depth counters.

        Jobs older than the maximum possible request lifetime (routing timeout
        + per-attempt timeout * max attempts + margin) are orphans whose
        completion/failure callbacks were never invoked.  Evicting them and
        reconciling ``_backend_queue_depths`` prevents phantom queue buildup.
        """
        try:
            from backend.app.settings import get_settings
            settings = get_settings()

            # Maximum plausible lifetime: route wait + all retry attempts
            max_lifetime_s = (
                settings.backend_request_timeout  # route_timeout default
                + settings.backend_request_timeout_per_attempt
                  * settings.backend_retry_max_attempts
                + 60  # margin
            )

            all_jobs = await self.queue.get_all_jobs()
            now = datetime.now(timezone.utc)
            evicted = 0

            for job in all_jobs:
                age_s = (now - job.time_submitted).total_seconds()
                if age_s > max_lifetime_s:
                    await self.queue.cancel_job(job.request_id)
                    evicted += 1

            # Reconcile _backend_queue_depths: recompute from live jobs only.
            # Any counter that was incremented by route_job() for a now-evicted
            # job is stale.  Rebuild from the ground truth (remaining jobs).
            remaining_jobs = await self.queue.get_all_jobs()
            async with self._lock:
                new_depths: Dict[int, int] = {}
                for job in remaining_jobs:
                    if job.assigned_backend_id is not None:
                        new_depths[job.assigned_backend_id] = (
                            new_depths.get(job.assigned_backend_id, 0) + 1
                        )

                # Detect and log corrections
                old_total = sum(self._backend_queue_depths.values())
                new_total = sum(new_depths.values())
                if old_total != new_total:
                    logger.warning(
                        "gc_queue_depth_corrected",
                        old_total=old_total,
                        new_total=new_total,
                        evicted_jobs=evicted,
                    )

                self._backend_queue_depths = new_depths

            if evicted > 0:
                logger.warning(
                    "gc_stale_jobs_evicted",
                    evicted=evicted,
                    remaining=len(remaining_jobs),
                    max_lifetime_s=max_lifetime_s,
                )

                # Wake any waiters that may have been blocked by phantom depth
                async with self._capacity_condition:
                    self._capacity_condition.notify_all()

        except Exception as e:
            logger.error("gc_stale_jobs_error", error=str(e))
