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

    # Stale job warning threshold (seconds) — jobs older than this are
    # flagged in queue health even if not yet eligible for GC eviction.
    STALE_WARNING_THRESHOLD_S = 120.0

    # Number of depth samples to retain for trend computation (30s each).
    DEPTH_HISTORY_SIZE = 10

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

        # Queue health monitoring
        self._queue_depth_history: List[Tuple[datetime, int]] = []
        self._gc_last_run: Optional[datetime] = None
        self._gc_last_evicted: int = 0

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

        # Increment queue depth for selected backend and record the
        # assignment on the job so the GC can attribute it correctly.
        job.assigned_backend_id = selected_backend.id
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
        health = await self.get_queue_health()

        async with self._lock:
            backend_queues = dict(self._backend_queue_depths)

        return {
            "queue": queue_stats,
            "fair_share": fair_share_stats,
            "backend_queues": backend_queues,
            "health": health,
        }

    async def get_queue_health(self) -> Dict:
        """Assess queue health for monitoring.

        Returns a dict with status, trend, stale job count, and GC metadata.
        """
        all_jobs = await self.queue.get_all_jobs()
        now = datetime.now(timezone.utc)
        queue_total = len(all_jobs)

        # Stale job analysis
        stale_jobs = 0
        oldest_age_s: Optional[float] = None
        for job in all_jobs:
            age_s = (now - job.time_submitted).total_seconds()
            if oldest_age_s is None or age_s > oldest_age_s:
                oldest_age_s = age_s
            if age_s > self.STALE_WARNING_THRESHOLD_S:
                stale_jobs += 1

        # Trend from depth history
        trend = "stable"
        trend_rate = 0.0  # jobs/min
        history = list(self._queue_depth_history)
        if len(history) >= 2:
            first_time, first_depth = history[0]
            last_time, last_depth = history[-1]
            elapsed_min = (last_time - first_time).total_seconds() / 60.0
            if elapsed_min > 0:
                trend_rate = round((last_depth - first_depth) / elapsed_min, 2)
                if trend_rate > 0.5:
                    trend = "growing"
                elif trend_rate < -0.5:
                    trend = "draining"

        # Growing duration — how long has trend been positive?
        growing_duration_s = 0.0
        if trend == "growing" and len(history) >= 2:
            # Walk backwards to find when growth started
            for i in range(len(history) - 1, 0, -1):
                if history[i][1] <= history[i - 1][1]:
                    break
                growing_duration_s = (history[-1][0] - history[i - 1][0]).total_seconds()

        # Backend depths
        async with self._lock:
            backend_depths = dict(self._backend_queue_depths)

        # Health status
        if queue_total == 0:
            health_status = "healthy"
        elif queue_total > 0 and stale_jobs == queue_total:
            health_status = "critical"
        elif growing_duration_s >= 300:
            health_status = "critical"
        elif stale_jobs > 0 or trend == "growing":
            health_status = "warning"
        else:
            health_status = "healthy"

        return {
            "status": health_status,
            "queue_total": queue_total,
            "trend": trend,
            "trend_rate": trend_rate,
            "stale_jobs": stale_jobs,
            "oldest_job_age_seconds": round(oldest_age_s, 1) if oldest_age_s is not None else None,
            "backend_depths": backend_depths,
            "gc_last_run": self._gc_last_run.isoformat() if self._gc_last_run else None,
            "gc_last_evicted": self._gc_last_evicted,
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

                # Record queue depth sample for trend analysis
                queue_stats = await self.queue.get_queue_stats()
                now = datetime.now(timezone.utc)
                self._queue_depth_history.append((now, queue_stats["total"]))
                if len(self._queue_depth_history) > self.DEPTH_HISTORY_SIZE:
                    self._queue_depth_history = self._queue_depth_history[-self.DEPTH_HISTORY_SIZE:]

                # Check for idle cluster and accumulate burst credits
                if queue_stats["total"] == 0:
                    await self.fair_share.accumulate_burst_credits(30)
                else:
                    # Contention detected, decay burst credits
                    await self.fair_share.reset_burst_credits()

                # GC sweep every 60 seconds: evict stale jobs and reconcile
                # backend queue depth counters to prevent phantom queue buildup.
                if gc_counter % 2 == 0:
                    await self._gc_stale_jobs()

                # DB orphan cleanup every 5 minutes: mark queued requests in
                # the DB that aren't tracked by the in-memory scheduler as
                # failed. These accumulate when the app restarts or a request
                # falls through the cracks.
                if gc_counter % 10 == 0:
                    await self._cleanup_db_orphans()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("maintenance_error", error=str(e))

    async def _gc_stale_jobs(self) -> None:
        """Garbage-collect stale jobs and correct queue depth counters.

        Jobs older than the maximum possible request lifetime (routing timeout
        + per-attempt timeout * max attempts + margin) are orphans whose
        completion/failure callbacks were never invoked.  Evicting them and
        decrementing their backend queue depth prevents phantom queue buildup.
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

            self._gc_last_run = datetime.now(timezone.utc)

            all_jobs = await self.queue.get_all_jobs()
            now = self._gc_last_run
            evicted = 0

            for job in all_jobs:
                age_s = (now - job.time_submitted).total_seconds()
                if age_s > max_lifetime_s:
                    await self.queue.cancel_job(job.request_id)

                    # Decrement queue depth for the assigned backend.
                    # assigned_backend_id may be None if the job was queued
                    # but never routed — in that case the depth was never
                    # incremented so there is nothing to undo.
                    if job.assigned_backend_id is not None:
                        async with self._lock:
                            bid = job.assigned_backend_id
                            if bid in self._backend_queue_depths:
                                self._backend_queue_depths[bid] = max(
                                    0, self._backend_queue_depths[bid] - 1
                                )

                    evicted += 1

            # Safety net: if queue is empty but depth counters are non-zero,
            # they are entirely phantom.  Reset them.
            remaining_count = len(await self.queue.get_all_jobs())
            if remaining_count == 0:
                async with self._lock:
                    stale_total = sum(self._backend_queue_depths.values())
                    if stale_total > 0:
                        logger.warning(
                            "gc_queue_depth_reset",
                            stale_total=stale_total,
                        )
                        self._backend_queue_depths.clear()

            self._gc_last_evicted = evicted

            if evicted > 0:
                logger.warning(
                    "gc_stale_jobs_evicted",
                    evicted=evicted,
                    remaining=remaining_count,
                    max_lifetime_s=max_lifetime_s,
                )

                # Wake any waiters that may have been blocked by phantom depth
                async with self._capacity_condition:
                    self._capacity_condition.notify_all()

        except Exception as e:
            logger.error("gc_stale_jobs_error", error=str(e))

    async def _cleanup_db_orphans(self) -> None:
        """Mark DB requests stuck in 'queued' that aren't in the scheduler.

        Requests can become orphaned when the app restarts (losing the
        in-memory queue) or when a routing path fails without updating
        the DB status.  This runs periodically to catch them.
        """
        try:
            from datetime import timedelta

            from sqlalchemy import select, update

            from backend.app.db.models import Request, RequestStatus
            from backend.app.db.session import get_async_db_context

            # Only clean up requests older than 5 minutes to avoid racing
            # with requests that are still being routed
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)

            # Get request IDs that the in-memory scheduler knows about
            all_jobs = await self.queue.get_all_jobs()
            known_request_ids = {job.request_id for job in all_jobs}

            async with get_async_db_context() as db:
                # Find DB requests in queued status older than cutoff
                stale_rows = (
                    await db.execute(
                        select(Request.id, Request.request_uuid)
                        .where(Request.status == RequestStatus.QUEUED)
                        .where(Request.created_at < cutoff)
                    )
                ).fetchall()

                if not stale_rows:
                    return

                # Filter to only those NOT tracked by the scheduler
                orphan_ids = [
                    r.id for r in stale_rows
                    if r.request_uuid not in known_request_ids
                ]

                if not orphan_ids:
                    return

                result = await db.execute(
                    update(Request)
                    .where(Request.id.in_(orphan_ids))
                    .values(
                        status=RequestStatus.FAILED,
                        error_message="Orphaned: queued request not tracked by scheduler",
                    )
                )
                await db.commit()

                if result.rowcount > 0:
                    logger.warning(
                        "db_orphan_requests_cleaned",
                        count=result.rowcount,
                    )

        except Exception as e:
            logger.error("db_orphan_cleanup_error", error=str(e))
