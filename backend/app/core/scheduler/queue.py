############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# queue.py: Request queue management with priority ordering
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Request queue management for the scheduler."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from collections import defaultdict
import heapq


class JobModality(str, Enum):
    """Request modality types."""
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    MULTIMODAL = "multimodal"
    RERANKING = "reranking"


@dataclass
class Job:
    """Represents a single inference request in the queue."""

    # Identifiers
    request_id: str
    user_id: int
    api_key_id: int

    # Request details
    model: str
    modality: JobModality
    is_streaming: bool = False
    requires_multimodal: bool = False
    requires_structured_output: bool = False

    # Size estimates
    estimated_prompt_tokens: int = 0
    estimated_completion_tokens: int = 0
    image_bytes: int = 0

    # Timing
    time_submitted: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    time_started: Optional[datetime] = None
    time_completed: Optional[datetime] = None

    # Scheduling state
    priority: float = 0.0  # Computed by fair-share algorithm
    assigned_backend_id: Optional[int] = None

    # Original request data
    request_data: Optional[Dict[str, Any]] = None

    def __lt__(self, other: "Job") -> bool:
        """Compare jobs for priority queue ordering (higher priority first)."""
        return self.priority > other.priority

    def get_estimated_cost(self) -> int:
        """Get estimated token cost for this job."""
        return self.estimated_prompt_tokens + self.estimated_completion_tokens

    def get_queue_time_seconds(self) -> float:
        """Get time spent waiting in queue."""
        if self.time_started:
            return (self.time_started - self.time_submitted).total_seconds()
        return (datetime.now(timezone.utc) - self.time_submitted).total_seconds()


class RequestQueue:
    """
    Manages the global request queue with per-user sub-queues.

    Provides efficient access patterns for:
    - Global queue ordering (by priority)
    - Per-user queue access
    - Per-model queue access (optional)
    """

    def __init__(self):
        self._global_queue: List[Job] = []  # Min-heap (but we use negative priority)
        self._user_queues: Dict[int, List[Job]] = defaultdict(list)
        self._model_queues: Dict[str, List[Job]] = defaultdict(list)
        self._job_map: Dict[str, Job] = {}  # request_id -> Job
        self._lock = asyncio.Lock()

    async def enqueue(self, job: Job) -> int:
        """
        Add a job to the queue.

        Args:
            job: The job to enqueue

        Returns:
            Queue position (0-indexed)
        """
        async with self._lock:
            # Add to global queue
            heapq.heappush(self._global_queue, job)

            # Add to user queue
            self._user_queues[job.user_id].append(job)

            # Add to model queue
            self._model_queues[job.model].append(job)

            # Add to lookup map
            self._job_map[job.request_id] = job

            return len(self._global_queue) - 1

    async def dequeue(self) -> Optional[Job]:
        """
        Remove and return the highest priority job.

        Returns:
            The job with highest priority, or None if queue is empty
        """
        async with self._lock:
            while self._global_queue:
                job = heapq.heappop(self._global_queue)

                # Verify job still exists (might have been cancelled)
                if job.request_id in self._job_map:
                    # Remove from user queue
                    user_queue = self._user_queues[job.user_id]
                    if job in user_queue:
                        user_queue.remove(job)

                    # Remove from model queue
                    model_queue = self._model_queues[job.model]
                    if job in model_queue:
                        model_queue.remove(job)

                    # Remove from lookup map
                    del self._job_map[job.request_id]

                    return job

            return None

    async def peek(self) -> Optional[Job]:
        """
        Return the highest priority job without removing it.

        Returns:
            The job with highest priority, or None if queue is empty
        """
        async with self._lock:
            for job in self._global_queue:
                if job.request_id in self._job_map:
                    return job
            return None

    async def get_job(self, request_id: str) -> Optional[Job]:
        """Get a job by request ID."""
        async with self._lock:
            return self._job_map.get(request_id)

    async def cancel_job(self, request_id: str) -> bool:
        """
        Cancel a job by request ID.

        Args:
            request_id: The request ID to cancel

        Returns:
            True if job was found and cancelled
        """
        async with self._lock:
            job = self._job_map.get(request_id)
            if not job:
                return False

            # Remove from user queue
            user_queue = self._user_queues[job.user_id]
            if job in user_queue:
                user_queue.remove(job)

            # Remove from model queue
            model_queue = self._model_queues[job.model]
            if job in model_queue:
                model_queue.remove(job)

            # Remove from lookup map
            del self._job_map[request_id]

            # Note: We don't remove from global_queue heap immediately
            # It will be skipped during dequeue

            return True

    async def update_priority(self, request_id: str, priority: float) -> bool:
        """
        Update priority for a job.

        Args:
            request_id: The request ID to update
            priority: New priority value

        Returns:
            True if job was found and updated
        """
        async with self._lock:
            job = self._job_map.get(request_id)
            if not job:
                return False

            job.priority = priority

            # Rebuild heap to maintain ordering
            heapq.heapify(self._global_queue)

            return True

    async def get_user_queue(self, user_id: int) -> List[Job]:
        """Get all jobs for a specific user."""
        async with self._lock:
            return list(self._user_queues.get(user_id, []))

    async def get_model_queue(self, model: str) -> List[Job]:
        """Get all jobs for a specific model."""
        async with self._lock:
            return list(self._model_queues.get(model, []))

    async def get_user_pending_count(self, user_id: int) -> int:
        """Get count of pending jobs for a user."""
        async with self._lock:
            return len(self._user_queues.get(user_id, []))

    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        async with self._lock:
            total = len(self._job_map)
            by_user = {uid: len(jobs) for uid, jobs in self._user_queues.items() if jobs}
            by_model = {model: len(jobs) for model, jobs in self._model_queues.items() if jobs}

            # Calculate average wait time
            avg_wait = 0.0
            if total > 0:
                now = datetime.now(timezone.utc)
                total_wait = sum(
                    (now - job.time_submitted).total_seconds()
                    for job in self._job_map.values()
                )
                avg_wait = total_wait / total

            return {
                "total": total,
                "by_user": by_user,
                "by_model": by_model,
                "average_wait_seconds": avg_wait,
            }

    async def get_all_jobs(self) -> List[Job]:
        """Get all jobs in the queue."""
        async with self._lock:
            return list(self._job_map.values())

    def __len__(self) -> int:
        """Get total number of jobs in queue."""
        return len(self._job_map)
