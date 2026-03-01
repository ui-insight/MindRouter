############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# fairshare.py: Weighted Deficit Round Robin (WDRR) implementation
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Fair-share scheduling with Weighted Deficit Round Robin (WDRR)."""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from backend.app.core.scheduler.queue import Job
from backend.app.settings import get_settings


@dataclass
class UserState:
    """Tracks scheduling state for a user."""

    user_id: int
    weight: float = 1.0  # Role-based weight

    # Deficit counter for WDRR
    deficit: float = 0.0

    # Burst credits (accumulated when idle)
    burst_credits: float = 0.0
    max_burst_credits: float = 1000.0

    # Recent usage tracking for deprioritization
    recent_tokens: int = 0
    recent_requests: int = 0

    # Active tracking
    active_requests: int = 0
    last_request_time: Optional[datetime] = None


@dataclass
class UsageWindow:
    """Sliding window for usage tracking."""

    tokens: int = 0
    requests: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FairShareManager:
    """
    Implements Weighted Deficit Round Robin (WDRR) for fair scheduling.

    Key concepts:
    - Each user has a weight based on their role (faculty > staff > student)
    - Deficit counters track how much service debt each user has
    - Users with higher deficit (more owed) get priority
    - Burst credits allow users to consume freely when cluster is idle
    - Big users are deprioritized when they exceed threshold in recent window
    """

    def __init__(self):
        self._settings = get_settings()
        self._user_states: Dict[int, UserState] = {}
        self._usage_history: Dict[int, List[UsageWindow]] = defaultdict(list)
        self._lock = asyncio.Lock()

        # Configuration
        self._fairness_window = self._settings.scheduler_fairness_window
        self._deprioritize_threshold = self._settings.scheduler_deprioritize_threshold

        # Track global usage for deprioritization
        self._global_recent_tokens = 0
        self._last_cleanup = datetime.now(timezone.utc)

    def get_role_weight(self, role: str) -> float:
        """Get the weight for a role."""
        return float(self._settings.get_scheduler_weight(role))

    async def register_user(
        self,
        user_id: int,
        role: str = "",
        weight_override: Optional[float] = None,
        weight: Optional[float] = None,
    ) -> UserState:
        """
        Register or update a user's scheduling state.

        Args:
            user_id: User ID
            role: User role (deprecated, kept for backward compat)
            weight_override: Optional weight override
            weight: Direct weight value (preferred over role-based lookup)

        Returns:
            UserState for the user
        """
        async with self._lock:
            resolved_weight = weight_override or weight or (self.get_role_weight(role) if role else 1.0)

            if user_id in self._user_states:
                state = self._user_states[user_id]
                state.weight = resolved_weight
            else:
                state = UserState(user_id=user_id, weight=resolved_weight)
                self._user_states[user_id] = state

            return state

    async def compute_priority(self, job: Job, role: str = "", weight: Optional[float] = None) -> float:
        """
        Compute priority for a job using WDRR algorithm.

        Priority formula:
        priority = (deficit + burst_credits) / weight * deprioritization_factor

        Args:
            job: The job to compute priority for
            role: User's role (deprecated)
            weight: Direct weight value (preferred)

        Returns:
            Priority score (higher = more priority)
        """
        async with self._lock:
            # Ensure user is registered
            if job.user_id not in self._user_states:
                resolved_weight = weight or (self.get_role_weight(role) if role else 1.0)
                self._user_states[job.user_id] = UserState(
                    user_id=job.user_id, weight=resolved_weight
                )

            state = self._user_states[job.user_id]

            # Base priority from deficit and weight
            base_priority = (state.deficit + state.burst_credits) / max(state.weight, 0.1)

            # Apply deprioritization for heavy users
            deprioritization_factor = await self._compute_deprioritization(job.user_id)

            # Add small time-based factor to prevent starvation
            wait_bonus = job.get_queue_time_seconds() * 0.1

            priority = base_priority * deprioritization_factor + wait_bonus

            return priority

    async def on_job_queued(self, job: Job, role: str = "", weight: Optional[float] = None) -> None:
        """
        Called when a job is added to the queue.

        Args:
            job: The queued job
            role: User's role (deprecated)
            weight: Direct weight value (preferred)
        """
        async with self._lock:
            if job.user_id not in self._user_states:
                resolved_weight = weight or (self.get_role_weight(role) if role else 1.0)
                self._user_states[job.user_id] = UserState(
                    user_id=job.user_id, weight=resolved_weight
                )

            state = self._user_states[job.user_id]
            state.recent_requests += 1
            state.last_request_time = datetime.now(timezone.utc)

    async def on_job_started(self, job: Job) -> None:
        """
        Called when a job starts processing.

        Args:
            job: The started job
        """
        async with self._lock:
            if job.user_id in self._user_states:
                state = self._user_states[job.user_id]
                state.active_requests += 1

    async def on_job_completed(
        self,
        job: Job,
        tokens_used: int,
    ) -> None:
        """
        Called when a job completes.

        Updates deficit counter and usage tracking.

        Args:
            job: The completed job
            tokens_used: Total tokens consumed
        """
        async with self._lock:
            if job.user_id not in self._user_states:
                return

            state = self._user_states[job.user_id]

            # Update deficit (subtract cost)
            cost = tokens_used or job.get_estimated_cost()
            state.deficit -= cost

            # Ensure deficit doesn't go too negative
            state.deficit = max(state.deficit, -state.weight * 10000)

            # Update usage tracking
            state.recent_tokens += cost
            state.active_requests = max(0, state.active_requests - 1)

            # Track global usage
            self._global_recent_tokens += cost

            # Record in usage history
            self._usage_history[job.user_id].append(
                UsageWindow(tokens=cost, requests=1)
            )

    async def accumulate_burst_credits(self, idle_seconds: float) -> None:
        """
        Accumulate burst credits for all users when cluster is idle.

        Called periodically when there's unused cluster capacity.

        Args:
            idle_seconds: How long the cluster has been idle
        """
        async with self._lock:
            # Credit proportional to idle time and weight
            credit_rate = 100  # tokens per second of idleness

            for state in self._user_states.values():
                credit = idle_seconds * credit_rate * state.weight
                state.burst_credits = min(
                    state.burst_credits + credit,
                    state.max_burst_credits,
                )

    async def reset_burst_credits(self) -> None:
        """Reset burst credits when contention is detected."""
        async with self._lock:
            for state in self._user_states.values():
                # Decay burst credits rather than zeroing
                state.burst_credits *= 0.5

    async def rebalance_deficits(self) -> None:
        """
        Rebalance deficit counters to prevent overflow.

        Called periodically to normalize the deficit values.
        """
        async with self._lock:
            if not self._user_states:
                return

            # Find the user with lowest deficit
            min_deficit = min(s.deficit for s in self._user_states.values())

            # Shift all deficits up
            if min_deficit < -1000:
                shift = -min_deficit - 500
                for state in self._user_states.values():
                    state.deficit += shift

    async def cleanup_old_history(self) -> None:
        """Clean up old usage history entries."""
        async with self._lock:
            cutoff = datetime.now(timezone.utc) - timedelta(
                seconds=self._fairness_window
            )

            for user_id in list(self._usage_history.keys()):
                history = self._usage_history[user_id]
                self._usage_history[user_id] = [
                    w for w in history if w.timestamp >= cutoff
                ]

                # Update recent totals
                if user_id in self._user_states:
                    state = self._user_states[user_id]
                    state.recent_tokens = sum(w.tokens for w in self._usage_history[user_id])
                    state.recent_requests = sum(w.requests for w in self._usage_history[user_id])

            # Recalculate global tokens
            self._global_recent_tokens = sum(
                state.recent_tokens for state in self._user_states.values()
            )

            self._last_cleanup = datetime.now(timezone.utc)

    async def _compute_deprioritization(self, user_id: int) -> float:
        """
        Compute deprioritization factor for a user.

        Users who have consumed more than threshold% of recent cluster
        capacity get deprioritized.

        Args:
            user_id: User ID

        Returns:
            Factor between 0 and 1 (1 = no deprioritization)
        """
        if self._global_recent_tokens <= 0:
            return 1.0

        state = self._user_states.get(user_id)
        if not state:
            return 1.0

        usage_fraction = state.recent_tokens / max(self._global_recent_tokens, 1)

        if usage_fraction > self._deprioritize_threshold:
            # Reduce priority proportionally to excess usage
            excess = usage_fraction - self._deprioritize_threshold
            # Map excess to 0-1 range for deprioritization
            return max(0.1, 1.0 - (excess * 2))

        return 1.0

    async def get_user_state(self, user_id: int) -> Optional[UserState]:
        """Get the current state for a user."""
        async with self._lock:
            return self._user_states.get(user_id)

    async def get_all_states(self) -> Dict[int, UserState]:
        """Get all user states."""
        async with self._lock:
            return dict(self._user_states)

    async def get_stats(self) -> Dict:
        """Get fair-share statistics."""
        async with self._lock:
            states = list(self._user_states.values())

            return {
                "total_users": len(states),
                "global_recent_tokens": self._global_recent_tokens,
                "fairness_window_seconds": self._fairness_window,
                "user_stats": [
                    {
                        "user_id": s.user_id,
                        "weight": s.weight,
                        "deficit": s.deficit,
                        "burst_credits": s.burst_credits,
                        "recent_tokens": s.recent_tokens,
                        "active_requests": s.active_requests,
                    }
                    for s in states
                ],
            }
