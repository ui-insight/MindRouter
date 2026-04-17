############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# policy.py: Scheduler policy and main scheduling entry point
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Scheduler policy - main entry point for scheduling decisions."""

from typing import Dict, List, Optional, Set
import tiktoken

from backend.app.core.scheduler.routing import BackendRouter, RoutingDecision
from backend.app.core.scheduler.queue import Job, JobModality
from backend.app.core.canonical_schemas import (
    CanonicalChatRequest,
    CanonicalCompletionRequest,
    CanonicalEmbeddingRequest,
    CanonicalImageRequest,
    CanonicalRerankRequest,
    CanonicalScoreRequest,
)
from backend.app.db.models import Backend, Model
from backend.app.settings import get_settings
from backend.app.logging_config import get_logger

logger = get_logger(__name__)


class SchedulerPolicy:
    """
    Main scheduler policy class.

    Coordinates job creation, queuing, routing, and completion tracking.
    This is the primary interface for the API layer to interact with scheduling.
    """

    def __init__(self):
        self.router = BackendRouter()
        self._settings = get_settings()

        # Tokenizer for estimation
        self._tokenizer = None

    async def start(self) -> None:
        """Start the scheduler."""
        await self.router.start()
        logger.info("Scheduler policy started")

    async def stop(self) -> None:
        """Stop the scheduler."""
        await self.router.stop()
        logger.info("Scheduler policy stopped")

    def _get_tokenizer(self):
        """Get or create tokenizer for token estimation."""
        if self._tokenizer is None:
            try:
                self._tokenizer = tiktoken.get_encoding(self._settings.default_tokenizer)
            except Exception:
                # Fallback to cl100k_base if configured tokenizer not found
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
        return self._tokenizer

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        try:
            tokenizer = self._get_tokenizer()
            return len(tokenizer.encode(text))
        except Exception:
            # Rough estimate: ~4 chars per token
            return len(text) // 4

    def create_job_from_chat_request(
        self,
        request: CanonicalChatRequest,
        user_id: int,
        api_key_id: int,
    ) -> Job:
        """
        Create a Job from a chat request.

        Args:
            request: The canonical chat request
            user_id: User ID
            api_key_id: API key ID

        Returns:
            Job instance
        """
        # Estimate prompt tokens
        prompt_text = ""
        for msg in request.messages:
            prompt_text += msg.get_text_content() + " "

        estimated_prompt_tokens = self.estimate_tokens(prompt_text)

        # Estimate completion tokens (use max_tokens or default)
        estimated_completion_tokens = request.max_tokens or 1000

        # Determine modality
        modality = JobModality.CHAT
        requires_multimodal = request.requires_multimodal()
        if requires_multimodal:
            modality = JobModality.MULTIMODAL

        # Calculate image bytes if present
        image_bytes = 0
        for msg in request.messages:
            if msg.has_images():
                # Rough estimate - actual size depends on image data
                image_bytes += 100000  # 100KB estimate per image

        return Job(
            request_id=request.request_id or "",
            user_id=user_id,
            api_key_id=api_key_id,
            model=request.model,
            modality=modality,
            is_streaming=request.stream,
            requires_multimodal=requires_multimodal,
            requires_structured_output=request.requires_structured_output(),
            estimated_prompt_tokens=estimated_prompt_tokens,
            estimated_completion_tokens=estimated_completion_tokens,
            image_bytes=image_bytes,
            request_data=request.model_dump(),
        )

    def create_job_from_completion_request(
        self,
        request: CanonicalCompletionRequest,
        user_id: int,
        api_key_id: int,
    ) -> Job:
        """Create a Job from a completion request."""
        prompt_text = request.prompt if isinstance(request.prompt, str) else " ".join(request.prompt)
        estimated_prompt_tokens = self.estimate_tokens(prompt_text)
        estimated_completion_tokens = request.max_tokens or 1000

        return Job(
            request_id=request.request_id or "",
            user_id=user_id,
            api_key_id=api_key_id,
            model=request.model,
            modality=JobModality.COMPLETION,
            is_streaming=request.stream,
            requires_multimodal=False,
            requires_structured_output=False,
            estimated_prompt_tokens=estimated_prompt_tokens,
            estimated_completion_tokens=estimated_completion_tokens,
            request_data=request.model_dump(),
        )

    def create_job_from_embedding_request(
        self,
        request: CanonicalEmbeddingRequest,
        user_id: int,
        api_key_id: int,
    ) -> Job:
        """Create a Job from an embedding request."""
        input_text = request.input if isinstance(request.input, str) else " ".join(request.input)
        estimated_tokens = self.estimate_tokens(input_text)

        return Job(
            request_id=request.request_id or "",
            user_id=user_id,
            api_key_id=api_key_id,
            model=request.model,
            modality=JobModality.EMBEDDING,
            is_streaming=False,
            requires_multimodal=False,
            requires_structured_output=False,
            estimated_prompt_tokens=estimated_tokens,
            estimated_completion_tokens=0,
            request_data=request.model_dump(),
        )

    def create_job_from_rerank_request(
        self,
        request: CanonicalRerankRequest,
        user_id: int,
        api_key_id: int,
    ) -> Job:
        """Create a Job from a rerank or score request."""
        # Estimate tokens from query + all documents
        text = request.query + " " + " ".join(request.documents)
        estimated_tokens = self.estimate_tokens(text)

        return Job(
            request_id=request.request_id or "",
            user_id=user_id,
            api_key_id=api_key_id,
            model=request.model,
            modality=JobModality.RERANKING,
            is_streaming=False,
            requires_multimodal=False,
            requires_structured_output=False,
            estimated_prompt_tokens=estimated_tokens,
            estimated_completion_tokens=0,
            request_data=request.model_dump(),
        )

    def create_job_from_score_request(
        self,
        request: CanonicalScoreRequest,
        user_id: int,
        api_key_id: int,
    ) -> Job:
        """Create a Job from a score request."""
        text_2 = request.text_2 if isinstance(request.text_2, str) else " ".join(request.text_2)
        text = request.text_1 + " " + text_2
        estimated_tokens = self.estimate_tokens(text)

        return Job(
            request_id=request.request_id or "",
            user_id=user_id,
            api_key_id=api_key_id,
            model=request.model,
            modality=JobModality.RERANKING,
            is_streaming=False,
            requires_multimodal=False,
            requires_structured_output=False,
            estimated_prompt_tokens=estimated_tokens,
            estimated_completion_tokens=0,
            request_data=request.model_dump(),
        )

    def create_job_from_image_request(
        self,
        request: CanonicalImageRequest,
        user_id: int,
        api_key_id: int,
    ) -> Job:
        """Create a Job from an image generation request."""
        # Token estimation is not meaningful for diffusion models,
        # but we estimate prompt tokens for fair-share scheduling.
        estimated_prompt_tokens = self.estimate_tokens(request.prompt)

        return Job(
            request_id=request.request_id or "",
            user_id=user_id,
            api_key_id=api_key_id,
            model=request.model,
            modality=JobModality.IMAGE_GENERATION,
            is_streaming=False,
            requires_multimodal=False,
            requires_structured_output=False,
            estimated_prompt_tokens=estimated_prompt_tokens,
            estimated_completion_tokens=0,
            request_data=request.model_dump(),
        )

    async def submit_job(
        self,
        job: Job,
        user_role: str,
        user_weight: float = 1.0,
    ) -> int:
        """
        Submit a job for scheduling.

        Args:
            job: The job to submit
            user_role: User's role for priority weighting
            user_weight: Direct weight from group.scheduler_weight

        Returns:
            Queue position
        """
        return await self.router.submit_job(job, user_role, user_weight=user_weight)

    async def route_job(
        self,
        job: Job,
        backends: List[Backend],
        backend_models: Dict[int, List[Model]],
        gpu_utilizations: Optional[Dict[int, Optional[float]]] = None,
        exclude_backend_ids: Optional[Set[int]] = None,
        latency_emas: Optional[Dict[int, float]] = None,
    ) -> RoutingDecision:
        """
        Route a job to the best available backend.

        Args:
            job: The job to route
            backends: Available backends
            backend_models: Models per backend
            gpu_utilizations: Optional GPU utilization data
            exclude_backend_ids: Backend IDs to exclude (retry failover)
            latency_emas: Per-backend latency EMA for scoring

        Returns:
            RoutingDecision
        """
        return await self.router.route_job(
            job=job,
            backends=backends,
            backend_models=backend_models,
            gpu_utilizations=gpu_utilizations or {},
            exclude_backend_ids=exclude_backend_ids,
            latency_emas=latency_emas,
        )

    async def on_job_started(self, job: Job, backend_id: int) -> None:
        """Notify scheduler that a job has started."""
        await self.router.on_job_started(job, backend_id)

    async def on_job_completed(
        self,
        job: Job,
        backend_id: int,
        tokens_used: int,
    ) -> None:
        """Notify scheduler that a job has completed."""
        await self.router.on_job_completed(job, backend_id, tokens_used)

    async def on_job_failed(self, job: Job, backend_id: int) -> None:
        """Notify scheduler that a job has failed."""
        await self.router.on_job_failed(job, backend_id)

    async def wait_for_capacity(self, timeout: float = 5.0) -> bool:
        """Wait until backend capacity becomes available."""
        return await self.router.wait_for_capacity(timeout)

    def register_waiter(self, job: Job) -> None:
        """Register a job as waiting for capacity."""
        self.router.register_waiter(job)

    def unregister_waiter(self, job: Job) -> None:
        """Remove a job from the waiter set."""
        self.router.unregister_waiter(job)

    def is_highest_priority_waiter(self, job: Job) -> bool:
        """Check if job has the highest priority among waiters for its model."""
        return self.router.is_highest_priority_waiter(job)

    async def recompute_priority(self, job: Job, role: str = "", weight: float = 1.0) -> None:
        """Recompute and update a job's priority using current fair-share state."""
        job.priority = await self.router.fair_share.compute_priority(job, role=role, weight=weight)

    async def cancel_job(self, request_id: str) -> bool:
        """Cancel a queued job."""
        return await self.router.cancel_job(request_id)

    async def get_stats(self) -> Dict:
        """Get scheduler statistics."""
        return await self.router.get_queue_stats()


# Global scheduler instance
_scheduler: Optional[SchedulerPolicy] = None


def get_scheduler() -> SchedulerPolicy:
    """Get the global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = SchedulerPolicy()
    return _scheduler


async def init_scheduler() -> SchedulerPolicy:
    """Initialize and start the global scheduler."""
    scheduler = get_scheduler()
    await scheduler.start()
    return scheduler


async def shutdown_scheduler() -> None:
    """Shutdown the global scheduler."""
    global _scheduler
    if _scheduler:
        await _scheduler.stop()
        _scheduler = None
