############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# scoring.py: Multi-factor backend scoring for routing decisions
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Backend scoring for scheduler routing decisions."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from backend.app.db.models import Backend, BackendStatus, Model, Modality
from backend.app.core.scheduler.queue import Job
from backend.app.settings import get_settings


@dataclass
class BackendScore:
    """Score breakdown for a backend."""

    backend_id: int
    backend_name: str
    total_score: float

    # Score components
    model_loaded_score: float = 0.0
    utilization_score: float = 0.0
    queue_score: float = 0.0
    throughput_score: float = 0.0
    latency_score: float = 0.0
    priority_score: float = 0.0

    # Constraint status
    passed_constraints: List[str] = None
    failed_constraints: List[str] = None

    def __post_init__(self):
        if self.passed_constraints is None:
            self.passed_constraints = []
        if self.failed_constraints is None:
            self.failed_constraints = []


@dataclass
class HardConstraints:
    """Hard constraints that must be met for a backend to be eligible."""

    model_available: bool = False
    supports_modality: bool = False
    supports_structured_output: bool = True  # Most backends do
    has_capacity: bool = False
    memory_fit: bool = True  # Assume true if unknown


class BackendScorer:
    """
    Scores backends for job assignment.

    Implements a multi-factor scoring system:
    - Hard constraints filter out ineligible backends
    - Soft scores rank eligible backends

    Scoring factors:
    - Model already loaded (major bonus)
    - Low GPU utilization (bonus)
    - Short queue depth (bonus)
    - High throughput GPU type (bonus)
    - Backend priority (admin-configured preference)
    """

    def __init__(self):
        self._settings = get_settings()

        # Score weights from settings
        self._weight_model_loaded = self._settings.scheduler_score_model_loaded
        self._weight_low_utilization = self._settings.scheduler_score_low_utilization
        self._weight_short_queue = self._settings.scheduler_score_short_queue
        self._weight_high_throughput = self._settings.scheduler_score_high_throughput
        self._weight_latency = self._settings.scheduler_score_latency

    def check_hard_constraints(
        self,
        backend: Backend,
        job: Job,
        backend_models: List[Model],
        current_queue_depth: int = 0,
    ) -> Tuple[bool, HardConstraints]:
        """
        Check if a backend meets hard constraints for a job.

        Args:
            backend: The backend to check
            job: The job to assign
            backend_models: Models available on this backend
            current_queue_depth: Current queue depth at this backend

        Returns:
            Tuple of (is_eligible, constraint_details)
        """
        constraints = HardConstraints()

        # Check if backend is healthy
        if backend.status != BackendStatus.HEALTHY:
            return False, constraints

        # Check model availability
        model_names = [m.name for m in backend_models]
        constraints.model_available = job.model in model_names

        # Check modality support
        if job.modality == Modality.MULTIMODAL or job.requires_multimodal:
            constraints.supports_modality = backend.supports_multimodal
            # Model-level check is authoritative — if the specific model
            # doesn't support multimodal, override the backend-level flag.
            for m in backend_models:
                if m.name == job.model:
                    constraints.supports_modality = m.supports_multimodal
                    break
        elif job.modality == Modality.EMBEDDING:
            constraints.supports_modality = backend.supports_embeddings
        else:
            constraints.supports_modality = True

        # Check structured output support
        if job.requires_structured_output:
            constraints.supports_structured_output = backend.supports_structured_output
            # Also check model-level support
            for m in backend_models:
                if m.name == job.model:
                    constraints.supports_structured_output = m.supports_structured_output
                    break

        # Check capacity
        constraints.has_capacity = (
            backend.current_concurrent + current_queue_depth < backend.max_concurrent
        )

        # Memory fit check (if we have info)
        if backend.gpu_memory_gb:
            for m in backend_models:
                if m.name == job.model and m.vram_required_gb:
                    constraints.memory_fit = m.vram_required_gb <= backend.gpu_memory_gb
                    break

        # All hard constraints must pass
        is_eligible = (
            constraints.model_available
            and constraints.supports_modality
            and constraints.supports_structured_output
            and constraints.has_capacity
            and constraints.memory_fit
        )

        return is_eligible, constraints

    def compute_score(
        self,
        backend: Backend,
        job: Job,
        backend_models: List[Model],
        gpu_utilization: Optional[float] = None,
        current_queue_depth: int = 0,
        latency_ema_ms: Optional[float] = None,
    ) -> BackendScore:
        """
        Compute soft score for a backend.

        Args:
            backend: The backend to score
            job: The job to assign
            backend_models: Models available on this backend
            gpu_utilization: Current GPU utilization (0-100, or None if unknown)
            current_queue_depth: Current queue depth at this backend
            latency_ema_ms: Real-world latency EMA in ms (lower = better)

        Returns:
            BackendScore with component breakdown
        """
        score = BackendScore(
            backend_id=backend.id,
            backend_name=backend.name,
            total_score=0.0,
        )

        # Model loaded bonus
        for m in backend_models:
            if m.name == job.model and m.is_loaded:
                score.model_loaded_score = self._weight_model_loaded
                break

        # Utilization score (lower is better)
        if gpu_utilization is not None:
            # Score inversely proportional to utilization
            # 0% utilization = full bonus, 100% utilization = no bonus
            score.utilization_score = (
                self._weight_low_utilization * (100 - gpu_utilization) / 100
            )
        else:
            # Unknown utilization gets half bonus
            score.utilization_score = self._weight_low_utilization * 0.5

        # Queue depth score (shorter is better)
        if current_queue_depth == 0:
            score.queue_score = self._weight_short_queue
        else:
            # Decay score based on queue depth
            score.queue_score = self._weight_short_queue / (1 + current_queue_depth * 0.5)

        # Throughput score from backend configuration
        score.throughput_score = self._weight_high_throughput * backend.throughput_score

        # Latency score (lower latency = higher bonus)
        if latency_ema_ms is not None and latency_ema_ms > 0:
            latency_factor = 1.0 / (1.0 + latency_ema_ms / 5000.0)
            score.latency_score = self._weight_latency * latency_factor
        else:
            # Unknown latency = neutral (half bonus)
            score.latency_score = self._weight_latency * 0.5

        # Priority score from backend configuration
        score.priority_score = backend.priority * 10  # Scale priority

        # Compute total
        score.total_score = (
            score.model_loaded_score
            + score.utilization_score
            + score.queue_score
            + score.throughput_score
            + score.latency_score
            + score.priority_score
        )

        return score

    def rank_backends(
        self,
        backends: List[Backend],
        job: Job,
        backend_models: Dict[int, List[Model]],
        gpu_utilizations: Dict[int, Optional[float]] = None,
        queue_depths: Dict[int, int] = None,
        latency_emas: Optional[Dict[int, float]] = None,
    ) -> List[BackendScore]:
        """
        Rank backends for a job, filtering by hard constraints.

        Args:
            backends: List of backends to consider
            job: The job to assign
            backend_models: Dict mapping backend_id to list of models
            gpu_utilizations: Dict mapping backend_id to GPU utilization
            queue_depths: Dict mapping backend_id to queue depth
            latency_emas: Dict mapping backend_id to latency EMA (ms)

        Returns:
            List of BackendScore, sorted by total_score descending (best first)
        """
        gpu_utilizations = gpu_utilizations or {}
        queue_depths = queue_depths or {}
        latency_emas = latency_emas or {}

        eligible_scores = []
        ineligible_scores = []

        for backend in backends:
            models = backend_models.get(backend.id, [])
            queue_depth = queue_depths.get(backend.id, 0)

            # Check hard constraints
            is_eligible, constraints = self.check_hard_constraints(
                backend, job, models, queue_depth
            )

            if not is_eligible:
                # Track why backend was rejected (for debugging)
                score = BackendScore(
                    backend_id=backend.id,
                    backend_name=backend.name,
                    total_score=-1,  # Indicates ineligible
                )
                if not constraints.model_available:
                    score.failed_constraints.append("model_not_available")
                if not constraints.supports_modality:
                    score.failed_constraints.append("modality_not_supported")
                if not constraints.supports_structured_output:
                    score.failed_constraints.append("structured_output_not_supported")
                if not constraints.has_capacity:
                    score.failed_constraints.append("no_capacity")
                if not constraints.memory_fit:
                    score.failed_constraints.append("memory_insufficient")
                ineligible_scores.append(score)
                continue

            # Compute soft score
            score = self.compute_score(
                backend=backend,
                job=job,
                backend_models=models,
                gpu_utilization=gpu_utilizations.get(backend.id),
                current_queue_depth=queue_depth,
                latency_ema_ms=latency_emas.get(backend.id),
            )

            # Record passed constraints
            score.passed_constraints = [
                "model_available",
                "modality_supported",
                "structured_output_supported",
                "has_capacity",
                "memory_fit",
            ]

            eligible_scores.append(score)

        # If no eligible backends, return ineligible scores so the caller
        # can distinguish permanent constraint failures from empty backend lists.
        if not eligible_scores and ineligible_scores:
            return ineligible_scores

        # Sort by total score descending
        eligible_scores.sort(key=lambda s: s.total_score, reverse=True)

        return eligible_scores
