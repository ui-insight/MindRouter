############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# models.py: Telemetry data models for backends and GPUs
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Telemetry data models."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional


@dataclass
class ModelInfo:
    """Information about a model on a backend."""

    name: str
    family: Optional[str] = None
    parameter_count: Optional[str] = None  # "7B", "70B", etc.
    quantization: Optional[str] = None  # "Q4_K_M", "FP16", etc.
    context_length: Optional[int] = None
    model_max_context: Optional[int] = None
    supports_multimodal: bool = False
    supports_thinking: bool = False
    supports_structured_output: bool = True
    is_loaded: bool = False
    vram_required_gb: Optional[float] = None

    # Rich metadata from /api/show (Ollama)
    model_format: Optional[str] = None  # "gguf", etc.
    capabilities: Optional[List[str]] = None  # ["completion", "vision", "tools"]
    embedding_length: Optional[int] = None
    head_count: Optional[int] = None
    layer_count: Optional[int] = None
    feed_forward_length: Optional[int] = None
    parent_model: Optional[str] = None


@dataclass
class GPUInfo:
    """GPU information from a backend."""

    utilization: Optional[float] = None  # 0-100
    memory_used_gb: Optional[float] = None
    memory_total_gb: Optional[float] = None
    temperature: Optional[float] = None
    name: Optional[str] = None


@dataclass
class BackendCapabilities:
    """Capabilities discovered from a backend."""

    engine_version: Optional[str] = None
    models: List[ModelInfo] = field(default_factory=list)
    loaded_models: List[str] = field(default_factory=list)
    gpu_info: Optional[GPUInfo] = None

    max_concurrent: int = 4
    is_healthy: bool = False
    error_message: Optional[str] = None

    discovered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TelemetrySnapshot:
    """Point-in-time telemetry snapshot from a backend."""

    backend_id: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # GPU metrics
    gpu_utilization: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    gpu_temperature: Optional[float] = None

    # Request metrics
    active_requests: int = 0
    queued_requests: int = 0
    requests_per_second: Optional[float] = None

    # Model state
    loaded_models: List[str] = field(default_factory=list)

    # Health
    is_healthy: bool = True
    latency_ms: Optional[float] = None


@dataclass
class GPUDeviceSnapshot:
    """Detailed per-GPU device snapshot from sidecar agent."""

    index: int = 0
    name: Optional[str] = None
    uuid: Optional[str] = None
    pci_bus_id: Optional[str] = None
    compute_capability: Optional[str] = None
    memory_total_gb: Optional[float] = None
    memory_used_gb: Optional[float] = None
    memory_free_gb: Optional[float] = None
    utilization_gpu: Optional[float] = None
    utilization_memory: Optional[float] = None
    temperature_gpu: Optional[float] = None
    temperature_memory: Optional[float] = None
    power_draw_watts: Optional[float] = None
    power_limit_watts: Optional[float] = None
    fan_speed_percent: Optional[float] = None
    clock_sm_mhz: Optional[int] = None
    clock_memory_mhz: Optional[int] = None


@dataclass
class SidecarResponse:
    """Full response from GPU sidecar agent."""

    hostname: Optional[str] = None
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    gpu_count: int = 0
    gpus: List[GPUDeviceSnapshot] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sidecar_version: Optional[str] = None


@dataclass
class BackendHealth:
    """Health check result for a backend."""

    is_healthy: bool
    status_code: Optional[int] = None
    latency_ms: float = 0.0
    error_message: Optional[str] = None
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CircuitBreakerState:
    """Per-backend circuit breaker state for reactive health management."""

    live_failure_count: int = 0
    circuit_open_until: Optional[datetime] = None
    last_failure_time: Optional[datetime] = None

    @property
    def is_open(self) -> bool:
        """Circuit is open (rejecting requests) if open_until is in the future."""
        if self.circuit_open_until is None:
            return False
        open_until = self.circuit_open_until
        if open_until.tzinfo is None:
            open_until = open_until.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) < open_until

    @property
    def is_half_open(self) -> bool:
        """Circuit is half-open (allowing a probe) if open_until has passed."""
        if self.circuit_open_until is None:
            return False
        open_until = self.circuit_open_until
        if open_until.tzinfo is None:
            open_until = open_until.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) >= open_until


