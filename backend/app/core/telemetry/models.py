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

import socket
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
    supports_tools: bool = False
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
class ServerPowerSnapshot:
    """Server-level power reading from IPMI DCMI."""

    instantaneous_watts: Optional[int] = None
    minimum_watts: Optional[int] = None
    maximum_watts: Optional[int] = None
    average_watts: Optional[int] = None
    error: Optional[str] = None


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
    server_power: Optional[ServerPowerSnapshot] = None


# Failure classes for a health check or telemetry poll. A "local" fault is one
# THIS host caused — name resolution failed, the local network is unreachable —
# and says nothing about the remote backend, so callers must not charge it
# against that backend.
ERROR_KIND_DNS = "dns"
ERROR_KIND_CONNECT = "connect"
ERROR_KIND_TIMEOUT = "timeout"
ERROR_KIND_HTTP = "http"

# Kinds that are this host's problem rather than the backend's.
LOCAL_FAULT_KINDS = frozenset({ERROR_KIND_DNS})

_TIMEOUT_TYPE_NAMES = frozenset(
    {"TimeoutException", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout"}
)

# Text of a resolver failure when the cause chain has been flattened and the
# original socket.gaierror is no longer attached (glibc, macOS wordings).
_DNS_MESSAGE_MARKERS = (
    "temporary failure in name resolution",
    "name or service not known",
    "nodename nor servname provided",
    "no address associated with hostname",
)


def classify_transport_error(exc: BaseException) -> Optional[str]:
    """Classify a transport exception raised while contacting a backend.

    Returns one of the ERROR_KIND_* values, or None when unrecognised.

    None is deliberate and load-bearing: callers treat an unclassified failure
    as the backend's own fault, so anything this function does not positively
    recognise still counts against the backend. Misclassifying a genuinely sick
    backend as a local fault would keep routing traffic to it, which is far
    worse than the reverse, so this errs toward blaming the backend.

    Duck-typed on class name rather than importing httpx: this module holds
    plain data structures and must not take a dependency on the HTTP client.
    """
    seen = 0
    current: Optional[BaseException] = exc
    while current is not None and seen < 5:
        if isinstance(current, socket.gaierror):
            return ERROR_KIND_DNS

        name = type(current).__name__
        if name in _TIMEOUT_TYPE_NAMES:
            return ERROR_KIND_TIMEOUT
        if name == "ConnectError":
            text = str(current).lower()
            if any(marker in text for marker in _DNS_MESSAGE_MARKERS):
                return ERROR_KIND_DNS
            return ERROR_KIND_CONNECT

        current = current.__cause__ or current.__context__
        seen += 1

    return None


@dataclass
class BackendHealth:
    """Health check result for a backend."""

    is_healthy: bool
    status_code: Optional[int] = None
    latency_ms: float = 0.0
    error_message: Optional[str] = None
    # Failure class (see classify_transport_error). None means unclassified,
    # which callers treat as the backend's own fault.
    error_kind: Optional[str] = None
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


