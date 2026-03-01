############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# vllm.py: vLLM backend adapter for telemetry collection
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""vLLM backend adapter for capability discovery and telemetry."""

import time
from typing import Optional

import httpx

from backend.app.core.telemetry.models import (
    BackendCapabilities,
    BackendHealth,
    GPUInfo,
    ModelInfo,
    TelemetrySnapshot,
)
from backend.app.logging_config import get_logger
from backend.app.settings import get_settings

logger = get_logger(__name__)


class VLLMAdapter:
    """
    Adapter for vLLM backend telemetry and capability discovery.

    vLLM API endpoints used:
    - GET /health - Health check
    - GET /v1/models - List models (OpenAI-compatible)
    - GET /metrics - Prometheus metrics (optional)
    """

    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._settings = get_settings()
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def health_check(self) -> BackendHealth:
        """
        Perform a health check on the vLLM backend.

        Returns:
            BackendHealth result
        """
        start_time = time.monotonic()
        try:
            client = await self._get_client()

            # vLLM has /health endpoint
            response = await client.get("/health")
            latency_ms = (time.monotonic() - start_time) * 1000

            if response.status_code == 200:
                return BackendHealth(
                    is_healthy=True,
                    status_code=response.status_code,
                    latency_ms=latency_ms,
                )
            else:
                return BackendHealth(
                    is_healthy=False,
                    status_code=response.status_code,
                    latency_ms=latency_ms,
                    error_message=f"HTTP {response.status_code}",
                )

        except httpx.TimeoutException:
            latency_ms = (time.monotonic() - start_time) * 1000
            return BackendHealth(
                is_healthy=False,
                latency_ms=latency_ms,
                error_message="Connection timeout",
            )
        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000

            # Try /v1/models as fallback health check
            try:
                client = await self._get_client()
                response = await client.get("/v1/models")
                if response.status_code == 200:
                    return BackendHealth(
                        is_healthy=True,
                        status_code=response.status_code,
                        latency_ms=latency_ms,
                    )
            except Exception:
                pass

            return BackendHealth(
                is_healthy=False,
                latency_ms=latency_ms,
                error_message=str(e),
            )

    async def discover_capabilities(self) -> BackendCapabilities:
        """
        Discover backend capabilities.

        Returns:
            BackendCapabilities with models, version, and features
        """
        caps = BackendCapabilities()

        try:
            client = await self._get_client()

            # Get version from headers or /version endpoint
            version = await self._get_version(client)
            caps.engine_version = version

            # Get models
            models = await self._get_models(client)
            caps.models = models

            # vLLM models are always loaded (one model per instance typically)
            caps.loaded_models = [m.name for m in models]
            for model in caps.models:
                model.is_loaded = True

            # Determine capabilities based on models
            for model in caps.models:
                if model.supports_multimodal:
                    caps.supports_multimodal = True
                # vLLM can run embedding models
                if "embed" in model.name.lower():
                    caps.supports_embeddings = True

            # vLLM generally supports structured output well
            caps.supports_structured_output = True
            caps.is_healthy = True

        except Exception as e:
            logger.warning("vllm_capability_discovery_failed", error=str(e))
            caps.is_healthy = False
            caps.error_message = str(e)

        return caps

    async def get_telemetry(self, backend_id: int) -> TelemetrySnapshot:
        """
        Get current telemetry from the backend.

        Args:
            backend_id: Backend ID for the snapshot

        Returns:
            TelemetrySnapshot with current metrics
        """
        snapshot = TelemetrySnapshot(backend_id=backend_id)

        try:
            client = await self._get_client()

            # Get models (to know what's loaded)
            models = await self._get_models(client)
            snapshot.loaded_models = [m.name for m in models]

            # Try to get metrics from Prometheus endpoint
            metrics = await self._get_prometheus_metrics(client)
            if metrics:
                snapshot.gpu_utilization = metrics.get("gpu_utilization")
                snapshot.gpu_memory_used_gb = metrics.get("gpu_memory_used_gb")
                snapshot.gpu_memory_total_gb = metrics.get("gpu_memory_total_gb")
                snapshot.active_requests = metrics.get("active_requests", 0)
                snapshot.queued_requests = metrics.get("queued_requests", 0)
                snapshot.requests_per_second = metrics.get("requests_per_second")

            snapshot.is_healthy = True

        except Exception as e:
            logger.debug("vllm_telemetry_failed", error=str(e))
            snapshot.is_healthy = False

        return snapshot

    async def _get_version(self, client: httpx.AsyncClient) -> Optional[str]:
        """Get vLLM version."""
        try:
            # Try /version endpoint first
            response = await client.get("/version")
            if response.status_code == 200:
                data = response.json()
                return data.get("version")
        except Exception:
            pass

        try:
            # Check server header
            response = await client.get("/v1/models")
            server = response.headers.get("server", "")
            if "vllm" in server.lower():
                # Try to extract version
                parts = server.split("/")
                if len(parts) > 1:
                    return parts[1]
        except Exception:
            pass

        return None

    async def _get_models(self, client: httpx.AsyncClient) -> list[ModelInfo]:
        """Get list of available models."""
        models = []

        try:
            response = await client.get("/v1/models")
            if response.status_code != 200:
                return models

            data = response.json()
            for model_data in data.get("data", []):
                model_id = model_data.get("id", "")
                if not model_id:
                    continue

                # Determine capabilities from model name
                model_lower = model_id.lower()
                supports_multimodal = any(
                    x in model_lower
                    for x in ["llava", "vision", "vl", "multimodal"]
                )

                # Auto-detect thinking support for known model families
                supports_thinking = any(
                    x in model_lower
                    for x in ["qwen3", "gpt-oss", "deepseek-r1"]
                )

                # Try to extract parameter count
                param_count = None
                for size in ["70b", "34b", "13b", "7b", "3b", "1b"]:
                    if size in model_id.lower():
                        param_count = size.upper()
                        break

                # max_model_len is a vLLM extension to the OpenAI /v1/models response
                context_length = model_data.get("max_model_len")

                models.append(
                    ModelInfo(
                        name=model_id,
                        parameter_count=param_count,
                        context_length=context_length,
                        supports_multimodal=supports_multimodal,
                        supports_thinking=supports_thinking,
                        supports_structured_output=True,
                        is_loaded=True,  # vLLM models are always loaded
                    )
                )

        except Exception as e:
            logger.warning("vllm_get_models_failed", error=str(e))

        return models

    async def _get_prometheus_metrics(self, client: httpx.AsyncClient) -> Optional[dict]:
        """
        Parse Prometheus metrics from /metrics endpoint.

        Returns parsed metrics dict or None if unavailable.
        """
        try:
            response = await client.get("/metrics")
            if response.status_code != 200:
                return None

            metrics = {}
            text = response.text

            # Parse key metrics from Prometheus format
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("#") or not line:
                    continue

                # Parse metric line: metric_name{labels} value
                try:
                    if " " in line:
                        metric_part, value = line.rsplit(" ", 1)
                        metric_name = metric_part.split("{")[0]

                        # Extract relevant metrics
                        if metric_name == "vllm:num_requests_running":
                            metrics["active_requests"] = int(float(value))
                        elif metric_name == "vllm:num_requests_waiting":
                            metrics["queued_requests"] = int(float(value))
                        elif metric_name == "vllm:gpu_cache_usage_perc":
                            metrics["gpu_utilization"] = float(value) * 100
                        # Add more metrics as needed

                except (ValueError, IndexError):
                    continue

            return metrics if metrics else None

        except Exception:
            return None
