############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# ollama.py: Ollama backend adapter for telemetry collection
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Ollama backend adapter for capability discovery and telemetry."""

import asyncio
from typing import Optional
import time

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


class OllamaAdapter:
    """
    Adapter for Ollama backend telemetry and capability discovery.

    Ollama API endpoints used:
    - GET /api/version - Engine version
    - GET /api/tags - List available models
    - POST /api/ps - List running/loaded models
    - GET /api/show - Get model details (optional)
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
        Perform a health check on the Ollama backend.

        Returns:
            BackendHealth result
        """
        start_time = time.monotonic()
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")

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

            # Get version
            version = await self._get_version(client)
            caps.engine_version = version

            # Get models
            models = await self._get_models(client)
            caps.models = models

            # Get loaded models
            loaded = await self._get_loaded_models(client)
            caps.loaded_models = loaded

            # Mark which models are loaded
            for model in caps.models:
                model.is_loaded = model.name in loaded

            # Determine capabilities based on models
            for model in caps.models:
                if model.supports_multimodal:
                    caps.supports_multimodal = True
                # Check for embedding models
                if "embed" in model.name.lower() or "embedding" in model.name.lower():
                    caps.supports_embeddings = True

            caps.is_healthy = True

        except Exception as e:
            logger.warning("ollama_capability_discovery_failed", error=str(e))
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

            # Get loaded models
            loaded = await self._get_loaded_models(client)
            snapshot.loaded_models = loaded

            # Try to get GPU info if available
            # Ollama doesn't expose this directly, but some setups have a sidecar
            gpu_info = await self._try_get_gpu_info(client)
            if gpu_info:
                snapshot.gpu_utilization = gpu_info.utilization
                snapshot.gpu_memory_used_gb = gpu_info.memory_used_gb
                snapshot.gpu_memory_total_gb = gpu_info.memory_total_gb
                snapshot.gpu_temperature = gpu_info.temperature

            snapshot.is_healthy = True

        except Exception as e:
            logger.debug("ollama_telemetry_failed", error=str(e))
            snapshot.is_healthy = False

        return snapshot

    async def _get_version(self, client: httpx.AsyncClient) -> Optional[str]:
        """Get Ollama version."""
        try:
            response = await client.get("/api/version")
            if response.status_code == 200:
                data = response.json()
                return data.get("version")
        except Exception:
            pass

        # Try parsing from headers
        try:
            response = await client.get("/api/tags")
            version = response.headers.get("x-ollama-version")
            return version
        except Exception:
            pass

        return None

    async def _get_models(self, client: httpx.AsyncClient) -> list[ModelInfo]:
        """Get list of available models, enriched with /api/show data."""
        models = []

        try:
            response = await client.get("/api/tags")
            if response.status_code != 200:
                return models

            data = response.json()
            for model_data in data.get("models", []):
                name = model_data.get("name", "")
                if not name:
                    continue

                # Parse model details from /api/tags
                details = model_data.get("details", {})

                # Estimate parameter count from name or details
                param_count = details.get("parameter_size")
                if not param_count:
                    for size in ["70b", "34b", "13b", "7b", "3b", "1b"]:
                        if size in name.lower():
                            param_count = size.upper()
                            break

                models.append(
                    ModelInfo(
                        name=name,
                        family=details.get("family"),
                        parameter_count=param_count,
                        quantization=details.get("quantization_level"),
                        context_length=details.get("context_length"),
                        supports_multimodal=False,  # Will be set from capabilities below
                        supports_structured_output=True,
                    )
                )

            # Enrich models with /api/show data in parallel
            if models:
                show_tasks = [
                    self._get_model_details(client, m.name) for m in models
                ]
                show_results = await asyncio.gather(*show_tasks, return_exceptions=True)

                for model, show_data in zip(models, show_results):
                    if isinstance(show_data, Exception) or not show_data:
                        # Fall back to name-based multimodal heuristic
                        name_lower = model.name.lower()
                        if any(x in name_lower for x in ["llava", "vision", "-vl-", "-vl:"]):
                            model.supports_multimodal = True
                        continue

                    # Capabilities from /api/show
                    caps = show_data.get("capabilities", [])
                    if caps:
                        model.capabilities = caps
                        model.supports_multimodal = "vision" in caps
                        model.supports_thinking = "thinking" in caps

                    # Format and parent model from details
                    show_details = show_data.get("details", {})
                    model.model_format = show_details.get("format")
                    model.parent_model = show_details.get("parent_model") or None

                    # Families — use first family if available
                    families = show_details.get("families")
                    if families and isinstance(families, list) and families:
                        # Keep the family from /api/tags if set, otherwise use first from /api/show
                        if not model.family:
                            model.family = families[0]

                    # Architecture fields from model_info
                    # Parse num_ctx from Modelfile parameters
                    num_ctx = self._parse_num_ctx(show_data.get("parameters", ""))

                    model_info = show_data.get("model_info", {})
                    if model_info:
                        arch_fields = self._extract_arch_fields(model_info)
                        # Set model_max_context to the architectural maximum
                        if arch_fields.get("context_length") is not None:
                            model.model_max_context = arch_fields["context_length"]
                        # Effective context_length: use num_ctx if configured,
                        # else model's architectural max capped at 32768 to
                        # prevent small models from consuming excessive VRAM
                        _MAX_DEFAULT_CTX = 32768
                        if num_ctx is not None:
                            model.context_length = num_ctx
                        elif model.model_max_context is not None:
                            model.context_length = min(model.model_max_context, _MAX_DEFAULT_CTX)
                        else:
                            model.context_length = 4096
                        model.embedding_length = arch_fields.get("embedding_length")
                        model.head_count = arch_fields.get("head_count")
                        model.layer_count = arch_fields.get("layer_count")
                        model.feed_forward_length = arch_fields.get("feed_forward_length")

        except Exception as e:
            logger.warning("ollama_get_models_failed", error=str(e))

        return models

    async def _get_model_details(self, client: httpx.AsyncClient, model_name: str) -> dict:
        """Fetch rich model details from /api/show.

        Args:
            client: HTTP client
            model_name: Model name to query

        Returns:
            Parsed JSON response or empty dict on failure
        """
        try:
            response = await client.post(
                "/api/show",
                json={"name": model_name},
                timeout=15.0,
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.debug("ollama_show_failed", model=model_name, error=str(e))
        return {}

    @staticmethod
    def _parse_num_ctx(parameters: str) -> Optional[int]:
        """Parse num_ctx from Ollama Modelfile parameters string.

        The parameters string is newline-delimited key-value pairs, e.g.:
            stop           "<|eot_id|>"
            num_ctx        8192
            temperature    0.7
        """
        if not parameters:
            return None
        for line in parameters.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "num_ctx":
                try:
                    return int(parts[1])
                except (ValueError, IndexError):
                    pass
        return None

    @staticmethod
    def _extract_arch_fields(model_info: dict) -> dict:
        """Extract architecture fields from model_info dict.

        model_info keys are prefixed with the architecture name,
        e.g. "qwen3.context_length", "qwen3.embedding_length".
        """
        arch = model_info.get("general.architecture", "")
        return {
            "context_length": model_info.get(f"{arch}.context_length"),
            "embedding_length": model_info.get(f"{arch}.embedding_length"),
            "head_count": model_info.get(f"{arch}.attention.head_count"),
            "layer_count": model_info.get(f"{arch}.block_count"),
            "feed_forward_length": model_info.get(f"{arch}.feed_forward_length"),
        }

    async def _get_loaded_models(self, client: httpx.AsyncClient) -> list[str]:
        """Get list of currently loaded models."""
        loaded = []

        try:
            # Ollama 0.1.24+ supports /api/ps
            response = await client.post("/api/ps", json={})
            if response.status_code == 200:
                data = response.json()
                for model in data.get("models", []):
                    name = model.get("name")
                    if name:
                        loaded.append(name)
        except Exception:
            # Older Ollama versions don't have /api/ps
            pass

        return loaded

    async def _try_get_gpu_info(self, client: httpx.AsyncClient) -> Optional[GPUInfo]:
        """
        Try to get GPU info from a sidecar or custom endpoint.

        Some Ollama deployments have a GPU metrics sidecar at /gpu-info.
        """
        try:
            response = await client.get("/gpu-info")
            if response.status_code == 200:
                data = response.json()
                return GPUInfo(
                    utilization=data.get("utilization"),
                    memory_used_gb=data.get("memory_used_gb"),
                    memory_total_gb=data.get("memory_total_gb"),
                    temperature=data.get("temperature"),
                    name=data.get("name"),
                )
        except Exception:
            pass

        return None
