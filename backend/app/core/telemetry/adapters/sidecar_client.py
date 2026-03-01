############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# sidecar_client.py: Client for GPU sidecar agent
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Client for the GPU sidecar agent running on inference nodes."""

from typing import Optional

import httpx

from backend.app.core.telemetry.models import (
    GPUDeviceSnapshot,
    SidecarResponse,
)
from backend.app.logging_config import get_logger

logger = get_logger(__name__)


class SidecarClient:
    """
    HTTP client for the GPU metrics sidecar agent.

    Each inference node can optionally run a sidecar agent that
    exposes per-GPU hardware metrics via GET /gpu-info. This client
    fetches and parses those metrics for storage in MindRouter2.
    """

    def __init__(self, sidecar_url: str, timeout: float = 5.0, sidecar_key: Optional[str] = None):
        self.base_url = sidecar_url.rstrip("/")
        self.timeout = timeout
        self.sidecar_key = sidecar_key
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            headers = {}
            if self.sidecar_key:
                headers["X-Sidecar-Key"] = self.sidecar_key
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers=headers,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def get_gpu_info(self) -> Optional[SidecarResponse]:
        """
        Fetch GPU info from the sidecar agent.

        Returns:
            SidecarResponse with per-GPU details, or None if unavailable
        """
        try:
            client = await self._get_client()
            response = await client.get("/gpu-info")

            if response.status_code != 200:
                logger.warning(
                    "sidecar_bad_status",
                    url=self.base_url,
                    status=response.status_code,
                    body=response.text[:200],
                )
                return None

            data = response.json()
            return self._parse_response(data)

        except httpx.TimeoutException:
            logger.warning("sidecar_timeout", url=self.base_url)
            return None
        except Exception as e:
            logger.warning("sidecar_error", url=self.base_url, error=str(e))
            return None

    async def health_check(self) -> bool:
        """Check if the sidecar agent is reachable."""
        try:
            client = await self._get_client()
            response = await client.get("/health")
            return response.status_code == 200
        except Exception:
            return False

    async def ollama_pull(self, ollama_url: str, model: str) -> Optional[dict]:
        """Start a model pull on the sidecar's Ollama instance."""
        try:
            client = await self._get_client()
            response = await client.post(
                "/ollama/pull",
                json={"ollama_url": ollama_url, "model": model},
                timeout=30.0,
            )
            if response.status_code != 200:
                logger.warning("sidecar_ollama_pull_failed", status=response.status_code, body=response.text[:200])
                return None
            return response.json()
        except httpx.TimeoutException:
            logger.warning("sidecar_ollama_pull_timeout", url=self.base_url)
            return None
        except Exception as e:
            logger.warning("sidecar_ollama_pull_error", url=self.base_url, error=str(e))
            return None

    async def ollama_pull_status(self, job_id: str) -> Optional[dict]:
        """Poll pull progress from the sidecar."""
        try:
            client = await self._get_client()
            response = await client.get(f"/ollama/pull/{job_id}", timeout=15.0)
            if response.status_code == 404:
                return {"error": "Pull job not found", "status": "error"}
            if response.status_code != 200:
                logger.warning("sidecar_pull_status_failed", status=response.status_code)
                return None
            return response.json()
        except httpx.TimeoutException:
            logger.warning("sidecar_pull_status_timeout", url=self.base_url)
            return None
        except Exception as e:
            logger.warning("sidecar_pull_status_error", url=self.base_url, error=str(e))
            return None

    async def ollama_delete(self, ollama_url: str, model: str) -> Optional[dict]:
        """Delete a model from an Ollama instance via the sidecar."""
        try:
            client = await self._get_client()
            response = await client.post(
                "/ollama/delete",
                json={"ollama_url": ollama_url, "model": model},
                timeout=120.0,
            )
            if response.status_code != 200:
                logger.warning("sidecar_ollama_delete_failed", status=response.status_code, body=response.text[:200])
                return {"error": response.text[:500], "status_code": response.status_code}
            return response.json()
        except httpx.TimeoutException:
            logger.warning("sidecar_ollama_delete_timeout", url=self.base_url)
            return None
        except Exception as e:
            logger.warning("sidecar_ollama_delete_error", url=self.base_url, error=str(e))
            return None

    def _parse_response(self, data: dict) -> SidecarResponse:
        """Parse raw sidecar response into SidecarResponse dataclass."""
        gpus = []
        for gpu_data in data.get("gpus", []):
            gpus.append(
                GPUDeviceSnapshot(
                    index=gpu_data.get("index", 0),
                    name=gpu_data.get("name"),
                    uuid=gpu_data.get("uuid"),
                    pci_bus_id=gpu_data.get("pci_bus_id"),
                    compute_capability=gpu_data.get("compute_capability"),
                    memory_total_gb=gpu_data.get("memory_total_gb"),
                    memory_used_gb=gpu_data.get("memory_used_gb"),
                    memory_free_gb=gpu_data.get("memory_free_gb"),
                    utilization_gpu=gpu_data.get("utilization_gpu"),
                    utilization_memory=gpu_data.get("utilization_memory"),
                    temperature_gpu=gpu_data.get("temperature_gpu"),
                    temperature_memory=gpu_data.get("temperature_memory"),
                    power_draw_watts=gpu_data.get("power_draw_watts"),
                    power_limit_watts=gpu_data.get("power_limit_watts"),
                    fan_speed_percent=gpu_data.get("fan_speed_percent"),
                    clock_sm_mhz=gpu_data.get("clock_sm_mhz"),
                    clock_memory_mhz=gpu_data.get("clock_memory_mhz"),
                )
            )

        return SidecarResponse(
            hostname=data.get("hostname"),
            driver_version=data.get("driver_version"),
            cuda_version=data.get("cuda_version"),
            gpu_count=data.get("gpu_count", 0),
            gpus=gpus,
            sidecar_version=data.get("sidecar_version"),
        )

