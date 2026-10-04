############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# decision.py: System One decision-server backend adapter
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Adapter for a System One decision server (Clef behind clef_service, Laya).

A decision server answers ``POST /v1/systemone`` and nothing else, so it is
not a chat backend. Registering it with engine ``decision`` makes it a fleet
member: health-polled, shown on the backends page, alerted on, and (through
the node's GPU sidecar) charted, exactly like the DLP scan service. Like DLP
it discovers ZERO models, which keeps it out of chat routing and the model
catalog. ``/v1/systemone`` reaches it through ``decisions.upstreams``, matched
by URL, and consults this health status before dialing (see decisions_api).

Health is ``GET /health`` without a key: 200 with ``{"status": "ok"}``.
clef_service reports ``"loading"`` while the weights load, which is NOT
healthy. A server whose health body has no ``status`` field is healthy on 200.
"""

import time
from typing import Optional

import httpx

from backend.app.core.telemetry.models import (
    ERROR_KIND_TIMEOUT,
    BackendCapabilities,
    BackendHealth,
    TelemetrySnapshot,
    classify_transport_error,
)
from backend.app.logging_config import get_logger

logger = get_logger(__name__)


class DecisionAdapter:
    """Health and telemetry for a System One decision server. Same interface
    as VLLMAdapter / DlpAdapter so the registry treats it uniformly."""

    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def health_check(self) -> BackendHealth:
        start = time.monotonic()
        try:
            client = await self._get_client()
            response = await client.get("/health")
            latency_ms = (time.monotonic() - start) * 1000
            if response.status_code != 200:
                return BackendHealth(is_healthy=False, status_code=response.status_code, latency_ms=latency_ms,
                                     error_message=f"HTTP {response.status_code}")
            state = self._state(response)
            if state not in (None, "ok"):
                return BackendHealth(is_healthy=False, status_code=200, latency_ms=latency_ms,
                                     error_message=f"not ready ({str(state)[:40]})")
            return BackendHealth(is_healthy=True, status_code=200, latency_ms=latency_ms)
        except httpx.TimeoutException:
            return BackendHealth(is_healthy=False, latency_ms=(time.monotonic() - start) * 1000,
                                 error_message="Connection timeout", error_kind=ERROR_KIND_TIMEOUT)
        except Exception as e:
            return BackendHealth(is_healthy=False, latency_ms=(time.monotonic() - start) * 1000,
                                 error_message=str(e), error_kind=classify_transport_error(e))

    async def discover_capabilities(self) -> BackendCapabilities:
        """ALWAYS zero models: that is what keeps a decision server out of
        chat routing and the model catalog."""
        caps = BackendCapabilities()
        caps.models = []
        caps.loaded_models = []
        health = await self.health_check()
        caps.is_healthy = health.is_healthy
        if not health.is_healthy:
            caps.error_message = health.error_message
        return caps

    async def get_telemetry(self, backend_id: int) -> TelemetrySnapshot:
        """Liveness only; GPU and power come from the node's sidecar. Never raises."""
        snapshot = TelemetrySnapshot(backend_id=backend_id)
        snapshot.is_healthy = (await self.health_check()).is_healthy
        return snapshot

    @staticmethod
    def _state(response: httpx.Response) -> Optional[str]:
        """The health body's ``status`` value, or None when there is none."""
        try:
            data = response.json()
        except Exception:
            return None
        return data.get("status") if isinstance(data, dict) else None
