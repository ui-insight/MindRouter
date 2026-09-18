############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# dlp.py: DLP (GLiNER) scan-service backend adapter
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""DLP (GLiNER) backend adapter for capability discovery and telemetry.

The DLP scan service (dlp_service/) is a fleet member for status and — through
the per-node GPU sidecar — GPU/power telemetry, but it serves NO models and
must NEVER be eligible for inference routing or appear in the model catalog.

To guarantee that, discover_capabilities() ALWAYS returns an empty `models`
list: crud.get_backends_with_model requires a `models` row, the /v1/models
catalog only lists backends that have `models` rows, and the scheduler only
scores a backend for a model that is in its model set. A backend that
discovers zero models is therefore automatically non-routable and invisible.

DLP service endpoints (see dlp_service/server.py):
  - GET  /healthz  (NO auth) -> {status, model, device, replicas, warm, ...}
  - POST /scan     (X-Worker-Key)   -- not used here
  - GET  /stats    (X-Worker-Key)   -- not used here (would 401 without key)

Note the health path is /healthz, distinct from vLLM's /health. "warm" is the
liveness signal: a 200 with warm=false means the process is up but the model is
still loading, which we report as NOT healthy.
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


class DlpAdapter:
    """
    Adapter for DLP (GLiNER) scan-service health and telemetry.

    Mirrors the VLLMAdapter interface (__init__, _get_client, close,
    health_check, discover_capabilities, get_telemetry) so the registry can
    treat it uniformly — but it deliberately discovers zero models so the DLP
    backend never becomes routable or catalog-visible.

    TLS: the DLP node presents a valid certificate, so we use httpx's default
    verification exactly as VLLMAdapter does.
    """

    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
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
        Perform a health check on the DLP scan service.

        Authoritative liveness: healthy only when the service answers HTTP 200
        on /healthz AND reports warm=True. A 200 with warm=False means the
        process is up but the GLiNER model is still loading — report NOT
        healthy so the fleet does not treat a cold service as ready.

        Returns:
            BackendHealth result
        """
        start_time = time.monotonic()
        try:
            client = await self._get_client()

            response = await client.get("/healthz")
            latency_ms = (time.monotonic() - start_time) * 1000

            if response.status_code == 200:
                warm = self._is_warm(response)
                if warm:
                    return BackendHealth(
                        is_healthy=True,
                        status_code=response.status_code,
                        latency_ms=latency_ms,
                    )
                return BackendHealth(
                    is_healthy=False,
                    status_code=response.status_code,
                    latency_ms=latency_ms,
                    error_message="not warm (model loading)",
                )

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
                error_kind=ERROR_KIND_TIMEOUT,
            )
        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            return BackendHealth(
                is_healthy=False,
                latency_ms=latency_ms,
                error_message=str(e),
                error_kind=classify_transport_error(e),
            )

    async def discover_capabilities(self) -> BackendCapabilities:
        """
        Discover backend capabilities.

        ALWAYS returns models=[] and loaded_models=[]. This is the mechanism
        that keeps the DLP backend out of inference routing and the model
        catalog. `engine_version` is filled from the /healthz "model" field so
        the fleet UI can show which GLiNER model is loaded.

        Returns:
            BackendCapabilities with is_healthy set, and NO models.
        """
        caps = BackendCapabilities()
        # Invariant: a DLP backend serves no models. Do not populate these.
        caps.models = []
        caps.loaded_models = []

        try:
            client = await self._get_client()
            response = await client.get("/healthz")

            if response.status_code == 200:
                caps.is_healthy = self._is_warm(response)
                body = self._json(response)
                caps.engine_version = body.get("model") or None
                if not caps.is_healthy:
                    caps.error_message = "not warm (model loading)"
            else:
                caps.is_healthy = False
                caps.error_message = f"HTTP {response.status_code}"

        except Exception as e:
            logger.warning("dlp_capability_discovery_failed", error=str(e))
            caps.is_healthy = False
            caps.error_message = str(e)

        return caps

    async def get_telemetry(self, backend_id: int) -> TelemetrySnapshot:
        """
        Get a minimal, best-effort telemetry snapshot.

        Deliberately cheap and never raises. We do NOT call /stats (it requires
        the worker key and would 401); GPU/power fields are supplied by the
        per-node sidecar, not here. We only probe /healthz for liveness.

        Args:
            backend_id: Backend ID for the snapshot

        Returns:
            TelemetrySnapshot with is_healthy set and no GPU fields.
        """
        snapshot = TelemetrySnapshot(backend_id=backend_id)
        try:
            client = await self._get_client()
            response = await client.get("/healthz")
            snapshot.is_healthy = (
                response.status_code == 200 and self._is_warm(response)
            )
        except Exception as e:
            logger.debug("dlp_telemetry_failed", error=str(e))
            snapshot.is_healthy = False

        return snapshot

    @staticmethod
    def _json(response: httpx.Response) -> dict:
        """Parse a JSON body, tolerating a non-JSON response."""
        try:
            data = response.json()
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @classmethod
    def _is_warm(cls, response: httpx.Response) -> bool:
        """True only when the /healthz body explicitly reports warm is True."""
        return cls._json(response).get("warm") is True
