############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# test_sidecar_client.py: Unit tests for GPU sidecar client
#
############################################################

"""Unit tests for SidecarClient."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---- isolation: bypass heavy package __init__ imports ----
for mod_name in [
    "backend.app.db",
    "backend.app.db.session",
    "backend.app.db.models",
    "backend.app.db.crud",
    "backend.app.settings",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# Load telemetry models directly to get real dataclasses
_models_path = Path(__file__).resolve().parents[2] / "core" / "telemetry" / "models.py"
_spec = importlib.util.spec_from_file_location("telemetry_models", _models_path)
telemetry_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(telemetry_models)

GPUDeviceSnapshot = telemetry_models.GPUDeviceSnapshot
SidecarResponse = telemetry_models.SidecarResponse

# Ensure the telemetry models module is accessible
sys.modules["backend.app.core.telemetry.models"] = telemetry_models

# Mock logging
mock_logging = MagicMock()
mock_logging.get_logger = MagicMock(return_value=MagicMock())
sys.modules["backend.app.logging_config"] = mock_logging

# Now load the sidecar client module directly
_client_path = (
    Path(__file__).resolve().parents[2]
    / "core"
    / "telemetry"
    / "adapters"
    / "sidecar_client.py"
)
_client_spec = importlib.util.spec_from_file_location("sidecar_client", _client_path)
sidecar_client_mod = importlib.util.module_from_spec(_client_spec)
_client_spec.loader.exec_module(sidecar_client_mod)

SidecarClient = sidecar_client_mod.SidecarClient


# ---- Sample sidecar response data ----
SAMPLE_GPU_RESPONSE = {
    "hostname": "gpu-node-01",
    "timestamp": "2026-02-09T20:30:00Z",
    "driver_version": "550.54.15",
    "cuda_version": "12.4",
    "gpu_count": 2,
    "gpus": [
        {
            "index": 0,
            "name": "NVIDIA A100-SXM4-80GB",
            "uuid": "GPU-abc123",
            "pci_bus_id": "00000000:3B:00.0",
            "compute_capability": "8.0",
            "memory_total_gb": 80.0,
            "memory_used_gb": 45.2,
            "memory_free_gb": 34.8,
            "utilization_gpu": 73.0,
            "utilization_memory": 56.5,
            "temperature_gpu": 62,
            "temperature_memory": 58,
            "power_draw_watts": 285.0,
            "power_limit_watts": 400.0,
            "fan_speed_percent": 45,
            "clock_sm_mhz": 1410,
            "clock_memory_mhz": 1593,
        },
        {
            "index": 1,
            "name": "NVIDIA A100-SXM4-80GB",
            "uuid": "GPU-def456",
            "pci_bus_id": "00000000:86:00.0",
            "compute_capability": "8.0",
            "memory_total_gb": 80.0,
            "memory_used_gb": 12.0,
            "memory_free_gb": 68.0,
            "utilization_gpu": 15.0,
            "utilization_memory": 10.0,
            "temperature_gpu": 48,
            "temperature_memory": 44,
            "power_draw_watts": 120.0,
            "power_limit_watts": 400.0,
            "fan_speed_percent": 30,
            "clock_sm_mhz": 1200,
            "clock_memory_mhz": 1593,
        },
    ],
}


class TestSidecarClientInit:
    """Test SidecarClient initialization."""

    def test_strips_trailing_slash(self):
        client = SidecarClient("http://gpu-node:8007/")
        assert client.base_url == "http://gpu-node:8007"

    def test_sets_timeout(self):
        client = SidecarClient("http://gpu-node:8007", timeout=10.0)
        assert client.timeout == 10.0

    def test_default_timeout(self):
        client = SidecarClient("http://gpu-node:8007")
        assert client.timeout == 5.0

    def test_stores_sidecar_key(self):
        client = SidecarClient("http://gpu-node:8007", sidecar_key="secret123")
        assert client.sidecar_key == "secret123"

    def test_default_sidecar_key_is_none(self):
        client = SidecarClient("http://gpu-node:8007")
        assert client.sidecar_key is None


class TestParseResponse:
    """Test _parse_response method."""

    def test_parses_full_response(self):
        client = SidecarClient("http://gpu-node:8007")
        result = client._parse_response(SAMPLE_GPU_RESPONSE)

        assert isinstance(result, SidecarResponse)
        assert result.hostname == "gpu-node-01"
        assert result.driver_version == "550.54.15"
        assert result.cuda_version == "12.4"
        assert result.gpu_count == 2
        assert len(result.gpus) == 2

    def test_parses_gpu_details(self):
        client = SidecarClient("http://gpu-node:8007")
        result = client._parse_response(SAMPLE_GPU_RESPONSE)

        gpu0 = result.gpus[0]
        assert isinstance(gpu0, GPUDeviceSnapshot)
        assert gpu0.index == 0
        assert gpu0.name == "NVIDIA A100-SXM4-80GB"
        assert gpu0.uuid == "GPU-abc123"
        assert gpu0.compute_capability == "8.0"
        assert gpu0.memory_total_gb == 80.0
        assert gpu0.memory_used_gb == 45.2
        assert gpu0.utilization_gpu == 73.0
        assert gpu0.temperature_gpu == 62
        assert gpu0.power_draw_watts == 285.0
        assert gpu0.clock_sm_mhz == 1410

    def test_parses_second_gpu(self):
        client = SidecarClient("http://gpu-node:8007")
        result = client._parse_response(SAMPLE_GPU_RESPONSE)

        gpu1 = result.gpus[1]
        assert gpu1.index == 1
        assert gpu1.uuid == "GPU-def456"
        assert gpu1.utilization_gpu == 15.0
        assert gpu1.memory_used_gb == 12.0

    def test_handles_empty_gpus(self):
        client = SidecarClient("http://gpu-node:8007")
        result = client._parse_response({"hostname": "node", "gpu_count": 0, "gpus": []})

        assert result.gpu_count == 0
        assert result.gpus == []

    def test_handles_missing_fields(self):
        client = SidecarClient("http://gpu-node:8007")
        data = {
            "hostname": "node",
            "gpus": [{"index": 0, "name": "Test GPU"}],
        }
        result = client._parse_response(data)

        assert result.gpu_count == 0
        assert result.driver_version is None
        assert result.cuda_version is None
        assert len(result.gpus) == 1
        assert result.gpus[0].uuid is None
        assert result.gpus[0].memory_total_gb is None


class TestSidecarKeyHeader:
    """Test that X-Sidecar-Key header is sent when a key is configured."""

    @pytest.mark.asyncio
    async def test_get_client_includes_key_header(self):
        client = SidecarClient("http://gpu-node:8007", sidecar_key="my-secret")
        http_client = await client._get_client()
        assert http_client.headers.get("X-Sidecar-Key") == "my-secret"  # type: ignore[union-attr]
        await client.close()

    @pytest.mark.asyncio
    async def test_get_client_no_key_header_when_none(self):
        client = SidecarClient("http://gpu-node:8007")
        http_client = await client._get_client()
        assert "X-Sidecar-Key" not in http_client.headers  # type: ignore[union-attr]
        await client.close()


class TestGetGpuInfo:
    """Test get_gpu_info HTTP interaction."""

    @pytest.mark.asyncio
    async def test_successful_fetch(self):
        client = SidecarClient("http://gpu-node:8007")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_GPU_RESPONSE

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=mock_response)
        mock_http_client.is_closed = False
        client._client = mock_http_client

        result = await client.get_gpu_info()

        assert result is not None
        assert isinstance(result, SidecarResponse)
        assert result.gpu_count == 2
        mock_http_client.get.assert_called_once_with("/gpu-info")

    @pytest.mark.asyncio
    async def test_bad_status_returns_none(self):
        client = SidecarClient("http://gpu-node:8007")

        mock_response = MagicMock()
        mock_response.status_code = 503

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=mock_response)
        mock_http_client.is_closed = False
        client._client = mock_http_client

        result = await client.get_gpu_info()
        assert result is None

    @pytest.mark.asyncio
    async def test_timeout_returns_none(self):
        import httpx

        client = SidecarClient("http://gpu-node:8007")

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
        mock_http_client.is_closed = False
        client._client = mock_http_client

        result = await client.get_gpu_info()
        assert result is None

    @pytest.mark.asyncio
    async def test_connection_error_returns_none(self):
        import httpx

        client = SidecarClient("http://gpu-node:8007")

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        mock_http_client.is_closed = False
        client._client = mock_http_client

        result = await client.get_gpu_info()
        assert result is None


class TestHealthCheck:
    """Test health_check method."""

    @pytest.mark.asyncio
    async def test_healthy_returns_true(self):
        client = SidecarClient("http://gpu-node:8007")

        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=mock_response)
        mock_http_client.is_closed = False
        client._client = mock_http_client

        assert await client.health_check() is True

    @pytest.mark.asyncio
    async def test_unhealthy_returns_false(self):
        client = SidecarClient("http://gpu-node:8007")

        mock_response = MagicMock()
        mock_response.status_code = 503

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=mock_response)
        mock_http_client.is_closed = False
        client._client = mock_http_client

        assert await client.health_check() is False

    @pytest.mark.asyncio
    async def test_error_returns_false(self):
        client = SidecarClient("http://gpu-node:8007")

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(side_effect=Exception("unreachable"))
        mock_http_client.is_closed = False
        client._client = mock_http_client

        assert await client.health_check() is False


class TestClose:
    """Test client cleanup."""

    @pytest.mark.asyncio
    async def test_close_client(self):
        client = SidecarClient("http://gpu-node:8007")

        mock_http_client = AsyncMock()
        mock_http_client.is_closed = False
        client._client = mock_http_client

        await client.close()
        mock_http_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_when_already_closed(self):
        client = SidecarClient("http://gpu-node:8007")

        mock_http_client = AsyncMock()
        mock_http_client.is_closed = True
        client._client = mock_http_client

        await client.close()
        mock_http_client.aclose.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_when_no_client(self):
        client = SidecarClient("http://gpu-node:8007")
        await client.close()  # Should not raise
