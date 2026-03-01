############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# test_gpu_agent.py: Unit tests for GPU sidecar agent
#
############################################################

"""Unit tests for the GPU sidecar agent with mocked pynvml."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---- Set required env var before importing the agent ----
TEST_SECRET_KEY = "test-secret-key-for-unit-tests"
os.environ["SIDECAR_SECRET_KEY"] = TEST_SECRET_KEY

# ---- Create mock pynvml before importing the agent ----
mock_pynvml = MagicMock()
mock_pynvml.NVML_TEMPERATURE_GPU = 0
mock_pynvml.NVML_CLOCK_SM = 0
mock_pynvml.NVML_CLOCK_MEM = 1
sys.modules["pynvml"] = mock_pynvml

# Now import the agent
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from gpu_agent import app, _init_nvml, _get_gpu_info
import gpu_agent

# Auth header for all authenticated requests
AUTH_HEADER = {"X-Sidecar-Key": TEST_SECRET_KEY}


# ---- Fixtures ----
@pytest.fixture(autouse=True)
def reset_agent_state():
    """Reset agent global state and mock side_effects before each test."""
    gpu_agent._initialized = False
    gpu_agent._init_error = None
    gpu_agent._driver_version = None
    gpu_agent._cuda_version = None
    gpu_agent._device_count = 0
    # Reset all mock side_effects so they don't leak between tests
    mock_pynvml.nvmlInit.side_effect = None
    mock_pynvml.nvmlInit.return_value = None
    mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = None
    mock_pynvml.nvmlDeviceGetTemperature.side_effect = None
    mock_pynvml.nvmlDeviceGetFanSpeed.side_effect = None
    yield


@pytest.fixture
def client():
    """Create a test client for the sidecar agent."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def mock_gpu_handle():
    """Create a mock GPU handle with realistic values."""
    handle = MagicMock()
    return handle


def setup_mock_pynvml(device_count=2):
    """Configure mock pynvml with realistic GPU data."""
    mock_pynvml.nvmlInit.return_value = None
    mock_pynvml.nvmlSystemGetDriverVersion.return_value = "550.54.15"
    mock_pynvml.nvmlSystemGetCudaDriverVersion_v2.return_value = 12040
    mock_pynvml.nvmlDeviceGetCount.return_value = device_count

    handles = [MagicMock() for _ in range(device_count)]
    mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = lambda i: handles[i]

    for i, handle in enumerate(handles):
        mock_pynvml.nvmlDeviceGetName.side_effect = lambda h: "NVIDIA A100-SXM4-80GB"
        mock_pynvml.nvmlDeviceGetUUID.side_effect = lambda h: f"GPU-uuid-{id(h)}"

        pci = MagicMock()
        pci.busId = b"00000000:3B:00.0"
        mock_pynvml.nvmlDeviceGetPciInfo.return_value = pci
        mock_pynvml.nvmlDeviceGetCudaComputeCapability.return_value = (8, 0)

        mem = MagicMock()
        mem.total = 80 * (1024**3)
        mem.used = 45 * (1024**3)
        mem.free = 35 * (1024**3)
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mem

        util = MagicMock()
        util.gpu = 73
        util.memory = 56
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = util

        mock_pynvml.nvmlDeviceGetTemperature.return_value = 62
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 285000  # milliwatts
        mock_pynvml.nvmlDeviceGetPowerManagementLimit.return_value = 400000
        mock_pynvml.nvmlDeviceGetFanSpeed.return_value = 45
        mock_pynvml.nvmlDeviceGetClockInfo.return_value = 1410
        mock_pynvml.nvmlDeviceGetComputeRunningProcesses.return_value = []

    return handles


class TestAuth:
    """Test sidecar key authentication."""

    def test_health_returns_401_without_key(self, client):
        setup_mock_pynvml(device_count=1)
        _init_nvml()

        response = client.get("/health")
        assert response.status_code == 401

    def test_health_returns_401_with_wrong_key(self, client):
        setup_mock_pynvml(device_count=1)
        _init_nvml()

        response = client.get("/health", headers={"X-Sidecar-Key": "wrong-key"})
        assert response.status_code == 401

    def test_health_returns_200_with_correct_key(self, client):
        setup_mock_pynvml(device_count=1)
        _init_nvml()

        response = client.get("/health", headers=AUTH_HEADER)
        assert response.status_code == 200

    def test_gpu_info_returns_401_without_key(self, client):
        setup_mock_pynvml(device_count=1)
        _init_nvml()

        response = client.get("/gpu-info")
        assert response.status_code == 401

    def test_gpu_info_returns_401_with_wrong_key(self, client):
        setup_mock_pynvml(device_count=1)
        _init_nvml()

        response = client.get("/gpu-info", headers={"X-Sidecar-Key": "bad"})
        assert response.status_code == 401

    def test_gpu_info_returns_200_with_correct_key(self, client):
        setup_mock_pynvml(device_count=1)
        _init_nvml()

        response = client.get("/gpu-info", headers=AUTH_HEADER)
        assert response.status_code == 200


class TestHealth:
    """Test /health endpoint."""

    def test_health_when_initialized(self, client):
        setup_mock_pynvml(device_count=2)
        _init_nvml()

        response = client.get("/health", headers=AUTH_HEADER)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["gpu_count"] == 2

    def test_health_when_not_initialized(self, client):
        response = client.get("/health", headers=AUTH_HEADER)
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "error"

    def test_health_after_init_failure(self, client):
        mock_pynvml.nvmlInit.side_effect = Exception("No NVIDIA driver")
        _init_nvml()

        response = client.get("/health", headers=AUTH_HEADER)
        assert response.status_code == 503
        data = response.json()
        assert "No NVIDIA driver" in data["error"]


class TestGpuInfo:
    """Test /gpu-info endpoint."""

    def test_gpu_info_when_not_initialized(self, client):
        response = client.get("/gpu-info", headers=AUTH_HEADER)
        assert response.status_code == 503
        data = response.json()
        assert data["gpu_count"] == 0
        assert data["gpus"] == []

    def test_gpu_info_returns_all_gpus(self, client):
        setup_mock_pynvml(device_count=2)
        _init_nvml()

        response = client.get("/gpu-info", headers=AUTH_HEADER)
        assert response.status_code == 200
        data = response.json()

        assert data["gpu_count"] == 2
        assert data["driver_version"] == "550.54.15"
        assert data["cuda_version"] == "12.4"
        assert len(data["gpus"]) == 2
        assert "hostname" in data
        assert "timestamp" in data

    def test_gpu_info_contains_metrics(self, client):
        setup_mock_pynvml(device_count=1)
        _init_nvml()

        response = client.get("/gpu-info", headers=AUTH_HEADER)
        data = response.json()
        gpu = data["gpus"][0]

        assert gpu["index"] == 0
        assert gpu["name"] == "NVIDIA A100-SXM4-80GB"
        assert gpu["compute_capability"] == "8.0"
        assert gpu["memory_total_gb"] == 80.0
        assert gpu["utilization_gpu"] == 73
        assert gpu["temperature_gpu"] == 62
        assert gpu["power_draw_watts"] == 285.0
        assert gpu["power_limit_watts"] == 400.0
        assert gpu["fan_speed_percent"] == 45
        assert gpu["clock_sm_mhz"] == 1410

    def test_gpu_info_handles_per_gpu_failure(self, client):
        setup_mock_pynvml(device_count=2)
        _init_nvml()

        # Make second GPU's handle raise
        original_side_effect = mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect

        def failing_handle(i):
            if i == 1:
                raise Exception("GPU 1 failed")
            return original_side_effect(i)

        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = failing_handle

        response = client.get("/gpu-info", headers=AUTH_HEADER)
        assert response.status_code == 200
        data = response.json()

        assert len(data["gpus"]) == 2
        # First GPU should be fine
        assert data["gpus"][0]["name"] == "NVIDIA A100-SXM4-80GB"
        # Second GPU should have error
        assert "error" in data["gpus"][1]


class TestInitNvml:
    """Test _init_nvml function."""

    def test_successful_init(self):
        setup_mock_pynvml(device_count=4)
        _init_nvml()

        assert gpu_agent._initialized is True
        assert gpu_agent._driver_version == "550.54.15"
        assert gpu_agent._cuda_version == "12.4"
        assert gpu_agent._device_count == 4

    def test_cuda_version_conversion(self):
        """Test that integer CUDA version is converted to string."""
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlSystemGetDriverVersion.return_value = "550.54.15"
        mock_pynvml.nvmlSystemGetCudaDriverVersion_v2.return_value = 11080
        mock_pynvml.nvmlDeviceGetCount.return_value = 1

        _init_nvml()

        assert gpu_agent._cuda_version == "11.8"

    def test_failed_init(self):
        mock_pynvml.nvmlInit.side_effect = Exception("No NVIDIA driver found")
        _init_nvml()

        assert gpu_agent._initialized is False
        assert "No NVIDIA driver found" in gpu_agent._init_error


class TestGetGpuInfoFunction:
    """Test _get_gpu_info helper function."""

    def test_collects_all_fields(self):
        handles = setup_mock_pynvml(device_count=1)

        info = _get_gpu_info(0)

        assert info["index"] == 0
        assert info["name"] == "NVIDIA A100-SXM4-80GB"
        assert info["compute_capability"] == "8.0"
        assert info["memory_total_gb"] == 80.0
        assert info["utilization_gpu"] == 73
        assert info["processes"] == []

    def test_handles_individual_metric_failures(self):
        """Each metric failure should result in None, not crash."""
        handles = setup_mock_pynvml(device_count=1)

        # Make temperature fail
        mock_pynvml.nvmlDeviceGetTemperature.side_effect = Exception("Not supported")
        # Make fan speed fail
        mock_pynvml.nvmlDeviceGetFanSpeed.side_effect = Exception("Not supported")

        info = _get_gpu_info(0)

        # Should still get other metrics
        assert info["name"] == "NVIDIA A100-SXM4-80GB"
        assert info["memory_total_gb"] == 80.0
        # Failed metrics should be None
        assert info["temperature_gpu"] is None
        assert info["fan_speed_percent"] is None
