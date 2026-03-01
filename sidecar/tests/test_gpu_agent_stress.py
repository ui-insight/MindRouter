############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# test_gpu_agent_stress.py: 60-second stress test for sidecar auth
#
############################################################

"""60-second stress test for sidecar key authentication.

Hammers /health and /gpu-info concurrently with valid keys, invalid keys,
and missing keys. Verifies auth holds up under sustained load.
"""

import os
import sys
import time
import threading
from collections import Counter
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


# ---- Set required env var before importing the agent ----
TEST_SECRET_KEY = "stress-test-secret-key-abc123"
os.environ["SIDECAR_SECRET_KEY"] = TEST_SECRET_KEY

# ---- Create mock pynvml before importing the agent ----
mock_pynvml = MagicMock()
mock_pynvml.NVML_TEMPERATURE_GPU = 0
mock_pynvml.NVML_CLOCK_SM = 0
mock_pynvml.NVML_CLOCK_MEM = 1
sys.modules["pynvml"] = mock_pynvml

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from gpu_agent import app, _init_nvml
import gpu_agent

AUTH_HEADER = {"X-Sidecar-Key": TEST_SECRET_KEY}
BAD_HEADER = {"X-Sidecar-Key": "wrong-key"}
DURATION_SECONDS = 60


def _setup_nvml():
    """Set up mock NVML with 2 GPUs."""
    gpu_agent._initialized = False
    mock_pynvml.nvmlInit.side_effect = None
    mock_pynvml.nvmlInit.return_value = None
    mock_pynvml.nvmlSystemGetDriverVersion.return_value = "550.54.15"
    mock_pynvml.nvmlSystemGetCudaDriverVersion_v2.return_value = 12040
    mock_pynvml.nvmlDeviceGetCount.return_value = 2
    mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = None
    mock_pynvml.nvmlDeviceGetTemperature.side_effect = None
    mock_pynvml.nvmlDeviceGetFanSpeed.side_effect = None

    handles = [MagicMock(), MagicMock()]
    mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = lambda i: handles[i]
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
    mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 285000
    mock_pynvml.nvmlDeviceGetPowerManagementLimit.return_value = 400000
    mock_pynvml.nvmlDeviceGetFanSpeed.return_value = 45
    mock_pynvml.nvmlDeviceGetClockInfo.return_value = 1410
    mock_pynvml.nvmlDeviceGetComputeRunningProcesses.return_value = []

    _init_nvml()


class TestStress60Seconds:
    """60-second concurrent stress test for sidecar auth."""

    def test_stress_auth_60s(self):
        _setup_nvml()
        client = TestClient(app, raise_server_exceptions=False)

        results = Counter()
        errors = []
        stop = threading.Event()

        def worker_valid_health():
            while not stop.is_set():
                try:
                    r = client.get("/health", headers=AUTH_HEADER)
                    results[("health", "valid", r.status_code)] += 1
                except Exception as e:
                    errors.append(("health_valid", str(e)))

        def worker_valid_gpu_info():
            while not stop.is_set():
                try:
                    r = client.get("/gpu-info", headers=AUTH_HEADER)
                    results[("gpu-info", "valid", r.status_code)] += 1
                except Exception as e:
                    errors.append(("gpu_info_valid", str(e)))

        def worker_bad_key():
            while not stop.is_set():
                try:
                    r = client.get("/health", headers=BAD_HEADER)
                    results[("health", "bad_key", r.status_code)] += 1
                except Exception as e:
                    errors.append(("health_bad", str(e)))

        def worker_no_key():
            while not stop.is_set():
                try:
                    r = client.get("/gpu-info")
                    results[("gpu-info", "no_key", r.status_code)] += 1
                except Exception as e:
                    errors.append(("gpu_info_no_key", str(e)))

        # Launch 8 threads: 2 valid health, 2 valid gpu-info, 2 bad key, 2 no key
        threads = []
        for fn in [
            worker_valid_health, worker_valid_health,
            worker_valid_gpu_info, worker_valid_gpu_info,
            worker_bad_key, worker_bad_key,
            worker_no_key, worker_no_key,
        ]:
            t = threading.Thread(target=fn, daemon=True)
            t.start()
            threads.append(t)

        # Run for 60 seconds
        time.sleep(DURATION_SECONDS)
        stop.set()

        for t in threads:
            t.join(timeout=5)

        # Print summary
        total = sum(results.values())
        print(f"\n{'='*60}")
        print(f"Stress test completed: {total:,} requests in {DURATION_SECONDS}s")
        print(f"  Rate: {total / DURATION_SECONDS:,.0f} req/s")
        print(f"  Errors: {len(errors)}")
        for key, count in sorted(results.items()):
            endpoint, auth_type, status = key
            print(f"  {endpoint:10s} {auth_type:8s} -> {status}: {count:>8,}")
        print(f"{'='*60}\n")

        # Assertions
        assert len(errors) == 0, f"Unexpected errors: {errors[:10]}"

        # All valid requests should return 200
        valid_health_200 = results.get(("health", "valid", 200), 0)
        valid_gpu_200 = results.get(("gpu-info", "valid", 200), 0)
        assert valid_health_200 > 0, "No successful health checks"
        assert valid_gpu_200 > 0, "No successful gpu-info requests"

        # No valid requests should get non-200
        for key, count in results.items():
            endpoint, auth_type, status_code = key
            if auth_type == "valid":
                assert status_code == 200, (
                    f"Valid key got status {status_code} on {endpoint} ({count} times)"
                )

        # All bad-key requests should return 401
        for key, count in results.items():
            endpoint, auth_type, status_code = key
            if auth_type in ("bad_key", "no_key"):
                assert status_code == 401, (
                    f"{auth_type} got status {status_code} on {endpoint} ({count} times)"
                )

        # Should have done a meaningful number of requests
        assert total > 1000, f"Only {total} requests in {DURATION_SECONDS}s — too slow"
