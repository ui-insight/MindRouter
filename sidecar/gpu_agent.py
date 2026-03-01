############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# gpu_agent.py: GPU metrics sidecar agent using pynvml
#
# Runs on each GPU inference node to expose per-GPU hardware
# metrics via a lightweight HTTP API. MindRouter's backend
# registry polls this endpoint to collect telemetry.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""GPU metrics sidecar agent using NVIDIA Management Library (pynvml)."""

import asyncio
import os
import secrets
import socket
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel


def _read_sidecar_version() -> str:
    """Read version from VERSION file in the sidecar directory."""
    try:
        version_file = Path(__file__).resolve().parent / "VERSION"
        return version_file.read_text().strip()
    except Exception:
        return "0.0.0"


SIDECAR_VERSION = _read_sidecar_version()

# Require SIDECAR_SECRET_KEY at startup
SIDECAR_SECRET_KEY = os.environ.get("SIDECAR_SECRET_KEY", "").strip()
if not SIDECAR_SECRET_KEY:
    print(
        "FATAL: SIDECAR_SECRET_KEY environment variable is required but not set.\n"
        "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\"",
        file=sys.stderr,
    )
    sys.exit(1)


async def verify_sidecar_key(x_sidecar_key: Optional[str] = Header(None)) -> None:
    """Validate the X-Sidecar-Key header against the configured secret."""
    if x_sidecar_key is None or not secrets.compare_digest(
        x_sidecar_key, SIDECAR_SECRET_KEY
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing sidecar key")


app = FastAPI(title="MindRouter GPU Sidecar Agent", version=SIDECAR_VERSION)

# GPU state cached at startup
_initialized = False
_init_error: Optional[str] = None
_driver_version: Optional[str] = None
_cuda_version: Optional[str] = None
_device_count: int = 0


def _init_nvml() -> None:
    """Initialize NVML library and cache static info."""
    global _initialized, _init_error, _driver_version, _cuda_version, _device_count

    try:
        import pynvml
        pynvml.nvmlInit()
        _driver_version = pynvml.nvmlSystemGetDriverVersion()
        _cuda_version = pynvml.nvmlSystemGetCudaDriverVersion_v2()
        # Convert CUDA version from int (e.g., 12040) to string (e.g., "12.4")
        if isinstance(_cuda_version, int):
            major = _cuda_version // 1000
            minor = (_cuda_version % 1000) // 10
            _cuda_version = f"{major}.{minor}"
        _device_count = pynvml.nvmlDeviceGetCount()
        _initialized = True
    except Exception as e:
        _init_error = str(e)
        _initialized = False


def _get_gpu_info(index: int) -> Dict[str, Any]:
    """Collect metrics for a single GPU device."""
    import pynvml

    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    info: Dict[str, Any] = {"index": index}

    # Device name
    try:
        info["name"] = pynvml.nvmlDeviceGetName(handle)
    except Exception:
        info["name"] = None

    # UUID
    try:
        info["uuid"] = pynvml.nvmlDeviceGetUUID(handle)
    except Exception:
        info["uuid"] = None

    # PCI bus ID
    try:
        pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
        info["pci_bus_id"] = pci_info.busId.decode() if isinstance(pci_info.busId, bytes) else pci_info.busId
    except Exception:
        info["pci_bus_id"] = None

    # Compute capability
    try:
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        info["compute_capability"] = f"{major}.{minor}"
    except Exception:
        info["compute_capability"] = None

    # Memory
    try:
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        info["memory_total_gb"] = round(mem.total / (1024**3), 2)
        info["memory_used_gb"] = round(mem.used / (1024**3), 2)
        info["memory_free_gb"] = round(mem.free / (1024**3), 2)
    except Exception:
        info["memory_total_gb"] = None
        info["memory_used_gb"] = None
        info["memory_free_gb"] = None

    # Utilization
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        info["utilization_gpu"] = util.gpu
        info["utilization_memory"] = util.memory
    except Exception:
        info["utilization_gpu"] = None
        info["utilization_memory"] = None

    # Temperature
    try:
        info["temperature_gpu"] = pynvml.nvmlDeviceGetTemperature(
            handle, pynvml.NVML_TEMPERATURE_GPU
        )
    except Exception:
        info["temperature_gpu"] = None

    # Memory temperature (not available on all GPUs)
    try:
        info["temperature_memory"] = pynvml.nvmlDeviceGetTemperature(
            handle, 2  # NVML_TEMPERATURE_MEM = 2 (not always in pynvml constants)
        )
    except Exception:
        info["temperature_memory"] = None

    # Power
    try:
        info["power_draw_watts"] = round(
            pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0, 1
        )  # milliwatts -> watts
    except Exception:
        info["power_draw_watts"] = None

    try:
        info["power_limit_watts"] = round(
            pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0, 1
        )
    except Exception:
        info["power_limit_watts"] = None

    # Fan speed
    try:
        info["fan_speed_percent"] = pynvml.nvmlDeviceGetFanSpeed(handle)
    except Exception:
        info["fan_speed_percent"] = None

    # Clocks
    try:
        info["clock_sm_mhz"] = pynvml.nvmlDeviceGetClockInfo(
            handle, pynvml.NVML_CLOCK_SM
        )
    except Exception:
        info["clock_sm_mhz"] = None

    try:
        info["clock_memory_mhz"] = pynvml.nvmlDeviceGetClockInfo(
            handle, pynvml.NVML_CLOCK_MEM
        )
    except Exception:
        info["clock_memory_mhz"] = None

    # Running processes
    try:
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        info["processes"] = [
            {
                "pid": p.pid,
                "gpu_memory_used_mb": round(p.usedGpuMemory / (1024**2), 1)
                if p.usedGpuMemory
                else None,
            }
            for p in procs
        ]
    except Exception:
        info["processes"] = []

    return info


@app.on_event("startup")
async def startup():
    """Initialize NVML on startup."""
    _init_nvml()


@app.get("/health")
async def health(_: None = Depends(verify_sidecar_key)):
    """Health check endpoint."""
    if _initialized:
        return {"status": "ok", "gpu_count": _device_count, "sidecar_version": SIDECAR_VERSION}
    return JSONResponse(
        status_code=503,
        content={
            "status": "error",
            "error": _init_error or "NVML not initialized",
            "sidecar_version": SIDECAR_VERSION,
        },
    )


@app.get("/gpu-info")
async def gpu_info(_: None = Depends(verify_sidecar_key)):
    """Return detailed GPU metrics for all devices on this node."""
    if not _initialized:
        return JSONResponse(
            status_code=503,
            content={
                "error": _init_error or "NVML not initialized",
                "hostname": socket.gethostname(),
                "gpu_count": 0,
                "gpus": [],
            },
        )

    gpus: List[Dict[str, Any]] = []
    for i in range(_device_count):
        try:
            gpus.append(_get_gpu_info(i))
        except Exception as e:
            gpus.append({"index": i, "error": str(e)})

    return {
        "hostname": socket.gethostname(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "driver_version": _driver_version,
        "cuda_version": _cuda_version,
        "gpu_count": _device_count,
        "gpus": gpus,
        "sidecar_version": SIDECAR_VERSION,
    }


# ---------------------------------------------------------------------------
# Ollama model management endpoints
# ---------------------------------------------------------------------------

_pull_jobs: Dict[str, dict] = {}


class OllamaPullRequest(BaseModel):
    ollama_url: str
    model: str


class OllamaDeleteRequest(BaseModel):
    ollama_url: str
    model: str


async def _run_pull(job_id: str, ollama_url: str, model: str) -> None:
    """Background task: stream an Ollama pull and update job progress."""
    job = _pull_jobs[job_id]
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=None)) as client:
            async with client.stream(
                "POST",
                f"{ollama_url.rstrip('/')}/api/pull",
                json={"name": model, "stream": True},
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    job["status"] = "error"
                    job["error"] = f"Ollama returned {response.status_code}: {body.decode(errors='replace')[:500]}"
                    job["completed_at"] = datetime.now(timezone.utc).isoformat()
                    return

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        import json
                        data = json.loads(line)
                    except Exception:
                        continue

                    status_str = data.get("status", "")
                    job["progress"]["status"] = status_str

                    if "digest" in data:
                        job["progress"]["digest"] = data["digest"]
                    if "total" in data:
                        job["progress"]["total"] = data["total"]
                    if "completed" in data:
                        job["progress"]["completed"] = data["completed"]

                    if data.get("error"):
                        job["status"] = "error"
                        job["error"] = data["error"]
                        job["completed_at"] = datetime.now(timezone.utc).isoformat()
                        return

        job["status"] = "success"
        job["completed_at"] = datetime.now(timezone.utc).isoformat()

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        job["completed_at"] = datetime.now(timezone.utc).isoformat()


@app.post("/ollama/pull")
async def ollama_pull(req: OllamaPullRequest, _: None = Depends(verify_sidecar_key)):
    """Start a background model pull from the Ollama library."""
    job_id = str(uuid.uuid4())
    _pull_jobs[job_id] = {
        "job_id": job_id,
        "model": req.model,
        "ollama_url": req.ollama_url,
        "status": "pulling",
        "progress": {"status": "", "digest": None, "total": None, "completed": None},
        "error": None,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
    }
    asyncio.create_task(_run_pull(job_id, req.ollama_url, req.model))
    return {"job_id": job_id, "status": "pulling"}


@app.get("/ollama/pull/{job_id}")
async def ollama_pull_status(job_id: str, _: None = Depends(verify_sidecar_key)):
    """Poll progress of a model pull job."""
    job = _pull_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Pull job not found")
    return job


@app.post("/ollama/delete")
async def ollama_delete(req: OllamaDeleteRequest, _: None = Depends(verify_sidecar_key)):
    """Delete a model from an Ollama instance."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                "DELETE",
                f"{req.ollama_url.rstrip('/')}/api/delete",
                json={"name": req.model},
            )
        if response.status_code == 200:
            return {"status": "deleted", "model": req.model}
        elif response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Model '{req.model}' not found on Ollama instance")
        else:
            raise HTTPException(
                status_code=502,
                detail=f"Ollama returned {response.status_code}: {response.text[:500]}",
            )
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Failed to reach Ollama: {e}")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("GPU_AGENT_PORT", "8007"))
    host = os.environ.get("GPU_AGENT_HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
