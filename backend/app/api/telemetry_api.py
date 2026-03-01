############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# telemetry_api.py: GPU and inference telemetry API endpoints
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Telemetry API endpoints for GPU and inference metrics."""

from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.auth import require_admin_or_session
from backend.app.core.telemetry.registry import get_registry
from backend.app.db import crud
from backend.app.db.models import User
from backend.app.db.session import get_async_db
from backend.app.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Resolution string to minutes mapping
RESOLUTION_MAP = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "6h": 360,
    "1d": 1440,
}


def _parse_time_range(
    start: Optional[str],
    end: Optional[str],
    range_str: Optional[str] = None,
) -> tuple[datetime, datetime]:
    """Parse time range from query params."""
    now = datetime.now(timezone.utc)

    if range_str:
        range_map = {
            "1h": timedelta(hours=1),
            "6h": timedelta(hours=6),
            "24h": timedelta(hours=24),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30),
        }
        delta = range_map.get(range_str, timedelta(hours=1))
        return now - delta, now

    start_dt = datetime.fromisoformat(start) if start else now - timedelta(hours=1)
    end_dt = datetime.fromisoformat(end) if end else now

    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)

    return start_dt, end_dt


def _build_gpu_info(device, gpu_t):
    """Build a GPU info dict from device and telemetry."""
    return {
        "id": device.id,
        "index": device.gpu_index,
        "name": device.name,
        "uuid": device.uuid,
        "compute_capability": device.compute_capability,
        "memory_total_gb": device.memory_total_gb,
        "power_limit_watts": device.power_limit_watts,
        "utilization_gpu": gpu_t.utilization_gpu if gpu_t else None,
        "utilization_memory": gpu_t.utilization_memory if gpu_t else None,
        "memory_used_gb": gpu_t.memory_used_gb if gpu_t else None,
        "memory_free_gb": gpu_t.memory_free_gb if gpu_t else None,
        "temperature_gpu": gpu_t.temperature_gpu if gpu_t else None,
        "temperature_memory": gpu_t.temperature_memory if gpu_t else None,
        "power_draw_watts": gpu_t.power_draw_watts if gpu_t else None,
        "fan_speed_percent": gpu_t.fan_speed_percent if gpu_t else None,
        "clock_sm_mhz": gpu_t.clock_sm_mhz if gpu_t else None,
        "clock_memory_mhz": gpu_t.clock_memory_mhz if gpu_t else None,
    }


@router.get("/overview")
async def telemetry_overview(
    admin: User = Depends(require_admin_or_session()),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Cluster-wide telemetry overview.

    Returns nodes (with all GPUs), backends (with assigned GPU subset),
    and aggregate cluster metrics. GPU totals come from nodes (deduplicated).
    """
    registry = get_registry()
    backends = await registry.get_all_backends()
    nodes = await registry.get_all_nodes()

    cluster = {
        "total_backends": len(backends),
        "healthy_backends": 0,
        "total_nodes": len(nodes),
        "total_gpus": 0,
        "total_gpu_memory_gb": 0.0,
        "used_gpu_memory_gb": 0.0,
        "avg_gpu_utilization": None,
        "total_power_draw_watts": 0.0,
        "total_active_requests": 0,
        "total_queued_requests": 0,
    }

    all_utils = []

    # Build nodes list with all GPUs
    node_list = []
    node_gpu_telemetry_map = {}  # node_id -> {device_id: telemetry}

    for node in nodes:
        gpu_devices = await crud.get_gpu_devices_for_node(db, node.id)
        latest_gpu_telemetry = await crud.get_latest_gpu_device_telemetry(db, node.id)
        gpu_telemetry_map = {t.gpu_device_id: t for t in latest_gpu_telemetry}
        node_gpu_telemetry_map[node.id] = gpu_telemetry_map

        gpu_list = []
        for device in gpu_devices:
            gpu_t = gpu_telemetry_map.get(device.id)
            gpu_list.append(_build_gpu_info(device, gpu_t))

            # Aggregate into cluster totals (from nodes = deduplicated)
            if device.memory_total_gb:
                cluster["total_gpu_memory_gb"] += device.memory_total_gb
            if gpu_t and gpu_t.memory_used_gb:
                cluster["used_gpu_memory_gb"] += gpu_t.memory_used_gb
            if gpu_t and gpu_t.utilization_gpu is not None:
                all_utils.append(gpu_t.utilization_gpu)
            if gpu_t and gpu_t.power_draw_watts:
                cluster["total_power_draw_watts"] += gpu_t.power_draw_watts

        cluster["total_gpus"] += len(gpu_devices) or (node.gpu_count or 0)

        node_list.append({
            "id": node.id,
            "name": node.name,
            "hostname": node.hostname,
            "status": node.status.value,
            "sidecar_url": node.sidecar_url,
            "gpu_count": node.gpu_count or len(gpu_devices),
            "driver_version": node.driver_version,
            "cuda_version": node.cuda_version,
            "gpus": gpu_list,
        })

    # Build backends list with assigned GPU subset
    backend_list = []
    for backend in backends:
        if backend.status.value == "healthy":
            cluster["healthy_backends"] += 1

        telemetry = await registry.get_telemetry(backend.id)
        models = await registry.get_backend_models(backend.id)

        if telemetry:
            cluster["total_active_requests"] += telemetry.active_requests
            cluster["total_queued_requests"] += telemetry.queued_requests

        # Get this backend's assigned GPUs
        gpu_list = []
        if backend.node_id and backend.node_id in node_gpu_telemetry_map:
            backend_gpu_devices = await crud.get_gpu_devices_for_backend(db, backend.id)
            gpu_telemetry_map = node_gpu_telemetry_map[backend.node_id]
            for device in backend_gpu_devices:
                gpu_t = gpu_telemetry_map.get(device.id)
                gpu_list.append(_build_gpu_info(device, gpu_t))

        # Resolve node name
        node_name = None
        if backend.node:
            node_name = backend.node.name

        backend_info = {
            "id": backend.id,
            "name": backend.name,
            "status": backend.status.value,
            "engine": backend.engine.value,
            "url": backend.url,
            "node_id": backend.node_id,
            "node_name": node_name,
            "gpu_indices": backend.gpu_indices,
            "gpu_type": backend.gpu_type,
            "version": backend.version,
            "max_concurrent": backend.max_concurrent,
            "current_concurrent": backend.current_concurrent,
            "gpus": gpu_list,
            "loaded_models": [m.name for m in models if m.is_loaded],
            "all_models": [m.name for m in models],
            "active_requests": telemetry.active_requests if telemetry else 0,
            "queued_requests": telemetry.queued_requests if telemetry else 0,
            "gpu_utilization": telemetry.gpu_utilization if telemetry else None,
            "gpu_memory_used_gb": telemetry.gpu_memory_used_gb if telemetry else None,
            "gpu_memory_total_gb": telemetry.gpu_memory_total_gb if telemetry else None,
            "latency_ema_ms": backend.latency_ema_ms,
            "ttft_ema_ms": backend.ttft_ema_ms,
        }
        backend_list.append(backend_info)

    if all_utils:
        cluster["avg_gpu_utilization"] = round(sum(all_utils) / len(all_utils), 1)

    cluster["total_gpu_memory_gb"] = round(cluster["total_gpu_memory_gb"], 1)
    cluster["used_gpu_memory_gb"] = round(cluster["used_gpu_memory_gb"], 1)
    cluster["total_power_draw_watts"] = round(cluster["total_power_draw_watts"], 1)

    return {
        "cluster": cluster,
        "nodes": node_list,
        "backends": backend_list,
    }


@router.get("/latest")
async def telemetry_latest(
    admin: User = Depends(require_admin_or_session()),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Lightweight endpoint for dashboard live polling.

    Returns the same structure as /overview but intended to be
    polled every 10 seconds by the dashboard JS.
    """
    # Reuse overview logic — it's fast enough for polling
    return await telemetry_overview(admin=admin, db=db)


@router.get("/backends/{backend_id}/history")
async def backend_telemetry_history(
    backend_id: int,
    metric: str = Query("gpu_utilization", description="Metric to retrieve"),
    start: Optional[str] = Query(None, description="Start time (ISO 8601)"),
    end: Optional[str] = Query(None, description="End time (ISO 8601)"),
    range: Optional[str] = Query(None, description="Time range: 1h, 6h, 24h, 7d, 30d"),
    resolution: str = Query("5m", description="Time resolution: 1m, 5m, 15m, 1h"),
    admin: User = Depends(require_admin_or_session()),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Get time-series telemetry history for a backend.

    Supports aggregated time buckets for efficient rendering of charts.
    """
    # Validate backend exists
    backend = await crud.get_backend_by_id(db, backend_id)
    if not backend:
        raise HTTPException(status_code=404, detail="Backend not found")

    start_dt, end_dt = _parse_time_range(start, end, range)
    res_minutes = RESOLUTION_MAP.get(resolution, 5)

    rows = await crud.get_backend_telemetry_history(
        db=db,
        backend_id=backend_id,
        start=start_dt,
        end=end_dt,
        resolution_minutes=res_minutes,
    )

    # Extract the requested metric from each row
    series = []
    for row in rows:
        point = {"timestamp": row["timestamp"]}

        if metric == "gpu_utilization":
            point["value"] = row.get("gpu_utilization")
            point["min"] = row.get("gpu_utilization_min")
            point["max"] = row.get("gpu_utilization_max")
        elif metric == "gpu_memory":
            point["value"] = row.get("gpu_memory_used_gb")
            point["total"] = row.get("gpu_memory_total_gb")
        elif metric == "power":
            point["value"] = row.get("gpu_power_draw_watts")
        elif metric == "temperature":
            point["value"] = row.get("gpu_temperature")
        elif metric == "requests":
            point["value"] = row.get("active_requests")
            point["queued"] = row.get("queued_requests")
        elif metric == "throughput":
            point["value"] = row.get("requests_per_second")
        else:
            # Return all metrics
            point.update(row)

        series.append(point)

    return {
        "backend_id": backend_id,
        "backend_name": backend.name,
        "metric": metric,
        "resolution": resolution,
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "series": series,
    }


@router.get("/gpus/{gpu_device_id}/history")
async def gpu_device_telemetry_history(
    gpu_device_id: int,
    metric: str = Query("utilization", description="Metric to retrieve"),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    range: Optional[str] = Query(None, description="Time range: 1h, 6h, 24h, 7d"),
    resolution: str = Query("5m"),
    admin: User = Depends(require_admin_or_session()),
    db: AsyncSession = Depends(get_async_db),
):
    """Get time-series telemetry history for a specific GPU device."""
    start_dt, end_dt = _parse_time_range(start, end, range)
    res_minutes = RESOLUTION_MAP.get(resolution, 5)

    rows = await crud.get_gpu_device_telemetry_history(
        db=db,
        gpu_device_id=gpu_device_id,
        start=start_dt,
        end=end_dt,
        resolution_minutes=res_minutes,
    )

    series = []
    for row in rows:
        point = {"timestamp": row["timestamp"]}

        if metric == "utilization":
            point["value"] = row.get("utilization_gpu")
            point["min"] = row.get("utilization_gpu_min")
            point["max"] = row.get("utilization_gpu_max")
            point["memory"] = row.get("utilization_memory")
        elif metric == "memory":
            point["value"] = row.get("memory_used_gb")
            point["free"] = row.get("memory_free_gb")
        elif metric == "power":
            point["value"] = row.get("power_draw_watts")
        elif metric == "temperature":
            point["value"] = row.get("temperature_gpu")
            point["memory"] = row.get("temperature_memory")
        elif metric == "clocks":
            point["sm"] = row.get("clock_sm_mhz")
            point["memory"] = row.get("clock_memory_mhz")
        else:
            point.update(row)

        series.append(point)

    return {
        "gpu_device_id": gpu_device_id,
        "metric": metric,
        "resolution": resolution,
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "series": series,
    }


@router.get("/nodes/{node_id}/history")
async def node_telemetry_history(
    node_id: int,
    metric: str = Query("utilization", description="Metric to retrieve"),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    range: Optional[str] = Query(None, description="Time range: 1h, 6h, 24h, 7d"),
    resolution: str = Query("5m"),
    admin: User = Depends(require_admin_or_session()),
    db: AsyncSession = Depends(get_async_db),
):
    """Get aggregated time-series telemetry history for a node (all GPUs)."""
    node = await crud.get_node_by_id(db, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    start_dt, end_dt = _parse_time_range(start, end, range)
    res_minutes = RESOLUTION_MAP.get(resolution, 5)

    # Get all GPU devices for this node
    gpu_devices = await crud.get_gpu_devices_for_node(db, node_id)
    if not gpu_devices:
        return {
            "node_id": node_id,
            "node_name": node.name,
            "metric": metric,
            "resolution": resolution,
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "series": [],
        }

    # Fetch history for each GPU and aggregate
    all_series = []
    for device in gpu_devices:
        rows = await crud.get_gpu_device_telemetry_history(
            db=db,
            gpu_device_id=device.id,
            start=start_dt,
            end=end_dt,
            resolution_minutes=res_minutes,
        )
        all_series.append(rows)

    # Aggregate across GPUs by timestamp
    ts_data = {}
    for rows in all_series:
        for row in rows:
            ts = row["timestamp"]
            if ts not in ts_data:
                ts_data[ts] = []
            ts_data[ts].append(row)

    series = []
    for ts in sorted(ts_data.keys()):
        rows = ts_data[ts]
        point = {"timestamp": ts}

        if metric == "utilization":
            vals = [r.get("utilization_gpu") for r in rows if r.get("utilization_gpu") is not None]
            point["value"] = sum(vals) / len(vals) if vals else None
        elif metric == "memory":
            used = [r.get("memory_used_gb") for r in rows if r.get("memory_used_gb") is not None]
            free = [r.get("memory_free_gb") for r in rows if r.get("memory_free_gb") is not None]
            point["value"] = sum(used) if used else None
            point["free"] = sum(free) if free else None
        elif metric == "power":
            vals = [r.get("power_draw_watts") for r in rows if r.get("power_draw_watts") is not None]
            point["value"] = sum(vals) if vals else None
        elif metric == "temperature":
            vals = [r.get("temperature_gpu") for r in rows if r.get("temperature_gpu") is not None]
            point["value"] = sum(vals) / len(vals) if vals else None
        else:
            # Average all numeric fields
            for key in ["utilization_gpu", "memory_used_gb", "power_draw_watts", "temperature_gpu"]:
                vals = [r.get(key) for r in rows if r.get(key) is not None]
                point[key] = sum(vals) / len(vals) if vals else None

        series.append(point)

    return {
        "node_id": node_id,
        "node_name": node.name,
        "metric": metric,
        "resolution": resolution,
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "series": series,
    }


@router.get("/export")
async def export_telemetry(
    backend_id: Optional[int] = Query(None),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    range: Optional[str] = Query("24h"),
    format: str = Query("json", description="Export format: json or csv"),
    admin: User = Depends(require_admin_or_session()),
    db: AsyncSession = Depends(get_async_db),
):
    """Export raw telemetry data as JSON or CSV."""
    start_dt, end_dt = _parse_time_range(start, end, range)

    if backend_id:
        rows = await crud.get_backend_telemetry_history(
            db=db,
            backend_id=backend_id,
            start=start_dt,
            end=end_dt,
            resolution_minutes=1,  # Raw resolution
        )
    else:
        # Export all backends
        registry = get_registry()
        all_backends = await registry.get_all_backends()
        rows = []
        for b in all_backends:
            b_rows = await crud.get_backend_telemetry_history(
                db=db,
                backend_id=b.id,
                start=start_dt,
                end=end_dt,
                resolution_minutes=1,
            )
            for r in b_rows:
                r["backend_id"] = b.id
                r["backend_name"] = b.name
            rows.extend(b_rows)

    if format == "csv":
        import csv
        import io

        output = io.StringIO()
        if rows:
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=telemetry_export.csv"},
        )

    return {
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "count": len(rows),
        "data": rows,
    }
