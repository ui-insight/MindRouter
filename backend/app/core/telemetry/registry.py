############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# registry.py: Backend registry for discovery, health, and telemetry
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Backend registry - manages backend discovery, health, and telemetry."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.telemetry.adapters.ollama import OllamaAdapter
from backend.app.core.telemetry.adapters.sidecar_client import SidecarClient
from backend.app.core.telemetry.adapters.vllm import VLLMAdapter
from backend.app.core.telemetry.latency_tracker import LatencyTracker
from backend.app.core.telemetry.models import (
    BackendCapabilities,
    BackendHealth,
    CircuitBreakerState,
    SidecarResponse,
    TelemetrySnapshot,
)
from backend.app.db import crud
from backend.app.db.models import Backend, BackendEngine, BackendStatus, Model, Modality, Node, NodeStatus
from backend.app.db.session import get_async_db_context
from backend.app.logging_config import get_logger
from backend.app.settings import get_settings

logger = get_logger(__name__)


class BackendRegistry:
    """
    Central registry for backend management.

    Responsibilities:
    - Maintain list of registered backends
    - Periodically poll backends for health and capabilities
    - Store telemetry snapshots
    - Provide backend/model lookup for scheduler
    """

    def __init__(self):
        self._settings = get_settings()
        self._adapters: Dict[int, OllamaAdapter | VLLMAdapter] = {}
        self._capabilities: Dict[int, BackendCapabilities] = {}
        self._telemetry: Dict[int, TelemetrySnapshot] = {}
        self._lock = asyncio.Lock()
        self._poll_task: Optional[asyncio.Task] = None
        self._persist_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._force_offline: bool = False

        # Sidecar clients keyed by node_id (one per physical server)
        self._sidecar_clients: Dict[int, SidecarClient] = {}

        # Node-backend mappings
        self._node_backends: Dict[int, set] = {}           # node_id → {backend_ids}
        self._backend_gpu_indices: Dict[int, Optional[list]] = {}  # backend_id → [0,1] or None

        # Latest per-node sidecar data (cached for backend telemetry phase)
        self._node_sidecar_data: Dict[int, Optional[SidecarResponse]] = {}

        # Circuit breaker state per backend
        self._circuit_breakers: Dict[int, CircuitBreakerState] = {}

        # Fast-poll set: backend_id -> fast_poll_until timestamp
        self._fast_poll_backends: Dict[int, datetime] = {}

        # Latency tracker
        self._latency_tracker = LatencyTracker(
            alpha=self._settings.latency_ema_alpha
        )

    @property
    def latency_tracker(self) -> LatencyTracker:
        """Access the latency tracker."""
        return self._latency_tracker

    async def start(self) -> None:
        """Start the registry and begin polling."""
        # Load existing backends from database
        await self._load_backends_from_db()

        # Load persisted latency EMAs
        async with get_async_db_context() as db:
            all_backends = await crud.get_all_backends(db=db)
        await self._latency_tracker.load_from_db(all_backends)

        # Load persisted circuit breaker state
        for b in all_backends:
            cb = CircuitBreakerState(
                live_failure_count=getattr(b, "live_failure_count", 0) or 0,
                circuit_open_until=getattr(b, "circuit_open_until", None),
            )
            self._circuit_breakers[b.id] = cb

        # Start background tasks
        self._poll_task = asyncio.create_task(self._poll_loop())
        self._persist_task = asyncio.create_task(self._persist_latency_loop())
        self._cleanup_task = asyncio.create_task(self._telemetry_cleanup_loop())
        logger.info("Backend registry started")

    async def stop(self) -> None:
        """Stop the registry."""
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

        if self._persist_task:
            self._persist_task.cancel()
            try:
                await self._persist_task
            except asyncio.CancelledError:
                pass

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Final persist of latency data before shutdown
        try:
            await self._persist_latency_data()
        except Exception as e:
            logger.warning("shutdown_persist_error", error=str(e))

        # Close all adapters and sidecar clients
        for adapter in self._adapters.values():
            await adapter.close()
        for client in self._sidecar_clients.values():
            await client.close()

        logger.info("Backend registry stopped")

    @property
    def is_force_offline(self) -> bool:
        """Whether the system is forced offline by admin."""
        return self._force_offline

    async def force_offline(self) -> None:
        """Force the system offline - stop polling and mark all backends unhealthy."""
        self._force_offline = True

        # Stop the poll loop
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        # Mark all backends as unhealthy in DB
        async with get_async_db_context() as db:
            backends = await crud.get_all_backends(db=db)
            for b in backends:
                await crud.update_backend_status(db, b.id, BackendStatus.UNHEALTHY)
            await db.commit()

        logger.info("system_forced_offline")

    async def force_online(self) -> None:
        """Force the system back online - reload backends and restart polling."""
        self._force_offline = False

        # Close existing adapters and sidecar clients
        for adapter in self._adapters.values():
            await adapter.close()
        for client in self._sidecar_clients.values():
            await client.close()

        self._adapters.clear()
        self._sidecar_clients.clear()
        self._node_backends.clear()
        self._backend_gpu_indices.clear()
        self._node_sidecar_data.clear()

        # Reload from DB
        await self._load_backends_from_db()

        # Restart poll loop
        if self._poll_task is None or self._poll_task.done():
            self._poll_task = asyncio.create_task(self._poll_loop())

        # Immediate full poll
        await self._poll_all_backends()

        logger.info("system_forced_online")

    async def register_backend(
        self,
        name: str,
        url: str,
        engine: BackendEngine,
        max_concurrent: int = 4,
        gpu_memory_gb: Optional[float] = None,
        gpu_type: Optional[str] = None,
        node_id: Optional[int] = None,
        gpu_indices: Optional[list] = None,
        db: Optional[AsyncSession] = None,
    ) -> Backend:
        """
        Register a new backend.

        Args:
            name: Unique backend name
            url: Backend URL
            engine: Backend engine type
            max_concurrent: Max concurrent requests
            gpu_memory_gb: GPU memory in GB
            gpu_type: GPU type description
            node_id: Optional node this backend runs on
            gpu_indices: Optional list of GPU indices assigned to this backend
            db: Optional database session

        Returns:
            Created Backend record
        """
        async with self._lock:
            # Create database record
            if db:
                backend = await crud.create_backend(
                    db=db,
                    name=name,
                    url=url,
                    engine=engine,
                    max_concurrent=max_concurrent,
                    gpu_memory_gb=gpu_memory_gb,
                    gpu_type=gpu_type,
                    node_id=node_id,
                    gpu_indices=gpu_indices,
                )
            else:
                async with get_async_db_context() as db_session:
                    backend = await crud.create_backend(
                        db=db_session,
                        name=name,
                        url=url,
                        engine=engine,
                        max_concurrent=max_concurrent,
                        gpu_memory_gb=gpu_memory_gb,
                        gpu_type=gpu_type,
                        node_id=node_id,
                        gpu_indices=gpu_indices,
                    )

            # Create adapter
            adapter = self._create_adapter(backend)
            self._adapters[backend.id] = adapter

            # Track node-backend mappings
            if node_id:
                self._node_backends.setdefault(node_id, set()).add(backend.id)
                self._backend_gpu_indices[backend.id] = gpu_indices

                # Create sidecar client for node if not already created
                if node_id not in self._sidecar_clients:
                    async with get_async_db_context() as db_session:
                        node = await crud.get_node_by_id(db_session, node_id)
                    if node and node.sidecar_url:
                        self._sidecar_clients[node_id] = SidecarClient(
                            node.sidecar_url,
                            timeout=self._settings.sidecar_timeout,
                            sidecar_key=node.sidecar_key,
                        )

        # Run discovery outside the lock to avoid blocking telemetry reads
        await self._discover_backend(backend.id)

        logger.info(
            "backend_registered",
            backend_id=backend.id,
            name=name,
            engine=engine.value,
            node_id=node_id,
        )

        return backend

    async def refresh_backend(self, backend_id: int) -> bool:
        """
        Force refresh capabilities for a backend.

        Args:
            backend_id: Backend ID to refresh

        Returns:
            True if refresh succeeded
        """
        if backend_id not in self._adapters:
            return False

        await self._discover_backend(backend_id)
        return True

    async def disable_backend(self, backend_id: int) -> bool:
        """Disable a backend."""
        async with get_async_db_context() as db:
            backend = await crud.update_backend_status(
                db=db,
                backend_id=backend_id,
                status=BackendStatus.DISABLED,
            )
            return backend is not None

    async def drain_backend(self, backend_id: int) -> bool:
        """Start draining a backend — stops new requests while in-flight finish.

        If no requests are in-flight, immediately transitions to DISABLED.
        """
        async with get_async_db_context() as db:
            backend = await crud.update_backend_status(
                db=db,
                backend_id=backend_id,
                status=BackendStatus.DRAINING,
            )
        if not backend:
            return False

        logger.info("backend_drain_started", backend_id=backend_id)

        # If queue depth is already 0, complete drain immediately
        try:
            from backend.app.core.scheduler.policy import get_scheduler
            scheduler = get_scheduler()
            async with scheduler.router._lock:
                depth = scheduler.router._backend_queue_depths.get(backend_id, 0)
            if depth == 0:
                await self.complete_drain(backend_id)
        except Exception as e:
            logger.debug("drain_immediate_check_error", backend_id=backend_id, error=str(e))

        return True

    async def complete_drain(self, backend_id: int) -> bool:
        """Auto-transition a draining backend to DISABLED when queue depth hits 0."""
        async with get_async_db_context() as db:
            backend = await crud.get_backend_by_id(db, backend_id)
            if not backend or backend.status != BackendStatus.DRAINING:
                return False
            await crud.update_backend_status(
                db=db,
                backend_id=backend_id,
                status=BackendStatus.DISABLED,
            )
        logger.info("backend_drain_complete", backend_id=backend_id)
        return True

    async def remove_backend(self, backend_id: int) -> bool:
        """Remove a backend entirely (unregister)."""
        async with self._lock:
            # Close and remove adapter
            adapter = self._adapters.pop(backend_id, None)
            if adapter:
                await adapter.close()

            # Remove from node-backend mappings
            self._backend_gpu_indices.pop(backend_id, None)
            for node_id, backend_ids in list(self._node_backends.items()):
                backend_ids.discard(backend_id)
                # If no more backends on this node, remove the sidecar client
                if not backend_ids:
                    del self._node_backends[node_id]
                    sidecar = self._sidecar_clients.pop(node_id, None)
                    if sidecar:
                        await sidecar.close()

            # Remove cached capabilities and telemetry
            self._capabilities.pop(backend_id, None)
            self._telemetry.pop(backend_id, None)

        # Delete from database
        async with get_async_db_context() as db:
            deleted = await crud.delete_backend(db=db, backend_id=backend_id)

        if deleted:
            logger.info("backend_removed", backend_id=backend_id)
        return deleted

    async def enable_backend(self, backend_id: int) -> bool:
        """Enable a previously disabled backend."""
        async with get_async_db_context() as db:
            backend = await crud.update_backend_status(
                db=db,
                backend_id=backend_id,
                status=BackendStatus.UNKNOWN,
            )

        # Trigger immediate health check
        if backend_id in self._adapters:
            await self._check_backend_health(backend_id)

        return True

    async def get_healthy_backends(
        self,
        engine: Optional[BackendEngine] = None,
    ) -> List[Backend]:
        """Get all healthy backends. Returns empty list if system is forced offline."""
        if self._force_offline:
            return []
        async with get_async_db_context() as db:
            return await crud.get_healthy_backends(db=db, engine=engine)

    async def get_all_backends(self) -> List[Backend]:
        """Get all backends."""
        async with get_async_db_context() as db:
            return await crud.get_all_backends(db=db)

    async def get_backend_models(self, backend_id: int) -> List[Model]:
        """Get models for a backend."""
        async with get_async_db_context() as db:
            return await crud.get_models_for_backend(db=db, backend_id=backend_id)

    async def get_backends_with_model(
        self,
        model_name: str,
        modality: Optional[Modality] = None,
    ) -> List[Backend]:
        """Get backends that have a specific model."""
        async with get_async_db_context() as db:
            return await crud.get_backends_with_model(
                db=db,
                model_name=model_name,
                modality=modality,
            )

    async def model_exists(self, model_name: str) -> bool:
        """Check if a model is available on any healthy backend."""
        backends = await self.get_backends_with_model(model_name)
        return len(backends) > 0

    async def get_gpu_utilizations(self) -> Dict[int, Optional[float]]:
        """Get current GPU utilization for all backends."""
        async with self._lock:
            return {
                bid: t.gpu_utilization
                for bid, t in self._telemetry.items()
            }

    async def get_telemetry(self, backend_id: int) -> Optional[TelemetrySnapshot]:
        """Get latest telemetry for a backend."""
        async with self._lock:
            return self._telemetry.get(backend_id)

    async def get_capabilities(self, backend_id: int) -> Optional[BackendCapabilities]:
        """Get discovered capabilities for a backend."""
        async with self._lock:
            return self._capabilities.get(backend_id)

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    async def register_node(
        self,
        name: str,
        hostname: Optional[str] = None,
        sidecar_url: Optional[str] = None,
        sidecar_key: Optional[str] = None,
    ) -> Node:
        """Register a new physical node."""
        async with get_async_db_context() as db:
            node = await crud.create_node(
                db=db,
                name=name,
                hostname=hostname,
                sidecar_url=sidecar_url,
                sidecar_key=sidecar_key,
            )

        async with self._lock:
            if sidecar_url:
                self._sidecar_clients[node.id] = SidecarClient(
                    sidecar_url,
                    timeout=self._settings.sidecar_timeout,
                    sidecar_key=sidecar_key,
                )

        logger.info("node_registered", node_id=node.id, name=name)

        # Immediately poll the sidecar to discover GPUs
        if sidecar_url and node.id in self._sidecar_clients:
            try:
                await self._collect_node_telemetry(node.id)
            except Exception as e:
                logger.warning("initial_sidecar_poll_failed", node_id=node.id, error=str(e))

        return node

    async def remove_node(self, node_id: int) -> bool:
        """Remove a node (fails if backends still reference it)."""
        async with self._lock:
            # Check if any backends reference this node
            if self._node_backends.get(node_id):
                return False

            sidecar = self._sidecar_clients.pop(node_id, None)
            if sidecar:
                await sidecar.close()
            self._node_sidecar_data.pop(node_id, None)

        async with get_async_db_context() as db:
            deleted = await crud.delete_node(db=db, node_id=node_id)

        if deleted:
            logger.info("node_removed", node_id=node_id)
        return deleted

    async def get_all_nodes(self) -> List[Node]:
        """Get all nodes."""
        async with get_async_db_context() as db:
            return await crud.get_all_nodes(db=db)

    async def update_backend(self, backend_id: int, **kwargs) -> "Backend":
        """Update a backend's editable fields and reconcile in-memory state.

        Supported kwargs: name, url, engine, max_concurrent, gpu_memory_gb,
        gpu_type, priority, node_id, gpu_indices, _clear_fields.
        """
        async with self._lock:
            # Capture old state for in-memory reconciliation
            async with get_async_db_context() as db:
                old_backend = await crud.get_backend_by_id(db, backend_id)
                if not old_backend:
                    raise ValueError(f"Backend {backend_id} not found")
                old_url = old_backend.url
                old_engine = old_backend.engine
                old_node_id = old_backend.node_id
                old_gpu_indices = old_backend.gpu_indices

            # Persist to DB
            async with get_async_db_context() as db:
                backend = await crud.update_backend(db, backend_id, **kwargs)
                if not backend:
                    raise ValueError(f"Backend {backend_id} not found")
                # Capture updated values while session is open
                new_url = backend.url
                new_engine = backend.engine
                new_node_id = backend.node_id
                new_gpu_indices = backend.gpu_indices

            # Reconcile adapter if url or engine changed
            new_engine_val = kwargs.get("engine")
            new_url_val = kwargs.get("url")
            if new_url_val or new_engine_val:
                old_adapter = self._adapters.pop(backend_id, None)
                if old_adapter:
                    await old_adapter.close()
                # Need a lightweight object to pass to _create_adapter
                class _Stub:
                    pass
                stub = _Stub()
                stub.url = new_url
                stub.engine = new_engine
                self._adapters[backend_id] = self._create_adapter(stub)

            # Reconcile node-backend mapping if node_id changed
            clear_fields = set(kwargs.get("_clear_fields", []))
            node_changed = "node_id" in kwargs or "node_id" in clear_fields
            if node_changed:
                # Remove from old node
                if old_node_id and old_node_id in self._node_backends:
                    self._node_backends[old_node_id].discard(backend_id)
                    if not self._node_backends[old_node_id]:
                        del self._node_backends[old_node_id]
                        sidecar = self._sidecar_clients.pop(old_node_id, None)
                        if sidecar:
                            await sidecar.close()
                # Add to new node
                if new_node_id:
                    self._node_backends.setdefault(new_node_id, set()).add(backend_id)
                    if new_node_id not in self._sidecar_clients:
                        async with get_async_db_context() as db:
                            node = await crud.get_node_by_id(db, new_node_id)
                        if node and node.sidecar_url:
                            self._sidecar_clients[new_node_id] = SidecarClient(
                                node.sidecar_url,
                                timeout=self._settings.sidecar_timeout,
                                sidecar_key=node.sidecar_key,
                            )

            # Reconcile gpu_indices if changed
            if "gpu_indices" in kwargs or "gpu_indices" in clear_fields:
                self._backend_gpu_indices[backend_id] = new_gpu_indices

        # Re-read the final object outside the lock for the return value
        async with get_async_db_context() as db:
            backend = await crud.get_backend_by_id(db, backend_id)

        logger.info("backend_updated", backend_id=backend_id, fields=list(kwargs.keys()))
        return backend

    async def update_node(self, node_id: int, **kwargs) -> "Node":
        """Update a node's editable fields and reconcile in-memory state.

        Supported kwargs: name, hostname, sidecar_url, sidecar_key, _clear_fields.
        """
        async with self._lock:
            # Persist to DB
            async with get_async_db_context() as db:
                node = await crud.update_node(db, node_id, **kwargs)
                if not node:
                    raise ValueError(f"Node {node_id} not found")
                new_sidecar_url = node.sidecar_url
                new_sidecar_key = node.sidecar_key

            # Reconcile sidecar client if url or key changed
            clear_fields = set(kwargs.get("_clear_fields", []))
            sidecar_changed = (
                "sidecar_url" in kwargs or "sidecar_key" in kwargs
                or "sidecar_url" in clear_fields or "sidecar_key" in clear_fields
            )
            if sidecar_changed:
                old_client = self._sidecar_clients.pop(node_id, None)
                if old_client:
                    await old_client.close()
                if new_sidecar_url:
                    self._sidecar_clients[node_id] = SidecarClient(
                        new_sidecar_url,
                        timeout=self._settings.sidecar_timeout,
                        sidecar_key=new_sidecar_key,
                    )
                self._node_sidecar_data.pop(node_id, None)

        # Re-read the final object for the return value
        async with get_async_db_context() as db:
            node = await crud.get_node_by_id(db, node_id)

        logger.info("node_updated", node_id=node_id, fields=list(kwargs.keys()))

        # Trigger immediate sidecar poll if URL/key changed
        if sidecar_changed and node_id in self._sidecar_clients:
            try:
                await self._collect_node_telemetry(node_id)
            except Exception as e:
                logger.warning("sidecar_poll_after_update_failed", node_id=node_id, error=str(e))

        return node

    async def refresh_node(self, node_id: int) -> bool:
        """Force refresh sidecar data for a node."""
        if node_id not in self._sidecar_clients:
            return False
        await self._collect_node_telemetry(node_id)
        return True

    # ------------------------------------------------------------------
    # Circuit breaker & reactive health
    # ------------------------------------------------------------------

    async def report_live_failure(self, backend_id: int) -> None:
        """Called when a live request to a backend fails.

        Increments failure count. If threshold reached, opens circuit and
        marks backend UNHEALTHY immediately. Activates fast-polling.
        """
        cb = self._circuit_breakers.setdefault(backend_id, CircuitBreakerState())
        cb.live_failure_count += 1
        cb.last_failure_time = datetime.now(timezone.utc)

        threshold = self._settings.backend_circuit_breaker_threshold
        if cb.live_failure_count >= threshold and not cb.is_open:
            recovery = self._settings.backend_circuit_breaker_recovery_seconds
            cb.circuit_open_until = datetime.now(timezone.utc) + timedelta(seconds=recovery)

            # Mark UNHEALTHY in DB immediately
            try:
                async with get_async_db_context() as db:
                    await crud.update_backend_status(
                        db=db,
                        backend_id=backend_id,
                        status=BackendStatus.UNHEALTHY,
                    )
                    await crud.update_backend_circuit_breaker(
                        db=db,
                        backend_id=backend_id,
                        live_failure_count=cb.live_failure_count,
                        circuit_open_until=cb.circuit_open_until,
                    )
            except Exception as e:
                logger.warning("circuit_breaker_db_error", backend_id=backend_id, error=str(e))

            # Activate fast polling
            fast_duration = self._settings.backend_adaptive_poll_fast_duration
            self._fast_poll_backends[backend_id] = (
                datetime.now(timezone.utc) + timedelta(seconds=fast_duration)
            )

            logger.warning(
                "circuit_breaker_opened",
                backend_id=backend_id,
                failures=cb.live_failure_count,
                recovery_seconds=recovery,
            )

    async def report_live_success(self, backend_id: int) -> None:
        """Called when a live request to a backend succeeds.

        Resets failure count. If circuit was half-open, closes it and
        marks backend HEALTHY.
        """
        cb = self._circuit_breakers.get(backend_id)
        if cb is None:
            return

        was_half_open = cb.is_half_open
        cb.live_failure_count = 0
        cb.circuit_open_until = None
        cb.last_failure_time = None

        if was_half_open:
            # Circuit recovered — mark healthy
            try:
                async with get_async_db_context() as db:
                    await crud.update_backend_status(
                        db=db,
                        backend_id=backend_id,
                        status=BackendStatus.HEALTHY,
                    )
                    await crud.update_backend_circuit_breaker(
                        db=db,
                        backend_id=backend_id,
                        live_failure_count=0,
                        circuit_open_until=None,
                    )
            except Exception as e:
                logger.warning("circuit_close_db_error", backend_id=backend_id, error=str(e))

            # Remove from fast-poll set
            self._fast_poll_backends.pop(backend_id, None)

            logger.info("circuit_breaker_closed", backend_id=backend_id)

    async def is_backend_available(self, backend_id: int) -> bool:
        """Check if backend is available (circuit not open).

        Half-open circuits ARE available (they allow a single probe request).
        """
        cb = self._circuit_breakers.get(backend_id)
        if cb is None:
            return True
        return not cb.is_open

    # ------------------------------------------------------------------
    # Latency persistence
    # ------------------------------------------------------------------

    async def _persist_latency_loop(self) -> None:
        """Periodically persist latency EMA values to DB."""
        while True:
            try:
                await asyncio.sleep(self._settings.latency_ema_persist_interval)
                await self._persist_latency_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("persist_latency_error", error=str(e))

    async def _persist_latency_data(self) -> None:
        """Write current latency EMA and throughput scores to DB."""
        all_latencies = await self._latency_tracker.get_all_latencies()
        if not all_latencies:
            return

        async with get_async_db_context() as db:
            for backend_id, latency_ema in all_latencies.items():
                ttft_ema = await self._latency_tracker.get_ttft_ema(backend_id)
                throughput = self._latency_tracker.compute_throughput_score(backend_id)
                await crud.update_backend_latency_ema(
                    db=db,
                    backend_id=backend_id,
                    latency_ema_ms=latency_ema,
                    ttft_ema_ms=ttft_ema,
                    throughput_score=throughput,
                )

    def _create_adapter(self, backend: Backend) -> OllamaAdapter | VLLMAdapter:
        """Create the appropriate adapter for a backend."""
        timeout = self._settings.backend_health_timeout

        if backend.engine == BackendEngine.OLLAMA:
            return OllamaAdapter(backend.url, timeout=timeout)
        else:
            return VLLMAdapter(backend.url, timeout=timeout)

    async def _load_backends_from_db(self) -> None:
        """Load existing backends and nodes from database on startup."""
        async with get_async_db_context() as db:
            backends = await crud.get_all_backends(db=db)
            nodes = await crud.get_all_nodes(db=db)

        # Create sidecar clients per node
        for node in nodes:
            if node.sidecar_url:
                self._sidecar_clients[node.id] = SidecarClient(
                    node.sidecar_url,
                    timeout=self._settings.sidecar_timeout,
                    sidecar_key=node.sidecar_key,
                )

        # Create adapters and build node-backend mappings
        for backend in backends:
            adapter = self._create_adapter(backend)
            self._adapters[backend.id] = adapter

            if backend.node_id:
                self._node_backends.setdefault(backend.node_id, set()).add(backend.id)
                self._backend_gpu_indices[backend.id] = backend.gpu_indices

        logger.info(
            "loaded_backends",
            count=len(backends),
            nodes=len(nodes),
            sidecar_count=len(self._sidecar_clients),
        )

    async def _poll_loop(self) -> None:
        """Background polling loop with adaptive intervals for troubled backends."""
        last_full_poll = 0.0
        while True:
            try:
                now = datetime.now(timezone.utc)

                # Clean up expired fast-poll entries
                self._fast_poll_backends = {
                    bid: until
                    for bid, until in self._fast_poll_backends.items()
                    if now < until
                }

                # Fast-poll troubled backends at the fast interval
                if self._fast_poll_backends:
                    fast_tasks = [
                        self._check_backend_health(bid)
                        for bid in self._fast_poll_backends
                    ]
                    await asyncio.gather(*fast_tasks, return_exceptions=True)

                # Determine sleep interval
                if self._fast_poll_backends:
                    interval = self._settings.backend_adaptive_poll_fast_interval
                else:
                    interval = self._settings.backend_poll_interval

                await asyncio.sleep(interval)

                # Full poll on normal interval
                import time as _time
                elapsed = _time.monotonic() - last_full_poll
                if elapsed >= self._settings.backend_poll_interval:
                    await self._poll_all_backends()
                    last_full_poll = _time.monotonic()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("poll_loop_error", error=str(e))

    async def _poll_all_backends(self) -> None:
        """Poll all nodes' sidecars, then all backends for health and telemetry."""
        # Phase A: Poll all unique nodes' sidecars (deduplicated)
        async with self._lock:
            node_ids = list(self._sidecar_clients.keys())
            backend_ids = list(self._adapters.keys())

        if node_ids:
            node_tasks = [
                self._collect_node_telemetry(nid)
                for nid in node_ids
            ]
            await asyncio.gather(*node_tasks, return_exceptions=True)

        # Phase B: Poll all backends for health/request metrics
        tasks = [
            self._check_backend_health(bid)
            for bid in backend_ids
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_backend_health(self, backend_id: int) -> None:
        """Check health of a single backend."""
        adapter = self._adapters.get(backend_id)
        if not adapter:
            return

        try:
            health = await adapter.health_check()

            # Update database
            async with get_async_db_context() as db:
                if health.is_healthy:
                    # Don't overwrite admin-set statuses (DISABLED, DRAINING)
                    current = await crud.get_backend_by_id(db, backend_id)
                    if current and current.status not in (
                        BackendStatus.DISABLED,
                        BackendStatus.DRAINING,
                    ):
                        await crud.update_backend_status(
                            db=db,
                            backend_id=backend_id,
                            status=BackendStatus.HEALTHY,
                        )

                    # If circuit was open/half-open, close it on successful health check
                    cb = self._circuit_breakers.get(backend_id)
                    if cb and (cb.is_open or cb.is_half_open):
                        cb.live_failure_count = 0
                        cb.circuit_open_until = None
                        cb.last_failure_time = None
                        await crud.update_backend_circuit_breaker(
                            db=db,
                            backend_id=backend_id,
                            live_failure_count=0,
                            circuit_open_until=None,
                        )
                        self._fast_poll_backends.pop(backend_id, None)
                        logger.info("circuit_breaker_closed_by_health_check", backend_id=backend_id)
                else:
                    # Check consecutive failures
                    backend = await crud.get_backend_by_id(db, backend_id)
                    if backend:
                        failures = backend.consecutive_failures + 1
                        if failures >= self._settings.backend_unhealthy_threshold:
                            await crud.update_backend_status(
                                db=db,
                                backend_id=backend_id,
                                status=BackendStatus.UNHEALTHY,
                            )
                        else:
                            # Increment failure count but don't mark unhealthy yet
                            backend.consecutive_failures = failures
                            await db.flush()

            # Get telemetry if healthy
            if health.is_healthy:
                # Auto-discover if no capabilities stored yet
                if backend_id not in self._capabilities:
                    await self._discover_backend(backend_id)
                await self._collect_telemetry(backend_id)

        except Exception as e:
            logger.warning(
                "health_check_error",
                backend_id=backend_id,
                error=str(e),
            )

    async def _discover_backend(self, backend_id: int) -> None:
        """Discover capabilities for a backend."""
        adapter = self._adapters.get(backend_id)
        if not adapter:
            return

        try:
            caps = await adapter.discover_capabilities()

            async with self._lock:
                self._capabilities[backend_id] = caps

            # Update database with discovered info
            async with get_async_db_context() as db:
                backend = await crud.get_backend_by_id(db, backend_id)
                if backend:
                    # Update backend capabilities
                    backend.supports_multimodal = caps.supports_multimodal
                    backend.supports_embeddings = caps.supports_embeddings
                    backend.supports_structured_output = caps.supports_structured_output
                    backend.version = caps.engine_version

                    # Don't overwrite admin-set statuses (DISABLED, DRAINING)
                    if backend.status not in (
                        BackendStatus.DISABLED,
                        BackendStatus.DRAINING,
                    ):
                        if caps.is_healthy:
                            backend.status = BackendStatus.HEALTHY
                        else:
                            backend.status = BackendStatus.UNHEALTHY

                    await db.flush()

                    # Update models
                    discovered_names = []
                    for model_info in caps.models:
                        modality = Modality.CHAT
                        if model_info.supports_multimodal:
                            modality = Modality.MULTIMODAL
                        elif "rerank" in model_info.name.lower():
                            modality = Modality.RERANKING
                        elif "embed" in model_info.name.lower():
                            modality = Modality.EMBEDDING

                        # Serialize capabilities list to JSON string for DB
                        caps_json = None
                        if model_info.capabilities:
                            caps_json = json.dumps(model_info.capabilities)

                        await crud.upsert_model(
                            db=db,
                            backend_id=backend_id,
                            name=model_info.name,
                            modality=modality,
                            context_length=model_info.context_length,
                            supports_multimodal=model_info.supports_multimodal,
                            supports_thinking=model_info.supports_thinking,
                            supports_structured_output=model_info.supports_structured_output,
                            is_loaded=model_info.is_loaded,
                            quantization=model_info.quantization,
                            model_format=model_info.model_format,
                            capabilities_json=caps_json,
                            embedding_length=model_info.embedding_length,
                            head_count=model_info.head_count,
                            layer_count=model_info.layer_count,
                            feed_forward_length=model_info.feed_forward_length,
                            parent_model=model_info.parent_model,
                            model_max_context=model_info.model_max_context,
                        )
                        discovered_names.append(model_info.name)

                    # Remove models no longer present on this backend
                    if discovered_names:
                        removed = await crud.remove_stale_models(
                            db, backend_id, discovered_names
                        )
                        if removed:
                            logger.info(
                                "stale_models_removed",
                                backend_id=backend_id,
                                count=removed,
                            )

            logger.info(
                "backend_discovered",
                backend_id=backend_id,
                models=len(caps.models),
                version=caps.engine_version,
            )

        except Exception as e:
            logger.warning(
                "discovery_error",
                backend_id=backend_id,
                error=str(e),
            )

    async def _collect_node_telemetry(self, node_id: int) -> None:
        """Phase A: Poll sidecar once per node, upsert GPU devices, store telemetry."""
        sidecar_client = self._sidecar_clients.get(node_id)
        if not sidecar_client:
            return

        sidecar_data: Optional[SidecarResponse] = None
        try:
            sidecar_data = await sidecar_client.get_gpu_info()
        except Exception as e:
            logger.debug("sidecar_collect_error", node_id=node_id, error=str(e))
            # Mark node offline
            try:
                async with get_async_db_context() as db:
                    await crud.update_node_status(db, node_id, NodeStatus.OFFLINE)
            except Exception:
                pass
            self._node_sidecar_data[node_id] = None
            return

        # Cache for backend telemetry phase
        self._node_sidecar_data[node_id] = sidecar_data

        if not sidecar_data:
            logger.warning("sidecar_no_response", node_id=node_id)
            try:
                async with get_async_db_context() as db:
                    await crud.update_node_status(db, node_id, NodeStatus.OFFLINE)
            except Exception:
                pass
            return

        if not sidecar_data.gpus:
            logger.warning("sidecar_no_gpus", node_id=node_id, gpu_count=sidecar_data.gpu_count)
            return

        try:
            async with get_async_db_context() as db:
                # Update node hardware info and status
                await crud.update_node_hardware(
                    db, node_id,
                    gpu_count=sidecar_data.gpu_count,
                    driver_version=sidecar_data.driver_version,
                    cuda_version=sidecar_data.cuda_version,
                    sidecar_version=sidecar_data.sidecar_version,
                )
                await crud.update_node_status(db, node_id, NodeStatus.ONLINE)

                # Upsert GPU devices and store per-GPU telemetry
                for gpu in sidecar_data.gpus:
                    device = await crud.upsert_gpu_device(
                        db=db,
                        node_id=node_id,
                        gpu_index=gpu.index,
                        uuid=gpu.uuid,
                        name=gpu.name,
                        pci_bus_id=gpu.pci_bus_id,
                        compute_capability=gpu.compute_capability,
                        memory_total_gb=gpu.memory_total_gb,
                        power_limit_watts=gpu.power_limit_watts,
                    )

                    await crud.create_gpu_device_telemetry(
                        db=db,
                        gpu_device_id=device.id,
                        utilization_gpu=gpu.utilization_gpu,
                        utilization_memory=gpu.utilization_memory,
                        memory_used_gb=gpu.memory_used_gb,
                        memory_free_gb=gpu.memory_free_gb,
                        temperature_gpu=gpu.temperature_gpu,
                        temperature_memory=gpu.temperature_memory,
                        power_draw_watts=gpu.power_draw_watts,
                        fan_speed_percent=gpu.fan_speed_percent,
                        clock_sm_mhz=gpu.clock_sm_mhz,
                        clock_memory_mhz=gpu.clock_memory_mhz,
                    )

        except Exception as e:
            logger.debug("node_telemetry_store_error", node_id=node_id, error=str(e))

    async def _collect_telemetry(self, backend_id: int) -> None:
        """Phase B: Collect per-backend telemetry, enriched with assigned GPU data."""
        adapter = self._adapters.get(backend_id)
        if not adapter:
            return

        try:
            snapshot = await adapter.get_telemetry(backend_id)

            # Look up node sidecar data for GPU enrichment
            gpu_indices = self._backend_gpu_indices.get(backend_id)
            node_id = None
            for nid, bids in self._node_backends.items():
                if backend_id in bids:
                    node_id = nid
                    break

            sidecar_data = self._node_sidecar_data.get(node_id) if node_id else None

            # Enrich snapshot with this backend's assigned GPU subset
            if sidecar_data and sidecar_data.gpus:
                # Filter to assigned GPUs only
                if gpu_indices is not None:
                    gpus = [g for g in sidecar_data.gpus if g.index in gpu_indices]
                else:
                    gpus = sidecar_data.gpus

                # Average utilization across assigned GPUs
                utils = [g.utilization_gpu for g in gpus if g.utilization_gpu is not None]
                if utils:
                    snapshot.gpu_utilization = sum(utils) / len(utils)
                # Sum memory across assigned GPUs
                mem_used = [g.memory_used_gb for g in gpus if g.memory_used_gb is not None]
                mem_total = [g.memory_total_gb for g in gpus if g.memory_total_gb is not None]
                if mem_used:
                    snapshot.gpu_memory_used_gb = sum(mem_used)
                if mem_total:
                    snapshot.gpu_memory_total_gb = sum(mem_total)
                # Average temperature
                temps = [g.temperature_gpu for g in gpus if g.temperature_gpu is not None]
                if temps:
                    snapshot.gpu_temperature = sum(temps) / len(temps)

            async with self._lock:
                self._telemetry[backend_id] = snapshot

            # Store backend telemetry snapshot
            async with get_async_db_context() as db:
                # Compute aggregate power/fan from assigned GPUs
                total_power = None
                avg_fan = None
                avg_temp = None
                if sidecar_data and sidecar_data.gpus:
                    if gpu_indices is not None:
                        gpus = [g for g in sidecar_data.gpus if g.index in gpu_indices]
                    else:
                        gpus = sidecar_data.gpus
                    powers = [g.power_draw_watts for g in gpus if g.power_draw_watts is not None]
                    fans = [g.fan_speed_percent for g in gpus if g.fan_speed_percent is not None]
                    temps = [g.temperature_gpu for g in gpus if g.temperature_gpu is not None]
                    if powers:
                        total_power = sum(powers)
                    if fans:
                        avg_fan = sum(fans) / len(fans)
                    if temps:
                        avg_temp = sum(temps) / len(temps)

                await crud.create_telemetry_snapshot(
                    db=db,
                    backend_id=backend_id,
                    gpu_utilization=snapshot.gpu_utilization,
                    gpu_memory_used_gb=snapshot.gpu_memory_used_gb,
                    gpu_memory_total_gb=snapshot.gpu_memory_total_gb,
                    gpu_temperature=avg_temp,
                    gpu_power_draw_watts=total_power,
                    gpu_fan_speed_percent=avg_fan,
                    active_requests=snapshot.active_requests,
                    queued_requests=snapshot.queued_requests,
                    loaded_models=snapshot.loaded_models,
                )

        except Exception as e:
            logger.debug(
                "telemetry_error",
                backend_id=backend_id,
                error=str(e),
            )

    # ------------------------------------------------------------------
    # Telemetry data retention cleanup
    # ------------------------------------------------------------------

    async def _telemetry_cleanup_loop(self) -> None:
        """Periodically purge old telemetry data."""
        while True:
            try:
                await asyncio.sleep(self._settings.telemetry_cleanup_interval)
                await self._cleanup_old_telemetry()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("telemetry_cleanup_error", error=str(e))

    async def _cleanup_old_telemetry(self) -> None:
        """Delete telemetry data older than retention period and expired API keys."""
        cutoff = datetime.now(timezone.utc) - timedelta(
            days=self._settings.telemetry_retention_days
        )

        async with get_async_db_context() as db:
            deleted_bt = await crud.delete_old_telemetry(db, older_than=cutoff)
            deleted_gdt = await crud.delete_old_gpu_telemetry(db, older_than=cutoff)
            deleted_keys = await crud.delete_expired_api_keys(db, grace_days=15)

        if deleted_keys:
            logger.info("expired_api_keys_cleaned_up", deleted=deleted_keys)

        if deleted_bt or deleted_gdt:
            logger.info(
                "telemetry_cleaned_up",
                backend_telemetry_deleted=deleted_bt,
                gpu_telemetry_deleted=deleted_gdt,
                cutoff=cutoff.isoformat(),
            )


# Global registry instance
_registry: Optional[BackendRegistry] = None


def get_registry() -> BackendRegistry:
    """Get the global registry instance."""
    global _registry
    if _registry is None:
        _registry = BackendRegistry()
    return _registry


async def init_registry() -> BackendRegistry:
    """Initialize and start the global registry."""
    registry = get_registry()
    await registry.start()
    return registry


async def shutdown_registry() -> None:
    """Shutdown the global registry."""
    global _registry
    if _registry:
        await _registry.stop()
        _registry = None
