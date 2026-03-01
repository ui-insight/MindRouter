############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# latency_tracker.py: Per-backend latency EMA tracking
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Per-backend latency tracking using Exponential Moving Average (EMA).

EMA formula: ema_new = alpha * observation + (1 - alpha) * ema_old

With alpha=0.3:
  - Last observation contributes 30%
  - Previous EMA contributes 70%
  - Effective window ~= 2/alpha - 1 ~= 5.7 observations
"""

import asyncio
from typing import Dict, List, Optional


class LatencyTracker:
    """Tracks per-backend latency using Exponential Moving Average."""

    def __init__(self, alpha: float = 0.3):
        self._alpha = alpha
        self._lock = asyncio.Lock()
        self._latency_ema: Dict[int, float] = {}
        self._ttft_ema: Dict[int, float] = {}
        self._observation_count: Dict[int, int] = {}

    async def record_latency(self, backend_id: int, latency_ms: float) -> float:
        """Record a total-latency observation, return updated EMA."""
        async with self._lock:
            if backend_id not in self._latency_ema:
                self._latency_ema[backend_id] = latency_ms
                self._observation_count[backend_id] = 1
            else:
                old = self._latency_ema[backend_id]
                self._latency_ema[backend_id] = (
                    self._alpha * latency_ms + (1 - self._alpha) * old
                )
                self._observation_count[backend_id] += 1
            return self._latency_ema[backend_id]

    async def record_ttft(self, backend_id: int, ttft_ms: float) -> float:
        """Record a time-to-first-token observation, return updated EMA."""
        async with self._lock:
            if backend_id not in self._ttft_ema:
                self._ttft_ema[backend_id] = ttft_ms
            else:
                old = self._ttft_ema[backend_id]
                self._ttft_ema[backend_id] = (
                    self._alpha * ttft_ms + (1 - self._alpha) * old
                )
            return self._ttft_ema[backend_id]

    async def get_latency_ema(self, backend_id: int) -> Optional[float]:
        """Get current latency EMA for a backend."""
        async with self._lock:
            return self._latency_ema.get(backend_id)

    async def get_ttft_ema(self, backend_id: int) -> Optional[float]:
        """Get current TTFT EMA for a backend."""
        async with self._lock:
            return self._ttft_ema.get(backend_id)

    async def get_all_latencies(self) -> Dict[int, float]:
        """Get all latency EMAs as {backend_id: ema_ms}."""
        async with self._lock:
            return dict(self._latency_ema)

    async def get_all_ttfts(self) -> Dict[int, float]:
        """Get all TTFT EMAs as {backend_id: ema_ms}."""
        async with self._lock:
            return dict(self._ttft_ema)

    async def load_from_db(self, backends: List) -> None:
        """Load persisted EMA values on startup.

        Args:
            backends: List of Backend ORM objects with latency_ema_ms / ttft_ema_ms attrs.
        """
        async with self._lock:
            for b in backends:
                if getattr(b, "latency_ema_ms", None) is not None:
                    self._latency_ema[b.id] = b.latency_ema_ms
                    self._observation_count[b.id] = 1
                if getattr(b, "ttft_ema_ms", None) is not None:
                    self._ttft_ema[b.id] = b.ttft_ema_ms

    def compute_throughput_score(self, backend_id: int) -> float:
        """Convert latency EMA into a 0.0-1.0 throughput score.

        Uses sigmoid-like mapping: score = 1.0 / (1.0 + latency_ms / 5000.0)

        Examples:
          - 100ms  -> 0.98
          - 1000ms -> 0.83
          - 5000ms -> 0.50
          - 10000ms -> 0.33
          - 30000ms -> 0.14
        """
        ema = self._latency_ema.get(backend_id)
        if ema is None:
            return 1.0  # Unknown = benefit of the doubt
        return 1.0 / (1.0 + ema / 5000.0)
