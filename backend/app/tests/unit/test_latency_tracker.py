############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# test_latency_tracker.py: Unit tests for latency EMA tracker
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for LatencyTracker.

Uses importlib to load latency_tracker directly, bypassing the
telemetry package __init__ which triggers the full DB import chain.
"""

import asyncio
import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Load latency_tracker module directly to avoid the telemetry __init__ chain
# tests/unit/test_latency_tracker.py -> parents[2] = app/ -> app/core/telemetry/
_mod_path = Path(__file__).resolve().parents[2] / "core" / "telemetry" / "latency_tracker.py"
_spec = importlib.util.spec_from_file_location("latency_tracker", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
LatencyTracker = _mod.LatencyTracker


@pytest.fixture
def tracker():
    """Create a LatencyTracker with default alpha=0.3."""
    return LatencyTracker(alpha=0.3)


@pytest.mark.asyncio
async def test_first_observation_is_exact(tracker):
    """First observation should set the EMA exactly."""
    result = await tracker.record_latency(1, 500.0)
    assert result == 500.0


@pytest.mark.asyncio
async def test_second_observation_is_weighted(tracker):
    """Second observation blends with alpha weighting."""
    await tracker.record_latency(1, 500.0)
    result = await tracker.record_latency(1, 1000.0)
    # EMA = 0.3 * 1000 + 0.7 * 500 = 300 + 350 = 650
    assert abs(result - 650.0) < 0.01


@pytest.mark.asyncio
async def test_multiple_observations(tracker):
    """Multiple observations converge toward recent values."""
    await tracker.record_latency(1, 100.0)
    await tracker.record_latency(1, 100.0)
    await tracker.record_latency(1, 100.0)
    # After 3 identical observations the EMA should equal the value
    result = await tracker.get_latency_ema(1)
    assert abs(result - 100.0) < 0.01


@pytest.mark.asyncio
async def test_ttft_tracking(tracker):
    """TTFT EMA is tracked separately from latency."""
    await tracker.record_latency(1, 1000.0)
    await tracker.record_ttft(1, 200.0)

    latency = await tracker.get_latency_ema(1)
    ttft = await tracker.get_ttft_ema(1)

    assert abs(latency - 1000.0) < 0.01
    assert abs(ttft - 200.0) < 0.01


@pytest.mark.asyncio
async def test_get_unknown_backend_returns_none(tracker):
    """Unknown backend returns None."""
    result = await tracker.get_latency_ema(999)
    assert result is None


@pytest.mark.asyncio
async def test_get_all_latencies(tracker):
    """get_all_latencies returns dict of all tracked backends."""
    await tracker.record_latency(1, 100.0)
    await tracker.record_latency(2, 200.0)

    result = await tracker.get_all_latencies()
    assert result == {1: 100.0, 2: 200.0}


@pytest.mark.asyncio
async def test_load_from_db(tracker):
    """load_from_db restores EMA from backend objects."""
    backend1 = MagicMock()
    backend1.id = 1
    backend1.latency_ema_ms = 500.0
    backend1.ttft_ema_ms = 100.0

    backend2 = MagicMock()
    backend2.id = 2
    backend2.latency_ema_ms = None
    backend2.ttft_ema_ms = None

    await tracker.load_from_db([backend1, backend2])

    assert await tracker.get_latency_ema(1) == 500.0
    assert await tracker.get_ttft_ema(1) == 100.0
    assert await tracker.get_latency_ema(2) is None


@pytest.mark.asyncio
async def test_load_from_db_no_attrs(tracker):
    """load_from_db handles objects without latency attrs gracefully."""
    backend = MagicMock(spec=[])  # no attributes
    backend.id = 1
    # Shouldn't raise
    await tracker.load_from_db([backend])
    assert await tracker.get_latency_ema(1) is None


class TestThroughputScore:
    """Tests for compute_throughput_score sigmoid mapping."""

    def test_unknown_returns_1(self):
        t = LatencyTracker()
        assert t.compute_throughput_score(999) == 1.0

    def test_low_latency_high_score(self):
        t = LatencyTracker()
        t._latency_ema[1] = 100.0
        score = t.compute_throughput_score(1)
        # 1 / (1 + 100/5000) = 1/1.02 ~ 0.98
        assert 0.97 < score < 0.99

    def test_medium_latency(self):
        t = LatencyTracker()
        t._latency_ema[1] = 5000.0
        score = t.compute_throughput_score(1)
        # 1 / (1 + 1.0) = 0.5
        assert abs(score - 0.5) < 0.01

    def test_high_latency_low_score(self):
        t = LatencyTracker()
        t._latency_ema[1] = 30000.0
        score = t.compute_throughput_score(1)
        # 1 / (1 + 6) = ~0.143
        assert 0.13 < score < 0.15


@pytest.mark.asyncio
async def test_custom_alpha():
    """Different alpha values weight observations differently."""
    t = LatencyTracker(alpha=0.5)
    await t.record_latency(1, 100.0)
    result = await t.record_latency(1, 200.0)
    # EMA = 0.5 * 200 + 0.5 * 100 = 150
    assert abs(result - 150.0) < 0.01


@pytest.mark.asyncio
async def test_multiple_backends_independent(tracker):
    """Each backend's EMA is tracked independently."""
    await tracker.record_latency(1, 100.0)
    await tracker.record_latency(2, 9000.0)

    assert await tracker.get_latency_ema(1) == 100.0
    assert await tracker.get_latency_ema(2) == 9000.0
