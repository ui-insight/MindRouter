############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# test_circuit_breaker.py: Unit tests for circuit breaker
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for CircuitBreakerState.

Uses importlib to load the telemetry models directly, bypassing the
telemetry package __init__ which triggers the full DB import chain.
"""

import importlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# Load telemetry models module directly to avoid the __init__ chain
_mod_path = Path(__file__).resolve().parents[2] / "core" / "telemetry" / "models.py"
_spec = importlib.util.spec_from_file_location("telemetry_models", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CircuitBreakerState = _mod.CircuitBreakerState


class TestCircuitBreakerState:
    """Tests for the CircuitBreakerState dataclass."""

    def test_default_state_not_open(self):
        cb = CircuitBreakerState()
        assert not cb.is_open
        assert not cb.is_half_open

    def test_open_when_future(self):
        cb = CircuitBreakerState(
            circuit_open_until=datetime.now(timezone.utc) + timedelta(seconds=30)
        )
        assert cb.is_open
        assert not cb.is_half_open

    def test_half_open_when_past(self):
        cb = CircuitBreakerState(
            circuit_open_until=datetime.now(timezone.utc) - timedelta(seconds=1)
        )
        assert not cb.is_open
        assert cb.is_half_open

    def test_none_means_closed(self):
        cb = CircuitBreakerState(circuit_open_until=None)
        assert not cb.is_open
        assert not cb.is_half_open

    def test_failure_count_increment(self):
        cb = CircuitBreakerState()
        cb.live_failure_count += 1
        assert cb.live_failure_count == 1
        cb.live_failure_count += 1
        assert cb.live_failure_count == 2

    def test_reset_clears_state(self):
        cb = CircuitBreakerState(
            live_failure_count=5,
            circuit_open_until=datetime.now(timezone.utc) + timedelta(seconds=30),
            last_failure_time=datetime.now(timezone.utc),
        )
        assert cb.is_open

        # Reset
        cb.live_failure_count = 0
        cb.circuit_open_until = None
        cb.last_failure_time = None

        assert not cb.is_open
        assert not cb.is_half_open
        assert cb.live_failure_count == 0
