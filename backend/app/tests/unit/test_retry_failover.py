############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# test_retry_failover.py: Unit tests for retry-with-failover logic
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for retry-with-failover and exclude_backend_ids.

Mocks the DB layer to avoid the pymysql import chain, then imports
the scheduler scoring module for testing.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Pre-mock the DB chain to avoid pymysql requirement
# (see MEMORY.md for rationale)
#
# IMPORTANT: Save and restore sys.modules so we don't pollute later test
# modules (e.g. test_scheduler.py) that import the real db.models.

# Create a sentinel for HEALTHY status that the scorer will compare against
_HEALTHY_SENTINEL = object()

# Build a mock db.models module with the real enum-like values the scorer uses
_mock_db_models = MagicMock()
_mock_db_models.BackendStatus.HEALTHY = _HEALTHY_SENTINEL
_mock_db_models.Modality.MULTIMODAL = "MULTIMODAL"
_mock_db_models.Modality.EMBEDDING = "EMBEDDING"

_MOCKED_MODULES = [
    "backend.app.db",
    "backend.app.db.session",
    "backend.app.db.crud",
    "backend.app.db.models",
    "backend.app.settings",
]

# Save originals before mocking
_saved_modules = {k: sys.modules[k] for k in _MOCKED_MODULES if k in sys.modules}

for mod_name in [
    "backend.app.db",
    "backend.app.db.session",
    "backend.app.db.crud",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# Set db.models to our controlled mock
sys.modules["backend.app.db.models"] = _mock_db_models

# Mock settings to avoid pydantic_settings import
_mock_settings = MagicMock()
_mock_settings.scheduler_score_model_loaded = 100
_mock_settings.scheduler_score_low_utilization = 50
_mock_settings.scheduler_score_latency = 40
_mock_settings.scheduler_score_short_queue = 30
_mock_settings.scheduler_score_high_throughput = 20
sys.modules.setdefault("backend.app.settings", MagicMock())

# Now safe to import
with patch("backend.app.core.scheduler.scoring.get_settings", return_value=_mock_settings):
    from backend.app.core.scheduler.scoring import BackendScorer, BackendScore

# Restore sys.modules so subsequent test modules get real imports.
# Also evict the scoring module itself — it was loaded with mocked references
# to BackendStatus/Modality in its global namespace, so test_scheduler.py
# (which uses the real db.models) needs a fresh import.
_EVICT_MODULES = [
    "backend.app.core.scheduler.scoring",
    "backend.app.core.scheduler",
]
for mod_name in _MOCKED_MODULES:
    if mod_name in _saved_modules:
        sys.modules[mod_name] = _saved_modules[mod_name]
    else:
        sys.modules.pop(mod_name, None)
for mod_name in _EVICT_MODULES:
    sys.modules.pop(mod_name, None)


def _make_backend(bid, name="test", throughput_score=1.0):
    """Create a mock Backend with fields the scorer checks."""
    b = MagicMock()
    b.id = bid
    b.name = name
    b.status = _HEALTHY_SENTINEL
    b.supports_multimodal = False
    b.supports_embeddings = False
    b.supports_structured_output = True
    b.current_concurrent = 0
    b.max_concurrent = 10
    b.gpu_memory_gb = None
    b.throughput_score = throughput_score
    b.priority = 0
    return b


def _make_model(name, is_loaded=True):
    """Create a mock Model."""
    m = MagicMock()
    m.name = name
    m.is_loaded = is_loaded
    m.supports_multimodal = False
    m.supports_structured_output = True
    m.modality = None
    m.vram_required_gb = None
    return m


def _make_job(model="llama3"):
    """Create a mock Job."""
    j = MagicMock()
    j.model = model
    j.modality = None
    j.requires_multimodal = False
    j.requires_structured_output = False
    j.estimated_prompt_tokens = 100
    j.estimated_completion_tokens = 50
    return j


@pytest.fixture
def scorer():
    """Create a BackendScorer with controlled settings.

    We can't use patch() on the scoring module because it was evicted from
    sys.modules after import (to avoid polluting test_scheduler.py).  Instead,
    construct the scorer and directly set the weight attributes that __init__
    would have read from get_settings().
    """
    s = BackendScorer.__new__(BackendScorer)
    s._settings = _mock_settings
    s._weight_model_loaded = _mock_settings.scheduler_score_model_loaded
    s._weight_low_utilization = _mock_settings.scheduler_score_low_utilization
    s._weight_short_queue = _mock_settings.scheduler_score_short_queue
    s._weight_high_throughput = _mock_settings.scheduler_score_high_throughput
    s._weight_latency = _mock_settings.scheduler_score_latency
    return s


class TestExcludeBackendIds:
    """Test that exclude_backend_ids filters backends in routing."""

    def test_rank_with_no_exclusions(self, scorer):
        b1 = _make_backend(1, "b1")
        b2 = _make_backend(2, "b2")
        model = _make_model("llama3")
        job = _make_job()

        scores = scorer.rank_backends(
            backends=[b1, b2],
            job=job,
            backend_models={1: [model], 2: [model]},
        )
        assert len(scores) == 2

    def test_rank_after_filtering_exclusions(self, scorer):
        """Simulate routing layer pre-filter with exclude_backend_ids."""
        b1 = _make_backend(1, "b1")
        b2 = _make_backend(2, "b2")
        b3 = _make_backend(3, "b3")
        model = _make_model("llama3")
        job = _make_job()

        excluded = {1, 2}
        filtered = [b for b in [b1, b2, b3] if b.id not in excluded]

        scores = scorer.rank_backends(
            backends=filtered,
            job=job,
            backend_models={3: [model]},
        )
        assert len(scores) == 1
        assert scores[0].backend_id == 3

    def test_all_excluded_returns_empty(self, scorer):
        """If all backends are excluded, no scores returned."""
        b1 = _make_backend(1)
        excluded = {1}
        filtered = [b for b in [b1] if b.id not in excluded]

        scores = scorer.rank_backends(
            backends=filtered,
            job=_make_job(),
            backend_models={},
        )
        assert len(scores) == 0


class TestLatencyAwareScoring:
    """Test that latency EMA affects backend ranking."""

    def test_lower_latency_scores_higher(self, scorer):
        b1 = _make_backend(1, "fast")
        b2 = _make_backend(2, "slow")
        model = _make_model("llama3")
        job = _make_job()

        scores = scorer.rank_backends(
            backends=[b1, b2],
            job=job,
            backend_models={1: [model], 2: [model]},
            latency_emas={1: 100.0, 2: 20000.0},
        )
        assert scores[0].backend_id == 1
        assert scores[0].latency_score > scores[1].latency_score

    def test_unknown_latency_gets_neutral_score(self, scorer):
        b1 = _make_backend(1, "known")
        b2 = _make_backend(2, "unknown")
        model = _make_model("llama3")
        job = _make_job()

        scores = scorer.rank_backends(
            backends=[b1, b2],
            job=job,
            backend_models={1: [model], 2: [model]},
            latency_emas={1: 100.0},
        )
        for s in scores:
            assert s.latency_score > 0

    def test_latency_score_values(self, scorer):
        b = _make_backend(1)
        model = _make_model("llama3")
        job = _make_job()

        score_low = scorer.compute_score(b, job, [model], latency_ema_ms=100.0)
        score_high = scorer.compute_score(b, job, [model], latency_ema_ms=30000.0)
        score_none = scorer.compute_score(b, job, [model], latency_ema_ms=None)

        assert score_low.latency_score > score_none.latency_score
        assert score_none.latency_score > score_high.latency_score

    def test_latency_included_in_total(self, scorer):
        b = _make_backend(1)
        model = _make_model("llama3")
        job = _make_job()

        score = scorer.compute_score(b, job, [model], latency_ema_ms=100.0)
        assert score.latency_score > 0
        # latency_score should be part of total_score
        expected_total = (
            score.model_loaded_score
            + score.utilization_score
            + score.queue_score
            + score.throughput_score
            + score.latency_score
            + score.priority_score
        )
        assert abs(score.total_score - expected_total) < 0.01

    def test_latency_does_not_override_model_loaded(self, scorer):
        """Model-loaded bonus (100pts) should still dominate over latency (40pts max)."""
        b1 = _make_backend(1, "fast_no_model")
        b2 = _make_backend(2, "slow_with_model")
        model_loaded = _make_model("llama3", is_loaded=True)
        model_not_loaded = _make_model("llama3", is_loaded=False)
        job = _make_job()

        scores = scorer.rank_backends(
            backends=[b1, b2],
            job=job,
            backend_models={1: [model_not_loaded], 2: [model_loaded]},
            latency_emas={1: 50.0, 2: 20000.0},
        )
        # b2 should still win because model-loaded (100pts) > latency advantage
        assert scores[0].backend_id == 2
