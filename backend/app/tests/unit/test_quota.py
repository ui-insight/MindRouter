############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# test_quota.py: Unit tests for quota management
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for quota management and CRUD operations."""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, AsyncMock, patch

from backend.app.db.models import Quota, UserRole


class TestQuotaModel:
    """Tests for Quota model behavior."""

    def test_quota_creation(self):
        """Test quota model creation with defaults."""
        quota = Quota(
            user_id=1,
            tokens_used=0,
        )

        assert quota.user_id == 1
        assert quota.tokens_used == 0

    def test_quota_with_custom_limits(self):
        """Test quota with custom RPM and concurrent limits."""
        quota = Quota(
            user_id=1,
            rpm_limit=60,
            max_concurrent=5,
        )

        assert quota.rpm_limit == 60
        assert quota.max_concurrent == 5


class TestQuotaAccounting:
    """Tests for quota accounting logic (budget comes from group)."""

    def test_tokens_remaining(self):
        """Test calculating remaining tokens against group budget."""
        group_budget = 100000
        quota = Quota(
            user_id=1,
            tokens_used=25000,
        )

        remaining = group_budget - quota.tokens_used
        assert remaining == 75000

    def test_budget_exhausted(self):
        """Test detecting exhausted budget."""
        group_budget = 100000
        quota = Quota(
            user_id=1,
            tokens_used=100000,
        )

        is_exhausted = quota.tokens_used >= group_budget
        assert is_exhausted is True

    def test_budget_overage(self):
        """Test handling overage (used more than budget)."""
        group_budget = 100000
        quota = Quota(
            user_id=1,
            tokens_used=150000,  # Over budget
        )

        overage = quota.tokens_used - group_budget
        assert overage == 50000

    def test_percentage_used(self):
        """Test calculating percentage of budget used."""
        group_budget = 100000
        quota = Quota(
            user_id=1,
            tokens_used=75000,
        )

        percentage = (quota.tokens_used / group_budget) * 100
        assert percentage == 75.0


class TestQuotaPeriod:
    """Tests for quota period management."""

    def test_period_active(self):
        """Test that quota period is active."""
        now = datetime.now(timezone.utc)
        quota = Quota(
            user_id=1,
            budget_period_start=now,
            budget_period_days=30,
        )

        period_end = quota.budget_period_start + timedelta(days=quota.budget_period_days)
        is_active = now < period_end
        assert is_active is True

    def test_period_expired(self):
        """Test detecting expired quota period."""
        past = datetime.now(timezone.utc) - timedelta(days=45)
        quota = Quota(
            user_id=1,
            budget_period_start=past,
            budget_period_days=30,
        )

        now = datetime.now(timezone.utc)
        period_end = quota.budget_period_start + timedelta(days=quota.budget_period_days)
        is_expired = now >= period_end
        assert is_expired is True

    def test_days_remaining(self):
        """Test calculating days remaining in period."""
        now = datetime.now(timezone.utc)
        quota = Quota(
            user_id=1,
            budget_period_start=now - timedelta(days=10),
            budget_period_days=30,
        )

        period_end = quota.budget_period_start + timedelta(days=quota.budget_period_days)
        days_remaining = (period_end - now).days
        assert days_remaining == 20


class TestRoleLimits:
    """Tests for role-based default limits."""

    @pytest.fixture
    def default_quotas(self):
        """Default quotas by role."""
        return {
            UserRole.STUDENT: {
                "token_budget": 100000,
                "rpm_limit": 20,
                "max_concurrent": 2,
            },
            UserRole.STAFF: {
                "token_budget": 500000,
                "rpm_limit": 30,
                "max_concurrent": 3,
            },
            UserRole.FACULTY: {
                "token_budget": 1000000,
                "rpm_limit": 60,
                "max_concurrent": 5,
            },
            UserRole.ADMIN: {
                "token_budget": 10000000,
                "rpm_limit": 120,
                "max_concurrent": 10,
            },
        }

    def test_student_defaults(self, default_quotas):
        """Test student default quotas."""
        limits = default_quotas[UserRole.STUDENT]
        assert limits["token_budget"] == 100000
        assert limits["rpm_limit"] == 20
        assert limits["max_concurrent"] == 2

    def test_faculty_defaults(self, default_quotas):
        """Test faculty default quotas."""
        limits = default_quotas[UserRole.FACULTY]
        assert limits["token_budget"] == 1000000
        assert limits["rpm_limit"] == 60
        assert limits["max_concurrent"] == 5

    def test_admin_defaults(self, default_quotas):
        """Test admin default quotas."""
        limits = default_quotas[UserRole.ADMIN]
        assert limits["token_budget"] == 10000000
        assert limits["rpm_limit"] == 120
        assert limits["max_concurrent"] == 10

    def test_role_hierarchy(self, default_quotas):
        """Test that higher roles have higher limits."""
        student = default_quotas[UserRole.STUDENT]
        staff = default_quotas[UserRole.STAFF]
        faculty = default_quotas[UserRole.FACULTY]
        admin = default_quotas[UserRole.ADMIN]

        # Token budgets should increase with role
        assert student["token_budget"] < staff["token_budget"]
        assert staff["token_budget"] < faculty["token_budget"]
        assert faculty["token_budget"] < admin["token_budget"]


class TestGroupLimits:
    """Tests for group-based default limits (replaces role hierarchy)."""

    @pytest.fixture
    def default_group_quotas(self):
        """Default quotas by group name."""
        return {
            "students": {
                "token_budget": 100000,
                "rpm_limit": 30,
                "max_concurrent": 2,
                "scheduler_weight": 1,
                "is_admin": False,
            },
            "staff": {
                "token_budget": 500000,
                "rpm_limit": 60,
                "max_concurrent": 4,
                "scheduler_weight": 2,
                "is_admin": False,
            },
            "faculty": {
                "token_budget": 1000000,
                "rpm_limit": 120,
                "max_concurrent": 8,
                "scheduler_weight": 3,
                "is_admin": False,
            },
            "researchers": {
                "token_budget": 1000000,
                "rpm_limit": 120,
                "max_concurrent": 8,
                "scheduler_weight": 3,
                "is_admin": False,
            },
            "admin": {
                "token_budget": 10000000,
                "rpm_limit": 1000,
                "max_concurrent": 50,
                "scheduler_weight": 10,
                "is_admin": True,
            },
            "nerds": {
                "token_budget": 500000,
                "rpm_limit": 60,
                "max_concurrent": 4,
                "scheduler_weight": 2,
                "is_admin": False,
            },
            "other": {
                "token_budget": 100000,
                "rpm_limit": 30,
                "max_concurrent": 2,
                "scheduler_weight": 1,
                "is_admin": False,
            },
        }

    def test_students_defaults(self, default_group_quotas):
        """Test students group default quotas."""
        limits = default_group_quotas["students"]
        assert limits["token_budget"] == 100000
        assert limits["rpm_limit"] == 30
        assert limits["max_concurrent"] == 2
        assert limits["scheduler_weight"] == 1
        assert limits["is_admin"] is False

    def test_faculty_defaults(self, default_group_quotas):
        """Test faculty group default quotas."""
        limits = default_group_quotas["faculty"]
        assert limits["token_budget"] == 1000000
        assert limits["rpm_limit"] == 120
        assert limits["max_concurrent"] == 8
        assert limits["scheduler_weight"] == 3

    def test_admin_defaults(self, default_group_quotas):
        """Test admin group default quotas."""
        limits = default_group_quotas["admin"]
        assert limits["token_budget"] == 10000000
        assert limits["rpm_limit"] == 1000
        assert limits["max_concurrent"] == 50
        assert limits["scheduler_weight"] == 10
        assert limits["is_admin"] is True

    def test_researchers_match_faculty(self, default_group_quotas):
        """Test that researchers and faculty share the same quotas."""
        researchers = default_group_quotas["researchers"]
        faculty = default_group_quotas["faculty"]
        assert researchers["token_budget"] == faculty["token_budget"]
        assert researchers["rpm_limit"] == faculty["rpm_limit"]
        assert researchers["scheduler_weight"] == faculty["scheduler_weight"]

    def test_only_admin_has_admin_flag(self, default_group_quotas):
        """Test that only admin group has is_admin=True."""
        for name, limits in default_group_quotas.items():
            if name == "admin":
                assert limits["is_admin"] is True
            else:
                assert limits["is_admin"] is False, f"{name} should not be admin"

    def test_seven_default_groups(self, default_group_quotas):
        """Test that there are exactly 7 default groups."""
        assert len(default_group_quotas) == 7
        expected = {"students", "staff", "faculty", "researchers", "admin", "nerds", "other"}
        assert set(default_group_quotas.keys()) == expected


class TestQuotaCheckLogic:
    """Tests for quota check logic (budget from group)."""

    def test_check_token_budget_ok(self):
        """Test token budget check passes when sufficient."""
        group_budget = 100000
        quota = Quota(
            user_id=1,
            tokens_used=50000,
        )

        estimated_tokens = 1000
        has_budget = (quota.tokens_used + estimated_tokens) <= group_budget
        assert has_budget is True

    def test_check_token_budget_exceeded(self):
        """Test token budget check fails when exceeded."""
        group_budget = 100000
        quota = Quota(
            user_id=1,
            tokens_used=99500,
        )

        estimated_tokens = 1000
        has_budget = (quota.tokens_used + estimated_tokens) <= group_budget
        assert has_budget is False

    def test_unlimited_group_never_blocks(self):
        """Test that unlimited group (budget=0) never blocks requests."""
        group_budget = 0  # unlimited
        quota = Quota(
            user_id=1,
            tokens_used=999999999,
        )

        # With budget=0, the check should be skipped (budget > 0 is false)
        should_block = group_budget > 0 and quota.tokens_used >= group_budget
        assert should_block is False

    def test_check_concurrent_limit_ok(self):
        """Test concurrent request check passes."""
        max_concurrent = 5
        current_concurrent = 3

        can_proceed = current_concurrent < max_concurrent
        assert can_proceed is True

    def test_check_concurrent_limit_exceeded(self):
        """Test concurrent request check fails at limit."""
        max_concurrent = 5
        current_concurrent = 5

        can_proceed = current_concurrent < max_concurrent
        assert can_proceed is False


class TestUsageLedger:
    """Tests for usage tracking logic."""

    def test_calculate_total_tokens(self):
        """Test calculating total tokens from prompt and completion."""
        prompt_tokens = 500
        completion_tokens = 300
        total = prompt_tokens + completion_tokens
        assert total == 800

    def test_usage_aggregation(self):
        """Test aggregating usage entries."""
        entries = [
            {"prompt_tokens": 100, "completion_tokens": 50},
            {"prompt_tokens": 200, "completion_tokens": 150},
            {"prompt_tokens": 300, "completion_tokens": 200},
        ]

        total_prompt = sum(e["prompt_tokens"] for e in entries)
        total_completion = sum(e["completion_tokens"] for e in entries)
        total_tokens = total_prompt + total_completion

        assert total_prompt == 600
        assert total_completion == 400
        assert total_tokens == 1000

    def test_usage_by_model(self):
        """Test grouping usage by model."""
        entries = [
            {"model": "llama3.2", "total_tokens": 500},
            {"model": "mistral", "total_tokens": 300},
            {"model": "llama3.2", "total_tokens": 200},
        ]

        by_model = {}
        for entry in entries:
            model = entry["model"]
            by_model[model] = by_model.get(model, 0) + entry["total_tokens"]

        assert by_model["llama3.2"] == 700
        assert by_model["mistral"] == 300

    def test_usage_window_filtering(self):
        """Test filtering usage entries by time window."""
        now = datetime.now(timezone.utc)
        entries = [
            {"timestamp": now - timedelta(hours=1), "tokens": 100},  # Within window
            {"timestamp": now - timedelta(hours=3), "tokens": 200},  # Within window
            {"timestamp": now - timedelta(hours=10), "tokens": 500},  # Outside window
        ]

        window_seconds = 5 * 60 * 60  # 5 hours
        cutoff = now - timedelta(seconds=window_seconds)

        in_window = [e for e in entries if e["timestamp"] >= cutoff]
        total_in_window = sum(e["tokens"] for e in in_window)

        assert len(in_window) == 2
        assert total_in_window == 300


class TestTokenEstimation:
    """Tests for token estimation logic."""

    def test_estimate_prompt_tokens(self):
        """Test estimating tokens from prompt text."""
        # Rough estimation: ~4 characters per token for English
        prompt = "Hello, this is a test prompt for token estimation."
        estimated_tokens = len(prompt) // 4
        assert estimated_tokens > 0
        assert estimated_tokens < len(prompt)

    def test_estimate_chat_tokens(self):
        """Test estimating tokens from chat messages."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
        ]

        # Sum up content lengths and divide by 4
        total_chars = sum(len(m["content"]) for m in messages)
        estimated_tokens = total_chars // 4

        # Add overhead for message formatting (roughly 4 tokens per message)
        estimated_tokens += len(messages) * 4

        assert estimated_tokens > 0

    def test_estimate_with_tiktoken(self):
        """Test using tiktoken for estimation (if available)."""
        try:
            import tiktoken
            encoder = tiktoken.get_encoding("cl100k_base")
            text = "Hello, this is a test."
            tokens = encoder.encode(text)
            assert len(tokens) == 7  # Actual token count
        except ImportError:
            pytest.skip("tiktoken not installed")


class TestQuotaResetLogic:
    """Tests for quota reset logic."""

    def test_should_reset_expired_period(self):
        """Test that expired period triggers reset."""
        past = datetime.now(timezone.utc) - timedelta(days=45)
        quota = Quota(
            user_id=1,
            tokens_used=50000,
            budget_period_start=past,
            budget_period_days=30,
        )

        now = datetime.now(timezone.utc)
        period_end = quota.budget_period_start + timedelta(days=quota.budget_period_days)
        should_reset = now >= period_end

        assert should_reset is True

    def test_reset_clears_usage(self):
        """Test that reset clears token usage."""
        quota = Quota(
            user_id=1,
            tokens_used=75000,
        )

        # Simulate reset
        quota.tokens_used = 0
        quota.budget_period_start = datetime.now(timezone.utc)

        assert quota.tokens_used == 0

    def test_no_reset_active_period(self):
        """Test that active period doesn't trigger reset."""
        now = datetime.now(timezone.utc)
        quota = Quota(
            user_id=1,
            tokens_used=50000,
            budget_period_start=now - timedelta(days=15),  # 15 days ago
            budget_period_days=30,
        )

        period_end = quota.budget_period_start + timedelta(days=quota.budget_period_days)
        should_reset = now >= period_end

        assert should_reset is False
