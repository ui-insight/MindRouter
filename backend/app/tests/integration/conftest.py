"""Integration test configuration — shared CLI options and fixtures."""

import pytest


def pytest_addoption(parser):
    """Add CLI options for integration tests."""
    parser.addoption(
        "--api-key",
        action="store",
        default=None,
        help="MindRouter2 API key for live tests",
    )
    parser.addoption(
        "--base-url",
        action="store",
        default="https://localhost:8000",
        help="MindRouter2 base URL (default: https://localhost:8000)",
    )


@pytest.fixture(scope="session")
def api_key(request):
    """API key from --api-key CLI option."""
    key = request.config.getoption("--api-key")
    if not key:
        pytest.skip("--api-key not provided")
    return key


@pytest.fixture(scope="session")
def base_url(request):
    """Base URL from --base-url CLI option."""
    return request.config.getoption("--base-url")
