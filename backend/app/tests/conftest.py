############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# conftest.py: Pytest configuration and shared test fixtures
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Pytest configuration and shared fixtures for MindRouter2 tests."""

import asyncio
import pytest
from datetime import datetime, timezone
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

# Configure pytest-asyncio
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = MagicMock()

    # Database settings
    settings.database_url = "sqlite+aiosqlite:///:memory:"

    # Scheduler settings
    settings.scheduler_fairness_window = 300
    settings.scheduler_deprioritize_threshold = 0.5
    settings.scheduler_score_model_loaded = 100
    settings.scheduler_score_low_utilization = 50
    settings.scheduler_score_short_queue = 30
    settings.scheduler_score_high_throughput = 20

    # Role weights
    settings.get_scheduler_weight = MagicMock(
        side_effect=lambda role: {
            "student": 1,
            "staff": 2,
            "faculty": 3,
            "admin": 10,
        }.get(role, 1)
    )

    # Quota defaults
    settings.default_quota_student = 100000
    settings.default_quota_staff = 500000
    settings.default_quota_faculty = 1000000
    settings.default_quota_admin = 10000000

    settings.default_rpm_student = 20
    settings.default_rpm_staff = 30
    settings.default_rpm_faculty = 60
    settings.default_rpm_admin = 120

    return settings


@pytest.fixture
def sample_chat_messages():
    """Sample chat messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
        {"role": "user", "content": "What's the weather like?"},
    ]


@pytest.fixture
def sample_openai_request():
    """Sample OpenAI-format request for testing."""
    return {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        "temperature": 0.7,
        "max_tokens": 100,
        "stream": False,
    }


@pytest.fixture
def sample_ollama_request():
    """Sample Ollama-format request for testing."""
    return {
        "model": "llama3.2",
        "messages": [
            {"role": "user", "content": "Hello!"},
        ],
        "stream": True,
        "options": {
            "temperature": 0.7,
            "num_predict": 100,
        },
    }


@pytest.fixture
def sample_multimodal_request():
    """Sample multimodal request with image for testing."""
    return {
        "model": "gpt-4-vision",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.png"},
                    },
                ],
            }
        ],
    }


@pytest.fixture
def sample_json_schema():
    """Sample JSON schema for structured output testing."""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0},
            "email": {"type": "string"},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["name", "email"],
    }


@pytest.fixture
def mock_backend():
    """Create a mock backend for testing."""
    from backend.app.db.models import Backend, BackendEngine, BackendStatus

    backend = MagicMock(spec=Backend)
    backend.id = 1
    backend.name = "test-backend"
    backend.url = "http://localhost:11434"
    backend.engine = BackendEngine.OLLAMA
    backend.status = BackendStatus.HEALTHY
    backend.supports_multimodal = True
    backend.supports_embeddings = True
    backend.supports_structured_output = True
    backend.current_concurrent = 0
    backend.max_concurrent = 10
    backend.gpu_memory_gb = 24.0
    backend.gpu_type = "NVIDIA RTX 4090"
    backend.throughput_score = 1.0
    backend.priority = 0
    backend.consecutive_failures = 0

    return backend


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    from backend.app.db.models import Model, Modality

    model = MagicMock(spec=Model)
    model.id = 1
    model.backend_id = 1
    model.name = "llama3.2"
    model.modality = Modality.CHAT
    model.context_length = 4096
    model.supports_multimodal = True
    model.supports_structured_output = True
    model.is_loaded = True
    model.vram_required_gb = 8.0

    return model


@pytest.fixture
def mock_group():
    """Create a mock group for testing."""
    group = MagicMock()
    group.id = 3
    group.name = "faculty"
    group.display_name = "Faculty"
    group.token_budget = 1000000
    group.rpm_limit = 120
    group.max_concurrent = 8
    group.scheduler_weight = 3
    group.is_admin = False
    return group


@pytest.fixture
def mock_admin_group():
    """Create a mock admin group for testing."""
    group = MagicMock()
    group.id = 5
    group.name = "admin"
    group.display_name = "Admin"
    group.token_budget = 10000000
    group.rpm_limit = 1000
    group.max_concurrent = 50
    group.scheduler_weight = 10
    group.is_admin = True
    return group


@pytest.fixture
def mock_user(mock_group):
    """Create a mock user for testing."""
    from backend.app.db.models import User, UserRole

    user = MagicMock(spec=User)
    user.id = 1
    user.uuid = "test-user-uuid"
    user.username = "testuser"
    user.email = "test@example.com"
    user.role = UserRole.FACULTY
    user.is_active = True
    user.created_at = datetime.now(timezone.utc)
    user.group_id = mock_group.id
    user.group = mock_group

    return user


@pytest.fixture
def mock_api_key():
    """Create a mock API key for testing."""
    from backend.app.db.models import ApiKey, ApiKeyStatus

    api_key = MagicMock(spec=ApiKey)
    api_key.id = 1
    api_key.user_id = 1
    api_key.key_prefix = "mr_test"
    api_key.name = "Test Key"
    api_key.status = ApiKeyStatus.ACTIVE
    api_key.expires_at = None
    api_key.usage_count = 0

    return api_key


@pytest.fixture
def mock_quota():
    """Create a mock quota for testing."""
    from backend.app.db.models import Quota

    quota = MagicMock(spec=Quota)
    quota.id = 1
    quota.user_id = 1
    quota.tokens_used = 0
    quota.budget_period_start = datetime.now(timezone.utc)
    quota.budget_period_days = 30
    quota.rpm_limit = 60
    quota.max_concurrent = 5
    quota.weight_override = None

    return quota


# --- Structured output fixtures ---


@pytest.fixture
def simple_json_schema():
    """Flat object schema: {name: str, age: int}."""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
    }


@pytest.fixture
def nested_json_schema():
    """Nested object schema with address and tags."""
    return {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "address": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "zip": {"type": "string"},
                        },
                        "required": ["city", "zip"],
                    },
                },
                "required": ["name", "address"],
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["user", "tags"],
    }


@pytest.fixture
def enum_json_schema():
    """Schema with enum values."""
    return {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["active", "inactive"],
            },
            "priority": {"type": "integer"},
        },
        "required": ["status", "priority"],
    }


@pytest.fixture
def array_schema():
    """Array of objects schema."""
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "quantity": {"type": "integer"},
                    },
                    "required": ["name", "quantity"],
                },
            },
        },
        "required": ["items"],
    }


@pytest.fixture
def complex_schema():
    """Realistic LLM use case schema — list of countries."""
    return {
        "type": "object",
        "properties": {
            "countries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "capital": {"type": "string"},
                        "population": {"type": "integer"},
                        "languages": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["name", "capital", "population", "languages"],
                },
            },
        },
        "required": ["countries"],
    }


# --- Mock streaming data fixtures ---


@pytest.fixture
def ollama_stream_chunks():
    """Simulated Ollama ndjson stream chunks."""
    import json

    return [
        json.dumps({
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "Hello"},
            "done": False,
        }).encode() + b"\n",
        json.dumps({
            "model": "llama3.2",
            "message": {"role": "assistant", "content": " there"},
            "done": False,
        }).encode() + b"\n",
        json.dumps({
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "!"},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 3,
            "total_duration": 500000000,
        }).encode() + b"\n",
    ]


@pytest.fixture
def vllm_sse_chunks():
    """Simulated vLLM/OpenAI SSE stream chunks."""
    import json

    return [
        (
            "data: "
            + json.dumps({
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "llama3.2",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
            })
            + "\n\n"
        ).encode(),
        (
            "data: "
            + json.dumps({
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "llama3.2",
                "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
            })
            + "\n\n"
        ).encode(),
        (
            "data: "
            + json.dumps({
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "llama3.2",
                "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": None}],
            })
            + "\n\n"
        ).encode(),
        (
            "data: "
            + json.dumps({
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "llama3.2",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            })
            + "\n\n"
        ).encode(),
        b"data: [DONE]\n\n",
    ]


# --- Cross-engine mock backend fixtures ---


@pytest.fixture
def mock_ollama_backend():
    """Mock backend with engine=OLLAMA."""
    backend = MagicMock()
    backend.id = 1
    backend.name = "ollama-backend"
    backend.url = "http://localhost:11434"
    # Use a simple object with .value for engine to avoid importing db.models
    engine = MagicMock()
    engine.value = "ollama"
    engine.__eq__ = lambda self, other: getattr(other, 'value', other) == "ollama"
    backend.engine = engine
    status = MagicMock()
    status.value = "healthy"
    backend.status = status
    backend.supports_multimodal = True
    backend.supports_embeddings = True
    backend.supports_structured_output = True
    backend.current_concurrent = 0
    backend.max_concurrent = 10
    backend.gpu_memory_gb = 24.0
    backend.throughput_score = 1.0
    backend.priority = 0
    backend.consecutive_failures = 0
    return backend


@pytest.fixture
def mock_vllm_backend():
    """Mock backend with engine=VLLM."""
    backend = MagicMock()
    backend.id = 2
    backend.name = "vllm-backend"
    backend.url = "http://localhost:8000"
    engine = MagicMock()
    engine.value = "vllm"
    engine.__eq__ = lambda self, other: getattr(other, 'value', other) == "vllm"
    backend.engine = engine
    status = MagicMock()
    status.value = "healthy"
    backend.status = status
    backend.supports_multimodal = True
    backend.supports_embeddings = True
    backend.supports_structured_output = True
    backend.current_concurrent = 0
    backend.max_concurrent = 10
    backend.gpu_memory_gb = 24.0
    backend.throughput_score = 1.0
    backend.priority = 0
    backend.consecutive_failures = 0
    return backend


# --- Full parameter request fixtures ---


@pytest.fixture
def all_params_ollama_request():
    """Ollama request with every supported option set."""
    return {
        "model": "llama3.2",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        "stream": True,
        "format": "json",
        "options": {
            "temperature": 0.8,
            "top_p": 0.95,
            "num_predict": 256,
            "stop": ["\n", "END"],
            "presence_penalty": 0.5,
            "frequency_penalty": 0.3,
            "seed": 42,
        },
    }


@pytest.fixture
def all_params_openai_request():
    """OpenAI request with every supported parameter set."""
    return {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": 256,
        "stream": False,
        "stop": ["\n", "END"],
        "presence_penalty": 0.5,
        "frequency_penalty": 0.3,
        "seed": 42,
        "n": 1,
        "user": "test-user",
        "response_format": {"type": "json_object"},
    }


@pytest.fixture
def all_params_ollama_request_extended():
    """Ollama request with ALL options including extended params."""
    return {
        "model": "llama3.2",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        "stream": True,
        "format": "json",
        "think": True,
        "options": {
            "temperature": 0.8,
            "top_p": 0.95,
            "num_predict": 256,
            "stop": ["\n", "END"],
            "presence_penalty": 0.5,
            "frequency_penalty": 0.3,
            "seed": 42,
            "top_k": 40,
            "repeat_penalty": 1.2,
            "min_p": 0.05,
            "mirostat": 2,
            "mirostat_tau": 5.0,
            "num_ctx": 4096,
        },
    }


@pytest.fixture
def all_params_openai_request_extended():
    """OpenAI request with ALL params including extended params."""
    return {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": 256,
        "stream": False,
        "stop": ["\n", "END"],
        "presence_penalty": 0.5,
        "frequency_penalty": 0.3,
        "seed": 42,
        "top_k": 50,
        "repetition_penalty": 1.3,
        "min_p": 0.1,
        "n": 1,
        "user": "test-user",
        "response_format": {"type": "json_object"},
    }


@pytest.fixture
def sample_completion_request():
    """Sample OpenAI /v1/completions request."""
    return {
        "model": "gpt-3.5-turbo-instruct",
        "prompt": "Once upon a time",
        "temperature": 0.7,
        "max_tokens": 100,
        "top_k": 40,
        "repetition_penalty": 1.1,
        "min_p": 0.05,
    }


@pytest.fixture
def sample_generate_request():
    """Sample Ollama /api/generate request."""
    return {
        "model": "llama3.2",
        "prompt": "Once upon a time",
        "system": "You are a storyteller.",
        "stream": True,
        "options": {
            "temperature": 0.7,
            "num_predict": 100,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "min_p": 0.05,
        },
    }
