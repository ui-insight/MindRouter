############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# settings.py: Application configuration and environment settings
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Application settings using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _get_version() -> str:
    """Read version from pyproject.toml (single source of truth)."""
    try:
        from importlib.metadata import version
        return version("mindrouter")
    except Exception:
        pass
    # Fallback: read pyproject.toml directly (works in dev without pip install)
    try:
        import tomllib
        toml_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        return data["project"]["version"]
    except Exception:
        return "0.0.0"


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.prod"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "MindRouter"
    app_version: str = Field(default_factory=_get_version)
    app_base_url: str = "https://mindrouter.uidaho.edu"
    debug: bool = False
    reload: bool = False

    # MCP server (standalone single-worker process)
    mcp_server_url: str = "http://127.0.0.1:8001"

    # Database
    database_url: str = Field(
        default="mysql+pymysql://mindrouter:mindrouter_password@localhost:3306/mindrouter"
    )
    database_pool_size: int = 30
    database_max_overflow: int = 20
    database_echo: bool = False

    # Archive Database (optional — enables tiered data retention)
    archive_database_url: Optional[str] = None

    # Redis (optional)
    redis_url: Optional[str] = None

    # Security
    secret_key: str = Field(default="dev-secret-key-change-in-production")
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    session_cookie_name: str = "mindrouter_session"
    session_cookie_secure: bool = False
    session_cookie_httponly: bool = True
    session_cookie_samesite: str = "lax"
    api_key_hash_algorithm: str = "argon2"

    # Azure AD SSO
    azure_ad_client_id: Optional[str] = None
    azure_ad_client_secret: Optional[str] = None
    azure_ad_tenant_id: Optional[str] = None
    azure_ad_redirect_uri: str = "https://your-domain.example.com/login/azure/authorized"
    azure_ad_default_group: str = "other"

    # Artifact Storage
    artifact_storage_path: str = "/data/artifacts"
    artifact_max_size_mb: int = 50
    artifact_retention_days: int = 365

    # Video Generation
    # Every value here MUST also be added to docker-compose.yml as
    # `- NEW_VAR=${NEW_VAR:-<default>}` (pydantic-settings reads env_file only
    # inside the container; .env.prod is not mounted). See
    # docs/video-generation-plan.md. Per-model tunables (presets, quotas,
    # policy) live as vid.* rows in app_config, NOT here.
    video_storage_path: str = "/data/video"
    video_runner_enabled: bool = True
    video_runner_poll_interval_seconds: int = 5
    video_worker_timeout_seconds: int = 60          # control-plane calls (submit/poll/cancel)
    video_worker_fetch_timeout_seconds: int = 900   # artifact fetch (worker -> gateway, large)
    video_job_max_wall_seconds: int = 3600
    video_job_stale_heartbeat_seconds: int = 120
    video_max_upload_mb: int = 64
    video_webhook_signing_key: str = ""             # host .env only, never in repo

    # Default Quotas - per role (deprecated: use Group DB defaults instead)
    default_token_budget_student: int = 100000
    default_rpm_student: int = 30

    # Default Quotas - Staff
    default_token_budget_staff: int = 500000
    default_rpm_staff: int = 60

    # Default Quotas - Faculty
    default_token_budget_faculty: int = 1000000
    default_rpm_faculty: int = 120

    # Default Quotas - Admin
    default_token_budget_admin: int = 10000000
    default_rpm_admin: int = 1000

    # Scheduler Weights (deprecated: use Group.scheduler_weight instead)
    scheduler_weight_student: int = 1
    scheduler_weight_staff: int = 2
    scheduler_weight_faculty: int = 3
    scheduler_weight_admin: int = 10

    # Scheduler Configuration
    scheduler_fairness_window: int = 300  # seconds
    scheduler_deprioritize_threshold: float = 0.5

    # Scheduler Scoring
    scheduler_score_model_loaded: int = 100
    scheduler_score_low_utilization: int = 50
    scheduler_score_latency: int = 40
    scheduler_score_short_queue: int = 30
    scheduler_score_high_throughput: int = 20

    # Latency Tracking
    latency_ema_alpha: float = 0.3
    latency_ema_persist_interval: int = 30

    # Backend Registry
    backend_poll_interval: int = 30
    backend_health_timeout: int = 5
    backend_unhealthy_threshold: int = 3
    backend_circuit_breaker_threshold: int = 3
    backend_circuit_breaker_recovery_seconds: int = 30
    backend_adaptive_poll_fast_interval: int = 10
    backend_adaptive_poll_fast_duration: int = 120

    # Request Handling
    max_request_size: int = 52428800  # 50MB
    backend_request_timeout: int = 300
    backend_request_timeout_per_attempt: int = 180
    backend_retry_max_attempts: int = 3
    structured_output_retry_on_invalid: bool = True
    # Gateway policy: reasoning/thinking is OFF by default unless the client
    # explicitly opts in (think:true / thinking:{type:enabled} / reasoning_effort).
    # Applies to enable_thinking-style models (Qwen, Gemma, Nemotron); gpt-oss
    # uses reasoning_effort and is left untouched. Set false to restore the old
    # per-model launch defaults.
    thinking_off_by_default: bool = True

    # Public website publishing (mindrouter.ai static site). Selected blog posts
    # are committed to the mindrouter-website repo via the GitHub API. The repo
    # is additionally hard-allowlisted in website_publisher.py so it can never
    # target another repository. Needs a fine-grained PAT with Contents:RW.
    website_publish_enabled: bool = False
    website_publish_repo: str = "sheneman/mindrouter-website"
    website_publish_branch: str = "main"
    website_publish_github_token: str = ""

    # Startup: opt-in `alembic upgrade head` before serving (env RUN_MIGRATIONS=1)
    # so a fresh/unmigrated database doesn't crash-loop the app. Off by default
    # so existing deploys are unaffected; run single-worker on first boot.
    run_migrations: bool = False

    # Request-field validation: 'off' | 'log' | 'enforce'. Surfaces vLLM-dialect
    # or unknown request fields that would otherwise be silently dropped. Deploy
    # at 'log' to observe real traffic, then flip to 'enforce'.
    field_validation: str = "log"

    # OpenAI Responses API (/v1/responses)
    responses_api_enabled: bool = True
    responses_store_max_chain_depth: int = 20  # previous_response_id hops
    responses_store_max_payload_bytes: int = 5242880  # 5MB; 0 = uncapped
    responses_store_max_rows_per_user: int = 1000  # 0 = uncapped; oldest evicted
    # Hosted web_search tool ({"type":"web_search"}) — executed server-side
    # via the /v1/search provider stack
    responses_web_search_enabled: bool = True
    responses_web_search_max_calls: int = 4  # per response; max_tool_calls can lower it
    responses_web_search_max_results: int = 5  # results fed to the model per search
    # Conversations API (conv_* objects)
    conversations_max_per_user: int = 1000  # 0 = uncapped; create rejected beyond
    conversations_max_items: int = 10000  # per conversation; appends rejected beyond
    conversations_max_item_bytes: int = 2097152  # 2MB per item (post-offload); 0 = uncapped

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: Optional[str] = None

    # Audit Logging
    audit_log_enabled: bool = True
    audit_log_prompts: bool = True
    audit_log_responses: bool = True

    # Telemetry & GPU Metrics
    telemetry_retention_days: int = 30
    telemetry_cleanup_interval: int = 3600  # seconds
    sidecar_timeout: int = 15  # seconds for sidecar HTTP calls

    # Observability
    metrics_enabled: bool = True
    metrics_prefix: str = "mindrouter"
    otel_enabled: bool = False
    otel_exporter_otlp_endpoint: Optional[str] = None
    otel_service_name: str = "mindrouter"

    # CORS
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Chat UI
    chat_files_path: str = "/data/chat_files"
    chat_upload_max_size_mb: int = 10
    chat_upload_allowed_extensions: List[str] = [
        ".txt", ".md", ".csv", ".json", ".html", ".htm", ".log",
        ".docx", ".xlsx", ".pptx", ".pdf",
        ".jpg", ".jpeg", ".png", ".gif", ".webp",
    ]

    # Conversation Retention
    conversation_retention_days: int = 730  # 2 years
    conversation_cleanup_interval: int = 86400  # seconds (24 hours)

    # Web Search (Brave)
    brave_search_api_key: Optional[str] = None
    brave_search_max_results: int = 5

    # Tokenizer
    default_tokenizer: str = "cl100k_base"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [origin.strip() for origin in v.split(",")]
        return v

    @property
    def azure_ad_enabled(self) -> bool:
        """Check if Azure AD SSO is configured."""
        return bool(self.azure_ad_client_id and self.azure_ad_tenant_id)

    def get_quota_defaults(self, role: str) -> dict:
        """Get default quota settings for a role."""
        role_lower = role.lower()
        return {
            "token_budget": getattr(self, f"default_token_budget_{role_lower}", 100000),
            "rpm": getattr(self, f"default_rpm_{role_lower}", 30),
        }

    def get_scheduler_weight(self, role: str) -> int:
        """Get scheduler weight for a role."""
        role_lower = role.lower()
        return getattr(self, f"scheduler_weight_{role_lower}", 1)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
