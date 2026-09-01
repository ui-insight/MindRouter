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
    # Public HTTPS origin of THIS deployment (scheme + host, no trailing path).
    # Must be set per deployment: SSO redirect URIs (Azure/Google/OIDC) and SAML
    # Destination/Recipient validation are derived from it, so a wrong value
    # sends users to another org's host and fails as redirect_uri_mismatch.
    app_base_url: str = "https://your-domain.example.com"
    debug: bool = False
    reload: bool = False

    # MCP server (standalone single-worker process, legacy SSE transport only)
    mcp_server_url: str = "http://127.0.0.1:8001"

    # Streamable HTTP transport (POST /mcp), served from the main app.
    # Kill switch: turning this off leaves only the legacy /mcp/sse path.
    mcp_streamable_enabled: bool = True
    # False = SSE framing inside the POST response (spec default).
    # True = a single application/json body. Flip to True if the fronting
    # Apache proxy is found to buffer the streamed response.
    mcp_streamable_json_response: bool = False

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

    # Login throttling — cap failed login attempts per source within a rolling
    # window (brute-force / credential-stuffing guard). Enforcement lives in
    # the login path; these are the tunables it reads.
    login_max_attempts_per_window: int = 15
    login_attempt_window_seconds: int = 300

    # CSRF: for state-changing browser requests (POST/PUT/PATCH/DELETE) the
    # Origin/Referer host must be this deployment's own host or one of these
    # explicitly trusted origins (scheme+host, e.g. "https://app.example.com").
    # API-key/server-to-server clients send no Origin and are unaffected.
    csrf_trusted_origins: List[str] = []

    # Azure AD SSO
    azure_ad_client_id: Optional[str] = None
    azure_ad_client_secret: Optional[str] = None
    azure_ad_tenant_id: Optional[str] = None
    azure_ad_redirect_uri: str = "https://your-domain.example.com/login/azure/authorized"
    azure_ad_default_group: str = "other"

    # Google SSO (OIDC; set client id+secret to enable)
    google_sso_client_id: Optional[str] = None
    google_sso_client_secret: Optional[str] = None
    google_sso_redirect_uri: Optional[str] = None  # default: <public URL>/login/google/authorized
    google_sso_hosted_domain: Optional[str] = None  # restrict to a Workspace domain (hd claim)
    google_sso_default_group: str = "other"

    # Generic OIDC SSO (Okta, Keycloak, Auth0, CILogon/InCommon, ...; set issuer+client to enable)
    oidc_sso_issuer: Optional[str] = None  # issuer base URL; discovery at <issuer>/.well-known/openid-configuration
    oidc_sso_client_id: Optional[str] = None
    oidc_sso_client_secret: Optional[str] = None
    oidc_sso_redirect_uri: Optional[str] = None  # default: <public URL>/login/oidc/authorized
    oidc_sso_display_name: str = "SSO"  # login button label ("Sign in with <name>")
    oidc_sso_scopes: str = "openid profile email"
    oidc_sso_default_group: str = "other"

    # SAML 2.0 SSO (Shibboleth/InCommon IdPs, ADFS, ...; set SP entity id + IdP metadata to enable)
    saml_sp_entity_id: Optional[str] = None
    saml_sp_acs_url: Optional[str] = None  # default: <public URL>/login/saml/acs
    saml_idp_metadata_url: Optional[str] = None  # fetch IdP entity/SSO URL/cert from metadata
    saml_idp_entity_id: Optional[str] = None  # explicit alternative to metadata URL
    saml_idp_sso_url: Optional[str] = None
    saml_idp_x509_cert: Optional[str] = None
    saml_display_name: str = "SSO"
    saml_default_group: str = "other"
    # SP key pair — publishes a <KeyDescriptor> in SP metadata and enables
    # AuthnRequest signing / encrypted-assertion support.  Each value is
    # either inline PEM (literal "\n" escapes are accepted, for .env files)
    # or a path to a PEM file mounted into the container.  Conventionally a
    # long-lived SELF-SIGNED pair — trust comes from metadata registration,
    # not a CA chain — and NOT the web server's TLS certificate.
    saml_sp_x509_cert: Optional[str] = None
    saml_sp_private_key: Optional[str] = None
    # Both require a key pair; ignored (with a warning) without one.
    saml_authn_requests_signed: bool = False
    saml_want_assertions_encrypted: bool = False
    # Attribute names in the assertion (defaults follow eduPerson/InCommon conventions)
    saml_attr_email: str = "mail"
    saml_attr_name: str = "displayName"
    saml_attr_username: str = "eduPersonPrincipalName"

    # Artifact Storage.  Artifact rows/files are reaped with their parent
    # request by the retention cycle (Admin -> Retention) — retention/size
    # caps were never enforced from settings, so no such knobs exist here.
    artifact_storage_path: str = "/data/artifacts"

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
    # Shared secret sent to the worker as X-Worker-Key (host .env only, never in
    # repo). Empty = no header (legacy). Set the SAME value as the worker's
    # VIDEO_WORKER_API_KEY to enforce auth on the worker /v1/* routes.
    video_worker_api_key: str = ""
    # Cumulative byte cap for a single streamed worker artifact fetch, in MiB
    # (<=0 disables). Guards the gateway disk against a runaway/oversized render.
    video_worker_fetch_max_mb: int = 2048
    # Verify TLS on outbound calls to internal services (the video worker).
    # Secure default (verify ON); set INTERNAL_TLS_VERIFY=false for a worker on a
    # self-signed/private cert until it is fronted by nginx TLS with a trusted
    # chain. A no-op for plain http:// workers.
    internal_tls_verify: bool = True
    video_job_max_wall_seconds: int = 3600
    video_job_stale_heartbeat_seconds: int = 120
    video_reconcile_interval_seconds: int = 20  # ground-truth sweep for orphaned renders
    video_runner_lease_ttl_seconds: int = 30    # leader lease so only ONE runner is active
    video_max_upload_mb: int = 64
    video_webhook_signing_key: str = ""             # host .env only, never in repo

    # UI Branding / Theming
    # Admin-configurable look-and-feel (org name, logos, accent colors) stored as
    # branding.* rows in app_config; uploaded logo/favicon files live on disk here.
    # BRANDING_STORAGE_PATH must also be added to docker-compose.yml (see note on
    # video block above) and the named volume `branding_data` mounted there.
    branding_storage_path: str = "/data/branding"
    branding_max_logo_mb: int = 4                    # per-file cap for logo/favicon uploads

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
    # Max backend health checks run concurrently per poll sweep. Each check
    # opens a DB session, so an unbounded gather over every backend opens a
    # connection per backend at once — and with N uvicorn workers each polling,
    # that multiplies into the DB pool and can exhaust MariaDB max_connections.
    # Bounding it keeps peak poll connections ~= concurrency x workers instead
    # of backends x workers.
    backend_poll_concurrency: int = 8

    # Request Handling
    max_request_size: int = 52428800  # 50MB
    backend_request_timeout: int = 300
    backend_request_timeout_per_attempt: int = 180
    backend_retry_max_attempts: int = 3
    # Per-attempt ceiling for diffusion (image) backends. Image jobs are long
    # and legitimately variable (n x steps x size), so they get their own
    # budget instead of the chat per-attempt timeout, and a timeout is NOT
    # retried on another backend (see InferenceService._proxy_with_retry).
    backend_image_request_timeout: int = 600
    # A per-attempt timeout on a non-streaming chat request is almost always
    # the JOB (max_tokens 65536, a 30k-token answer racing the wall), not the
    # backend — vLLM aborts on disconnect and the engine is usually idle. So a
    # timeout neither counts toward the circuit breaker (5xx and connection
    # errors still do) nor is re-run on other replicas by default; the caller
    # gets a 504 telling it to stream or lower max_tokens. Both are settings
    # so the pre-2.9.62 behaviour (retry 3x, breaker trips) can be restored.
    backend_timeout_trips_breaker: bool = False
    backend_retry_on_timeout: bool = False
    structured_output_retry_on_invalid: bool = True
    # Upper bound on the OpenAI `n` parameter (completions per request) —
    # a large n multiplies backend load and cost.
    max_completions_n: int = 8
    # Rate limiting: when Redis is unavailable, fall back to a per-worker
    # in-process limiter instead of skipping the limit entirely. Default True.
    rate_limit_local_fallback: bool = True
    # Streaming write coalescing: buffer up to N framed SSE/NDJSON events per
    # socket write.  The T-ms bound applies while events keep arriving —
    # during a backend stall, buffered events flush on the next arrival or
    # end of stream (there is no idle timer).  The first event of a stream,
    # finish_reason/usage chunks, errors and [DONE] always flush immediately.
    # Set events to 0/1 (or ms to 0) to disable and restore per-event writes.
    stream_coalesce_events: int = 8
    stream_coalesce_ms: int = 50
    # Gateway policy: reasoning/thinking is OFF by default unless the client
    # explicitly opts in (think:true / thinking:{type:enabled} / reasoning_effort).
    # Applies to enable_thinking-style models (Qwen, Gemma, Nemotron); gpt-oss
    # uses reasoning_effort and is left untouched. Set false to restore the old
    # per-model launch defaults.
    thinking_off_by_default: bool = True

    # Blog syndication is PULL-only: selected posts are exposed read-only at
    # /blog/feed.xml (RSS) and /api/blog/syndicated (JSON); external sites pull
    # and render with their own templates. The old push-to-GitHub publisher
    # (site-specific chrome + a write credential inside the gateway) was
    # removed in 2.8.44.

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

    # Audit Logging — content capture into the requests/responses audit
    # tables. audit_log_enabled=False disables BOTH prompt and response
    # capture; the finer flags gate each side individually. Metadata
    # (model, tokens, timings, status) is always recorded. NOTE: the DLP
    # worker scans the stored content, so disabling capture also disables
    # DLP scanning of that content. Web-chat conversation storage is
    # user-facing state, not audit, and is unaffected.
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

    # API docs — /docs (Swagger), /redoc and /openapi.json. Disable to hide
    # the interactive schema on internet-facing deployments. Default True
    # preserves the current behavior.
    enable_api_docs: bool = True

    # Image URL fetching (image_url inputs / vision + OCR). When
    # image_url_block_private is True, targets resolving to private/loopback/
    # link-local addresses are refused (SSRF guard); allowed URL schemes are
    # restricted to image_url_allowed_schemes. ocr_max_frames caps how many
    # pages/frames a single OCR job will rasterize.
    image_url_block_private: bool = True
    image_url_allowed_schemes: List[str] = ["http", "https", "data"]
    ocr_max_frames: int = 500

    # CORS
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Chat UI
    chat_files_path: str = "/data/chat_files"
    chat_upload_max_size_mb: int = 10
    # Cap for the DECOMPRESSED size of an uploaded file (guards against
    # decompression-bomb archives whose on-disk size passes the size cap).
    chat_upload_max_uncompressed_mb: int = 100
    chat_upload_allowed_extensions: List[str] = [
        ".txt", ".md", ".csv", ".json", ".html", ".htm", ".log",
        ".docx", ".xlsx", ".pptx", ".pdf",
        ".jpg", ".jpeg", ".png", ".gif", ".webp",
    ]

    # (Chat retention lives in the runtime-editable app_config policies
    # at Admin -> Retention, not in environment settings.)

    # Web Search (Brave)
    brave_search_api_key: Optional[str] = None
    brave_search_max_results: int = 5

    # Tokenizer
    default_tokenizer: str = "cl100k_base"
    # Upper bound on characters accepted by the /v1/tokenize helper — caps the
    # work a single unauthenticated-cost tokenization request can trigger.
    tokenize_max_input_chars: int = 2000000

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

    @property
    def google_sso_enabled(self) -> bool:
        """Check if Google SSO is configured."""
        return bool(self.google_sso_client_id and self.google_sso_client_secret)

    @property
    def oidc_sso_enabled(self) -> bool:
        """Check if generic OIDC SSO is configured."""
        return bool(self.oidc_sso_issuer and self.oidc_sso_client_id and self.oidc_sso_client_secret)

    @property
    def saml_sso_enabled(self) -> bool:
        """Check if SAML SSO is configured."""
        return bool(
            self.saml_sp_entity_id
            and (self.saml_idp_metadata_url or (self.saml_idp_entity_id and self.saml_idp_sso_url and self.saml_idp_x509_cert))
        )

    @property
    def sso_enabled(self) -> bool:
        """True when at least one SSO provider is configured."""
        return self.azure_ad_enabled or self.google_sso_enabled or self.oidc_sso_enabled or self.saml_sso_enabled

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
