############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# models.py: SQLAlchemy ORM models for all database entities
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""SQLAlchemy database models for MindRouter."""

from datetime import datetime
from enum import Enum as PyEnum
from typing import List, Optional
import uuid

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base, TimestampMixin, SoftDeleteMixin

# Use enum values (lowercase) for database storage, not enum names (uppercase)
_enum_values = lambda obj: [e.value for e in obj]


# Enums
class UserRole(str, PyEnum):
    """User role types."""
    STUDENT = "student"
    STAFF = "staff"
    FACULTY = "faculty"
    ADMIN = "admin"


class ApiKeyStatus(str, PyEnum):
    """API key status types."""
    ACTIVE = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"


class BackendEngine(str, PyEnum):
    """Backend engine types."""
    OLLAMA = "ollama"
    VLLM = "vllm"


class BackendStatus(str, PyEnum):
    """Backend health status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DISABLED = "disabled"
    DRAINING = "draining"
    UNKNOWN = "unknown"


class NodeStatus(str, PyEnum):
    """Node health status."""
    ONLINE = "online"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class RequestStatus(str, PyEnum):
    """Request processing status."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QuotaRequestStatus(str, PyEnum):
    """Quota request status."""
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"


class ServiceKeyRequestStatus(str, PyEnum):
    """Service key request status."""
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"


class Modality(str, PyEnum):
    """Request modality types."""
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    MULTIMODAL = "multimodal"
    RERANKING = "reranking"
    TTS = "tts"
    STT = "stt"


# Group Model
class Group(Base, TimestampMixin):
    """User group for authorization and quota defaults."""

    __tablename__ = "groups"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    token_budget: Mapped[int] = mapped_column(BigInteger, nullable=False, default=100000)
    rpm_limit: Mapped[int] = mapped_column(Integer, nullable=False, default=30)
    max_concurrent: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    scheduler_weight: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    is_admin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_auditor: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    api_key_expiry_days: Mapped[int] = mapped_column(Integer, nullable=False, default=45)
    max_api_keys: Mapped[int] = mapped_column(Integer, nullable=False, default=16)

    @property
    def has_admin_read(self) -> bool:
        """True if this group grants read-only admin access (admin or auditor)."""
        return self.is_admin or self.is_auditor

    # Relationships
    users: Mapped[List["User"]] = relationship("User", back_populates="group")


# User and Authentication Models
class User(Base, TimestampMixin, SoftDeleteMixin):
    """User account model."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    uuid: Mapped[str] = mapped_column(
        String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4())
    )
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    azure_oid: Mapped[Optional[str]] = mapped_column(String(36), unique=True, nullable=True, index=True)
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, values_callable=_enum_values), nullable=False, default=UserRole.STUDENT
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Group membership
    group_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("groups.id"), nullable=True)

    # Profile fields
    college: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    department: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    intended_use: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Agreement tracking
    agreement_version_accepted: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    agreement_accepted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    group: Mapped[Optional["Group"]] = relationship("Group", back_populates="users")
    api_keys: Mapped[List["ApiKey"]] = relationship("ApiKey", back_populates="user", foreign_keys="ApiKey.user_id")
    quota: Mapped[Optional["Quota"]] = relationship("Quota", back_populates="user", uselist=False)
    quota_requests: Mapped[List["QuotaRequest"]] = relationship(
        "QuotaRequest", back_populates="user", foreign_keys="QuotaRequest.user_id"
    )
    requests: Mapped[List["Request"]] = relationship("Request", back_populates="user")

    __table_args__ = (
        Index("ix_users_role_active", "role", "is_active"),
        Index("ix_users_group_active", "group_id", "is_active"),
    )


class ApiKey(Base, TimestampMixin):
    """API key model."""

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    key_prefix: Mapped[str] = mapped_column(String(12), nullable=False)  # First 8 chars for identification
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[ApiKeyStatus] = mapped_column(
        Enum(ApiKeyStatus, values_callable=_enum_values), nullable=False, default=ApiKeyStatus.ACTIVE
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    usage_count: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)

    # Rate limiting overrides (null = use quota defaults)
    rpm_limit: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    max_concurrent: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Service key fields
    is_service: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    service_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    service_contacts: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    data_risk_level: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    compliance_tags: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    promoted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    promoted_by: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    revocation_requested_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    revocation_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="api_keys", foreign_keys=[user_id])
    promoted_by_user: Mapped[Optional["User"]] = relationship("User", foreign_keys=[promoted_by])
    requests: Mapped[List["Request"]] = relationship("Request", back_populates="api_key")

    __table_args__ = (
        Index("ix_api_keys_status_user", "status", "user_id"),
        Index("ix_api_keys_is_service", "is_service"),
    )


class Quota(Base, TimestampMixin):
    """User quota and limits."""

    __tablename__ = "quotas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), unique=True, nullable=False
    )

    # Token usage (budget comes from user's group)
    tokens_used: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    lifetime_tokens_used: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    budget_period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    budget_period_days: Mapped[int] = mapped_column(Integer, nullable=False, default=30)

    # Rate limits
    rpm_limit: Mapped[int] = mapped_column(Integer, nullable=False, default=30)
    max_concurrent: Mapped[int] = mapped_column(Integer, nullable=False, default=2)

    # Scheduler weight override (null = use role default)
    weight_override: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="quota")


class QuotaRequest(Base, TimestampMixin):
    """Request for quota increase or API key."""

    __tablename__ = "quota_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)

    # For new user API key requests
    requester_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    requester_email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    affiliation: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    request_type: Mapped[str] = mapped_column(String(50), nullable=False)  # "api_key" or "quota_increase"
    justification: Mapped[str] = mapped_column(Text, nullable=False)
    requested_tokens: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    requested_rpm: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    status: Mapped[QuotaRequestStatus] = mapped_column(
        Enum(QuotaRequestStatus, values_callable=_enum_values), nullable=False, default=QuotaRequestStatus.PENDING
    )

    # Admin review
    reviewed_by: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    review_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    user: Mapped[Optional["User"]] = relationship(
        "User", back_populates="quota_requests", foreign_keys=[user_id]
    )
    reviewer: Mapped[Optional["User"]] = relationship("User", foreign_keys=[reviewed_by])

    __table_args__ = (
        Index("ix_quota_requests_status", "status"),
    )


class ServiceKeyRequest(Base, TimestampMixin):
    """Request to promote an API key to a service key."""

    __tablename__ = "service_key_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    api_key_id: Mapped[int] = mapped_column(Integer, ForeignKey("api_keys.id"), nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    service_name: Mapped[str] = mapped_column(String(200), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    alternative_contacts: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    data_risk_level: Mapped[str] = mapped_column(String(20), nullable=False, default="low")
    compliance_tags: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    compliance_other: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[ServiceKeyRequestStatus] = mapped_column(
        Enum(ServiceKeyRequestStatus, values_callable=_enum_values),
        nullable=False,
        default=ServiceKeyRequestStatus.PENDING,
    )

    # Admin review
    reviewed_by: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    review_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    api_key: Mapped["ApiKey"] = relationship("ApiKey")
    user: Mapped["User"] = relationship("User", foreign_keys=[user_id])
    reviewer: Mapped[Optional["User"]] = relationship("User", foreign_keys=[reviewed_by])

    __table_args__ = (
        Index("ix_service_key_requests_status", "status"),
    )


# Node Models
class Node(Base, TimestampMixin):
    """Physical server node with GPU hardware and sidecar agent."""

    __tablename__ = "nodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    hostname: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    sidecar_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    status: Mapped[NodeStatus] = mapped_column(
        Enum(NodeStatus, values_callable=_enum_values), nullable=False, default=NodeStatus.UNKNOWN
    )

    # Sidecar authentication
    sidecar_key: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Hardware info (populated by sidecar)
    gpu_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    driver_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    cuda_version: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    sidecar_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Server power (IPMI DCMI, populated by sidecar)
    server_power_watts: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Relationships
    gpu_devices: Mapped[List["GPUDevice"]] = relationship("GPUDevice", back_populates="node")
    backends: Mapped[List["Backend"]] = relationship("Backend", back_populates="node")
    node_telemetry: Mapped[List["NodeTelemetry"]] = relationship(
        "NodeTelemetry", back_populates="node", cascade="all, delete-orphan"
    )


class NodeTelemetry(Base):
    """Time-series telemetry snapshots for nodes (server-level power)."""

    __tablename__ = "node_telemetry"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    node_id: Mapped[int] = mapped_column(Integer, ForeignKey("nodes.id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Server-level power from IPMI DCMI
    server_power_watts: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Sum of all GPU power draws on this node at this timestamp
    gpu_power_watts: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Relationships
    node: Mapped["Node"] = relationship("Node", back_populates="node_telemetry")

    __table_args__ = (
        Index("ix_node_telemetry_node_time", "node_id", "timestamp"),
        # Covering index for cluster-wide power history queries that
        # filter by timestamp only (no node_id) — see migration 049.
        Index(
            "ix_node_telemetry_time_covering",
            "timestamp",
            "node_id",
            "server_power_watts",
            "gpu_power_watts",
        ),
    )


# Backend Models
class Backend(Base, TimestampMixin):
    """Backend inference server registration."""

    __tablename__ = "backends"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    url: Mapped[str] = mapped_column(String(500), unique=True, nullable=False)
    engine: Mapped[BackendEngine] = mapped_column(Enum(BackendEngine, values_callable=_enum_values), nullable=False)
    status: Mapped[BackendStatus] = mapped_column(
        Enum(BackendStatus, values_callable=_enum_values), nullable=False, default=BackendStatus.UNKNOWN
    )

    # Capabilities
    max_concurrent: Mapped[int] = mapped_column(Integer, nullable=False, default=4)
    gpu_memory_gb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gpu_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    # Runtime state
    current_concurrent: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_health_check: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_success: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Priority/throughput hints
    priority: Mapped[int] = mapped_column(Integer, default=0, nullable=False)  # Higher = preferred
    throughput_score: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)

    # Latency EMA (persisted for restart resilience)
    latency_ema_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=None)
    ttft_ema_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=None)

    # Circuit breaker state
    live_failure_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    circuit_open_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Node association
    node_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("nodes.id"), nullable=True)
    gpu_indices: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)  # e.g. [0, 1] or null for all

    # Relationships
    node: Mapped[Optional["Node"]] = relationship("Node", back_populates="backends")
    models: Mapped[List["Model"]] = relationship("Model", back_populates="backend", cascade="all, delete-orphan")
    telemetry: Mapped[List["BackendTelemetry"]] = relationship("BackendTelemetry", back_populates="backend", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_backends_status_engine", "status", "engine"),
    )


class BackendTelemetry(Base):
    """Time-series telemetry snapshots for backends."""

    __tablename__ = "backend_telemetry"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    backend_id: Mapped[int] = mapped_column(Integer, ForeignKey("backends.id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # GPU metrics
    gpu_utilization: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 0-100
    gpu_memory_used_gb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gpu_memory_total_gb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gpu_temperature: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gpu_power_draw_watts: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gpu_fan_speed_percent: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Request metrics
    active_requests: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    queued_requests: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    requests_per_second: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Model info
    loaded_models: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)  # List of model names

    # Relationships
    backend: Mapped["Backend"] = relationship("Backend", back_populates="telemetry")

    __table_args__ = (
        Index("ix_backend_telemetry_backend_time", "backend_id", "timestamp"),
    )


class Model(Base, TimestampMixin):
    """Model catalog - models available on backends."""

    __tablename__ = "models"
    __table_args__ = (
        UniqueConstraint("backend_id", "name", name="uq_models_backend_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    backend_id: Mapped[int] = mapped_column(Integer, ForeignKey("backends.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    family: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)  # llama, mistral, etc.

    # Capabilities
    modality: Mapped[Modality] = mapped_column(Enum(Modality, values_callable=_enum_values), nullable=False, default=Modality.CHAT)
    context_length: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    model_max_context: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    supports_multimodal: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    supports_structured_output: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Admin override for multimodal (NULL = auto-detect, True/False = manual)
    multimodal_override: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    # Thinking/reasoning support
    supports_thinking: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    thinking_override: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    # Tool calling support
    supports_tools: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    tools_override: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    # Admin-editable fields
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    model_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    huggingface_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Size and performance
    parameter_count: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # "7B", "70B"
    quantization: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # "Q4_K_M", "FP16", etc.
    vram_required_gb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Rich metadata from /api/show (Ollama) — NULL for vLLM models
    model_format: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # "gguf", etc.
    capabilities: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array: ["completion","vision","tools"]
    embedding_length: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    head_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    layer_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    feed_forward_length: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    parent_model: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    # Admin overrides for metadata (NULL = use auto-detected, non-NULL = manual)
    context_length_override: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    embedding_length_override: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    head_count_override: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    layer_count_override: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    feed_forward_length_override: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    capabilities_override: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON text
    family_override: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    parameter_count_override: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    quantization_override: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    model_format_override: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    parent_model_override: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    # State
    is_loaded: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    last_used: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    backend: Mapped["Backend"] = relationship("Backend", back_populates="models")

    __table_args__ = (
        Index("ix_models_backend_name", "backend_id", "name"),
        Index("ix_models_name", "name"),
    )


# Model Description Cache
class ModelDescriptionCache(Base, TimestampMixin):
    """Cache of enriched model descriptions, keyed by model name.

    Survives model row deletion (e.g. remove_stale_models during node
    maintenance) so descriptions can be restored from cache when the
    model is rediscovered, avoiding redundant web search + LLM calls.
    """

    __tablename__ = "model_description_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)


# GPU Device Models
class GPUDevice(Base, TimestampMixin):
    """Individual GPU device on a node."""

    __tablename__ = "gpu_devices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    node_id: Mapped[int] = mapped_column(Integer, ForeignKey("nodes.id"), nullable=False)
    gpu_index: Mapped[int] = mapped_column(Integer, nullable=False)

    # Static device info (updated on discovery, not every poll)
    uuid: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    pci_bus_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    compute_capability: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    memory_total_gb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    power_limit_watts: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Relationships
    node: Mapped["Node"] = relationship("Node", back_populates="gpu_devices")
    telemetry: Mapped[List["GPUDeviceTelemetry"]] = relationship(
        "GPUDeviceTelemetry", back_populates="gpu_device", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_gpu_devices_node_index", "node_id", "gpu_index", unique=True),
    )


class GPUDeviceTelemetry(Base):
    """Per-GPU time-series telemetry snapshots."""

    __tablename__ = "gpu_device_telemetry"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    gpu_device_id: Mapped[int] = mapped_column(Integer, ForeignKey("gpu_devices.id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Utilization
    utilization_gpu: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    utilization_memory: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Memory
    memory_used_gb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    memory_free_gb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Thermal & Power
    temperature_gpu: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    temperature_memory: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    power_draw_watts: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fan_speed_percent: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Clocks
    clock_sm_mhz: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    clock_memory_mhz: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Relationships
    gpu_device: Mapped["GPUDevice"] = relationship("GPUDevice", back_populates="telemetry")

    __table_args__ = (
        Index("ix_gpu_device_telemetry_device_time", "gpu_device_id", "timestamp"),
    )


# Request/Response Audit Models
class Request(Base, TimestampMixin):
    """Audit log of all inference requests."""

    __tablename__ = "requests"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    request_uuid: Mapped[str] = mapped_column(
        String(36), unique=True, nullable=False, index=True, default=lambda: str(uuid.uuid4())
    )

    # User context
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    api_key_id: Mapped[int] = mapped_column(Integer, ForeignKey("api_keys.id"), nullable=False)

    # Request details
    endpoint: Mapped[str] = mapped_column(String(100), nullable=False)  # /v1/chat/completions
    model: Mapped[str] = mapped_column(String(200), nullable=False)
    modality: Mapped[Modality] = mapped_column(Enum(Modality, values_callable=_enum_values), nullable=False)
    is_streaming: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Request content (stored for audit)
    messages: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)  # Chat messages
    prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Completion prompt
    parameters: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)  # temperature, etc.
    response_format: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)  # structured output schema

    # Scheduling metadata
    status: Mapped[RequestStatus] = mapped_column(
        Enum(RequestStatus, values_callable=_enum_values), nullable=False, default=RequestStatus.QUEUED
    )
    backend_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("backends.id"), nullable=True)
    queue_position: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    queued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Timing
    queue_delay_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    processing_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Token counts
    prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    tokens_estimated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Error tracking
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_code: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Client info
    client_ip: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="requests")
    api_key: Mapped["ApiKey"] = relationship("ApiKey", back_populates="requests")
    response: Mapped[Optional["Response"]] = relationship("Response", back_populates="request", uselist=False)
    artifacts: Mapped[List["Artifact"]] = relationship("Artifact", back_populates="request")
    scheduler_decision: Mapped[Optional["SchedulerDecision"]] = relationship(
        "SchedulerDecision", back_populates="request", uselist=False
    )

    __table_args__ = (
        Index("ix_requests_user_created", "user_id", "created_at"),
        Index("ix_requests_status", "status"),
        Index("ix_requests_model", "model"),
        Index("ix_requests_created_model_status", "created_at", "model", "status"),
        # Covering index for per-user token total aggregations on the
        # user dashboard — see migration 049.
        Index(
            "ix_requests_user_tokens_covering",
            "user_id",
            "total_tokens",
            "prompt_tokens",
            "completion_tokens",
        ),
    )


class Response(Base, TimestampMixin):
    """Audit log of inference responses."""

    __tablename__ = "responses"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    request_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("requests.id"), unique=True, nullable=False
    )

    # Response content
    content: Mapped[Optional[str]] = mapped_column(MEDIUMTEXT, nullable=True)  # Final aggregated response
    finish_reason: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Streaming metadata
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    first_token_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Structured output validation
    structured_output_valid: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    validation_errors: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)

    # Raw backend response (for debugging)
    raw_response: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)

    # Relationships
    request: Mapped["Request"] = relationship("Request", back_populates="response")


class Artifact(Base, TimestampMixin):
    """Uploaded artifacts (images, documents) linked to requests."""

    __tablename__ = "artifacts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    request_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("requests.id"), nullable=False)

    # File info
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)

    # Storage
    storage_path: Mapped[str] = mapped_column(String(500), nullable=False)
    sha256_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # Processing info
    processed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    processing_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    request: Mapped["Request"] = relationship("Request", back_populates="artifacts")

    __table_args__ = (
        Index("ix_artifacts_request", "request_id"),
    )


class SchedulerDecision(Base):
    """Audit log of scheduler routing decisions."""

    __tablename__ = "scheduler_decisions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    request_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("requests.id"), unique=True, nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Decision details
    selected_backend_id: Mapped[int] = mapped_column(Integer, ForeignKey("backends.id"), nullable=False)
    candidate_backends: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)  # Backend IDs considered
    scores: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)  # Scores per backend

    # Fairness metrics
    user_deficit: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    user_weight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    user_recent_usage: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Constraints
    hard_constraints_passed: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    hard_constraints_failed: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)

    # Relationships
    request: Mapped["Request"] = relationship("Request", back_populates="scheduler_decision")


# Chat Models
class ChatConversation(Base, TimestampMixin):
    """Chat conversation for web UI."""

    __tablename__ = "chat_conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False, default="New Chat")
    model: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User")
    messages: Mapped[List["ChatMessage"]] = relationship(
        "ChatMessage", back_populates="conversation", cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
    )

    __table_args__ = (
        Index("ix_chat_conversations_user_updated", "user_id", "updated_at"),
    )


class ChatMessage(Base, TimestampMixin):
    """Individual message within a chat conversation."""

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("chat_conversations.id"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[Optional[str]] = mapped_column(MEDIUMTEXT, nullable=True)

    # Relationships
    conversation: Mapped["ChatConversation"] = relationship(
        "ChatConversation", back_populates="messages"
    )
    attachments: Mapped[List["ChatAttachment"]] = relationship(
        "ChatAttachment", back_populates="message"
    )

    __table_args__ = (
        Index("ix_chat_messages_conv_created", "conversation_id", "created_at"),
    )


class ChatAttachment(Base, TimestampMixin):
    """File attachment for a chat message."""

    __tablename__ = "chat_attachments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    message_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("chat_messages.id"), nullable=True
    )
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    is_image: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    storage_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    thumbnail_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    extracted_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    file_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Relationships
    message: Mapped[Optional["ChatMessage"]] = relationship(
        "ChatMessage", back_populates="attachments"
    )

    __table_args__ = (
        Index("ix_chat_attachments_message", "message_id"),
        Index("ix_chat_attachments_user_created", "user_id", "created_at"),
    )


# Blog Models
class BlogPost(Base, TimestampMixin, SoftDeleteMixin):
    """Blog post model."""

    __tablename__ = "blog_posts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    excerpt: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    author_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    is_published: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    author: Mapped["User"] = relationship("User")


class EmailStatus(str, PyEnum):
    """Email send status."""
    PENDING = "pending"
    SENDING = "sending"
    COMPLETED = "completed"
    FAILED = "failed"


class EmailLog(Base):
    """Audit log of sent emails."""

    __tablename__ = "email_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    subject: Mapped[str] = mapped_column(String(500), nullable=False)
    body_preview: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    recipient_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    success_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    fail_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    sent_by: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    blog_post_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("blog_posts.id"), nullable=True)
    status: Mapped[EmailStatus] = mapped_column(
        Enum(EmailStatus, values_callable=_enum_values), nullable=False, server_default="pending"
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    sender: Mapped["User"] = relationship("User", foreign_keys=[sent_by])
    blog_post: Mapped[Optional["BlogPost"]] = relationship("BlogPost")


class AdminAuditLog(Base):
    """Persistent audit log of admin actions."""

    __tablename__ = "admin_audit_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    before_value: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    after_value: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    user: Mapped["User"] = relationship("User", foreign_keys=[user_id])

    __table_args__ = (
        Index("ix_admin_audit_timestamp", "timestamp"),
        Index("ix_admin_audit_user_time", "user_id", "timestamp"),
        Index("ix_admin_audit_entity", "entity_type", "entity_id"),
        Index("ix_admin_audit_action", "action"),
    )


class DlpSeverity(str, PyEnum):
    """DLP alert severity levels."""
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"


class DlpAlert(Base):
    """DLP (Data Loss Prevention) alert for sensitive data detected in requests."""

    __tablename__ = "dlp_alerts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    request_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("requests.id"), nullable=True)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    scanner: Mapped[str] = mapped_column(String(20), nullable=False)
    categories: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    entities: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    scan_latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    scanned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    acknowledged: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("0"))
    acknowledged_by: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    user: Mapped[Optional["User"]] = relationship("User", foreign_keys=[user_id])
    request: Mapped[Optional["Request"]] = relationship("Request", foreign_keys=[request_id])
    acknowledger: Mapped[Optional["User"]] = relationship("User", foreign_keys=[acknowledged_by])

    __table_args__ = (
        Index("ix_dlp_alerts_severity", "severity"),
        Index("ix_dlp_alerts_user_time", "user_id", "scanned_at"),
        Index("ix_dlp_alerts_request", "request_id"),
        Index("ix_dlp_alerts_scanner", "scanner"),
    )


class RegistryMeta(Base):
    """Single-row table tracking registry mutation version for cross-worker sync."""

    __tablename__ = "registry_meta"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")


class AppConfig(Base, TimestampMixin):
    """Key-value application configuration stored in the database."""

    __tablename__ = "app_config"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)  # JSON-encoded
    description: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
