############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# archive_models.py: FK-free archive model definitions
#     for the tiered data retention system.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""SQLAlchemy models for the archive database.

These mirror the app DB tables but have:
- No ForeignKey constraints (referenced tables don't exist in archive DB)
- No relationship() declarations
- autoincrement=False on PKs (preserve original IDs)
- Added archived_at DateTime column
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class ArchiveBase(DeclarativeBase):
    """Separate declarative base for archive models (own MetaData)."""

    type_annotation_map = {
        datetime: DateTime(timezone=True),
    }


# ------------------------------------------------------------------
# Request-family archives
# ------------------------------------------------------------------


class ArchivedRequest(ArchiveBase):
    """Archived inference request."""

    __tablename__ = "archived_requests"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    request_uuid: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)

    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    api_key_id: Mapped[int] = mapped_column(Integer, nullable=False)

    endpoint: Mapped[str] = mapped_column(String(100), nullable=False)
    model: Mapped[str] = mapped_column(String(200), nullable=False)
    modality: Mapped[str] = mapped_column(String(20), nullable=False)
    is_streaming: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    messages: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    parameters: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    response_format: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)

    status: Mapped[str] = mapped_column(String(20), nullable=False)
    backend_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    queue_position: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    queued_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    queue_delay_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    processing_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    tokens_estimated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_code: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    client_ip: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    archived_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_arch_req_user_created", "user_id", "created_at"),
        Index("ix_arch_req_model", "model"),
        Index("ix_arch_req_archived_at", "archived_at"),
    )


class ArchivedResponse(ArchiveBase):
    """Archived inference response."""

    __tablename__ = "archived_responses"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    request_id: Mapped[int] = mapped_column(BigInteger, nullable=False)

    content: Mapped[Optional[str]] = mapped_column(MEDIUMTEXT, nullable=True)
    finish_reason: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    first_token_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    structured_output_valid: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    validation_errors: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    raw_response: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    archived_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_arch_resp_request", "request_id"),
        Index("ix_arch_resp_archived_at", "archived_at"),
    )


class ArchivedSchedulerDecision(ArchiveBase):
    """Archived scheduler routing decision."""

    __tablename__ = "archived_scheduler_decisions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    request_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    selected_backend_id: Mapped[int] = mapped_column(Integer, nullable=False)
    candidate_backends: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    scores: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)

    user_deficit: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    user_weight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    user_recent_usage: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    hard_constraints_passed: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    hard_constraints_failed: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)

    archived_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_arch_sched_request", "request_id"),
        Index("ix_arch_sched_archived_at", "archived_at"),
    )


class ArchivedArtifact(ArchiveBase):
    """Archived uploaded artifact."""

    __tablename__ = "archived_artifacts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    request_id: Mapped[int] = mapped_column(BigInteger, nullable=False)

    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)

    storage_path: Mapped[str] = mapped_column(String(500), nullable=False)
    sha256_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    processed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    processing_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    archived_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_arch_art_request", "request_id"),
        Index("ix_arch_art_archived_at", "archived_at"),
    )


# ------------------------------------------------------------------
# Chat-family archives
# ------------------------------------------------------------------


class ArchivedChatConversation(ArchiveBase):
    """Archived chat conversation."""

    __tablename__ = "archived_chat_conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    model: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    archived_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_arch_conv_user", "user_id"),
        Index("ix_arch_conv_archived_at", "archived_at"),
    )


class ArchivedChatMessage(ArchiveBase):
    """Archived chat message."""

    __tablename__ = "archived_chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    conversation_id: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[Optional[str]] = mapped_column(MEDIUMTEXT, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    archived_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_arch_msg_conv", "conversation_id"),
        Index("ix_arch_msg_archived_at", "archived_at"),
    )


class ArchivedChatAttachment(ArchiveBase):
    """Archived chat attachment."""

    __tablename__ = "archived_chat_attachments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    message_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    is_image: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    storage_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    thumbnail_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    extracted_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    file_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    archived_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_arch_att_message", "message_id"),
        Index("ix_arch_att_archived_at", "archived_at"),
    )


# ------------------------------------------------------------------
# Telemetry archives
# ------------------------------------------------------------------


class ArchivedBackendTelemetry(ArchiveBase):
    """Archived backend telemetry snapshot."""

    __tablename__ = "archived_backend_telemetry"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    backend_id: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    gpu_utilization: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gpu_memory_used_gb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gpu_memory_total_gb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gpu_temperature: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gpu_power_draw_watts: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gpu_fan_speed_percent: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    active_requests: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    queued_requests: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    requests_per_second: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    loaded_models: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)

    archived_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_arch_bt_backend_time", "backend_id", "timestamp"),
        Index("ix_arch_bt_archived_at", "archived_at"),
    )


class ArchivedGPUDeviceTelemetry(ArchiveBase):
    """Archived per-GPU device telemetry snapshot."""

    __tablename__ = "archived_gpu_device_telemetry"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    gpu_device_id: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    utilization_gpu: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    utilization_memory: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    memory_used_gb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    memory_free_gb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    temperature_gpu: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    temperature_memory: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    power_draw_watts: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fan_speed_percent: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    clock_sm_mhz: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    clock_memory_mhz: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    archived_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_arch_gdt_device_time", "gpu_device_id", "timestamp"),
        Index("ix_arch_gdt_archived_at", "archived_at"),
    )


class ArchivedNodeTelemetry(ArchiveBase):
    """Archived node-level telemetry snapshot."""

    __tablename__ = "archived_node_telemetry"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    node_id: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    server_power_watts: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    gpu_power_watts: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    archived_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_arch_nt_node_time", "node_id", "timestamp"),
        Index("ix_arch_nt_archived_at", "archived_at"),
    )
