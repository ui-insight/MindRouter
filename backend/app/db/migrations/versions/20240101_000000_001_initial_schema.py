############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# 001_initial_schema.py: Initial database schema migration
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Initial database schema

Revision ID: 001
Revises:
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('uuid', sa.String(36), nullable=False),
        sa.Column('username', sa.String(100), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('full_name', sa.String(255), nullable=True),
        sa.Column('role', sa.Enum('student', 'staff', 'faculty', 'admin', name='userrole'), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('last_login_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('uuid'),
        sa.UniqueConstraint('username'),
        sa.UniqueConstraint('email'),
    )
    op.create_index('ix_users_username', 'users', ['username'])
    op.create_index('ix_users_email', 'users', ['email'])
    op.create_index('ix_users_role_active', 'users', ['role', 'is_active'])

    # API Keys table
    op.create_table(
        'api_keys',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('key_hash', sa.String(255), nullable=False),
        sa.Column('key_prefix', sa.String(12), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('status', sa.Enum('active', 'revoked', 'expired', name='apikeystatus'), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('usage_count', sa.BigInteger(), nullable=False, default=0),
        sa.Column('rpm_limit', sa.Integer(), nullable=True),
        sa.Column('max_concurrent', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('key_hash'),
    )
    op.create_index('ix_api_keys_key_hash', 'api_keys', ['key_hash'])
    op.create_index('ix_api_keys_status_user', 'api_keys', ['status', 'user_id'])

    # Quotas table
    op.create_table(
        'quotas',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('token_budget', sa.BigInteger(), nullable=False, default=100000),
        sa.Column('tokens_used', sa.BigInteger(), nullable=False, default=0),
        sa.Column('budget_period_start', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('budget_period_days', sa.Integer(), nullable=False, default=30),
        sa.Column('rpm_limit', sa.Integer(), nullable=False, default=30),
        sa.Column('max_concurrent', sa.Integer(), nullable=False, default=2),
        sa.Column('weight_override', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id'),
    )

    # Quota Requests table
    op.create_table(
        'quota_requests',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('requester_name', sa.String(255), nullable=True),
        sa.Column('requester_email', sa.String(255), nullable=True),
        sa.Column('affiliation', sa.String(255), nullable=True),
        sa.Column('request_type', sa.String(50), nullable=False),
        sa.Column('justification', sa.Text(), nullable=False),
        sa.Column('requested_tokens', sa.BigInteger(), nullable=True),
        sa.Column('requested_rpm', sa.Integer(), nullable=True),
        sa.Column('status', sa.Enum('pending', 'approved', 'denied', name='quotarequeststatus'), nullable=False),
        sa.Column('reviewed_by', sa.Integer(), nullable=True),
        sa.Column('reviewed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('review_notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.ForeignKeyConstraint(['reviewed_by'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_quota_requests_status', 'quota_requests', ['status'])

    # Backends table
    op.create_table(
        'backends',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('url', sa.String(500), nullable=False),
        sa.Column('engine', sa.Enum('ollama', 'vllm', name='backendengine'), nullable=False),
        sa.Column('status', sa.Enum('healthy', 'unhealthy', 'disabled', 'unknown', name='backendstatus'), nullable=False),
        sa.Column('max_concurrent', sa.Integer(), nullable=False, default=4),
        sa.Column('gpu_memory_gb', sa.Float(), nullable=True),
        sa.Column('gpu_type', sa.String(100), nullable=True),
        sa.Column('supports_vision', sa.Boolean(), nullable=False, default=False),
        sa.Column('supports_embeddings', sa.Boolean(), nullable=False, default=False),
        sa.Column('supports_structured_output', sa.Boolean(), nullable=False, default=True),
        sa.Column('current_concurrent', sa.Integer(), nullable=False, default=0),
        sa.Column('consecutive_failures', sa.Integer(), nullable=False, default=0),
        sa.Column('last_health_check', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_success', sa.DateTime(timezone=True), nullable=True),
        sa.Column('version', sa.String(50), nullable=True),
        sa.Column('priority', sa.Integer(), nullable=False, default=0),
        sa.Column('throughput_score', sa.Float(), nullable=False, default=1.0),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name'),
    )
    op.create_index('ix_backends_status_engine', 'backends', ['status', 'engine'])

    # Backend Telemetry table
    op.create_table(
        'backend_telemetry',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('backend_id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('gpu_utilization', sa.Float(), nullable=True),
        sa.Column('gpu_memory_used_gb', sa.Float(), nullable=True),
        sa.Column('gpu_memory_total_gb', sa.Float(), nullable=True),
        sa.Column('gpu_temperature', sa.Float(), nullable=True),
        sa.Column('active_requests', sa.Integer(), nullable=False, default=0),
        sa.Column('queued_requests', sa.Integer(), nullable=False, default=0),
        sa.Column('requests_per_second', sa.Float(), nullable=True),
        sa.Column('loaded_models', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['backend_id'], ['backends.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_backend_telemetry_backend_time', 'backend_telemetry', ['backend_id', 'timestamp'])

    # Models table
    op.create_table(
        'models',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('backend_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('family', sa.String(100), nullable=True),
        sa.Column('modality', sa.Enum('chat', 'completion', 'embedding', 'vision', name='modality'), nullable=False),
        sa.Column('context_length', sa.Integer(), nullable=True),
        sa.Column('supports_vision', sa.Boolean(), nullable=False, default=False),
        sa.Column('supports_structured_output', sa.Boolean(), nullable=False, default=True),
        sa.Column('parameter_count', sa.String(50), nullable=True),
        sa.Column('vram_required_gb', sa.Float(), nullable=True),
        sa.Column('is_loaded', sa.Boolean(), nullable=False, default=False),
        sa.Column('last_used', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['backend_id'], ['backends.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_models_backend_name', 'models', ['backend_id', 'name'])
    op.create_index('ix_models_name', 'models', ['name'])

    # Requests table
    op.create_table(
        'requests',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('request_uuid', sa.String(36), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('api_key_id', sa.Integer(), nullable=False),
        sa.Column('endpoint', sa.String(100), nullable=False),
        sa.Column('model', sa.String(200), nullable=False),
        sa.Column('modality', sa.Enum('chat', 'completion', 'embedding', 'vision', name='modality'), nullable=False),
        sa.Column('is_streaming', sa.Boolean(), nullable=False, default=False),
        sa.Column('messages', sa.JSON(), nullable=True),
        sa.Column('prompt', sa.Text(), nullable=True),
        sa.Column('parameters', sa.JSON(), nullable=True),
        sa.Column('response_format', sa.JSON(), nullable=True),
        sa.Column('status', sa.Enum('queued', 'processing', 'completed', 'failed', 'cancelled', name='requeststatus'), nullable=False),
        sa.Column('backend_id', sa.Integer(), nullable=True),
        sa.Column('queue_position', sa.Integer(), nullable=True),
        sa.Column('queued_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('queue_delay_ms', sa.Integer(), nullable=True),
        sa.Column('processing_time_ms', sa.Integer(), nullable=True),
        sa.Column('total_time_ms', sa.Integer(), nullable=True),
        sa.Column('prompt_tokens', sa.Integer(), nullable=True),
        sa.Column('completion_tokens', sa.Integer(), nullable=True),
        sa.Column('total_tokens', sa.Integer(), nullable=True),
        sa.Column('tokens_estimated', sa.Boolean(), nullable=False, default=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('error_code', sa.String(50), nullable=True),
        sa.Column('client_ip', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.String(500), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.ForeignKeyConstraint(['api_key_id'], ['api_keys.id']),
        sa.ForeignKeyConstraint(['backend_id'], ['backends.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('request_uuid'),
    )
    op.create_index('ix_requests_request_uuid', 'requests', ['request_uuid'])
    op.create_index('ix_requests_user_created', 'requests', ['user_id', 'created_at'])
    op.create_index('ix_requests_status', 'requests', ['status'])
    op.create_index('ix_requests_model', 'requests', ['model'])

    # Usage Ledger table
    op.create_table(
        'usage_ledger',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('api_key_id', sa.Integer(), nullable=False),
        sa.Column('request_id', sa.BigInteger(), nullable=False),
        sa.Column('prompt_tokens', sa.Integer(), nullable=False, default=0),
        sa.Column('completion_tokens', sa.Integer(), nullable=False, default=0),
        sa.Column('total_tokens', sa.Integer(), nullable=False, default=0),
        sa.Column('is_estimated', sa.Boolean(), nullable=False, default=False),
        sa.Column('model', sa.String(100), nullable=False),
        sa.Column('backend_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.ForeignKeyConstraint(['api_key_id'], ['api_keys.id']),
        sa.ForeignKeyConstraint(['request_id'], ['requests.id']),
        sa.ForeignKeyConstraint(['backend_id'], ['backends.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_usage_ledger_user_id', 'usage_ledger', ['user_id'])
    op.create_index('ix_usage_ledger_user_created', 'usage_ledger', ['user_id', 'created_at'])

    # Responses table
    op.create_table(
        'responses',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('request_id', sa.BigInteger(), nullable=False),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('finish_reason', sa.String(50), nullable=True),
        sa.Column('chunk_count', sa.Integer(), nullable=False, default=0),
        sa.Column('first_token_time_ms', sa.Integer(), nullable=True),
        sa.Column('structured_output_valid', sa.Boolean(), nullable=True),
        sa.Column('validation_errors', sa.JSON(), nullable=True),
        sa.Column('raw_response', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['request_id'], ['requests.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('request_id'),
    )

    # Artifacts table
    op.create_table(
        'artifacts',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('request_id', sa.BigInteger(), nullable=False),
        sa.Column('filename', sa.String(255), nullable=False),
        sa.Column('content_type', sa.String(100), nullable=False),
        sa.Column('size_bytes', sa.BigInteger(), nullable=False),
        sa.Column('storage_path', sa.String(500), nullable=False),
        sa.Column('sha256_hash', sa.String(64), nullable=False),
        sa.Column('processed', sa.Boolean(), nullable=False, default=False),
        sa.Column('processing_notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['request_id'], ['requests.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_artifacts_request', 'artifacts', ['request_id'])
    op.create_index('ix_artifacts_sha256_hash', 'artifacts', ['sha256_hash'])

    # Scheduler Decisions table
    op.create_table(
        'scheduler_decisions',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('request_id', sa.BigInteger(), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('selected_backend_id', sa.Integer(), nullable=False),
        sa.Column('candidate_backends', sa.JSON(), nullable=True),
        sa.Column('scores', sa.JSON(), nullable=True),
        sa.Column('user_deficit', sa.Float(), nullable=True),
        sa.Column('user_weight', sa.Float(), nullable=True),
        sa.Column('user_recent_usage', sa.Integer(), nullable=True),
        sa.Column('hard_constraints_passed', sa.JSON(), nullable=True),
        sa.Column('hard_constraints_failed', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['request_id'], ['requests.id']),
        sa.ForeignKeyConstraint(['selected_backend_id'], ['backends.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('request_id'),
    )


def downgrade() -> None:
    op.drop_table('scheduler_decisions')
    op.drop_table('artifacts')
    op.drop_table('responses')
    op.drop_table('usage_ledger')
    op.drop_table('requests')
    op.drop_table('models')
    op.drop_table('backend_telemetry')
    op.drop_table('backends')
    op.drop_table('quota_requests')
    op.drop_table('quotas')
    op.drop_table('api_keys')
    op.drop_table('users')

    # Drop enums
    op.execute("DROP TYPE IF EXISTS userrole")
    op.execute("DROP TYPE IF EXISTS apikeystatus")
    op.execute("DROP TYPE IF EXISTS backendengine")
    op.execute("DROP TYPE IF EXISTS backendstatus")
    op.execute("DROP TYPE IF EXISTS requeststatus")
    op.execute("DROP TYPE IF EXISTS quotarequeststatus")
    op.execute("DROP TYPE IF EXISTS modality")
