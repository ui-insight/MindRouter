############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# 007_chat_tables.py: Add chat_conversations, chat_messages,
#                     chat_attachments tables
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add chat conversation, message, and attachment tables

Revision ID: 007
Revises: 006
Create Date: 2026-02-11 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '007'
down_revision: Union[str, None] = '006'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'chat_conversations',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(255), nullable=False, server_default='New Chat'),
        sa.Column('model', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], name='fk_chat_conversations_user_id'),
    )
    op.create_index('ix_chat_conversations_user_updated', 'chat_conversations', ['user_id', sa.text('updated_at DESC')])

    op.create_table(
        'chat_messages',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('conversation_id', sa.Integer(), nullable=False),
        sa.Column('role', sa.String(20), nullable=False),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['conversation_id'], ['chat_conversations.id'], name='fk_chat_messages_conversation_id'),
    )
    op.create_index('ix_chat_messages_conv_created', 'chat_messages', ['conversation_id', 'created_at'])

    op.create_table(
        'chat_attachments',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('message_id', sa.Integer(), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('filename', sa.String(255), nullable=False),
        sa.Column('content_type', sa.String(100), nullable=True),
        sa.Column('is_image', sa.Boolean(), nullable=False, server_default=sa.text('0')),
        sa.Column('storage_path', sa.String(500), nullable=True),
        sa.Column('thumbnail_base64', sa.Text(), nullable=True),
        sa.Column('extracted_text', sa.Text(), nullable=True),
        sa.Column('file_size', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['message_id'], ['chat_messages.id'], name='fk_chat_attachments_message_id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], name='fk_chat_attachments_user_id'),
    )
    op.create_index('ix_chat_attachments_message', 'chat_attachments', ['message_id'])
    op.create_index('ix_chat_attachments_user_created', 'chat_attachments', ['user_id', 'created_at'])


def downgrade() -> None:
    # Drop in reverse FK order
    op.drop_table('chat_attachments')
    op.drop_table('chat_messages')
    op.drop_table('chat_conversations')
