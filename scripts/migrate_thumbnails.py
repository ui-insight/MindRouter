#!/usr/bin/env python3
############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# migrate_thumbnails.py: Migrate thumbnail data from DB
#     (thumbnail_base64) to filesystem (thumbnail_path)
#
# Run AFTER migration 010 and BEFORE migration 011.
#
# Usage:
#   python scripts/migrate_thumbnails.py
#
# Environment:
#   DATABASE_URL - SQLAlchemy database URL (required)
#   CHAT_FILES_PATH - Base directory for chat files (default: /data/chat_files)
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Migrate thumbnail_base64 to filesystem thumbnail_path."""

import base64
import os
import sys

import sqlalchemy as sa
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session


def main():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL environment variable is required", file=sys.stderr)
        sys.exit(1)

    chat_files_path = os.environ.get("CHAT_FILES_PATH", "/data/chat_files")

    engine = create_engine(database_url)
    metadata = sa.MetaData()
    attachments = sa.Table("chat_attachments", metadata, autoload_with=engine)

    migrated = 0
    errors = 0

    with Session(engine) as session:
        # Find all attachments with base64 thumbnail but no filesystem path
        query = (
            select(attachments.c.id, attachments.c.thumbnail_base64)
            .where(
                attachments.c.thumbnail_base64.isnot(None),
                sa.or_(
                    attachments.c.thumbnail_path.is_(None),
                    attachments.c.thumbnail_path == "",
                ),
            )
        )
        rows = session.execute(query).fetchall()
        total = len(rows)
        print(f"Found {total} attachments to migrate")

        for row in rows:
            att_id = row.id
            b64_data = row.thumbnail_base64

            try:
                thumb_bytes = base64.b64decode(b64_data)
            except Exception as e:
                print(f"  ERROR decoding base64 for attachment {att_id}: {e}")
                errors += 1
                continue

            # Write to sharded path
            shard = att_id % 1000
            shard_dir = os.path.join(chat_files_path, str(shard))
            os.makedirs(shard_dir, exist_ok=True)
            thumb_path = os.path.join(shard_dir, f"{att_id}_thumb.png")

            try:
                with open(thumb_path, "wb") as f:
                    f.write(thumb_bytes)
            except Exception as e:
                print(f"  ERROR writing thumbnail for attachment {att_id}: {e}")
                errors += 1
                continue

            # Update DB row
            session.execute(
                update(attachments)
                .where(attachments.c.id == att_id)
                .values(thumbnail_path=thumb_path)
            )
            migrated += 1

            if migrated % 100 == 0:
                session.commit()
                print(f"  Migrated {migrated}/{total}...")

        session.commit()

    print(f"\nDone: {migrated} migrated, {errors} errors, {total} total")


if __name__ == "__main__":
    main()
