#!/usr/bin/env python3
############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# seed_dev_data.py: Seed database with development test data
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Seed development data for MindRouter."""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.db.session import get_async_db_context
from backend.app.db import crud
from backend.app.db.models import UserRole
from backend.app.security import hash_password, generate_api_key
from backend.app.security.api_keys import hash_api_key
from backend.app.settings import get_settings


async def ensure_groups(db):
    """Ensure default groups exist (migration may have already created them)."""
    default_groups = [
        {"name": "students", "display_name": "Students", "description": "University students",
         "token_budget": 100000, "rpm_limit": 30, "scheduler_weight": 1, "is_admin": False},
        {"name": "staff", "display_name": "Staff", "description": "University staff",
         "token_budget": 500000, "rpm_limit": 60, "scheduler_weight": 2, "is_admin": False},
        {"name": "faculty", "display_name": "Faculty", "description": "University faculty members",
         "token_budget": 1000000, "rpm_limit": 120, "scheduler_weight": 3, "is_admin": False},
        {"name": "researchers", "display_name": "Researchers", "description": "Research group members",
         "token_budget": 1000000, "rpm_limit": 120, "scheduler_weight": 3, "is_admin": False},
        {"name": "admin", "display_name": "Admin", "description": "System administrators",
         "token_budget": 10000000, "rpm_limit": 1000, "scheduler_weight": 10, "is_admin": True},
        {"name": "nerds", "display_name": "Nerds", "description": "Power users and enthusiasts",
         "token_budget": 500000, "rpm_limit": 60, "scheduler_weight": 2, "is_admin": False},
        {"name": "other", "display_name": "Other", "description": "General users",
         "token_budget": 100000, "rpm_limit": 30, "scheduler_weight": 1, "is_admin": False},
    ]

    groups = {}
    for gdata in default_groups:
        existing = await crud.get_group_by_name(db, gdata["name"])
        if existing:
            groups[gdata["name"]] = existing
            print(f"  Group '{gdata['name']}' already exists, skipping...")
        else:
            group = await crud.create_group(db, **gdata)
            groups[gdata["name"]] = group
            print(f"  Created group: {gdata['display_name']}")

    return groups


async def seed_users():
    """Create default users for development."""
    async with get_async_db_context() as db:
        # Ensure groups exist first
        print("Ensuring groups...")
        groups = await ensure_groups(db)
        await db.commit()

        # Automation-friendly overrides (env). ADMIN_PASSWORD sets the password;
        # ADMIN_API_KEY uses a specific key instead of minting; MINT_ADMIN_KEY=1
        # (re)mints a key for an already-existing admin.
        admin_password = os.environ.get("ADMIN_PASSWORD", "admin123")
        admin_api_key = os.environ.get("ADMIN_API_KEY")
        mint_admin_key = os.environ.get("MINT_ADMIN_KEY", "").lower() in ("1", "true", "yes")

        users_data = [
            {
                "username": "admin",
                "email": "admin@mindrouter.local",
                "password": admin_password,
                "role": UserRole.ADMIN,
                "group_name": "admin",
                "full_name": "Administrator",
            },
        ]

        for user_data in users_data:
            existing = await crud.get_user_by_username(db, user_data["username"])
            if existing:
                user = existing
                print(f"User {user_data['username']} already exists.")
                # Only (re)mint a key on demand, to avoid surprise key churn.
                if not (admin_api_key or mint_admin_key):
                    continue
            else:
                group = groups[user_data["group_name"]]
                user = await crud.create_user(
                    db=db,
                    username=user_data["username"],
                    email=user_data["email"],
                    password_hash=hash_password(user_data["password"]),
                    role=user_data["role"],
                    full_name=user_data["full_name"],
                    group_id=group.id,
                )
                print(f"Created user: {user.username} (group: {group.display_name})")
                await crud.create_quota(db=db, user_id=user.id, rpm_limit=group.rpm_limit)
                print(f"  Created quota: {group.token_budget} tokens")

            # Accept a supplied key (ADMIN_API_KEY) or mint one.
            if admin_api_key:
                full_key = admin_api_key
                key_hash = hash_api_key(admin_api_key)
                key_prefix = admin_api_key[:12]
            else:
                full_key, key_hash, key_prefix = generate_api_key()
            await crud.create_api_key(
                db=db,
                user_id=user.id,
                key_hash=key_hash,
                key_prefix=key_prefix,
                name="Default Key",
            )
            print(f"  Created API key: {key_prefix}...")
            # Full key on ONE parseable line (prefix and key were previously on
            # separate lines, so automation grabbed the prefix by mistake). It
            # cannot be re-shown; re-run with MINT_ADMIN_KEY=1 to add a new one.
            print(f"ADMIN_API_KEY={full_key}")
            print()

        await db.commit()


async def main():
    """Main entry point."""
    print("=" * 60)
    print("MindRouter Development Data Seeder")
    print("=" * 60)
    print()

    print("Creating users...")
    await seed_users()

    print()
    print("=" * 60)
    print("Seeding complete!")
    print()
    print("Default credentials:")
    print("  admin / admin123")
    print()
    print("Other users will be created via Azure AD SSO on first login.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
