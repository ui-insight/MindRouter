############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# api_keys.py: API key generation, hashing, and verification
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""API key generation and verification."""

import hashlib
import secrets
from typing import Optional, Tuple

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db import crud
from backend.app.db.models import ApiKey

# API key format: mr2_<random_string>
API_KEY_PREFIX = "mr2_"
API_KEY_LENGTH = 48  # Total length including prefix

# Use Argon2 for hashing
_hasher = PasswordHasher(
    time_cost=2,
    memory_cost=65536,
    parallelism=1,
)


def generate_api_key() -> Tuple[str, str, str]:
    """
    Generate a new API key.

    Returns:
        Tuple of (full_key, key_hash, key_prefix)
        - full_key: The complete API key to give to the user (store nowhere!)
        - key_hash: Argon2 hash to store in database
        - key_prefix: First 8 chars for identification
    """
    # Generate random bytes
    random_part = secrets.token_urlsafe(32)

    # Full key with prefix
    full_key = f"{API_KEY_PREFIX}{random_part}"

    # Hash for storage
    key_hash = hash_api_key(full_key)

    # Prefix for identification (first 8 chars of random part)
    key_prefix = f"{API_KEY_PREFIX}{random_part[:8]}"

    return full_key, key_hash, key_prefix


def hash_api_key(api_key: str) -> str:
    """
    Hash an API key using Argon2.

    For API keys, we use a faster hash since we need to look them up frequently.
    We use SHA-256 first to normalize, then Argon2 for the actual hash.

    Args:
        api_key: The raw API key

    Returns:
        Argon2 hash of the key
    """
    # First normalize with SHA-256 (fast, deterministic)
    normalized = hashlib.sha256(api_key.encode()).hexdigest()

    # Then hash with Argon2
    return _hasher.hash(normalized)


def _verify_key_hash(api_key: str, key_hash: str) -> bool:
    """
    Verify an API key against a stored hash.

    Args:
        api_key: The raw API key to verify
        key_hash: The stored Argon2 hash

    Returns:
        True if the key matches
    """
    try:
        # Normalize with SHA-256
        normalized = hashlib.sha256(api_key.encode()).hexdigest()

        # Verify against Argon2 hash
        _hasher.verify(key_hash, normalized)
        return True
    except VerifyMismatchError:
        return False
    except Exception:
        return False


async def verify_api_key(db: AsyncSession, api_key: str) -> Optional[ApiKey]:
    """
    Verify an API key and return the ApiKey record if valid.

    This performs a lookup by prefix first (fast), then verifies the hash.

    Args:
        db: Database session
        api_key: The raw API key to verify

    Returns:
        ApiKey record if valid, None otherwise
    """
    # Extract prefix for lookup
    if not api_key.startswith(API_KEY_PREFIX):
        return None

    # Get the prefix portion (first 8 chars after mr2_)
    random_part = api_key[len(API_KEY_PREFIX):]
    key_prefix = f"{API_KEY_PREFIX}{random_part[:8]}"

    # Look up by prefix
    db_key = await crud.get_api_key_by_prefix(db, key_prefix)

    if not db_key:
        return None

    # Verify the full hash
    if _verify_key_hash(api_key, db_key.key_hash):
        return db_key

    return None


def generate_secure_token(length: int = 32) -> str:
    """Generate a secure random token."""
    return secrets.token_urlsafe(length)
