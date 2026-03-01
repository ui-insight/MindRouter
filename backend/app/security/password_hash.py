############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# password_hash.py: Argon2 password hashing utilities
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Password hashing utilities."""

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

# Use Argon2id with recommended parameters
_hasher = PasswordHasher(
    time_cost=3,  # Number of iterations
    memory_cost=65536,  # 64 MB
    parallelism=4,  # Number of parallel threads
    hash_len=32,  # Length of the hash
    salt_len=16,  # Length of the salt
)


def hash_password(password: str) -> str:
    """
    Hash a password using Argon2id.

    Args:
        password: The plaintext password

    Returns:
        Argon2id hash string (includes salt and parameters)
    """
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """
    Verify a password against a stored hash.

    Args:
        password: The plaintext password to verify
        password_hash: The stored Argon2id hash

    Returns:
        True if the password matches
    """
    try:
        _hasher.verify(password_hash, password)
        return True
    except VerifyMismatchError:
        return False
    except Exception:
        return False


def needs_rehash(password_hash: str) -> bool:
    """
    Check if a password hash needs to be rehashed.

    This is useful when upgrading hash parameters.

    Args:
        password_hash: The stored hash

    Returns:
        True if the hash should be regenerated
    """
    return _hasher.check_needs_rehash(password_hash)
