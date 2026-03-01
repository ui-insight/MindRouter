############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# artifacts.py: Artifact storage for uploaded files and images
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Artifact storage for uploaded files."""

import hashlib
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Optional, Tuple

import aiofiles

from backend.app.settings import get_settings
from backend.app.logging_config import get_logger

logger = get_logger(__name__)


class ArtifactStorage:
    """
    Handles storage of uploaded artifacts (images, documents).

    Storage layout:
    /artifacts/
      /YYYY/MM/DD/
        /<sha256_prefix>/
          /<full_sha256>_<uuid>.<ext>
    """

    def __init__(self):
        self._settings = get_settings()
        self._base_path = Path(self._settings.artifact_storage_path)

    async def initialize(self) -> None:
        """Initialize storage directory."""
        self._base_path.mkdir(parents=True, exist_ok=True)
        logger.info("artifact_storage_initialized", path=str(self._base_path))

    async def store(
        self,
        data: bytes,
        filename: str,
        content_type: str,
    ) -> Tuple[str, str, int]:
        """
        Store an artifact.

        Args:
            data: File content as bytes
            filename: Original filename
            content_type: MIME type

        Returns:
            Tuple of (storage_path, sha256_hash, size_bytes)
        """
        # Calculate hash
        sha256_hash = hashlib.sha256(data).hexdigest()

        # Generate path
        now = datetime.now(timezone.utc)
        date_path = now.strftime("%Y/%m/%d")
        hash_prefix = sha256_hash[:4]

        # Extract extension
        ext = Path(filename).suffix or self._guess_extension(content_type)

        # Generate unique filename
        unique_id = uuid.uuid4().hex[:8]
        stored_filename = f"{sha256_hash}_{unique_id}{ext}"

        # Full path
        dir_path = self._base_path / date_path / hash_prefix
        dir_path.mkdir(parents=True, exist_ok=True)

        file_path = dir_path / stored_filename
        relative_path = f"{date_path}/{hash_prefix}/{stored_filename}"

        # Write file
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(data)

        logger.info(
            "artifact_stored",
            path=relative_path,
            size=len(data),
            hash=sha256_hash[:16],
        )

        return relative_path, sha256_hash, len(data)

    async def retrieve(self, storage_path: str) -> Optional[bytes]:
        """
        Retrieve an artifact by storage path.

        Args:
            storage_path: Relative storage path

        Returns:
            File content as bytes, or None if not found
        """
        file_path = self._base_path / storage_path

        if not file_path.exists():
            return None

        async with aiofiles.open(file_path, "rb") as f:
            return await f.read()

    async def delete(self, storage_path: str) -> bool:
        """
        Delete an artifact.

        Args:
            storage_path: Relative storage path

        Returns:
            True if deleted, False if not found
        """
        file_path = self._base_path / storage_path

        if not file_path.exists():
            return False

        file_path.unlink()
        logger.info("artifact_deleted", path=storage_path)
        return True

    async def exists(self, storage_path: str) -> bool:
        """Check if an artifact exists."""
        file_path = self._base_path / storage_path
        return file_path.exists()

    async def get_size(self, storage_path: str) -> Optional[int]:
        """Get artifact size in bytes."""
        file_path = self._base_path / storage_path

        if not file_path.exists():
            return None

        return file_path.stat().st_size

    def _guess_extension(self, content_type: str) -> str:
        """Guess file extension from content type."""
        mapping = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "application/pdf": ".pdf",
            "text/plain": ".txt",
            "application/json": ".json",
        }
        return mapping.get(content_type, ".bin")


# Global instance
_artifact_storage: Optional[ArtifactStorage] = None


def get_artifact_storage() -> ArtifactStorage:
    """Get the global artifact storage instance."""
    global _artifact_storage
    if _artifact_storage is None:
        _artifact_storage = ArtifactStorage()
    return _artifact_storage
