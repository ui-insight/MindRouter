############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# mesh_store.py: On-disk layout for image-to-mesh jobs.
#
# Every file a mesh job owns lives in <root>/<user_id>/ under a
# server-generated name derived from the job uuid, so no stored
# path string ever reaches a file operation: the API, the runner,
# deletion and retention all rebuild paths from (root, user_id,
# job_uuid) and check containment under the root.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
#
############################################################

"""Paths for a mesh job's input image, GLB and preview PNG."""

import os
import re
from typing import List, Optional

from backend.app.core.pathsafe import resolve_under

# Input image types the API accepts, and the extension each is stored under.
INPUT_EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}

# Server-generated ids only: "mesh-" + 24 hex characters.
_JOB_UUID = re.compile(r"^mesh-[0-9a-f]{24}$")


class InvalidJobId(ValueError):
    """A job id that is not a server-generated mesh id."""


def _checked_uuid(job_uuid: str) -> str:
    if not isinstance(job_uuid, str) or not _JOB_UUID.match(job_uuid):
        raise InvalidJobId("not a mesh job id")
    return job_uuid


def user_dir(root: str, user_id: int, *, create: bool = False) -> str:
    """<root>/<user_id>, contained under root (optionally created)."""
    path = os.path.join(root, str(int(user_id)))
    if create:
        os.makedirs(path, exist_ok=True)
    resolve_under(root, path)
    return path


def _file(root: str, user_id: int, job_uuid: str, suffix: str, create: bool) -> str:
    out = os.path.join(user_dir(root, user_id, create=create), f"{_checked_uuid(job_uuid)}{suffix}")
    resolve_under(root, out)  # raises PathEscapeError if it escapes the root
    return out


def input_path(root: str, user_id: int, job_uuid: str, content_type: Optional[str], *, create: bool = False) -> str:
    """The uploaded image: <root>/<user_id>/<job_uuid>.input.<ext>."""
    ext = INPUT_EXTENSIONS.get((content_type or "").lower(), "png")
    return _file(root, user_id, job_uuid, f".input.{ext}", create)


def output_path(root: str, user_id: int, job_uuid: str, *, create: bool = False) -> str:
    """The generated model: <root>/<user_id>/<job_uuid>.glb."""
    return _file(root, user_id, job_uuid, ".glb", create)


def preview_path(root: str, user_id: int, job_uuid: str, *, create: bool = False) -> str:
    """The preview render: <root>/<user_id>/<job_uuid>.png."""
    return _file(root, user_id, job_uuid, ".png", create)


def job_files(root: str, user_id: int, job_uuid: str, content_type: Optional[str]) -> List[str]:
    """Every file a job may own (some may not exist yet)."""
    return [
        input_path(root, user_id, job_uuid, content_type),
        output_path(root, user_id, job_uuid),
        preview_path(root, user_id, job_uuid),
    ]


def remove_job_files(root: str, user_id: int, job_uuid: str, content_type: Optional[str]) -> int:
    """Best-effort unlink of a job's files. Returns how many were removed."""
    removed = 0
    try:
        paths = job_files(root, user_id, job_uuid, content_type)
    except (InvalidJobId, ValueError):
        return 0
    for path in paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                removed += 1
        except OSError:
            pass
    return removed
