"""Unit tests for services/mesh_store.py: the on-disk layout of mesh jobs.

mesh_store imports only os/re and backend.app.core.pathsafe, so both are
spec-loaded directly (no backend package import chain); sys.modules is
restored afterwards.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_APP = Path(__file__).resolve().parents[2]
_MISSING = object()
_prev = sys.modules.get("backend.app.core.pathsafe", _MISSING)

_ps_spec = importlib.util.spec_from_file_location("backend.app.core.pathsafe", _APP / "core" / "pathsafe.py")
_pathsafe = importlib.util.module_from_spec(_ps_spec)
_ps_spec.loader.exec_module(_pathsafe)
sys.modules["backend.app.core.pathsafe"] = _pathsafe
_spec = importlib.util.spec_from_file_location("mesh_store_under_test", _APP / "services" / "mesh_store.py")
store = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(store)
if _prev is _MISSING:
    sys.modules.pop("backend.app.core.pathsafe", None)
else:
    sys.modules["backend.app.core.pathsafe"] = _prev

UUID = "mesh-0123456789abcdef01234567"


def test_layout_is_user_sharded_and_named_from_the_job_id(tmp_path):
    root = str(tmp_path)
    assert store.input_path(root, 7, UUID, "image/jpeg") == os.path.join(root, "7", f"{UUID}.input.jpg")
    assert store.input_path(root, 7, UUID, "image/webp").endswith(".input.webp")
    # Unknown or missing types fall back to png (the API only stores sniffed types).
    assert store.input_path(root, 7, UUID, None).endswith(".input.png")
    assert store.output_path(root, 7, UUID) == os.path.join(root, "7", f"{UUID}.glb")
    assert store.preview_path(root, 7, UUID) == os.path.join(root, "7", f"{UUID}.png")


def test_create_makes_the_user_directory(tmp_path):
    path = store.output_path(str(tmp_path), 42, UUID, create=True)
    assert os.path.isdir(os.path.dirname(path))


@pytest.mark.parametrize("bad", [
    "../etc/passwd", "mesh-../../x", "mesh-0123", "vid-0123456789abcdef01234567",
    "mesh-0123456789ABCDEF01234567", "", None, f"{UUID}/x",
])
def test_only_server_generated_ids_are_accepted(tmp_path, bad):
    with pytest.raises(ValueError):
        store.output_path(str(tmp_path), 1, bad)


def test_user_id_must_be_an_integer(tmp_path):
    with pytest.raises((TypeError, ValueError)):
        store.output_path(str(tmp_path), "1/../2", UUID)


def test_remove_job_files_removes_only_that_jobs_files(tmp_path):
    root = str(tmp_path)
    other = "mesh-ffffffffffffffffffffffff"
    paths = [
        store.input_path(root, 3, UUID, "image/png", create=True),
        store.output_path(root, 3, UUID),
        store.preview_path(root, 3, UUID),
        store.output_path(root, 3, other),
    ]
    for p in paths:
        Path(p).write_bytes(b"x")
    assert store.remove_job_files(root, 3, UUID, "image/png") == 3
    assert [os.path.exists(p) for p in paths] == [False, False, False, True]
    # Idempotent, and a bad id removes nothing instead of raising.
    assert store.remove_job_files(root, 3, UUID, "image/png") == 0
    assert store.remove_job_files(root, 3, "../x", "image/png") == 0


def test_a_symlinked_user_dir_cannot_escape_the_root(tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    os.symlink(outside, root / "9")
    with pytest.raises(_pathsafe.PathEscapeError):
        store.output_path(str(root), 9, UUID)
