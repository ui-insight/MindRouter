############################################################
#
# mindrouter mesh-worker - async image-to-3D mesh service
#
# engines/common.py: Helpers shared by the real (GPU) engines. Nothing here
#     imports torch/trimesh at module level — objects are duck-typed so the
#     helpers are unit-testable on mock deps.
#
# Luke Sheneman — University of Idaho RCDS — sheneman@uidaho.edu
#
############################################################

"""Shared helpers for the GPU engines."""

import contextlib
import logging
from typing import Any, Callable, Dict, Iterator, Tuple

import numpy as np

from engines.base import StageTracker
from preview import face_colors_from_texture, face_colors_from_vertices, render_preview

log = logging.getLogger(__name__)

# Background-removal weights we will never load: briaai/RMBG-2.0 (and 1.4) are
# CC BY-NC 4.0 — non-commercial — and TRELLIS.2 / Pixal3D's pipeline.json
# default to RMBG-2.0.
FORBIDDEN_REMBG_MARKERS = ("rmbg",)


def check_rembg_allowed(name: str) -> None:
    if any(marker in (name or "").lower() for marker in FORBIDDEN_REMBG_MARKERS):
        raise ValueError(
            f"refusing background-removal model {name!r}: briaai/RMBG-* is CC BY-NC "
            "(non-commercial). Use ZhengPeng7/BiRefNet (MIT) via MESH_WORKER_BIREFNET."
        )


@contextlib.contextmanager
def patched_attr(owner: Any, name: str, replacement: Any) -> Iterator[Any]:
    """Temporarily replace owner.name; yields the original."""
    original = getattr(owner, name)
    setattr(owner, name, replacement)
    try:
        yield original
    finally:
        setattr(owner, name, original)


@contextlib.contextmanager
def stage_hooks(target: Any, tracker: StageTracker, hooks: Dict[str, Tuple[str, float]]) -> Iterator[None]:
    """Wrap target's bound methods so each call is a stage boundary: cancel
    check + stage/progress report, then the original method. Installed as
    INSTANCE attributes (the upstream run() calls self.<method>(...)), and
    removed afterwards so the class methods show through again."""
    installed = []
    for method_name, (stage, progress) in hooks.items():
        original = getattr(target, method_name, None)
        if original is None:
            continue

        def make(orig: Callable, st: str, pr: float) -> Callable:
            def hooked(*args, **kwargs):
                tracker.enter(st, pr)
                return orig(*args, **kwargs)
            return hooked

        setattr(target, method_name, make(original, stage, progress))
        installed.append(method_name)
    try:
        yield
    finally:
        for method_name in installed:
            try:
                delattr(target, method_name)
            except AttributeError:
                pass


def gltf_axes(vertices: np.ndarray) -> np.ndarray:
    """The axis change o_voxel.postprocess.to_glb applies (Z-up -> glTF Y-up):
    (x, y, z) -> (x, z, -y). Applied to shape-only exports so they line up
    with textured ones."""
    v = np.asarray(vertices, dtype=np.float32)
    out = np.empty_like(v)
    out[:, 0] = v[:, 0]
    out[:, 1] = v[:, 2]
    out[:, 2] = -v[:, 1]
    return out


def trimesh_preview_png(mesh: Any) -> bytes:
    """512x512 preview of a trimesh.Trimesh via the shared software renderer.
    Colours: base-colour texture at UV centroids (trimesh stores OpenGL UVs,
    origin bottom-left), else vertex colours, else grey."""
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    colors = None
    visual = getattr(mesh, "visual", None)
    try:
        kind = getattr(visual, "kind", None)
        if kind == "texture":
            uv = getattr(visual, "uv", None)
            material = getattr(visual, "material", None)
            image = getattr(material, "baseColorTexture", None)
            if image is None:
                image = getattr(material, "image", None)
            if uv is not None and image is not None and len(uv) == len(vertices):
                uv = np.array(uv, dtype=np.float64)
                uv[:, 1] = 1.0 - uv[:, 1]
                colors = face_colors_from_texture(faces, uv, image)
        elif kind == "vertex":
            vc = np.asarray(visual.vertex_colors)[:, :3].astype(np.float32) / 255.0
            colors = face_colors_from_vertices(faces, vc)
    except Exception:  # a preview must never fail the job
        log.warning("preview colour extraction failed; rendering grey", exc_info=True)
        colors = None
    return render_preview(vertices, faces, colors)


def release_cuda_memory() -> None:  # pragma: no cover - GPU only
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
