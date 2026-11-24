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


# ── speed: decimation, texture gap filling, texture encoding ─────────────

def fast_decimate(vertices: Any, faces: Any, target_faces: int) -> Tuple[np.ndarray, np.ndarray]:
    """Quadric edge-collapse decimation with fast-simplification (MIT; Sven
    Forstmann's Fast-Quadric-Mesh-Simplification). Measured on Hunyuan3D
    meshes (H200 node CPU): 868k -> 40k faces in 0.65 s, where pymeshlab with
    upstream's settings took 38 s, with equal-looking geometry. Returns the
    input unchanged when it is already at or below the target."""
    v = np.asarray(vertices)
    f = np.asarray(faces)
    if target_faces <= 0 or len(f) <= target_faces:
        return v, f
    import fast_simplification  # installed in the Hunyuan image only

    v2, f2 = fast_simplification.simplify(
        v.astype(np.float32), f.astype(np.int64), target_reduction=1.0 - target_faces / len(f), agg=7,
    )
    return np.asarray(v2), np.asarray(f2)


def _sum2x2(a: np.ndarray) -> np.ndarray:
    """2x2 box sums (zero-padding odd edges): one pull level."""
    h, w = a.shape[:2]
    if h % 2 or w % 2:
        a = np.pad(a, ((0, h % 2), (0, w % 2)) + ((0, 0),) * (a.ndim - 2))
    return a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2, *a.shape[2:]).sum(axis=(1, 3))


def pull_push_fill(tex: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Fill unknown texels from coarser averages of the known ones (pull-push),
    the usual way to pad texture atlases so bilinear filtering and mipmaps
    never sample empty space. Known texels come back unchanged. CPU
    reference; pull_push_fill_gpu is the fast path.

    tex: (H, W, C) uint8; known: (H, W) bool."""
    tex = np.asarray(tex)
    known = np.asarray(known, dtype=bool)
    if known.all() or not known.any():
        return tex.copy()
    color = tex.astype(np.float32) * known[..., None]
    weight = known.astype(np.float32)[..., None]
    levels = [(color, weight)]
    while max(levels[-1][1].shape[:2]) > 1:
        c, w = levels[-1]
        levels.append((_sum2x2(c), _sum2x2(w)))
    c, w = levels[-1]
    fill = c / np.maximum(w, 1e-8)
    for c, w in reversed(levels[:-1]):
        up = np.repeat(np.repeat(fill, 2, axis=0), 2, axis=1)[: c.shape[0], : c.shape[1]]
        fill = np.where(w > 0, c / np.maximum(w, 1e-8), up)
    out = np.clip(np.rint(fill), 0, 255).astype(np.uint8)
    out[known] = tex[known]
    return out


def pull_push_fill_gpu(tex: np.ndarray, known: np.ndarray) -> np.ndarray:  # pragma: no cover - GPU path
    """pull_push_fill on the GPU with bilinear push (about 0.02 s for a 4096
    atlas on an H200, where upstream's OpenCV Navier-Stokes fill took 5-8 s).
    Falls back to the CPU reference without torch/CUDA."""
    try:
        import torch
        import torch.nn.functional as F
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA")
    except Exception:
        return pull_push_fill(tex, known)
    t = torch.from_numpy(np.ascontiguousarray(tex)).cuda().float().permute(2, 0, 1)[None]
    k = torch.from_numpy(np.ascontiguousarray(known, dtype=np.float32)).cuda()[None, None]
    levels = [(t * k, k)]
    while max(levels[-1][1].shape[-2:]) > 1:
        c, w = levels[-1]
        levels.append((F.avg_pool2d(c, 2, ceil_mode=True), F.avg_pool2d(w, 2, ceil_mode=True)))
    c, w = levels[-1]
    fill = c / w.clamp_min(1e-8)
    for c, w in reversed(levels[:-1]):
        up = F.interpolate(fill, size=c.shape[-2:], mode="bilinear", align_corners=False)
        fill = torch.where(w > 0, c / w.clamp_min(1e-8), up)
    out = fill[0].permute(1, 2, 0).clamp(0, 255).round().byte().cpu().numpy()
    out[np.asarray(known, dtype=bool)] = np.asarray(tex)[np.asarray(known, dtype=bool)]
    return out


# Texture slots that may be stored as JPEG (normal maps stay PNG: JPEG
# artefacts show up as lighting noise).
JPEG_SLOTS = ("baseColorTexture", "metallicRoughnessTexture", "occlusionTexture", "emissiveTexture")


def jpeg_textures(mesh: Any, quality: int = 92) -> int:
    """Re-wrap a trimesh mesh's (or scene's) opaque PBR textures as JPEG
    images, so trimesh's glTF export writes them as JPEG (it keeps a PIL
    image's JPEG format and encodes everything else as PNG). PNG encoding was
    5-10% of a TRELLIS-family job. The base colour of a material that blends
    or masks stays PNG (an OPAQUE material's alpha is ignored by glTF viewers,
    and TRELLIS.2 writes an unused alpha channel). 4:4:4 chroma, so the
    metallic/roughness channels are not subsampled. Returns how many textures
    were converted."""
    import io

    from PIL import Image

    geoms = list(mesh.geometry.values()) if hasattr(mesh, "geometry") else [mesh]
    converted = 0
    for geom in geoms:
        material = getattr(getattr(geom, "visual", None), "material", None)
        if material is None:
            continue
        alpha_mode = str(getattr(material, "alphaMode", None) or "OPAQUE").upper()
        for slot in JPEG_SLOTS:
            img = getattr(material, slot, None)
            if img is None or not hasattr(img, "save") or getattr(img, "format", None) == "JPEG":
                continue
            # glTF ignores base-colour alpha unless the material blends or masks
            # (alphaMode defaults to OPAQUE), so only those keep a PNG with alpha.
            if slot == "baseColorTexture" and alpha_mode in ("BLEND", "MASK"):
                continue
            buf = io.BytesIO()
            img.convert("RGB").save(buf, "JPEG", quality=quality, subsampling=0)
            data = buf.getvalue()
            jpg = Image.open(io.BytesIO(data))
            jpg.load()
            jpg.save = _jpeg_passthrough(data, jpg.save)
            setattr(material, slot, jpg)
            converted += 1
    return converted


def _jpeg_passthrough(data: bytes, original_save: Callable) -> Callable:
    """img.save replacement: a JPEG save writes the bytes we already encoded
    (trimesh would otherwise re-encode at PIL's default quality 75 with
    chroma subsampling); any other format falls through to PIL."""
    def save(fp, format=None, **params):
        if str(format or "").upper() in ("JPEG", "JPG") and not params:
            if hasattr(fp, "write"):
                fp.write(data)
            else:
                with open(fp, "wb") as fh:
                    fh.write(data)
            return None
        return original_save(fp, format=format, **params)
    return save


def release_cuda_memory() -> None:  # pragma: no cover - GPU only
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
